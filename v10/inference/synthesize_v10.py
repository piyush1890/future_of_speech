"""
v10 inference cascade:
  text → V10StylePlanner → per-phoneme style codes
       → V10Renderer (frame-rate AR with EOP) → (T, K) frame RVQ tokens
       → V10Tokenizer.tokens_to_frames → (T, 14) articulator features
       → SPARC → audio
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from g2p_en import G2p

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v10.models.v10_tokenizer import V10Tokenizer
from v10.models.v10_renderer import V10Renderer
from v10.models.v10_planner import V10StylePlanner
from models.phoneme_vocab import PhonemeVocab


DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"
EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


def text_to_phonemes(text: str, g2p) -> list:
    parts = re.split(r"(?<=[.!?,;:—–])\s+", text.strip())
    phs = ["<sil>"]
    for si, part in enumerate(parts):
        if not part.strip():
            continue
        if si > 0:
            phs.append("<sil>")
        for p in g2p(part):
            if p and p[0].isalpha() and p[0].isupper():
                phs.append(p)
    phs.append("<sil>")
    return phs


def build_knobs(emotion: str, intensity: float):
    eid = EMOTION_TO_ID.get(emotion.lower(), 0)
    one_hot = [0.0] * 5; one_hot[eid] = 1.0
    return np.asarray(one_hot + [float(intensity)], dtype=np.float32)


def sample_logits(logits, temperature: float, top_k: int):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / max(1e-6, temperature)
    if top_k > 0:
        v, _ = torch.topk(logits, top_k, dim=-1)
        thr = v[..., -1:].expand_as(logits)
        logits = torch.where(logits < thr, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def renderer_generate(
    renderer: V10Renderer,
    phoneme_ids: torch.Tensor,         # (B, N+2)
    style_codes: torch.Tensor,         # (B, N+2)
    spk_emb: torch.Tensor,             # (B, 64)
    knobs: torch.Tensor,               # (B, knob_dim) or empty
    phoneme_mask: torch.Tensor,        # (B, N+2)
    max_frames: int = 600,
    temperature: float = 1.0,
    top_k: int = 0,
    cfg_scale: float = 1.0,
    eop_threshold: float = 0.5,
):
    """Frame-rate AR loop with EOP-driven phoneme advancement.

    Returns dict with frame_codes (B, T, K), eop (B, T), frame_to_enc_pos (B, T), n_frames.
    """
    B, N = phoneme_ids.shape
    device = phoneme_ids.device
    K = renderer.K

    enc_cond = renderer.encode(phoneme_ids, style_codes, spk_emb, knobs, phoneme_mask,
                               force_drop_knobs=False)
    enc_un = renderer.encode(phoneme_ids, style_codes, spk_emb, knobs, phoneme_mask,
                             force_drop_knobs=True) if cfg_scale != 1.0 else None

    ph_idx = torch.ones(B, dtype=torch.long, device=device)        # current phoneme; first body = 1
    gen_codes = torch.zeros(B, max_frames, K, dtype=torch.long, device=device)
    gen_eop = torch.zeros(B, max_frames, dtype=torch.float, device=device)
    gen_pos = torch.ones(B, max_frames, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    # Real EOS position per row = (count of valid encoder positions) - 1.
    # phoneme_mask is True at BOS/body/EOS, False at right-padding.
    last_eos_pos = phoneme_mask.long().sum(dim=1) - 1              # (B,)

    f = 0
    while f < max_frames and not done.all():
        gen_pos[:, f] = ph_idx

        T = f + 1
        codes_so_far = gen_codes[:, :T, :]
        pos_so_far = gen_pos[:, :T]
        dec_inp = renderer._make_decoder_input(codes_so_far, pos_so_far)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
        h_full = renderer.decoder(
            dec_inp, enc_cond, tgt_mask=causal,
            memory_key_padding_mask=~phoneme_mask,
        )
        h_f = h_full[:, f, :]
        h_un_f = None
        if enc_un is not None:
            h_un_full = renderer.decoder(
                dec_inp, enc_un, tgt_mask=causal,
                memory_key_padding_mask=~phoneme_mask,
            )
            h_un_f = h_un_full[:, f, :]

        # Sample K RVQ tokens hierarchically
        for k in range(K):
            cl = renderer.heads.step_logits_one(h_f, k, gen_codes[:, f, :k])
            if h_un_f is not None:
                ul = renderer.heads.step_logits_one(h_un_f, k, gen_codes[:, f, :k])
                cl = ul + cfg_scale * (cl - ul)
            gen_codes[:, f, k] = sample_logits(cl, temperature, top_k)

        # Sample EOP
        eop_logit = renderer.eop_head(h_f).squeeze(-1)
        if h_un_f is not None:
            eop_un = renderer.eop_head(h_un_f).squeeze(-1)
            eop_logit = eop_un + cfg_scale * (eop_logit - eop_un)
        eop_prob = torch.sigmoid(eop_logit)
        eop_step = (eop_prob > eop_threshold).float()
        gen_eop[:, f] = eop_step

        advance = (eop_step > 0.5) & ~done
        ph_idx = ph_idx + advance.long()
        # Done when ph_idx has advanced PAST the last body phoneme (= reached EOS).
        done = done | (ph_idx >= last_eos_pos)                     # last_eos_pos is per-row
        f += 1

    return {
        "frame_codes": gen_codes[:, :f, :],
        "eop": gen_eop[:, :f],
        "frame_to_enc_pos": gen_pos[:, :f],
        "n_frames": f,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str)
    ap.add_argument("--tokenizer-checkpoint", default="v10/checkpoints/tokenizer/best.pt")
    ap.add_argument("--stage1-checkpoint", default="v10/checkpoints/stage1_renderer/best.pt")
    ap.add_argument("--stage2-checkpoint", default="v10/checkpoints/stage2_planner/best.pt")
    ap.add_argument("--vocab", default="data/processed_merged_v3/vocab_mfa.json")
    ap.add_argument("--norm-stats", default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--speaker-emb", default=None)
    ap.add_argument("--emotion", default="neutral", choices=list(EMOTION_TO_ID.keys()))
    ap.add_argument("--intensity", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--planner-temperature", type=float, default=1.0)
    ap.add_argument("--planner-cfg-scale", type=float, default=1.0)
    ap.add_argument("--eop-threshold", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Native:tokenized rate ratio (must match training). "
                         "Default 1 = no upsample (native SPARC = 50 Hz).")
    ap.add_argument("--device", default="mps",
                    help="Device for transformer + tokenizer ops. SPARC vocoder "
                         "always runs on CPU regardless.")
    ap.add_argument("--output", "-o", default="v10/outputs/v10.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Tokenizer
    tc = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    ta = tc["args"]
    print(f"Tokenizer: epoch={tc['epoch']} val_mse={tc.get('val_mse', float('nan')):.4f}")
    tokenizer = V10Tokenizer(
        d_model=ta.get("d_model", 256),
        num_encoder_layers=ta.get("enc_layers", 4),
        num_decoder_layers=ta.get("dec_layers", 4),
        codebook_size=ta.get("codebook_size", 1024),
        num_quantizers=ta.get("num_quantizers", 4),
        max_frames=ta.get("max_frames", 800) + 16,
    ).to(device)
    tokenizer.load_state_dict(tc["model"]); tokenizer.eval()

    # Stage 1: renderer (style encoder also lives in this checkpoint but we don't run it)
    s1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    s1a = s1["args"]
    print(f"Stage1: epoch={s1['epoch']} val_ce={s1.get('val_ce', float('nan')):.4f}")
    renderer = V10Renderer(
        codebook_size=s1a.get("codebook_size", 1024),
        num_quantizers=s1a.get("num_quantizers", 4),
        style_codebook_size=s1a.get("style_codebook_size", 64),
        d_model=s1a.get("d_model", 256),
        num_encoder_layers=s1a.get("enc_layers", 4),
        num_decoder_layers=s1a.get("dec_layers", 6),
        knob_dim=6 if s1a.get("knob_source", "emotion") == "emotion" else 0,
        knob_dropout=0.0,
        max_phonemes=s1a.get("max_phonemes", 200) + 4,
        max_frames=s1a.get("max_frames", 800) + 16,
    ).to(device)
    renderer.load_state_dict(s1["renderer"]); renderer.eval()

    # Stage 2: planner
    s2 = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
    s2a = s2["args"]
    print(f"Stage2: epoch={s2['epoch']} val_ce={s2.get('val_ce', float('nan')):.4f}")
    planner = V10StylePlanner(
        style_codebook_size=s2a.get("style_codebook_size", 64),
        d_model=s2a.get("d_model", 192),
        num_encoder_layers=s2a.get("num_layers", 4),
        num_decoder_layers=s2a.get("num_layers", 4),
        d_ff=s2a.get("d_ff", 768),
        knob_dim=s2.get("knob_dim", 6),
        knob_dropout=0.0,
        max_phonemes=s2a.get("max_phonemes", 200),
    ).to(device)
    planner.load_state_dict(s2["model"]); planner.eval()

    # Inputs
    vocab = PhonemeVocab(args.vocab)
    g2p = G2p()
    phs = text_to_phonemes(args.text, g2p)
    phoneme_ids = torch.tensor([vocab.encode(phs, add_bos_eos=True)], dtype=torch.long,
                                device=device)
    N = phoneme_ids.shape[1]
    phoneme_mask = phoneme_ids != 0
    print(f"phonemes: {N-2} body  ({' '.join(phs[:20])}{'...' if len(phs) > 20 else ''})")

    if args.speaker_emb:
        spk = np.load(args.speaker_emb).astype(np.float32)
    else:
        spk = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
        print("(default speaker)")
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)
    knobs_np = build_knobs(args.emotion, args.intensity)
    knobs_t = torch.from_numpy(knobs_np).unsqueeze(0).to(device)

    # Stage 2: planner → style codes
    style_codes = planner.generate(
        phoneme_ids, spk_t, knobs_t, phoneme_mask,
        temperature=args.planner_temperature, top_k=args.top_k,
        cfg_scale=args.planner_cfg_scale,
    )                                                              # (1, N)
    style_pad = s1a.get("style_codebook_size", 64)
    style_codes[:, 0] = style_pad
    style_codes[:, -1] = style_pad
    print(f"style codes (first 10 body): {style_codes[0, 1:11].tolist()}")

    # Stage 1: renderer → frame codes
    gen = renderer_generate(
        renderer, phoneme_ids, style_codes, spk_t, knobs_t, phoneme_mask,
        max_frames=args.max_frames, temperature=args.temperature,
        top_k=args.top_k, cfg_scale=args.cfg_scale,
        eop_threshold=args.eop_threshold,
    )
    n_frames = gen["n_frames"]
    token_rate = 100 // args.frame_stride
    print(f"generated {n_frames} frames  ({n_frames/token_rate:.2f}s @ {token_rate}Hz)  "
          f"EOPs fired: {int(gen['eop'].sum().item())}")
    print(f"first 5 frame_to_enc_pos: {gen['frame_to_enc_pos'][0, :5].tolist()}")

    # Tokenizer: tokens → articulator frames (at 50 Hz)
    frame_mask = torch.ones(1, n_frames, dtype=torch.bool, device=device)
    feats_norm_50 = tokenizer.tokens_to_frames(gen["frame_codes"], frame_mask)  # (1, T, 14)
    feats_norm_50 = feats_norm_50[0].cpu().numpy()                              # (T, 14) @ 50 Hz

    # Move to CPU for downstream numpy + SPARC.
    # Upsample 2× to 100 Hz for SPARC vocoder (linear interpolation per channel).
    T_50 = feats_norm_50.shape[0]
    T_100 = T_50 * args.frame_stride
    old_t = np.arange(T_50, dtype=np.float64)
    new_t = np.linspace(0, T_50 - 1, T_100, dtype=np.float64) if T_50 > 1 else np.zeros(T_100)
    feats_norm_100 = np.stack(
        [np.interp(new_t, old_t, feats_norm_50[:, d]) for d in range(feats_norm_50.shape[1])],
        axis=1,
    ).astype(np.float32)

    # Denormalize
    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)
    feats = feats_norm_100 * feat_std + feat_mean
    feats[:, 12] = np.exp(feats[:, 12]) - 1.0
    feats[feats[:, 12] < 30, 12] = 0.0
    print(f"frames @ 100 Hz: {feats.shape[0]}  duration: {feats.shape[0]/100:.2f}s  "
          f"(50 Hz tokens: {T_50})")

    # Vocoder
    print("loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    wav = sparc.decode(feats[:, :12], feats[:, 12], feats[:, 13], spk)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().squeeze().cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, sparc.sr)
    print(f"saved {args.output} ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
