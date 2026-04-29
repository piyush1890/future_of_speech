"""
v9 two-stage synthesis:
  text → V9StylePlanner → per-phoneme style code IDs
       → V9Renderer (with style codes as input) → per-phoneme RVQ tokens + durations
       → PhonemeRVQTokenizer.decode_indices_batch → frames
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

from v9.models.phoneme_rvq import PhonemeRVQTokenizer
from v9.models.style_encoder import V9PerPhonemeStyleEncoder, V9StyleCodebook
from v9.models.v9_renderer import V9Renderer
from v9.models.v9_style_planner import V9StylePlanner
from models.phoneme_vocab import PhonemeVocab


DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"
EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


def text_to_phonemes(text: str, g2p) -> list:
    parts = re.split(r"(?<=[.!?,;:—–])\s+", text.strip())
    phs = ["<sil>"]
    for si, part in enumerate(parts):
        if not part.strip(): continue
        if si > 0: phs.append("<sil>")
        for p in g2p(part):
            if p and p[0].isalpha() and p[0].isupper():
                phs.append(p)
    phs.append("<sil>")
    return phs


def build_knobs(emotion: str, intensity: float):
    eid = EMOTION_TO_ID.get(emotion.lower(), 0)
    one_hot = [0.0] * 5; one_hot[eid] = 1.0
    return np.asarray(one_hot + [float(intensity)], dtype=np.float32)


def sample_logits(logits, temperature, top_k):
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
def renderer_generate(renderer: V9Renderer, phoneme_ids, spk_emb, knobs, phoneme_mask,
                      style_codes, temperature=1.0, top_k=0, cfg_scale=1.0):
    """Half-phoneme rate AR over 2N decoder positions.

    Position 2i samples start[i], position 2i+1 samples end[i]. end[i] sees
    start[i] processed through the full decoder transformer (deep, symmetric
    to how start sees previous halves)."""
    B, N = phoneme_ids.shape
    device = phoneme_ids.device
    K = renderer.K

    enc_cond = renderer.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, style_codes,
                               force_drop_knobs=False)
    enc_un   = renderer.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, style_codes,
                               force_drop_knobs=True) if cfg_scale != 1.0 else None

    gen_start = torch.zeros(B, N, K, dtype=torch.long, device=device)
    gen_end   = torch.zeros(B, N, K, dtype=torch.long, device=device)
    gen_logdur = torch.zeros(B, N, device=device)

    L = 2 * N
    dec_pad_mask = (~phoneme_mask).repeat_interleave(2, dim=1)

    for d in range(L):
        # Build current decoder input from generated tokens so far
        dec_inp = renderer._make_decoder_input(gen_start, gen_end)
        dec_inp = renderer.decoder_pe(dec_inp)
        causal = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        h_full = renderer.decoder(
            dec_inp, enc_cond, tgt_mask=causal,
            tgt_key_padding_mask=dec_pad_mask,
            memory_key_padding_mask=~phoneme_mask,
        )
        h_d = h_full[:, d, :]
        h_un_d = None
        if enc_un is not None:
            h_un_full = renderer.decoder(
                dec_inp, enc_un, tgt_mask=causal,
                tgt_key_padding_mask=dec_pad_mask,
                memory_key_padding_mask=~phoneme_mask,
            )
            h_un_d = h_un_full[:, d, :]

        if d % 2 == 0:                                                       # start position
            i = d // 2
            for k in range(K):
                cl = renderer.start_heads.step_logits_one(h_d, k, gen_start[:, i, :k])
                if h_un_d is not None:
                    ul = renderer.start_heads.step_logits_one(h_un_d, k, gen_start[:, i, :k])
                    cl = ul + cfg_scale * (cl - ul)
                gen_start[:, i, k] = sample_logits(cl, temperature, top_k)
            # Duration is read off from start-position hidden state
            gen_logdur[:, i] = renderer.duration_head(h_d).squeeze(-1)
        else:                                                                # end position
            i = (d - 1) // 2
            for k in range(K):
                cl = renderer.end_heads.step_logits_one(h_d, k, gen_end[:, i, :k])
                if h_un_d is not None:
                    ul = renderer.end_heads.step_logits_one(h_un_d, k, gen_end[:, i, :k])
                    cl = ul + cfg_scale * (cl - ul)
                gen_end[:, i, k] = sample_logits(cl, temperature, top_k)

    return gen_start, gen_end, gen_logdur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str)
    ap.add_argument("--tokenizer-checkpoint", default="v9/checkpoints/phoneme_rvq/best.pt")
    ap.add_argument("--stage1-checkpoint",    default="v9/checkpoints/stage1_renderer/best.pt")
    ap.add_argument("--stage2-checkpoint",    default="v9/checkpoints/stage2_planner/best.pt")
    ap.add_argument("--vocab",                default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--norm-stats",           default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--speaker-emb",          default=None)
    ap.add_argument("--emotion",              default="neutral",
                    choices=list(EMOTION_TO_ID.keys()))
    ap.add_argument("--intensity",            type=float, default=0.5)
    ap.add_argument("--temperature",          type=float, default=1.0)
    ap.add_argument("--top-k",                type=int,   default=0)
    ap.add_argument("--cfg-scale",            type=float, default=1.0)
    ap.add_argument("--planner-temperature",  type=float, default=1.0)
    ap.add_argument("--planner-cfg-scale",    type=float, default=1.0)
    ap.add_argument("--duration-scale",       type=float, default=1.0)
    ap.add_argument("--blend-boundaries",     action="store_true", default=True,
                    help="Average overlapping boundary predictions across adjacent phonemes "
                         "(secondary refinement on top of training-time boundary loss).")
    ap.add_argument("--no-blend-boundaries",  dest="blend_boundaries", action="store_false")
    ap.add_argument("--device",               default="cpu")
    ap.add_argument("--output", "-o",         default="v9/outputs/v9_2stage.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Tokenizer
    tc = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    ta = tc["args"]
    print(f"Tokenizer: epoch={tc['epoch']} val={tc['val_loss']:.4f}")
    tokenizer = PhonemeRVQTokenizer(
        vocab_size=ta["vocab_size"], input_dim=14,
        latent_dim=ta["latent_dim"], hidden_dim=ta["hidden_dim"],
        codebook_size=ta["codebook_size"], num_quantizers=ta["num_quantizers"],
        decoder_d_model=ta["decoder_d_model"], decoder_nhead=ta["decoder_nhead"],
        decoder_layers=ta["decoder_layers"],
        commitment_weight=ta["commit_weight"], ema_decay=ta["ema_decay"],
    ).to(device)
    tokenizer.load_state_dict(tc["model"]); tokenizer.eval()

    # Stage 1: renderer (+ style encoder + codebook embedded inside checkpoint)
    s1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    s1a = s1["args"]
    print(f"Stage1: epoch={s1['epoch']} val={s1['val_loss']:.4f}")
    renderer = V9Renderer(
        vocab_size=s1a["vocab_size"], codebook_size=s1a["codebook_size"],
        num_quantizers=s1a["num_quantizers"],
        style_codebook_size=s1a["style_codebook_size"],
        d_model=s1a["d_model"], nhead=s1a["nhead"],
        num_encoder_layers=s1a["num_layers"], num_decoder_layers=s1a["num_layers"],
        d_ff=s1a["d_ff"], dropout=0.0, speaker_emb_dim=64,
        knob_dim=s1.get("knob_dim", 6), knob_dropout=0.0,
        max_phonemes=s1a["max_phonemes"],
    ).to(device)
    renderer.load_state_dict(s1["renderer"]); renderer.eval()

    # Stage 2: planner
    s2 = torch.load(args.stage2_checkpoint, map_location=device, weights_only=False)
    s2a = s2["args"]
    print(f"Stage2: epoch={s2['epoch']} val={s2['val_loss']:.4f}")
    planner = V9StylePlanner(
        vocab_size=s2a["vocab_size"], style_codebook_size=s2a["style_codebook_size"],
        d_model=s2a["d_model"], nhead=s2a["nhead"],
        num_encoder_layers=s2a["num_layers"], num_decoder_layers=s2a["num_layers"],
        d_ff=s2a["d_ff"], dropout=0.0, speaker_emb_dim=64,
        knob_dim=s2.get("knob_dim", 6), knob_dropout=0.0,
        max_phonemes=s2a["max_phonemes"],
    ).to(device)
    planner.load_state_dict(s2["model"]); planner.eval()

    # ── Inputs ──
    vocab = PhonemeVocab(args.vocab)
    g2p = G2p()
    phs = text_to_phonemes(args.text, g2p)
    phoneme_ids = torch.tensor([vocab.encode(phs, add_bos_eos=True)], dtype=torch.long, device=device)
    N = phoneme_ids.shape[1]
    phoneme_mask = phoneme_ids != 0

    if args.speaker_emb:
        spk = np.load(args.speaker_emb).astype(np.float32)
    else:
        spk = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
        print("(default speaker)")
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)
    knobs_np = build_knobs(args.emotion, args.intensity)
    knobs_t = torch.from_numpy(knobs_np).unsqueeze(0).to(device)

    # ── Stage 2: planner generates style codes ──
    style_codes = planner.generate(
        phoneme_ids, spk_t, knobs_t, phoneme_mask,
        temperature=args.planner_temperature, top_k=args.top_k,
        cfg_scale=args.planner_cfg_scale,
    )                                                                    # (1, N)
    # Override BOS/EOS positions to PAD_CODE (= style_codebook_size, matches Renderer)
    style_pad = s1a["style_codebook_size"]
    style_codes_full = style_codes.clone()
    style_codes_full[:, 0] = style_pad; style_codes_full[:, -1] = style_pad
    print(f"sampled style codes (first 10 body): {style_codes_full[0, 1:11].tolist()}")

    # ── Stage 1: renderer generates RVQ tokens ──
    gen_start, gen_end, gen_logdur = renderer_generate(
        renderer, phoneme_ids, spk_t, knobs_t, phoneme_mask, style_codes_full,
        temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale,
    )

    body_start = gen_start[0, 1:N-1, :]
    body_end   = gen_end[0,   1:N-1, :]
    body_logdur = gen_logdur[0, 1:N-1]
    durations = (torch.exp(body_logdur) * args.duration_scale).round().clamp(min=1).long()
    body_ph_ids = phoneme_ids[0, 1:N-1]

    with torch.no_grad():
        # Tokenizer was trained on EXTENDED blocks (body + 1 ctx each side).
        # Pass durations+2 so the decoder produces L+2 frames per phoneme.
        # Position 0 = predicted prev_ctx, positions 1..L = body, position L+1 = next_ctx.
        ext_durations = durations + 2
        decoded_blocks = tokenizer.decode_indices_batch(
            body_start, body_end, body_ph_ids, ext_durations,
        )
    ext_arrays = [c.cpu().numpy() for c in decoded_blocks]   # each (L+2, 14)

    # Extract body (positions 1..L) per phoneme, and AVERAGE boundary frames
    # with neighboring phoneme's context predictions.
    #   phoneme i body[0]   ↔  phoneme i-1 next_ctx prediction (its position L_{i-1}+1 = -1)
    #   phoneme i body[L-1] ↔  phoneme i+1 prev_ctx prediction (its position 0)
    body_arrays = []
    for i, ext in enumerate(ext_arrays):
        L = ext.shape[0] - 2
        body = ext[1:1+L].copy()
        if args.blend_boundaries:
            if i > 0:
                body[0] = 0.5 * body[0] + 0.5 * ext_arrays[i-1][-1]
            if i < len(ext_arrays) - 1:
                body[-1] = 0.5 * body[-1] + 0.5 * ext_arrays[i+1][0]
        body_arrays.append(body)
    body_norm = np.concatenate(body_arrays, axis=0)
    print(f"  ↳ {len(ext_arrays)} phoneme blocks  blend_boundaries={args.blend_boundaries}")

    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)
    body_feats = body_norm * feat_std + feat_mean
    body_feats[:, 12] = np.exp(body_feats[:, 12]) - 1.0
    body_feats[body_feats[:, 12] < 30, 12] = 0.0
    print(f"frames: {body_feats.shape[0]}  duration: {body_feats.shape[0]/50:.2f}s  "
          f"boundary_smooth={args.boundary_smooth}")

    print("Loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    wav = sparc.decode(body_feats[:, :12], body_feats[:, 12], body_feats[:, 13], spk)
    if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, sparc.sr)
    print(f"Saved {args.output} ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
