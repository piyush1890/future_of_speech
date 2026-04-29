"""
v9 end-to-end synthesis:
  text → phonemes (with <sil> at punctuation)
       → predictor (encoder + AR decoder) → per-phoneme (start_tokens, end_tokens, durations)
       → tokenizer decode → per-phoneme frames
       → concatenate frames → SPARC → audio
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
from v9.models.v9_predictor import V9Predictor
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
    one_hot = [0.0] * len(EMOTION_TO_ID); one_hot[eid] = 1.0
    return np.asarray(one_hot + [float(intensity)], dtype=np.float32)


def sample_logits(logits: torch.Tensor, temperature: float, top_k: int):
    """logits: (B, C). Returns (B,) sampled token ids."""
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / max(1e-6, temperature)
    if top_k > 0:
        v, _ = torch.topk(logits, top_k, dim=-1)
        thresh = v[..., -1:].expand_as(logits)
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def predictor_generate(predictor: V9Predictor, phoneme_ids, spk_emb, knobs, phoneme_mask,
                       temperature=1.0, top_k=0, cfg_scale=1.0):
    """AR generation of (start_tokens, end_tokens, log_dur) per phoneme."""
    B, N = phoneme_ids.shape
    device = phoneme_ids.device
    K = predictor.K

    # Encode once (conditional, and unconditional if CFG)
    enc_cond = predictor.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=False)
    enc_uncond = (predictor.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=True)
                  if cfg_scale != 1.0 else None)

    gen_start = torch.zeros(B, N, K, dtype=torch.long, device=device)
    gen_end   = torch.zeros(B, N, K, dtype=torch.long, device=device)
    gen_logdur = torch.zeros(B, N, device=device)

    for i in range(N):
        # Build shifted decoder input from tokens generated so far
        dec_inp = predictor._make_decoder_input(gen_start, gen_end)
        dec_inp = predictor.decoder_pe(dec_inp)
        causal_mask = torch.triu(
            torch.full((N, N), float("-inf"), device=device), diagonal=1
        )
        h_cond = predictor.decoder(
            dec_inp, enc_cond,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~phoneme_mask,
            memory_key_padding_mask=~phoneme_mask,
        )                                                                   # (B, N, D)
        h_pos = h_cond[:, i, :]                                              # (B, D)
        if enc_uncond is not None:
            h_uncond = predictor.decoder(
                dec_inp, enc_uncond,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=~phoneme_mask,
                memory_key_padding_mask=~phoneme_mask,
            )[:, i, :]
        else:
            h_uncond = None

        # Sample start tokens hierarchically
        for k in range(K):
            cond_logits = predictor.start_heads.step_logits_one(h_pos, k, gen_start[:, i, :k])
            if h_uncond is not None:
                un_logits = predictor.start_heads.step_logits_one(h_uncond, k, gen_start[:, i, :k])
                logits_k = un_logits + cfg_scale * (cond_logits - un_logits)
            else:
                logits_k = cond_logits
            gen_start[:, i, k] = sample_logits(logits_k, temperature, top_k)

        # Sample end tokens hierarchically
        for k in range(K):
            cond_logits = predictor.end_heads.step_logits_one(h_pos, k, gen_end[:, i, :k])
            if h_uncond is not None:
                un_logits = predictor.end_heads.step_logits_one(h_uncond, k, gen_end[:, i, :k])
                logits_k = un_logits + cfg_scale * (cond_logits - un_logits)
            else:
                logits_k = cond_logits
            gen_end[:, i, k] = sample_logits(logits_k, temperature, top_k)

        # Duration scalar
        gen_logdur[:, i] = predictor.duration_head(h_pos).squeeze(-1)

    return gen_start, gen_end, gen_logdur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str)
    ap.add_argument("--tokenizer-checkpoint", default="v9/checkpoints/phoneme_rvq/best.pt")
    ap.add_argument("--predictor-checkpoint", default="v9/checkpoints/predictor/best.pt")
    ap.add_argument("--vocab",                default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--norm-stats",           default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--speaker-emb",          default=None,
                    help="Path to .npy with 64-d speaker emb. Default uses DEFAULT_REFERENCE.")
    ap.add_argument("--emotion",              default="neutral",
                    choices=list(EMOTION_TO_ID.keys()))
    ap.add_argument("--intensity",            type=float, default=0.5)
    ap.add_argument("--temperature",          type=float, default=1.0)
    ap.add_argument("--top-k",                type=int,   default=0)
    ap.add_argument("--cfg-scale",            type=float, default=1.0)
    ap.add_argument("--duration-scale",       type=float, default=1.0)
    ap.add_argument("--device",               default="cpu")
    ap.add_argument("--output", "-o",         default="v9/outputs/v9_synth.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Tokenizer ──
    tc = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    ta = tc["args"]
    print(f"v9 tokenizer: epoch={tc['epoch']} val={tc['val_loss']:.4f}")
    tokenizer = PhonemeRVQTokenizer(
        vocab_size=ta["vocab_size"], input_dim=14,
        latent_dim=ta["latent_dim"], hidden_dim=ta["hidden_dim"],
        codebook_size=ta["codebook_size"], num_quantizers=ta["num_quantizers"],
        decoder_d_model=ta["decoder_d_model"], decoder_nhead=ta["decoder_nhead"],
        decoder_layers=ta["decoder_layers"],
        commitment_weight=ta["commit_weight"], ema_decay=ta["ema_decay"],
    ).to(device)
    tokenizer.load_state_dict(tc["model"]); tokenizer.eval()

    # ── Predictor ──
    pc = torch.load(args.predictor_checkpoint, map_location=device, weights_only=False)
    pa = pc["args"]
    print(f"v9 predictor: epoch={pc['epoch']} val={pc['val_loss']:.4f}")
    knob_dim = pc.get("knob_dim", 6)
    predictor = V9Predictor(
        vocab_size=pa["vocab_size"], codebook_size=pa["codebook_size"],
        num_quantizers=pa["num_quantizers"],
        d_model=pa["d_model"], nhead=pa["nhead"],
        num_encoder_layers=pa["num_layers"], num_decoder_layers=pa["num_layers"],
        d_ff=pa["d_ff"], dropout=0.0,
        speaker_emb_dim=64, knob_dim=knob_dim, knob_dropout=0.0,
        max_phonemes=pa["max_phonemes"],
    ).to(device)
    predictor.load_state_dict(pc["model"]); predictor.eval()

    # ── Inputs ──
    vocab = PhonemeVocab(args.vocab)
    g2p = G2p()
    phs = text_to_phonemes(args.text, g2p)
    phoneme_ids = torch.tensor([vocab.encode(phs, add_bos_eos=True)], dtype=torch.long, device=device)
    N = phoneme_ids.shape[1]
    phoneme_mask = phoneme_ids != 0
    print(f"phonemes ({N}): {phs[:20]}{'...' if len(phs) > 20 else ''}")

    if args.speaker_emb:
        spk = np.load(args.speaker_emb).astype(np.float32)
    else:
        spk = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
        print("(using default speaker)")
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)

    if knob_dim > 0:
        knobs_np = build_knobs(args.emotion, args.intensity)
        knobs_t = torch.from_numpy(knobs_np).unsqueeze(0).to(device)
        print(f"emotion={args.emotion}  intensity={args.intensity}  cfg_scale={args.cfg_scale}")
    else:
        knobs_t = None

    # ── AR generate tokens + durations ──
    gen_start, gen_end, gen_logdur = predictor_generate(
        predictor, phoneme_ids, spk_t, knobs_t, phoneme_mask,
        temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale,
    )

    # Body slice: positions 1..N-1 (exclude BOS at 0 and EOS at N-1, same as training)
    body_start = gen_start[0, 1:N-1, :]                      # (N_body, K)
    body_end   = gen_end[0,   1:N-1, :]
    body_logdur = gen_logdur[0, 1:N-1]

    # Convert log-dur → integer frame counts
    durations = (torch.exp(body_logdur) * args.duration_scale).round().clamp(min=1).long()
    print(f"sampled durations (first 20): {durations[:20].tolist()}")

    # ── Decode each body phoneme via tokenizer ──
    body_ph_ids = phoneme_ids[0, 1:N-1]                      # (N_body,)
    n_body = body_start.shape[0]
    F_max = int(durations.max().item())
    with torch.no_grad():
        decoded_blocks = tokenizer.decode_indices_batch(
            body_start, body_end, body_ph_ids, durations,
        )
    # Each entry: (length[b], 14) tensor
    body_norm = torch.cat(decoded_blocks, dim=0).cpu().numpy()    # (T, 14) normalized

    # Denormalize
    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32)
    feat_std  = stats["std"].astype(np.float32)
    body_feats = body_norm * feat_std + feat_mean
    body_feats[:, 12] = np.exp(body_feats[:, 12]) - 1.0       # log(p+1) → Hz
    body_feats[body_feats[:, 12] < 30, 12] = 0.0

    # Optional: prepend/append a frame of silence for BOS/EOS (zeros after denorm
    # would be unsafe; instead, just render body for now — the leading <sil> in
    # phonemes was already captured as a body phoneme).
    print(f"frames: {body_feats.shape[0]}  duration: {body_feats.shape[0]/50:.2f}s")

    # ── SPARC ──
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
