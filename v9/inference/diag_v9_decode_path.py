"""Test the EXACT inference decode path (decode_indices_batch) with GT tokens.

Compares:
  (A) GT frames → forward_batch (encode→quant→decode) → SPARC  ← diag does this
  (B) GT-saved tokens → decode_indices_batch → SPARC            ← synthesize_v9 does this

If (B) is bad but (A) is good, the inference path has a bug separate from the predictor.
If both are good, the inference path is fine and the predictor is the bottleneck.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.phoneme_rvq import PhonemeRVQTokenizer

CKPT = "v9/checkpoints/phoneme_rvq/best.pt"
TOKENS_DIR = Path("v9/data/phoneme_tokens")
PHONEMES = "data/processed_merged_v3/phonemes_mfa.json"
ALIGNMENTS = "data/processed_merged_v3/alignments_mfa.json"
NORM_STATS = "data/features_merged_logpitch_v2/norm_stats.npz"
ANCHORS = Path("v8/data/phoneme_anchors")
OUT = Path("v9/outputs/diag_decode_path")

UID = "0011_Neutral_0011_000313"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    c = torch.load(CKPT, map_location=device, weights_only=False)
    a = c["args"]
    print(f"v9 tokenizer: epoch={c['epoch']} val={c['val_loss']:.4f}")

    model = PhonemeRVQTokenizer(
        vocab_size=a["vocab_size"], input_dim=14,
        latent_dim=a["latent_dim"], hidden_dim=a["hidden_dim"],
        codebook_size=a["codebook_size"], num_quantizers=a["num_quantizers"],
        decoder_d_model=a["decoder_d_model"], decoder_nhead=a["decoder_nhead"],
        decoder_layers=a["decoder_layers"],
        commitment_weight=a["commit_weight"], ema_decay=a["ema_decay"],
    ).to(device)
    model.load_state_dict(c["model"]); model.eval()

    # Load saved GT tokens (extracted from this utterance during data prep)
    tok_path = TOKENS_DIR / f"{UID}.npz"
    if not tok_path.exists():
        print(f"FATAL: {tok_path} not found. Need extracted tokens for this UID.")
        return

    tok = np.load(tok_path, allow_pickle=False)
    phoneme_ids_full = tok["phoneme_ids"].astype(np.int64)         # (N+2,)  BOS+body+EOS
    start_idx = tok["start_idx"].astype(np.int64)                  # (N_body, K)
    end_idx = tok["end_idx"].astype(np.int64)
    durations = tok["durations"].astype(np.int64)                  # (N_body,) frames per body phoneme
    spk_emb = tok["spk_emb"].astype(np.float32)
    n_body = len(durations)

    body_ph_ids = torch.from_numpy(phoneme_ids_full[1:1 + n_body]).to(device)
    s_idx_t = torch.from_numpy(start_idx).to(device)
    e_idx_t = torch.from_numpy(end_idx).to(device)
    dur_t = torch.from_numpy(durations).to(device)

    print(f"\nUID: {UID}")
    print(f"n_body={n_body}  total_frames={int(durations.sum())}  K={start_idx.shape[1]}")
    print(f"start_idx range: [{int(start_idx.min())}, {int(start_idx.max())}]")
    print(f"end_idx range:   [{int(end_idx.min())}, {int(end_idx.max())}]")

    # === Path B: the actual inference path ===
    with torch.no_grad():
        decoded_blocks = model.decode_indices_batch(
            s_idx_t, e_idx_t, body_ph_ids, dur_t,
        )
    body_norm_B = torch.cat(decoded_blocks, dim=0).cpu().numpy()    # (T, 14)
    print(f"\nPath B (decode_indices_batch): T_recon = {body_norm_B.shape[0]}")

    # === Path A: forward_batch with GT frames (for comparison) ===
    feats_path = Path("data/features_merged_logpitch_v2") / f"{UID}.npz"
    f = np.load(feats_path, allow_pickle=False)
    T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
    gt_feats = np.concatenate([
        f["ema"][:T].astype(np.float32),
        f["pitch"][:T, None].astype(np.float32),
        f["loudness"][:T, None].astype(np.float32),
    ], axis=1)
    stats = np.load(NORM_STATS)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)
    gt_norm = (gt_feats - feat_mean) / (feat_std + 1e-8)
    body_T = int(durations.sum())
    gt_norm = gt_norm[:body_T]

    F_PAD = a.get("f_pad", 32)
    offsets = np.concatenate([[0], np.cumsum(durations)])
    blocks = []; clipped_lens = []; ph_ids_list = []
    for i in range(n_body):
        s, e = int(offsets[i]), int(offsets[i+1])
        L = min(e - s, F_PAD)
        block = np.zeros((F_PAD, 14), dtype=np.float32)
        block[:L] = gt_norm[s:s+L]
        blocks.append(block)
        clipped_lens.append(L)
        ph_ids_list.append(int(phoneme_ids_full[1 + i]))
    frames_t = torch.from_numpy(np.stack(blocks)).to(device)
    ph_ids_t = torch.tensor(ph_ids_list, dtype=torch.long, device=device)
    lengths_t = torch.tensor(clipped_lens, dtype=torch.long, device=device)

    with torch.no_grad():
        recon, info = model.forward_batch(frames_t, ph_ids_t, lengths_t)

    # Compare freshly-encoded indices vs saved indices
    fresh_s = info["start_idx"].cpu().numpy()
    fresh_e = info["end_idx"].cpu().numpy()
    s_match = (fresh_s == start_idx).mean() * 100
    e_match = (fresh_e == end_idx).mean() * 100
    print(f"\nfresh-encode vs saved tokens: start_match={s_match:.1f}%  end_match={e_match:.1f}%")
    if s_match < 99 or e_match < 99:
        print(f"  → STALE TOKENS: saved file was extracted with a different codebook")
        print(f"  → fresh start_idx[0]: {fresh_s[0].tolist()}")
        print(f"  → saved start_idx[0]: {start_idx[0].tolist()}")

    body_norm_A_chunks = [recon[i, :clipped_lens[i]].cpu().numpy() for i in range(n_body)]
    # Pad any truncated phoneme back to GT length with zero residual (matches diag behavior)
    for i in range(n_body):
        if clipped_lens[i] < int(durations[i]):
            extra = np.zeros((int(durations[i]) - clipped_lens[i], 14), dtype=np.float32)
            body_norm_A_chunks[i] = np.concatenate([body_norm_A_chunks[i], extra], axis=0)
    body_norm_A = np.concatenate(body_norm_A_chunks, axis=0)
    print(f"Path A (forward_batch): T_recon = {body_norm_A.shape[0]}")

    # Compare
    if body_norm_A.shape == body_norm_B.shape:
        diff = float(np.abs(body_norm_A - body_norm_B).mean())
        max_diff = float(np.abs(body_norm_A - body_norm_B).max())
        print(f"\n|A - B| mean = {diff:.6f}   max = {max_diff:.6f}")
        if diff < 1e-4:
            print("→ paths are IDENTICAL (no inference bug at decode level)")
        else:
            print(f"→ paths DIFFER materially — likely a bug or codebook drift")
    else:
        print(f"shape mismatch: A {body_norm_A.shape} vs B {body_norm_B.shape}")

    # Render both
    print("\nLoading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    for name, body_norm in (("A_forward_batch", body_norm_A), ("B_decode_indices", body_norm_B)):
        feats = body_norm * feat_std + feat_mean
        feats[:, 12] = np.exp(feats[:, 12]) - 1.0
        feats[feats[:, 12] < 30, 12] = 0.0
        wav = sparc.decode(feats[:, :12], feats[:, 12], feats[:, 13], spk_emb)
        if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
        out_path = OUT / f"{UID}_{name}.wav"
        sf.write(out_path, wav, sparc.sr)
        print(f"  → {out_path}  ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
