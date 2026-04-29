"""
Diagnose jitter in v9 output: measure frame-to-frame delta at phoneme
boundaries vs. within phonemes. If the hypothesis is right (boundaries
are uncoordinated since each phoneme decodes independently), boundary
deltas will be much larger than interior deltas.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.phoneme_rvq import PhonemeRVQTokenizer


CKPT       = "v9/checkpoints/phoneme_rvq/best.pt"
PHON_PATH  = "data/processed_merged_v3/phonemes_mfa.json"
ALIGN_PATH = "data/processed_merged_v3/alignments_mfa.json"
FEATS_DIR  = Path("data/features_merged_logpitch_v2")
NORM_STATS = "data/features_merged_logpitch_v2/norm_stats.npz"

UID = "0011_Angry_0011_000564"


def main():
    device = torch.device("cpu")
    c = torch.load(CKPT, map_location=device, weights_only=False)
    a = c["args"]
    model = PhonemeRVQTokenizer(
        vocab_size=a["vocab_size"], input_dim=14,
        latent_dim=a["latent_dim"], hidden_dim=a["hidden_dim"],
        codebook_size=a["codebook_size"], num_quantizers=a["num_quantizers"],
        decoder_d_model=a["decoder_d_model"], decoder_nhead=a["decoder_nhead"],
        decoder_layers=a["decoder_layers"],
        commitment_weight=a["commit_weight"], ema_decay=a["ema_decay"],
    ).to(device)
    model.load_state_dict(c["model"]); model.eval()

    phon = json.load(open(PHON_PATH))
    align = json.load(open(ALIGN_PATH))
    stats = np.load(NORM_STATS)
    feat_mean = stats["mean"].astype(np.float32)
    feat_std  = stats["std"].astype(np.float32)

    f = np.load(FEATS_DIR / f"{UID}.npz", allow_pickle=False)
    body_durs = np.asarray(align[UID]["durations"], dtype=np.int64)
    body_T = int(body_durs.sum())
    feats = np.concatenate([
        f["ema"][:body_T].astype(np.float32),
        f["pitch"][:body_T, None].astype(np.float32),
        f["loudness"][:body_T, None].astype(np.float32),
    ], axis=1)
    gt_norm = (feats - feat_mean) / (feat_std + 1e-8)

    phoneme_ids = np.asarray(phon[UID]["indices"], dtype=np.int64)
    body_phon = phoneme_ids[1:1+len(body_durs)]
    offsets = np.concatenate([[0], np.cumsum(body_durs)])

    F_PAD = a.get("f_pad", 32) if isinstance(a, dict) else 32
    n_body = len(body_durs)
    blocks = np.zeros((n_body, F_PAD, 14), dtype=np.float32)
    block_lens = np.zeros(n_body, dtype=np.int64)
    for i in range(n_body):
        s = int(offsets[i]); e = int(offsets[i+1])
        L = min(e - s, F_PAD)
        if L <= 0:
            block_lens[i] = 1
            continue
        blocks[i, :L] = gt_norm[s:s+L]
        block_lens[i] = L
    f_t = torch.from_numpy(blocks)
    p_t = torch.from_numpy(body_phon)
    l_t = torch.from_numpy(block_lens.clip(min=1))

    with torch.no_grad():
        recon, info = model.forward_batch(f_t, p_t, l_t)
    # Concatenate per-phoneme recon
    chunks = []
    for i in range(n_body):
        L = int(block_lens[i])
        chunks.append(recon[i, :L].numpy())
    recon_norm = np.concatenate(chunks, axis=0)

    # Compute frame-to-frame deltas for both GT and recon, separately
    # tag each frame as "interior" or "boundary"
    boundary_idx = set()
    cum = 0
    for L in block_lens:
        cum += int(L)
        if cum < recon_norm.shape[0]:
            boundary_idx.add(cum - 1)   # last frame of a phoneme; (cum-1)→cum is boundary

    deltas_gt = np.linalg.norm(np.diff(gt_norm, axis=0), axis=1)
    deltas_recon = np.linalg.norm(np.diff(recon_norm, axis=0), axis=1)

    interior_mask = np.array([(i not in boundary_idx) for i in range(len(deltas_gt))])
    boundary_mask = ~interior_mask

    print(f"Utterance: {UID}")
    print(f"Total frames: {recon_norm.shape[0]}, body phonemes: {n_body}")
    print(f"Boundary frames: {boundary_mask.sum()}, interior frames: {interior_mask.sum()}")
    print()
    print(f"  {'metric':<25} {'GT':>10} {'recon':>10} {'ratio':>8}")
    for name, mask in [("interior delta mean", interior_mask),
                       ("boundary delta mean", boundary_mask),
                       ("interior delta max",  interior_mask),
                       ("boundary delta max",  boundary_mask)]:
        agg = np.max if "max" in name else np.mean
        gt_v = agg(deltas_gt[mask]) if mask.any() else 0.0
        rc_v = agg(deltas_recon[mask]) if mask.any() else 0.0
        print(f"  {name:<25} {gt_v:>10.4f} {rc_v:>10.4f} {rc_v/max(gt_v,1e-9):>8.2f}")

    # Also: what does "interior delta" look like over time (within-phoneme smoothness)?
    print(f"\n  GT     mean interior frame-delta: {deltas_gt[interior_mask].mean():.4f}")
    print(f"  recon  mean interior frame-delta: {deltas_recon[interior_mask].mean():.4f}")
    print(f"  ↑ if recon >> GT here, the within-phoneme decoder is also adding jitter.")
    print(f"  GT     mean boundary frame-delta: {deltas_gt[boundary_mask].mean():.4f}")
    print(f"  recon  mean boundary frame-delta: {deltas_recon[boundary_mask].mean():.4f}")
    print(f"  ↑ if recon >> GT here, phoneme-boundary discontinuities cause jitter.")

    # Per-channel boundary delta to see which channel jitters most
    print(f"\nPer-channel boundary frame-delta (recon / GT ratio):")
    chan_names = ["EMA"+str(i) for i in range(12)] + ["log-pitch", "loudness"]
    for c_idx in range(14):
        gt_c = np.abs(np.diff(gt_norm[:, c_idx]))[boundary_mask].mean()
        rc_c = np.abs(np.diff(recon_norm[:, c_idx]))[boundary_mask].mean()
        print(f"  {chan_names[c_idx]:<14} GT={gt_c:.4f}  recon={rc_c:.4f}  ratio={rc_c/max(gt_c,1e-9):.2f}")


if __name__ == "__main__":
    main()
