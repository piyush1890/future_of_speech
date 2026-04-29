"""
Build features_merged_logpitch/ — features with log-scale pitch.

Transformation:  pitch_log = log(pitch + 1)
  unvoiced (pitch=0) → log(1)   = 0.0         (voicing preserved cleanly)
  low voiced (80 Hz) → log(81)  ≈ 4.39
  typical   (160 Hz) → log(161) ≈ 5.08
  high      (300 Hz) → log(301) ≈ 5.71

Rationale:
  - Voicing stays distinguishable: 0 maps to 0, not a varying per-speaker offset.
  - Perceptually aligned: humans hear pitch logarithmically; equal log-differences
    = equal perceptual intervals (octaves).
  - Speaker ranges partially equalize: male(120) and female(240) differ by 100 Hz
    linearly but only 0.69 in log space — model treats them as nearby in feature
    space, speaker identity goes through spk_emb instead.
  - No per-speaker stats file needed.
  - Fully reversible at inference: pitch = exp(pitch_log) - 1.

EMA, loudness, spk_emb copied unchanged.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main(features_dir, out_dir):
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(p for p in features_dir.glob("*.npz") if p.stem != "norm_stats")
    print(f"{len(files)} feature files found in {features_dir}")

    for f in tqdm(files, desc="Log-pitch transform"):
        d = dict(np.load(f))
        pitch = d["pitch"].astype(np.float32)
        # log(pitch + 1): unvoiced (0) -> 0 cleanly; voiced gets log-compressed
        d["pitch"] = np.log(pitch + 1.0).astype(np.float32)
        np.savez(out_dir / f.name, **d)

    # Copy speaker_embeddings.json through (needed by current dataset_rvq.py; will
    # be deprecated in favor of per-utterance embeddings in Concept 5's dataset edit).
    spk_src = features_dir / "speaker_embeddings.json"
    spk_dst = out_dir / "speaker_embeddings.json"
    if spk_src.exists():
        import shutil
        shutil.copy2(spk_src, spk_dst)
        print(f"Copied speaker_embeddings.json (note: will be bypassed when dataset "
              f"is updated to use per-utterance embeddings)")

    # Recompute global norm_stats over the new features
    print("Recomputing global norm_stats over log-pitch features...")
    all_feats = []
    for f in tqdm(list(out_dir.glob("*.npz")), desc="norm"):
        if f.stem == "norm_stats":
            continue
        d = np.load(f)
        feats = np.concatenate(
            [d["ema"], d["pitch"][:, None], d["loudness"][:, None]], axis=1
        )
        all_feats.append(feats)
    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std[std < 1e-6] = 1.0
    np.savez(out_dir / "norm_stats.npz", mean=mean, std=std)
    print(f"Wrote {out_dir / 'norm_stats.npz'}")
    print(f"  EMA channels mean: {mean[:12].round(3).tolist()}")
    print(f"  EMA channels std:  {std[:12].round(3).tolist()}")
    print(f"  pitch channel (log): mean={mean[12]:.4f} std={std[12]:.4f}")
    print(f"  loudness channel:    mean={mean[13]:.4f} std={std[13]:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="data/features_merged")
    p.add_argument("--out-dir",      type=str, default="data/features_merged_logpitch")
    a = p.parse_args()
    main(a.features_dir, a.out_dir)
