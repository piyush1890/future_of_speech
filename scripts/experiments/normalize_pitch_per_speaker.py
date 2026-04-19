"""
Per-speaker pitch normalization.

Reads features_merged/*.npz, computes per-speaker (mean, std) over voiced frames,
and writes a new features_merged_pitchnorm/ dir where the pitch channel has been
replaced by (pitch - spk_mean) / spk_std on voiced frames (unvoiced frames remain
at their original 0 value).

Also writes per_speaker_pitch_stats.json so we can denormalize at inference.

Rationale: pitch is the feature with the largest cross-speaker variance. If every
speaker's pitch distribution is collapsed to zero-mean-unit-variance, the transformer
sees consistent feature targets across all 500+ speakers, and the cross-speaker noise
that's currently inflating L1-L3 residuals drops out.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main(features_dir, out_dir, voiced_threshold=40.0):
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(p for p in features_dir.glob("*.npz") if p.stem != "norm_stats")
    print(f"{len(files)} feature files found")

    # Pass 1: gather per-speaker voiced pitch values
    per_spk_pitch = defaultdict(list)
    for f in tqdm(files, desc="Pass 1: gather stats"):
        spk = f.stem.split("-")[0]
        d = np.load(f)
        pitch = d["pitch"]
        voiced = pitch > voiced_threshold
        if voiced.any():
            per_spk_pitch[spk].append(pitch[voiced])

    print(f"Speakers with voiced data: {len(per_spk_pitch)}")

    # Compute stats (concatenate then compute mean/std)
    stats = {}
    for spk, chunks in per_spk_pitch.items():
        arr = np.concatenate(chunks)
        stats[spk] = {
            "pitch_mean": float(arr.mean()),
            "pitch_std":  float(arr.std()),
            "n_frames":   int(arr.size),
        }

    # Report distribution
    means = np.array([s["pitch_mean"] for s in stats.values()])
    stds = np.array([s["pitch_std"] for s in stats.values()])
    print(f"Per-speaker pitch_mean: min={means.min():.1f}  max={means.max():.1f}  "
          f"mean={means.mean():.1f}  median={np.median(means):.1f}")
    print(f"Per-speaker pitch_std:  min={stds.min():.1f}  max={stds.max():.1f}  "
          f"mean={stds.mean():.1f}  median={np.median(stds):.1f}")

    # Save per-speaker stats so inference can denormalize
    with open(out_dir / "per_speaker_pitch_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to {out_dir / 'per_speaker_pitch_stats.json'}")

    # Pass 2: write normalized features
    for f in tqdm(files, desc="Pass 2: write normalized"):
        spk = f.stem.split("-")[0]
        d = dict(np.load(f))
        pitch = d["pitch"].copy()
        voiced = pitch > voiced_threshold
        s = stats.get(spk)
        if s is None or s["pitch_std"] < 1.0:
            # fallback: leave pitch unchanged if speaker has too little voiced data
            pass
        else:
            pitch[voiced] = (pitch[voiced] - s["pitch_mean"]) / s["pitch_std"]
        d["pitch"] = pitch
        np.savez(out_dir / f.name, **d)

    # Also copy norm_stats.npz if present (recompute on normalized features)
    print("Recomputing global norm_stats over normalized features...")
    all_feats = []
    for f in tqdm(list(out_dir.glob("*.npz")), desc="global norm"):
        if f.stem == "norm_stats":
            continue
        d = np.load(f)
        # Stack: ema(12) + pitch(1) + loudness(1) = (T, 14)
        feats = np.concatenate([d["ema"], d["pitch"][:, None], d["loudness"][:, None]], axis=1)
        all_feats.append(feats)
    all_feats = np.concatenate(all_feats, axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std[std < 1e-6] = 1.0
    np.savez(out_dir / "norm_stats.npz", mean=mean, std=std)
    print(f"Wrote {out_dir / 'norm_stats.npz'}")
    print(f"  pitch-channel mean={mean[12]:.4f} std={std[12]:.4f}  (should be ~0 and ~1)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="data/features_merged")
    p.add_argument("--out-dir",      type=str, default="data/features_merged_pitchnorm")
    p.add_argument("--voiced-threshold", type=float, default=40.0)
    a = p.parse_args()
    main(a.features_dir, a.out_dir, a.voiced_threshold)
