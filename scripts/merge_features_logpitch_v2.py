"""
Merge multiple feature dirs into a single log-pitch feature dir for training.

Applies log1p to pitch channel (raw Hz → log Hz) and copies ema/loudness/spk_emb
unchanged. Skips files that already exist in output (idempotent).

Also recomputes norm_stats.npz over the merged set.
"""
import argparse
from pathlib import Path

import numpy as np


def transform_one(src: Path, dst: Path):
    d = np.load(src)
    pitch = np.log(d["pitch"].astype(np.float32) + 1.0)
    np.savez(
        dst,
        ema=d["ema"].astype(np.float32),
        pitch=pitch,
        loudness=d["loudness"].astype(np.float32),
        spk_emb=d["spk_emb"].astype(np.float32),
    )


def compute_norm_stats(out_dir: Path):
    """Compute per-channel mean/std over the full merged set on the 14-dim vector."""
    files = sorted(f for f in out_dir.glob("*.npz") if f.stem != "norm_stats")
    sums = np.zeros(14, dtype=np.float64)
    sq = np.zeros(14, dtype=np.float64)
    n = 0
    for i, f in enumerate(files):
        d = np.load(f)
        T = min(d["ema"].shape[0], d["pitch"].shape[0], d["loudness"].shape[0])
        feat = np.concatenate([
            d["ema"][:T],
            d["pitch"][:T, None],
            d["loudness"][:T, None],
        ], axis=1).astype(np.float64)
        sums += feat.sum(axis=0)
        sq += (feat ** 2).sum(axis=0)
        n += T
        if (i + 1) % 5000 == 0:
            print(f"  stats: {i+1}/{len(files)}")
    mean = sums / n
    var = sq / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    np.savez(out_dir / "norm_stats.npz", mean=mean.astype(np.float32),
             std=std.astype(np.float32), n_frames=n)
    print(f"Saved norm_stats over {n} frames from {len(files)} files")
    print(f"  mean pitch log: {mean[12]:.3f}  std: {std[12]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Source feature dirs to merge (e.g. data/features_all data/features_360_v2)")
    ap.add_argument("--out-dir", type=str, default="data/features_merged_logpitch_v2")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    for src in args.sources:
        s = Path(src)
        if not s.is_dir():
            print(f"  skip {src}: not a directory")
            continue
        files = list(s.glob("*.npz"))
        print(f"  {s}: {len(files)} files")
        for f in files:
            if f.name == "norm_stats.npz":
                continue
            tgt = out / f.name
            if args.skip_existing and tgt.exists():
                skipped += 1
                continue
            transform_one(f, tgt)
            total += 1
            if total % 5000 == 0:
                print(f"    processed {total}")
    print(f"\nMerged {total} new files (skipped {skipped} existing) → {out}")

    compute_norm_stats(out)


if __name__ == "__main__":
    main()
