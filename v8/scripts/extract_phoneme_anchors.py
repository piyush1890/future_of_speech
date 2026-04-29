"""
Extract per-phoneme anchor features (start, end) from GT articulatory features
+ MFA-aligned durations.

For each utterance:
  features (T_gt, 14)            from data/features_merged_logpitch_v2/<uid>.npz
  durations (N,)                 from data/processed_merged_v3/alignments_mfa.json
                                 — N = number of phonemes (after MFA, sums to T_gt)

Output for each utterance:
  start (N, 14)                  features at frame [cursor]
  end   (N, 14)                  features at frame [cursor + d - 1]
  durations (N,)                 GT durations (frames)

Saved to: v8/data/phoneme_anchors/<uid>.npz

Notes:
  - Pitch is stored as log(pitch+1) (matches v5 normalization).
  - Speaker emb is preserved from input npz for downstream conditioning.
  - For phonemes with duration=1, start=end (single frame).
  - For phonemes with duration=0 (rare boundary), we use neighboring frame as fallback.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_features(path):
    f = np.load(path)
    T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
    pitch = f["pitch"][:T].astype(np.float32)
    if pitch.max() > 20:
        pitch = np.log(pitch + 1.0)
    feats = np.concatenate([
        f["ema"][:T].astype(np.float32),
        pitch[:, None],
        f["loudness"][:T, None].astype(np.float32),
    ], axis=1)
    spk_emb = f["spk_emb"].astype(np.float32) if "spk_emb" in f.files else None
    return feats, spk_emb


def extract_anchors(feats: np.ndarray, durs: np.ndarray):
    """feats: (T, 14)  durs: (N,) summing to <= T.
    Returns body-only (start, mid, end), each (N, 14)."""
    N = len(durs)
    F = feats.shape[1]
    start = np.zeros((N, F), dtype=np.float32)
    mid   = np.zeros((N, F), dtype=np.float32)
    end   = np.zeros((N, F), dtype=np.float32)
    cursor = 0
    for i, d in enumerate(durs):
        d = int(d)
        if d == 0:
            idx = min(cursor, len(feats) - 1)
            start[i] = feats[idx]; mid[i] = feats[idx]; end[i] = feats[idx]
        else:
            start[i] = feats[cursor]
            mid[i]   = feats[min(cursor + d // 2, len(feats) - 1)]
            end[i]   = feats[min(cursor + d - 1, len(feats) - 1)]
            cursor  += d
    return start, mid, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/features_merged_logpitch_v2")
    ap.add_argument("--alignments",   default="data/processed_merged_v3/alignments_mfa.json")
    ap.add_argument("--phonemes",     default="data/processed_merged_v3/phonemes_mfa.json")
    ap.add_argument("--out-dir",      default="v8/data/phoneme_anchors")
    ap.add_argument("--max-utts",     type=int, default=None,
                    help="Limit (for sanity testing); default all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    align_data = json.load(open(args.alignments))
    phons_data = json.load(open(args.phonemes))
    feat_dir = Path(args.features_dir)

    utt_ids = sorted([u for u in phons_data
                      if u in align_data and (feat_dir / f"{u}.npz").exists()])
    if args.max_utts:
        utt_ids = utt_ids[:args.max_utts]
    print(f"Processing {len(utt_ids)} utterances → {out_dir}")

    skipped = 0
    written = 0
    for uid in tqdm(utt_ids, desc="extracting anchors"):
        try:
            feats, spk_emb = load_features(feat_dir / f"{uid}.npz")
            durs = np.asarray(align_data[uid]["durations"], dtype=np.int64)
            phon_strs = phons_data[uid]["phonemes"]
            if len(durs) != len(phon_strs):
                skipped += 1
                continue
            if durs.sum() > len(feats):
                skipped += 1
                continue
            start, mid, end = extract_anchors(feats, durs)
            np.savez_compressed(
                out_dir / f"{uid}.npz",
                start=start.astype(np.float32),
                mid=mid.astype(np.float32),
                end=end.astype(np.float32),
                durations=durs.astype(np.int32),
                spk_emb=spk_emb if spk_emb is not None else np.zeros(64, dtype=np.float32),
                n_phonemes=len(durs),
                total_frames=int(durs.sum()),
            )
            written += 1
        except Exception as e:
            skipped += 1

    print(f"\nDone: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
