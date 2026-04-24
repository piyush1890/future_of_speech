"""
Tag all LibriSpeech utterances with emotion labels using HuBERT-superb-er.

Output: data/librispeech_emotion_tags.json
  { "utt_id": {"label": "neu", "confidence": 0.93, "scores": {"neu": 0.93, "hap": 0.03, ...}}, ... }

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python -u scripts/tag_librispeech_emotions.py --device mps
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
from transformers import pipeline


def find_audio_path(utt_id: str, libri_root: Path) -> str:
    """Map utterance ID like '100-121669-0000' to FLAC path."""
    parts = utt_id.split("-")
    if len(parts) == 3:
        speaker, chapter, _ = parts
        # Try train-clean-100, train-clean-360, dev-clean
        for split in ["train-clean-100", "train-clean-360", "dev-clean",
                       "dev-other", "test-clean", "test-other"]:
            p = libri_root / split / speaker / chapter / f"{utt_id}.flac"
            if p.exists():
                return str(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=str,
                    default="data/features_merged_logpitch_v2")
    ap.add_argument("--libri-root", type=str, default="data/LibriSpeech")
    ap.add_argument("--out", type=str, default="data/librispeech_emotion_tags.json")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    features_dir = Path(args.features_dir)
    libri_root = Path(args.libri_root)
    out_path = Path(args.out)

    # Resume from partial results if they exist
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing tags")
    else:
        results = {}

    # All utterance IDs from features
    utt_ids = sorted([f.stem for f in features_dir.glob("*.npz")
                       if f.stem != "norm_stats"])
    remaining = [u for u in utt_ids if u not in results]
    print(f"Total: {len(utt_ids)}, already tagged: {len(results)}, "
          f"remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    print(f"Loading HuBERT-superb-er on {args.device}...")
    classifier = pipeline("audio-classification",
                          model="superb/hubert-large-superb-er",
                          device=args.device)
    label_names = classifier.model.config.id2label
    print(f"Labels: {label_names}")

    t0 = time.time()
    tagged = 0
    skipped = 0

    for i, utt_id in enumerate(remaining):
        audio_path = find_audio_path(utt_id, libri_root)
        if audio_path is None:
            skipped += 1
            continue

        try:
            wav, sr = sf.read(audio_path)
            preds = classifier(wav, sampling_rate=sr, top_k=len(label_names))
            scores = {p["label"]: round(p["score"], 4) for p in preds}
            top = preds[0]
            results[utt_id] = {
                "label": top["label"],
                "confidence": round(top["score"], 4),
                "scores": scores,
            }
            tagged += 1
        except Exception as e:
            skipped += 1
            continue

        if (tagged % args.save_every) == 0:
            elapsed = time.time() - t0
            rate = tagged / elapsed
            eta = (len(remaining) - tagged) / rate / 3600
            print(f"[{tagged}/{len(remaining)}] {rate:.1f} files/s, "
                  f"ETA: {eta:.1f}h, skipped: {skipped}")
            with open(out_path, "w") as f:
                json.dump(results, f)

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone! Tagged {tagged}, skipped {skipped}, "
          f"total time: {elapsed/3600:.1f}h")

    # Summary
    from collections import Counter
    labels = Counter(r["label"] for r in results.values())
    high_conf = sum(1 for r in results.values() if r["confidence"] > 0.7)
    print(f"\nDistribution ({len(results)} utterances):")
    for label, count in labels.most_common():
        pct = count / len(results) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print(f"High confidence (>0.7): {high_conf} ({high_conf/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
