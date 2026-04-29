"""
Tag ESD utterances with HuBERT-superb-er emotion scores.

ESD already has emotion labels in directory names (Happy, Angry, Neutral, Sad, Surprise).
We re-tag because:
  1. HuBERT confidence scores give continuous intensity (useful for intensity knob).
  2. Sub-emotion scores (full distribution) are richer than discrete labels.
  3. Cross-checks acted vs HuBERT-perceived emotion.

Output: data/esd_emotion_tags.json
  { "0011_Happy_0011_000927":
    { "label": "hap", "confidence": 0.93, "scores": {"hap": 0.93, "neu": 0.04, ...},
      "esd_speaker": "0011", "esd_emotion": "Happy", "esd_utt": "0011_000927" }, ... }

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python -u scripts/tag_esd_emotions.py --device mps
"""
import argparse
import json
import time
from pathlib import Path

import soundfile as sf
from transformers import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd-root", type=str,
                    default="data/esd_raw/Emotion Speech Dataset")
    ap.add_argument("--out", type=str, default="data/esd_emotion_tags.json")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--model", type=str,
                    default="superb/hubert-large-superb-er",
                    help="HF audio-classification model id")
    args = ap.parse_args()

    esd_root = Path(args.esd_root)
    out_path = Path(args.out)

    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing tags")
    else:
        results = {}

    # Walk ESD directory: speaker/emotion/utt.wav
    all_wavs = []
    for spk_dir in sorted(esd_root.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name.startswith("__"):
            continue
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir() or emo_dir.name.startswith("__"):
                continue
            for wav in sorted(emo_dir.glob("*.wav")):
                # ESD has English speakers 0011-0020. Skip Mandarin (0001-0010).
                if not (11 <= int(spk_dir.name) <= 20):
                    continue
                key = f"{spk_dir.name}_{emo_dir.name}_{wav.stem}"
                all_wavs.append((key, str(wav), spk_dir.name, emo_dir.name, wav.stem))

    remaining = [w for w in all_wavs if w[0] not in results]
    print(f"Total: {len(all_wavs)}, already tagged: {len(results)}, "
          f"remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    print(f"Loading {args.model} on {args.device}...")
    classifier = pipeline("audio-classification",
                          model=args.model,
                          device=args.device)
    label_names = classifier.model.config.id2label
    print(f"Labels: {label_names}")

    t0 = time.time()
    tagged = 0
    skipped = 0

    for key, wav_path, spk, emo, utt in remaining:
        try:
            wav, sr = sf.read(wav_path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            preds = classifier(wav, sampling_rate=sr, top_k=len(label_names))
            scores = {p["label"]: round(p["score"], 4) for p in preds}
            top = preds[0]
            results[key] = {
                "label": top["label"],
                "confidence": round(top["score"], 4),
                "scores": scores,
                "esd_speaker": spk,
                "esd_emotion": emo,
                "esd_utt": utt,
            }
            tagged += 1
        except Exception as e:
            skipped += 1
            continue

        if (tagged % args.save_every) == 0 and tagged > 0:
            elapsed = time.time() - t0
            rate = tagged / elapsed
            eta_sec = (len(remaining) - tagged) / rate
            print(f"[{tagged}/{len(remaining)}] {rate:.1f} files/s, "
                  f"ETA: {eta_sec/60:.1f} min, skipped: {skipped}")
            with open(out_path, "w") as f:
                json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone! Tagged {tagged}, skipped {skipped}, time: {elapsed/60:.1f} min")

    from collections import Counter
    labels = Counter(r["label"] for r in results.values())
    print(f"\nHuBERT label distribution:")
    for lab, cnt in labels.most_common():
        print(f"  {lab:>8}: {cnt:>5}")

    # Cross-check: HuBERT label vs ESD acted emotion
    matrix = {}
    for r in results.values():
        esd = r["esd_emotion"]
        hub = r["label"]
        matrix.setdefault(esd, Counter())[hub] += 1
    print(f"\nESD acted emotion → HuBERT label confusion:")
    print(f"  {'ESD':<12}", "  ".join(f"{l:>6}" for l in sorted(set(label_names.values()))))
    for esd, c in sorted(matrix.items()):
        print(f"  {esd:<12}", "  ".join(f"{c.get(l, 0):>6}" for l in sorted(set(label_names.values()))))


if __name__ == "__main__":
    main()
