"""
Build utterance_metadata_v5.json combining:
  - ESD: acted emotion from directory name + arousal as intensity
  - LibriSpeech: V/A/D-derived emotion + arousal as intensity

Output format expected by training/dataset_rvq.py:
  {
    "utt_id": {
      "emotion_label": "happy" | "sad" | "angry" | "neutral" | "surprise",
      "style_label":   "default",
      "intensity":     float in [0, 1]
    }, ...
  }
"""
import json
from pathlib import Path
from collections import Counter


def vad_to_emotion(valence: float, arousal: float) -> str:
    """Map V/A/D coordinates to one of {neutral, happy, sad, angry, surprise}."""
    if arousal < 0.35:
        return "sad"               # low arousal — usually sad/calm passage
    if arousal > 0.65:
        if valence > 0.55:
            return "happy"          # high arousal + positive
        elif valence < 0.42:
            return "angry"          # high arousal + negative
        else:
            return "surprise"       # high arousal + ambiguous valence
    return "neutral"               # mid arousal — default


def main():
    out = {}

    # --- ESD: acted labels + V/A/D intensity ---
    esd_vad_path = Path("data/esd_emotion_vad.json")
    with open(esd_vad_path) as f:
        esd_vad = json.load(f)

    for key, r in esd_vad.items():
        # ESD key format: "<speaker>_<EmotionDir>_<utt>"
        parts = key.split("_", 2)
        if len(parts) < 3:
            continue
        _, emo_dir, _ = parts
        # Map ESD acted directory name to our emotion vocab
        esd_emo_map = {
            "Angry": "angry", "Happy": "happy", "Neutral": "neutral",
            "Sad": "sad", "Surprise": "surprise",
        }
        emo = esd_emo_map.get(emo_dir, "neutral")
        out[key] = {
            "emotion_label": emo,
            "style_label":   "default",
            "intensity":     round(r["arousal"], 4),
        }

    # --- LibriSpeech: V/A/D-derived emotion + V/A/D intensity ---
    libri_vad_path = Path("data/librispeech_emotion_vad.json")
    with open(libri_vad_path) as f:
        libri_vad = json.load(f)

    for utt_id, r in libri_vad.items():
        emo = vad_to_emotion(r["valence"], r["arousal"])
        out[utt_id] = {
            "emotion_label": emo,
            "style_label":   "default",
            "intensity":     round(r["arousal"], 4),
        }

    out_path = Path("data/utterance_metadata_v5.json")
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"Wrote {len(out)} entries to {out_path}")

    # Stats
    print(f"\nEmotion distribution overall:")
    emo_counts = Counter(v["emotion_label"] for v in out.values())
    for e, c in emo_counts.most_common():
        print(f"  {e:>10}: {c:>6}  ({c/len(out):.1%})")

    print(f"\nESD only ({len(esd_vad)} entries):")
    esd_emo = Counter(out[k]["emotion_label"] for k in esd_vad if k in out)
    for e, c in esd_emo.most_common():
        print(f"  {e:>10}: {c:>5}")

    print(f"\nLibriSpeech only ({len(libri_vad)} entries):")
    libri_emo = Counter(out[k]["emotion_label"] for k in libri_vad if k in out)
    for e, c in libri_emo.most_common():
        print(f"  {e:>10}: {c:>5}  ({c/len(libri_vad):.1%})")

    # Intensity range
    import numpy as np
    intensities = np.array([v["intensity"] for v in out.values()])
    print(f"\nIntensity (arousal) distribution:")
    print(f"  min={intensities.min():.3f}  p25={np.percentile(intensities, 25):.3f}  "
          f"med={np.median(intensities):.3f}  p75={np.percentile(intensities, 75):.3f}  "
          f"max={intensities.max():.3f}")


if __name__ == "__main__":
    main()
