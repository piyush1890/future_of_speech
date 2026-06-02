"""
Compute phoneme-to-frame duration alignment.
MVP approach: proportional duration based on typical phoneme lengths.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Approximate relative durations for ARPAbet phonemes (from linguistic data)
# Vowels are longer than consonants, stops are shortest
PHONEME_RELATIVE_DURATIONS = {
    # Vowels (longer)
    "AA": 1.5, "AA0": 1.5, "AA1": 1.6, "AA2": 1.4,
    "AE": 1.4, "AE0": 1.4, "AE1": 1.5, "AE2": 1.3,
    "AH": 1.2, "AH0": 1.0, "AH1": 1.3, "AH2": 1.1,
    "AO": 1.5, "AO0": 1.5, "AO1": 1.6, "AO2": 1.4,
    "AW": 1.6, "AW0": 1.6, "AW1": 1.7, "AW2": 1.5,
    "AY": 1.6, "AY0": 1.6, "AY1": 1.7, "AY2": 1.5,
    "EH": 1.3, "EH0": 1.3, "EH1": 1.4, "EH2": 1.2,
    "ER": 1.4, "ER0": 1.4, "ER1": 1.5, "ER2": 1.3,
    "EY": 1.5, "EY0": 1.5, "EY1": 1.6, "EY2": 1.4,
    "IH": 1.1, "IH0": 1.0, "IH1": 1.2, "IH2": 1.0,
    "IY": 1.3, "IY0": 1.3, "IY1": 1.4, "IY2": 1.2,
    "OW": 1.5, "OW0": 1.5, "OW1": 1.6, "OW2": 1.4,
    "OY": 1.6, "OY0": 1.6, "OY1": 1.7, "OY2": 1.5,
    "UH": 1.2, "UH0": 1.2, "UH1": 1.3, "UH2": 1.1,
    "UW": 1.4, "UW0": 1.4, "UW1": 1.5, "UW2": 1.3,
    # Fricatives (medium)
    "F": 1.0, "V": 0.9, "TH": 1.0, "DH": 0.8,
    "S": 1.1, "Z": 1.0, "SH": 1.1, "ZH": 1.0,
    "HH": 0.8,
    # Nasals (medium)
    "M": 1.0, "N": 1.0, "NG": 1.0,
    # Liquids/glides (medium)
    "L": 1.0, "R": 1.0, "W": 0.9, "Y": 0.8,
    # Stops (short)
    "P": 0.7, "B": 0.7, "T": 0.7, "D": 0.7,
    "K": 0.8, "G": 0.7,
    # Affricates
    "CH": 0.9, "JH": 0.9,
    # Silence
    "<sil>": 1.5,
}

DEFAULT_DURATION = 1.0


def compute_durations(phonemes: list[str], total_frames: int) -> list[int]:
    """
    Compute per-phoneme frame durations that sum to total_frames.
    Uses relative duration weights to distribute frames proportionally.
    """
    if not phonemes:
        return []

    # Get relative weights
    weights = [PHONEME_RELATIVE_DURATIONS.get(p, DEFAULT_DURATION) for p in phonemes]
    total_weight = sum(weights)

    # Distribute frames proportionally
    raw_durations = [w / total_weight * total_frames for w in weights]

    # Round to integers while preserving total
    durations = [max(1, int(round(d))) for d in raw_durations]

    # Adjust to match total_frames exactly
    diff = total_frames - sum(durations)
    if diff > 0:
        # Add extra frames to longest phonemes
        indices = sorted(range(len(durations)), key=lambda i: -raw_durations[i])
        for i in range(diff):
            durations[indices[i % len(indices)]] += 1
    elif diff < 0:
        # Remove frames from longest phonemes (but keep min 1)
        indices = sorted(range(len(durations)), key=lambda i: -durations[i])
        for i in range(-diff):
            idx = indices[i % len(indices)]
            if durations[idx] > 1:
                durations[idx] -= 1

    assert sum(durations) == total_frames, f"Duration sum {sum(durations)} != {total_frames}"
    return durations


def align(features_dir: str, phonemes_path: str, output_path: str):
    features_dir = Path(features_dir)
    phonemes_path = Path(phonemes_path)

    with open(phonemes_path) as f:
        phoneme_data = json.load(f)

    alignments = {}
    skipped = 0

    for utt_id, pdata in tqdm(phoneme_data.items(), desc="Aligning"):
        npz_path = features_dir / f"{utt_id}.npz"
        if not npz_path.exists():
            skipped += 1
            continue

        data = np.load(npz_path)
        total_frames = data["ema"].shape[0]

        # Phonemes WITHOUT BOS/EOS (those don't have duration in the audio)
        phonemes = pdata["phonemes"]

        if len(phonemes) == 0:
            skipped += 1
            continue

        durations = compute_durations(phonemes, total_frames)

        alignments[utt_id] = {
            "durations": durations,
            "total_frames": total_frames,
            "num_phonemes": len(phonemes),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(alignments, f)

    print(f"Aligned {len(alignments)} utterances, skipped {skipped}")
    print(f"Saved to {output_path}")

    # Print stats
    if alignments:
        avg_frames = np.mean([a["total_frames"] for a in alignments.values()])
        avg_phonemes = np.mean([a["num_phonemes"] for a in alignments.values()])
        print(f"Avg frames/utterance: {avg_frames:.1f} ({avg_frames/50:.1f}s)")
        print(f"Avg phonemes/utterance: {avg_phonemes:.1f}")
        print(f"Avg frames/phoneme: {avg_frames/avg_phonemes:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--phonemes-path", type=str, default="data/processed/phonemes.json")
    parser.add_argument("--output-path", type=str, default="data/processed/alignments.json")
    args = parser.parse_args()

    align(args.features_dir, args.phonemes_path, args.output_path)
