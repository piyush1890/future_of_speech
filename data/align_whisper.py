"""
Whisper-based forced alignment: get word-level timestamps, then distribute
phoneme durations within each word proportionally.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

PHONEME_RELATIVE_DURATIONS = {
    "AA": 1.5, "AA0": 1.5, "AA1": 1.6, "AA2": 1.4, "AE": 1.4, "AE0": 1.4, "AE1": 1.5, "AE2": 1.3,
    "AH": 1.2, "AH0": 1.0, "AH1": 1.3, "AH2": 1.1, "AO": 1.5, "AO0": 1.5, "AO1": 1.6, "AO2": 1.4,
    "AW": 1.6, "AW0": 1.6, "AW1": 1.7, "AW2": 1.5, "AY": 1.6, "AY0": 1.6, "AY1": 1.7, "AY2": 1.5,
    "EH": 1.3, "EH0": 1.3, "EH1": 1.4, "EH2": 1.2, "ER": 1.4, "ER0": 1.4, "ER1": 1.5, "ER2": 1.3,
    "EY": 1.5, "EY0": 1.5, "EY1": 1.6, "EY2": 1.4, "IH": 1.1, "IH0": 1.0, "IH1": 1.2, "IH2": 1.0,
    "IY": 1.3, "IY0": 1.3, "IY1": 1.4, "IY2": 1.2, "OW": 1.5, "OW0": 1.5, "OW1": 1.6, "OW2": 1.4,
    "OY": 1.6, "OY0": 1.6, "OY1": 1.7, "OY2": 1.5, "UH": 1.2, "UH0": 1.2, "UH1": 1.3, "UH2": 1.1,
    "UW": 1.4, "UW0": 1.4, "UW1": 1.5, "UW2": 1.3,
    "F": 1.0, "V": 0.9, "TH": 1.0, "DH": 0.8, "S": 1.1, "Z": 1.0, "SH": 1.1, "ZH": 1.0, "HH": 0.8,
    "M": 1.0, "N": 1.0, "NG": 1.0, "L": 1.0, "R": 1.0, "W": 0.9, "Y": 0.8,
    "P": 0.7, "B": 0.7, "T": 0.7, "D": 0.7, "K": 0.8, "G": 0.7, "CH": 0.9, "JH": 0.9,
    "<sil>": 0.5,
}

ARTICULATORY_RATE = 50  # SPARC outputs 50 frames/sec


def get_whisper_word_times(whisper_model, audio_path: str) -> list[tuple[str, float, float]]:
    """Get word-level timestamps from Whisper."""
    result = whisper_model.transcribe(audio_path, word_timestamps=True, language="en")
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            word_text = w["word"].strip().upper().strip(".,;:!?\"'()-")
            if word_text:
                words.append((word_text, w["start"], w["end"]))
    return words


def compute_durations_from_word_times(
    phonemes: list[str],
    text: str,
    word_times: list[tuple[str, float, float]],
    total_frames: int,
) -> list[int]:
    """
    Convert Whisper word timestamps to per-phoneme frame durations.
    """
    if not word_times or not phonemes:
        return _proportional_fallback(phonemes, total_frames)

    audio_dur = total_frames / ARTICULATORY_RATE
    transcript_words = text.strip().split()

    # Split phonemes into word groups (separated by <sil>)
    word_groups = []  # list of (start_idx, list_of_phonemes)
    current = []
    current_start = 0
    for i, p in enumerate(phonemes):
        if p == "<sil>":
            if current:
                word_groups.append((current_start, current))
                current = []
            current_start = i + 1
        else:
            if not current:
                current_start = i
            current.append(p)
    if current:
        word_groups.append((current_start, current))

    # Match word groups to Whisper word times (by position)
    n_match = min(len(word_groups), len(word_times))

    # Build per-phoneme durations
    durations = [0] * len(phonemes)

    for i, p in enumerate(phonemes):
        if p == "<sil>":
            # Find which word boundary this silence is between
            words_before = sum(1 for j in range(i) if phonemes[j] == "<sil>")
            if phonemes[0] != "<sil>":
                words_before += 1

            if words_before == 0:
                # Leading silence
                if word_times:
                    dur_sec = max(0.01, word_times[0][1])
                else:
                    dur_sec = 0.02
            elif words_before <= len(word_times):
                prev_end = word_times[words_before - 1][2] if words_before - 1 < len(word_times) else 0
                next_start = word_times[words_before][1] if words_before < len(word_times) else audio_dur
                dur_sec = max(0.01, next_start - prev_end)
            else:
                dur_sec = 0.02

            durations[i] = max(1, int(round(dur_sec * ARTICULATORY_RATE)))

    # For speech phonemes, use word timing
    for group_idx, (start_idx, group_phons) in enumerate(word_groups):
        if group_idx < n_match:
            _, w_start, w_end = word_times[group_idx]
            word_dur_sec = max(0.02, w_end - w_start)
        else:
            # No matching word time — estimate
            word_dur_sec = len(group_phons) * 0.06  # ~60ms per phoneme

        # Distribute word duration among phonemes proportionally
        weights = [PHONEME_RELATIVE_DURATIONS.get(p, 1.0) for p in group_phons]
        total_w = sum(weights)

        for j, (p, w) in enumerate(zip(group_phons, weights)):
            dur_sec = word_dur_sec * w / total_w
            durations[start_idx + j] = max(1, int(round(dur_sec * ARTICULATORY_RATE)))

    # Adjust to match total_frames
    diff = total_frames - sum(durations)
    indices = sorted(range(len(durations)), key=lambda i: -durations[i])
    for j in range(abs(diff)):
        idx = indices[j % len(indices)]
        if diff > 0:
            durations[idx] += 1
        elif durations[idx] > 1:
            durations[idx] -= 1

    assert len(durations) == len(phonemes), f"{len(durations)} != {len(phonemes)}"
    assert all(d >= 1 for d in durations), f"Zero duration found"
    return durations


def _proportional_fallback(phonemes, total_frames):
    weights = [PHONEME_RELATIVE_DURATIONS.get(p, 1.0) for p in phonemes]
    total_w = sum(weights)
    if total_w == 0:
        return [max(1, total_frames // max(len(phonemes), 1))] * len(phonemes)
    durations = [max(1, int(round(w / total_w * total_frames))) for w in weights]
    diff = total_frames - sum(durations)
    indices = sorted(range(len(durations)), key=lambda i: -durations[i])
    for j in range(abs(diff)):
        idx = indices[j % len(indices)]
        if diff > 0:
            durations[idx] += 1
        elif durations[idx] > 1:
            durations[idx] -= 1
    return durations


def main(args):
    features_dir = Path(args.features_dir)

    with open(args.phonemes_path) as f:
        phoneme_data = json.load(f)

    # Build audio path lookup
    audio_paths = {}
    for base_dir in args.audio_dirs:
        for flac in Path(base_dir).rglob("*.flac"):
            audio_paths[flac.stem] = str(flac)

    # Load Whisper
    import whisper
    print(f"Loading Whisper model '{args.whisper_model}'...")
    whisper_model = whisper.load_model(args.whisper_model)
    print("Whisper loaded.")

    alignments = {}
    skipped = 0
    aligned = 0
    fallback = 0

    for utt_id, pdata in tqdm(phoneme_data.items(), desc="Aligning"):
        npz_path = features_dir / f"{utt_id}.npz"
        if not npz_path.exists():
            skipped += 1
            continue

        data = np.load(npz_path)
        total_frames = data["ema"].shape[0]
        phonemes = pdata["phonemes"]
        text = pdata["text"]

        if not phonemes:
            skipped += 1
            continue

        audio_path = audio_paths.get(utt_id)
        if audio_path:
            try:
                word_times = get_whisper_word_times(whisper_model, audio_path)
                durations = compute_durations_from_word_times(
                    phonemes, text, word_times, total_frames
                )
                aligned += 1
            except Exception as e:
                durations = _proportional_fallback(phonemes, total_frames)
                fallback += 1
        else:
            durations = _proportional_fallback(phonemes, total_frames)
            fallback += 1

        alignments[utt_id] = {
            "durations": durations,
            "total_frames": total_frames,
            "num_phonemes": len(phonemes),
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(alignments, f)

    print(f"\nDone: {len(alignments)} total ({aligned} whisper-aligned, {fallback} fallback, {skipped} skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features_combined")
    parser.add_argument("--phonemes-path", type=str, default="data/processed_combined/phonemes.json")
    parser.add_argument("--audio-dirs", nargs="+", default=[
        "data/LibriSpeech/dev-clean",
        "data/LibriSpeech/train-clean-100",
    ])
    parser.add_argument("--output-path", type=str, default="data/processed_combined/alignments_whisper.json")
    parser.add_argument("--whisper-model", type=str, default="tiny")
    args = parser.parse_args()

    main(args)
