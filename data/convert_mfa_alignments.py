"""
Convert MFA TextGrid phoneme alignments to our JSON duration format.
MFA TextGrids have exact phoneme start/end times from forced alignment.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

ARTICULATORY_RATE = 50  # SPARC = 50 Hz


def parse_textgrid(textgrid_path: str) -> tuple[list, list]:
    """
    Parse a Praat TextGrid file to extract word and phone tiers.
    Returns:
        words: list of (word, start_sec, end_sec)
        phones: list of (phone, start_sec, end_sec)
    """
    text = Path(textgrid_path).read_text()

    words = []
    phones = []

    # Split into tiers
    tiers = text.split('"IntervalTier"')

    for tier in tiers[1:]:  # skip header
        # Determine tier type by the NAME attribute (not arbitrary text occurrences)
        is_words = bool(re.search(r'name\s*=\s*"words"', tier))
        is_phones = bool(re.search(r'name\s*=\s*"phones"', tier))

        if not is_words and not is_phones:
            continue

        intervals = re.findall(
            r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
            tier,
        )

        for xmin, xmax, label in intervals:
            entry = (label.strip(), float(xmin), float(xmax))
            if is_words:
                words.append(entry)
            elif is_phones:
                phones.append(entry)

    return words, phones


def mfa_phones_to_durations(
    mfa_phones: list[tuple[str, float, float]],
    g2p_phonemes: list[str],
    total_frames: int,
) -> list[int]:
    """
    Convert MFA phone-level timestamps to per-phoneme frame durations.

    Key approach: Build a frame-level label array from MFA timestamps,
    then assign frames to g2p phonemes by walking both sequences.
    This correctly handles the offset from leading/trailing silence.
    """
    if not mfa_phones:
        return _proportional_fallback(g2p_phonemes, total_frames)

    # Build frame-level label from MFA: for each frame, what MFA phone covers it?
    # Frame i corresponds to time [i/RATE, (i+1)/RATE)
    frame_labels = [""] * total_frames
    for phone, start, end in mfa_phones:
        sf = max(0, int(round(start * ARTICULATORY_RATE)))
        ef = min(total_frames, int(round(end * ARTICULATORY_RATE)))
        label = "<SIL>" if phone in ("", "sil", "sp", "spn") else phone.upper()
        for f in range(sf, ef):
            frame_labels[f] = label

    # Build MFA speech-only sequence (for matching to g2p)
    mfa_speech = []
    for phone, start, end in mfa_phones:
        if phone not in ("", "sil", "sp", "spn"):
            mfa_speech.append((phone.upper(), start, end))

    # g2p speech-only sequence
    g2p_speech_indices = [i for i, p in enumerate(g2p_phonemes) if p != "<sil>"]

    # Match 1-to-1 by position: g2p speech phone j <-> MFA speech phone j
    n_match = min(len(g2p_speech_indices), len(mfa_speech))

    # For each g2p phoneme, compute its frame range from MFA timestamps
    # Speech phones: use MFA start/end directly
    # <sil> phones: use gap between adjacent MFA speech phones
    durations = [1] * len(g2p_phonemes)  # default 1

    # Assign speech phone durations from MFA
    phone_frame_ranges = {}  # g2p_idx -> (start_frame, end_frame)
    for j in range(n_match):
        g2p_idx = g2p_speech_indices[j]
        _, mfa_start, mfa_end = mfa_speech[j]
        sf = max(0, int(round(mfa_start * ARTICULATORY_RATE)))
        ef = min(total_frames, int(round(mfa_end * ARTICULATORY_RATE)))
        durations[g2p_idx] = max(1, ef - sf)
        phone_frame_ranges[g2p_idx] = (sf, ef)

    # Assign <sil> durations from gaps
    for i, p in enumerate(g2p_phonemes):
        if p != "<sil>":
            continue

        # Find adjacent speech phones
        prev_end = 0
        next_start = total_frames
        for idx in g2p_speech_indices:
            if idx < i and idx in phone_frame_ranges:
                prev_end = max(prev_end, phone_frame_ranges[idx][1])
            elif idx > i and idx in phone_frame_ranges:
                next_start = min(next_start, phone_frame_ranges[idx][0])
                break

        durations[i] = max(1, next_start - prev_end)

    # Adjust to match total_frames exactly
    current_sum = sum(durations)
    diff = total_frames - current_sum
    if diff != 0:
        indices = sorted(range(len(durations)), key=lambda i: -durations[i])
        for j in range(abs(diff)):
            idx = indices[j % len(indices)]
            if diff > 0:
                durations[idx] += 1
            elif durations[idx] > 1:
                durations[idx] -= 1

    assert len(durations) == len(g2p_phonemes)
    assert sum(durations) == total_frames, f"Sum {sum(durations)} != {total_frames}"
    assert all(d >= 1 for d in durations)
    return durations


def _proportional_fallback(phonemes, total_frames):
    from data.align import PHONEME_RELATIVE_DURATIONS, DEFAULT_DURATION
    weights = [PHONEME_RELATIVE_DURATIONS.get(p, DEFAULT_DURATION) for p in phonemes]
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

    # Find all TextGrid files
    textgrid_dir = Path(args.textgrid_dir)
    tg_lookup = {}
    for tg in textgrid_dir.rglob("*.TextGrid"):
        tg_lookup[tg.stem] = str(tg)

    alignments = {}
    mfa_aligned = 0
    fallback = 0
    skipped = 0
    errors = 0

    for utt_id, pdata in tqdm(phoneme_data.items(), desc="Converting MFA alignments"):
        npz_path = features_dir / f"{utt_id}.npz"
        if not npz_path.exists():
            skipped += 1
            continue

        data = np.load(npz_path)
        total_frames = data["ema"].shape[0]
        phonemes = pdata["phonemes"]

        if not phonemes:
            skipped += 1
            continue

        tg_path = tg_lookup.get(utt_id)
        if tg_path:
            try:
                _, mfa_phones = parse_textgrid(tg_path)
                durations = mfa_phones_to_durations(mfa_phones, phonemes, total_frames)
                mfa_aligned += 1
            except Exception as e:
                if errors < 5:
                    print(f"  Error on {utt_id}: {e}")
                durations = _proportional_fallback(phonemes, total_frames)
                fallback += 1
                errors += 1
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

    print(f"\nDone: {len(alignments)} total ({mfa_aligned} MFA-aligned, {fallback} fallback, {skipped} skipped, {errors} errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features_combined")
    parser.add_argument("--phonemes-path", type=str, default="data/processed_combined/phonemes.json")
    parser.add_argument("--textgrid-dir", type=str, default="data/mfa_alignments")
    parser.add_argument("--output-path", type=str, default="data/processed_combined/alignments_mfa.json")
    args = parser.parse_args()

    main(args)
