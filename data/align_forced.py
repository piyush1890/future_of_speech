"""
Forced alignment using torchaudio's CTC-based aligner with wav2vec2.
Produces precise per-word timestamps, then distributes to phonemes within each word.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from g2p_en import G2p


# Relative durations within a word (for distributing word-level time to phonemes)
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
}


def get_word_timestamps(waveform, transcript, bundle, model, device="cpu"):
    """
    Get word-level timestamps using wav2vec2 CTC forced alignment.
    Returns list of (word, start_frame, end_frame) in wav2vec2 frame rate.
    """
    labels = bundle.get_labels()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)

    with torch.no_grad():
        emissions, _ = model(waveform)
    emission = emissions[0]  # (T, C)

    # Build token sequence: word characters separated by |
    words = transcript.upper().strip().split()
    token_str = "|".join(words)
    # Add leading and trailing silence
    token_str = "|" + token_str + "|"

    tokens = []
    for c in token_str:
        if c in labels:
            tokens.append(labels.index(c))
        elif c == " ":
            tokens.append(labels.index("|"))
        # Skip characters not in vocab (apostrophes, etc.)

    if len(tokens) == 0:
        return []

    tokens_tensor = torch.tensor([tokens], dtype=torch.int32)

    try:
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emission.unsqueeze(0).cpu(),
            tokens_tensor,
        )
    except Exception:
        return []

    aligned_tokens = aligned_tokens[0]  # (T_aligned,)

    # Extract word boundaries from alignment
    word_segments = []
    current_word_idx = -1  # -1 = silence before first word
    word_start = 0
    char_pos = 0  # position in token_str

    # Walk through token_str and find word boundaries
    word_boundaries = []
    pos = 0
    for w in words:
        # Find start of this word in token_str (skip |)
        start_in_tokens = token_str.index(w, pos)
        end_in_tokens = start_in_tokens + len(w)
        word_boundaries.append((w, start_in_tokens, end_in_tokens))
        pos = end_in_tokens

    # Map token positions to emission frames
    # aligned_tokens contains token indices at each emission frame
    # Find frame ranges for each word
    token_to_frame = {}
    for frame_idx in range(len(aligned_tokens)):
        tok = aligned_tokens[frame_idx].item()
        if tok > 0:  # 0 = blank
            if tok - 1 not in token_to_frame:
                token_to_frame[tok - 1] = [frame_idx, frame_idx]
            else:
                token_to_frame[tok - 1][1] = frame_idx

    # Map character positions to frames for each word
    result = []
    for word, char_start, char_end in word_boundaries:
        frames = []
        for ci in range(char_start, char_end):
            if ci in token_to_frame:
                frames.extend(token_to_frame[ci])
        if frames:
            result.append((word, min(frames), max(frames)))

    return result


def word_timestamps_to_phoneme_durations(
    word_segments,
    phonemes,
    total_articulatory_frames,
    emission_frames,
    wav_sample_rate=16000,
    emission_rate=50,  # wav2vec2 outputs ~50 frames/sec
    articulatory_rate=50,  # SPARC outputs 50 frames/sec
):
    """
    Convert word-level timestamps to per-phoneme frame durations.

    Strategy:
    1. Map word boundaries to articulatory frame indices
    2. Split each word's frames among its phonemes proportionally
    3. Assign silence frames to <sil> tokens
    """
    if not word_segments:
        # Fallback to proportional
        return _proportional_fallback(phonemes, total_articulatory_frames)

    # Convert emission frames to articulatory frames
    # emission frame rate ≈ 50 Hz, articulatory rate = 50 Hz, so roughly 1:1
    # But need to scale: emission_frames maps to total_articulatory_frames
    scale = total_articulatory_frames / max(emission_frames, 1)

    # Split phonemes into words (separated by <sil>)
    word_phonemes = []
    current = []
    for p in phonemes:
        if p == "<sil>":
            if current:
                word_phonemes.append(current)
                current = []
        else:
            current.append(p)
    if current:
        word_phonemes.append(current)

    # Match word_phonemes to word_segments
    # There might be mismatches, so we do best-effort
    n_words = min(len(word_phonemes), len(word_segments))

    durations = []
    phoneme_idx = 0

    for i, p in enumerate(phonemes):
        if p == "<sil>":
            # Silence between words
            if i == 0:
                # Leading silence
                if word_segments:
                    sil_end = int(word_segments[0][1] * scale)
                    durations.append(max(1, sil_end))
                else:
                    durations.append(max(1, int(total_articulatory_frames * 0.02)))
            elif i == len(phonemes) - 1:
                # Trailing silence — will be adjusted at the end
                durations.append(1)
            else:
                # Find which word boundary we're between
                word_before = sum(1 for pp in phonemes[:i] if pp == "<sil>") - 1
                word_after = word_before + 1
                if word_before < len(word_segments) and word_after < len(word_segments):
                    gap_start = int(word_segments[word_before][2] * scale)
                    gap_end = int(word_segments[word_after][1] * scale)
                    durations.append(max(1, gap_end - gap_start))
                else:
                    durations.append(max(1, int(total_articulatory_frames * 0.02)))
        else:
            # Find which word this phoneme belongs to
            word_idx = sum(1 for pp in phonemes[:i] if pp == "<sil>")
            if phonemes[0] == "<sil>":
                word_idx -= 1

            if 0 <= word_idx < n_words:
                word_phons = word_phonemes[word_idx]
                ws = word_segments[word_idx]
                word_start_frame = int(ws[1] * scale)
                word_end_frame = int(ws[2] * scale)
                word_duration = max(1, word_end_frame - word_start_frame)

                # Distribute within word proportionally
                weight = PHONEME_RELATIVE_DURATIONS.get(p, 1.0)
                total_weight = sum(PHONEME_RELATIVE_DURATIONS.get(pp, 1.0) for pp in word_phons)
                dur = max(1, int(round(word_duration * weight / total_weight)))
                durations.append(dur)
            else:
                durations.append(max(1, int(total_articulatory_frames / len(phonemes))))

    # Adjust to match total
    diff = total_articulatory_frames - sum(durations)
    if diff != 0:
        indices = sorted(range(len(durations)), key=lambda i: -durations[i])
        for i in range(abs(diff)):
            idx = indices[i % len(indices)]
            if diff > 0:
                durations[idx] += 1
            elif durations[idx] > 1:
                durations[idx] -= 1

    assert len(durations) == len(phonemes)
    return durations


def _proportional_fallback(phonemes, total_frames):
    """Fallback to simple proportional alignment."""
    weights = [PHONEME_RELATIVE_DURATIONS.get(p, 1.0) for p in phonemes]
    total_w = sum(weights)
    durations = [max(1, int(round(w / total_w * total_frames))) for w in weights]
    diff = total_frames - sum(durations)
    indices = sorted(range(len(durations)), key=lambda i: -durations[i])
    for i in range(abs(diff)):
        idx = indices[i % len(indices)]
        if diff > 0:
            durations[idx] += 1
        elif durations[idx] > 1:
            durations[idx] -= 1
    return durations


def main(args):
    features_dir = Path(args.features_dir)
    device = args.device

    with open(args.phonemes_path) as f:
        phoneme_data = json.load(f)

    # Build audio path lookup
    audio_paths = {}
    for base_dir in args.audio_dirs:
        for flac in Path(base_dir).rglob("*.flac"):
            audio_paths[flac.stem] = str(flac)

    # Load wav2vec2 model
    print("Loading wav2vec2 alignment model...")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    model.eval()
    print("Model loaded.")

    alignments = {}
    skipped = 0
    forced = 0
    fallback = 0

    for utt_id, pdata in tqdm(phoneme_data.items(), desc="Forced aligning"):
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
                waveform, sr = torchaudio.load(audio_path)
                if sr != bundle.sample_rate:
                    waveform = torchaudio.transforms.Resample(sr, bundle.sample_rate)(waveform)

                word_segs = get_word_timestamps(waveform, text, bundle, model, device)

                if word_segs:
                    # Get emission frame count
                    with torch.no_grad():
                        emissions, _ = model(waveform.to(device))
                    emission_frames = emissions.shape[1]

                    durations = word_timestamps_to_phoneme_durations(
                        word_segs, phonemes, total_frames, emission_frames
                    )
                    forced += 1
                else:
                    durations = _proportional_fallback(phonemes, total_frames)
                    fallback += 1
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

    print(f"\nDone: {len(alignments)} aligned ({forced} forced, {fallback} fallback), {skipped} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features_combined")
    parser.add_argument("--phonemes-path", type=str, default="data/processed_combined/phonemes.json")
    parser.add_argument("--audio-dirs", nargs="+", default=[
        "data/LibriSpeech/dev-clean",
        "data/LibriSpeech/train-clean-100",
    ])
    parser.add_argument("--output-path", type=str, default="data/processed_combined/alignments_forced.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
