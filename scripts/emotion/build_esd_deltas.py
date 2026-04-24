"""
Build phoneme-indexed emotion deltas from ESD SPARC features.

INPUT:  data/esd_features/*.npz — 17,500 files from 10 English speakers × 5 emotions × 350 utterances.
OUTPUT: data/emotion_deltas.npz
          deltas     shape (V, E, 14)    per-(phoneme, emotion) articulatory delta vs neutral
          phonemes   list of V symbols   matches data/processed_all/vocab_mfa.json indexing
          emotions   list of E names     non-neutral emotions: Happy, Sad, Angry, Surprise
          dur_scale  shape (E,)          per-emotion global duration multiplier vs neutral
          counts     shape (V, E)        number of frames used per (phoneme, emotion) — diagnostic

ALGORITHM:
  Per utterance:
    1. Load features (ema 12d, pitch 1d raw Hz, loudness 1d)
    2. Apply log-pitch: pitch_log = log(pitch + 1) to match training distribution
    3. Construct 14-dim feature matrix
    4. Run g2p on transcript to get ARPABET phonemes; filter+prepend <sil>
    5. Map phonemes to vocab indices
    6. Proportionally align: each phoneme gets ceil(T/N) frames of the utterance
    7. For each frame, record (phoneme_idx, emotion, feature_vec) triple

  Aggregation:
    mean[p, e] = mean_over_matching_frames(features)
    delta[p, e] = mean[p, e] - mean[p, "Neutral"]

  Duration scale:
    For each (speaker, sentence_base) with both neutral and emotional recordings:
        ratio = frames_emotional / frames_neutral
    dur_scale[e] = mean over all such ratios

EDGE CASES:
  - Special tokens (<pad>, <bos>, <eos>, <sil>): zero delta (no emotion on silence)
  - Phonemes with fewer than MIN_COUNT frames in any (phoneme, emotion) cell:
    warn but still compute (may be noisy)
  - g2p phoneme count of 0 after filtering: skip utterance
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


NON_NEUTRAL_EMOTIONS = ["Happy", "Sad", "Angry", "Surprise"]
NEUTRAL = "Neutral"
SPECIAL_TOKENS = {"<pad>", "<bos>", "<eos>", "<sil>"}
FEATURE_DIMS = 14
MIN_COUNT_WARN = 50     # warn if a (phoneme, emotion) cell has fewer frames than this


def text_to_phonemes(g2p, text):
    """MFA-style phoneme sequence: drop punctuation/spaces, prepend a single <sil>."""
    raw = g2p(text)
    ph = [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
    return ["<sil>"] + ph


def features_14d(npz):
    """Assemble 14-dim features with log-pitch from a loaded npz."""
    ema = np.asarray(npz["ema"], dtype=np.float32)          # (T, 12)
    pitch = np.asarray(npz["pitch"], dtype=np.float32)      # (T,)
    loudness = np.asarray(npz["loudness"], dtype=np.float32) # (T,)
    # length reconcile (Colab script already did this, but be safe)
    T = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
    ema, pitch, loudness = ema[:T], pitch[:T], loudness[:T]
    pitch_log = np.log(pitch + 1.0)
    return np.concatenate([ema, pitch_log[:, None], loudness[:, None]], axis=1)  # (T, 14)


def assign_frames_to_phonemes(num_frames: int, num_phonemes: int):
    """Proportional alignment. Returns a (num_frames,) int array where each value
    is the phoneme-sequence index (0..num_phonemes-1) that owns that frame."""
    if num_phonemes == 0:
        return np.array([], dtype=np.int64)
    # Allocate frames as evenly as possible. e.g. 117 frames / 9 phonemes = 13 each.
    # Last phoneme gets the remainder.
    base = num_frames // num_phonemes
    remainder = num_frames - base * num_phonemes
    counts = np.full(num_phonemes, base, dtype=np.int64)
    # Distribute remainder across the first few phonemes
    counts[:remainder] += 1
    # Repeat phoneme indices according to counts
    indices = np.repeat(np.arange(num_phonemes, dtype=np.int64), counts)
    return indices  # shape (num_frames,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="data/esd_features")
    p.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    p.add_argument("--out-path", type=str, default="data/emotion_deltas.npz")
    args = p.parse_args()

    features_dir = Path(args.features_dir)
    vocab = PhonemeVocab(args.vocab_path)
    V = len(vocab)
    E = len(NON_NEUTRAL_EMOTIONS)

    g2p = G2p()

    # Accumulators:  sum[p, e] = running sum of 14-dim feature vectors
    #                count[p, e] = number of frames contributed
    # Indexed: [phoneme_index, emotion_label] where emotion_label is a string key.
    sums = {e: np.zeros((V, FEATURE_DIMS), dtype=np.float64) for e in NON_NEUTRAL_EMOTIONS + [NEUTRAL]}
    counts = {e: np.zeros(V, dtype=np.int64) for e in NON_NEUTRAL_EMOTIONS + [NEUTRAL]}

    # For duration scale: per (speaker, sentence_base) collect frame counts per emotion.
    # Parallel-text utterance ids in ESD follow a pattern where utt_id encodes
    # (speaker, index). Index range per emotion is mostly disjoint (e.g., Neutral=001-350,
    # Angry=351-700), but sentence TEXT is the same for parallel slots with offset +350.
    # So we key by (speaker, text) to find parallel pairs robustly.
    utt_frame_counts = defaultdict(dict)  # (speaker, text) -> {emotion: frame_count}

    files = sorted(features_dir.glob("*.npz"))
    print(f"Found {len(files)} ESD feature files")

    unknown_phonemes = set()
    skipped_no_text = 0
    skipped_no_phonemes = 0
    total_frames_processed = 0

    corrupt_files = []
    for f in tqdm(files, desc="Accumulating per-(phoneme, emotion) stats"):
        try:
            d = np.load(f, allow_pickle=True)
        except (EOFError, OSError, ValueError) as e:
            corrupt_files.append(f.name)
            continue
        emotion = str(d["emotion"])
        text = str(d["text"])
        speaker = str(d["speaker"])

        if emotion not in sums:
            # Unknown emotion label (shouldn't happen but safety)
            continue

        features = features_14d(d)
        T = features.shape[0]
        if T == 0:
            continue

        if not text:
            skipped_no_text += 1
            continue

        phonemes = text_to_phonemes(g2p, text)
        if not phonemes:
            skipped_no_phonemes += 1
            continue

        # Map to vocab indices, tracking unknowns
        phoneme_indices = []
        for ph in phonemes:
            idx = vocab.token2idx.get(ph)
            if idx is None:
                unknown_phonemes.add(ph)
                # Fall through: use <sil> as placeholder (conservative — contributes to <sil> stats which are zero'd later)
                idx = vocab.token2idx.get("<sil>", 0)
            phoneme_indices.append(idx)
        phoneme_indices = np.array(phoneme_indices, dtype=np.int64)
        N = len(phoneme_indices)

        # Proportional frame-to-phoneme alignment
        frame_phoneme_seq_idx = assign_frames_to_phonemes(T, N)         # (T,) indices into phoneme sequence
        frame_phoneme_vocab_idx = phoneme_indices[frame_phoneme_seq_idx]  # (T,) vocab indices

        # Accumulate per-phoneme-index, per-emotion
        for p_idx in range(V):
            mask = frame_phoneme_vocab_idx == p_idx
            if not mask.any():
                continue
            sums[emotion][p_idx] += features[mask].sum(axis=0).astype(np.float64)
            counts[emotion][p_idx] += mask.sum()

        total_frames_processed += T

        # Duration tracking (parallel pair math)
        key = (speaker, text)
        utt_frame_counts[key][emotion] = T

    print(f"\nTotal frames processed: {total_frames_processed:,}")
    if unknown_phonemes:
        print(f"Unknown phonemes (mapped to <sil>): {sorted(unknown_phonemes)}")
    print(f"Utterances skipped (no text): {skipped_no_text}")
    print(f"Utterances skipped (no phonemes after filter): {skipped_no_phonemes}")
    if corrupt_files:
        print(f"Corrupt files skipped: {len(corrupt_files)}")
        for cf in corrupt_files[:5]:
            print(f"    {cf}")

    # Compute means
    means = {}
    for e in NON_NEUTRAL_EMOTIONS + [NEUTRAL]:
        m = np.zeros((V, FEATURE_DIMS), dtype=np.float64)
        for p_idx in range(V):
            if counts[e][p_idx] > 0:
                m[p_idx] = sums[e][p_idx] / counts[e][p_idx]
        means[e] = m

    # Compute deltas (shape V, E, 14)
    deltas = np.zeros((V, E, FEATURE_DIMS), dtype=np.float32)
    counts_out = np.zeros((V, E), dtype=np.int64)
    for e_idx, emo in enumerate(NON_NEUTRAL_EMOTIONS):
        raw_delta = means[emo] - means[NEUTRAL]
        deltas[:, e_idx, :] = raw_delta.astype(np.float32)
        counts_out[:, e_idx] = np.minimum(counts[emo], counts[NEUTRAL])

    # Zero out deltas for special tokens
    phonemes_list = [p for p, _ in sorted(vocab.token2idx.items(), key=lambda x: x[1])]
    for p_idx, phoneme in enumerate(phonemes_list):
        if phoneme in SPECIAL_TOKENS:
            deltas[p_idx] = 0.0

    # Warn about low-count cells
    low_count_cells = []
    for e_idx, emo in enumerate(NON_NEUTRAL_EMOTIONS):
        for p_idx, phoneme in enumerate(phonemes_list):
            if phoneme in SPECIAL_TOKENS:
                continue
            c = counts_out[p_idx, e_idx]
            if 0 < c < MIN_COUNT_WARN:
                low_count_cells.append((phoneme, emo, int(c)))
    if low_count_cells:
        print(f"\n{len(low_count_cells)} (phoneme, emotion) cells with < {MIN_COUNT_WARN} frames:")
        for phoneme, emo, c in low_count_cells[:10]:
            print(f"    {phoneme:8s}  {emo:10s}  count={c}")
        if len(low_count_cells) > 10:
            print(f"    ... and {len(low_count_cells) - 10} more")

    # Duration scales: for each (speaker, sentence), compute emotional/neutral ratio
    dur_scale = np.zeros(E, dtype=np.float32)
    dur_ratios = {e: [] for e in NON_NEUTRAL_EMOTIONS}
    for key, emos in utt_frame_counts.items():
        if NEUTRAL not in emos:
            continue
        neutral_T = emos[NEUTRAL]
        for e in NON_NEUTRAL_EMOTIONS:
            if e in emos and neutral_T > 0:
                dur_ratios[e].append(emos[e] / neutral_T)
    for e_idx, emo in enumerate(NON_NEUTRAL_EMOTIONS):
        ratios = dur_ratios[emo]
        if ratios:
            dur_scale[e_idx] = float(np.mean(ratios))
        else:
            dur_scale[e_idx] = 1.0

    print(f"\nDuration scales (emotional / neutral mean duration):")
    for emo, s in zip(NON_NEUTRAL_EMOTIONS, dur_scale):
        print(f"  {emo}: {s:.3f}  (from {len(dur_ratios[emo])} parallel pairs)")

    # Save
    # Match handcrafted schema but with lowercase emotion keys for inference compatibility.
    emotions_out = [e.lower() for e in NON_NEUTRAL_EMOTIONS]
    np.savez(
        args.out_path,
        deltas=deltas,
        phonemes=np.array(phonemes_list),
        emotions=np.array(emotions_out),
        dur_scale=dur_scale,
        counts=counts_out,
    )
    print(f"\nWrote {args.out_path}")
    print(f"  deltas shape: {deltas.shape}")
    print(f"  emotions: {emotions_out}")

    # Quick sanity summary: per-emotion mean magnitude of delta across phonemes
    print(f"\nPer-emotion delta magnitudes (L2 norm of 14-dim delta, averaged over non-special phonemes):")
    special_mask = np.array([p not in SPECIAL_TOKENS for p in phonemes_list])
    for e_idx, emo in enumerate(NON_NEUTRAL_EMOTIONS):
        norms = np.linalg.norm(deltas[special_mask, e_idx, :], axis=1)
        print(f"  {emo}: mean L2={norms.mean():.4f}  max={norms.max():.4f}")

    # Per-channel mean delta (which channels does each emotion modify most?)
    print(f"\nMean delta per channel per emotion (averaged over non-special phonemes):")
    channel_labels = [f"EMA{i}" for i in range(12)] + ["log_pitch", "loudness"]
    print(f"  {'emotion':10s}  " + "  ".join(f"{c:>8s}" for c in channel_labels))
    for e_idx, emo in enumerate(NON_NEUTRAL_EMOTIONS):
        per_ch = deltas[special_mask, e_idx, :].mean(axis=0)
        print(f"  {emo:10s}  " + "  ".join(f"{v:+8.4f}" for v in per_ch))


if __name__ == "__main__":
    main()
