"""
Build a phoneme-indexed emotion delta table.

OUTPUT: data/emotion_deltas.npz with:
  - deltas:  shape (V, E, 14) where V=vocab_size, E=num_non_neutral_emotions, 14=feature dims
  - phonemes: list of phoneme symbols (length V)
  - emotions: list of emotion names (length E)

This file is used at inference time: for a frame whose phoneme is `p` and target
emotion is `e`, add `deltas[phoneme_to_idx[p], emotion_to_idx[e]]` to its 14-dim
feature vector.

MODES:
  --mode=handcrafted:
      Build deltas from hand-authored per-emotion recipes (NO training data required).
      Useful as a proof-of-concept to validate the inference application math before
      investing in ESD preprocessing. Each emotion is a single 14-dim vector broadcast
      across all phonemes (no per-phoneme variation yet).

  --mode=esd:
      Build deltas from ESD-processed features. Requires build_esd_features.py to
      have been run first (produces per-(phoneme, emotion) feature tuples). This is
      the real phoneme-indexed version.
"""
import argparse
import json
from pathlib import Path

import numpy as np


EMOTIONS = ["happy", "sad", "angry", "calm"]  # excludes neutral (delta=0 by definition)
FEATURE_DIMS = 14  # 12 EMA + log-pitch + loudness


# ---------------------------------------------------------------------------
# Hand-authored baseline (mode=handcrafted)
# ---------------------------------------------------------------------------
#
# These per-emotion 14-dim vectors encode rough expected articulatory shifts
# across the 14 channels. They are NOT derived from data and are a proof-of-
# concept baseline. Real values will be learned from ESD (mode=esd).
#
# Channel layout:
#   [0:12]  EMA dims (tongue-tip-x, tongue-tip-y, tongue-dorsum-x, ..., lip-aperture-x, ...)
#           We don't have exact semantics per channel without inspecting SPARC,
#           so we apply a modest uniform perturbation scaled by emotion intensity.
#   [12]    log-pitch (remember: features are in log space!)
#           pitch change factor = exp(delta_logpitch).
#           angry: +0.10 → pitch ×1.10 (+10%)
#           happy: +0.13 → pitch ×1.14 (+14%)
#           sad:   -0.10 → pitch ×0.90 (-10%)
#           calm:  -0.05 → pitch ×0.95 (-5%)
#   [13]    loudness. SPARC's loudness scale means different things at different
#           values; we use small increments to avoid out-of-distribution values
#           (recall: positive loudness offsets earlier caused buzz artifacts).
#
# Note: EMA increments are small per-channel uniform — real ESD deltas will vary
# by channel. This is a placeholder.
HANDCRAFTED_DELTAS = {
    "happy": np.concatenate([
        0.04 * np.ones(12, dtype=np.float32),   # slight articulation engagement
        np.array([+0.13, +0.05], dtype=np.float32),  # higher pitch, slightly louder
    ]),
    "sad": np.concatenate([
        -0.03 * np.ones(12, dtype=np.float32),  # slightly reduced articulation precision
        np.array([-0.10, -0.15], dtype=np.float32),  # lower pitch, quieter
    ]),
    "angry": np.concatenate([
        0.06 * np.ones(12, dtype=np.float32),   # tense articulation
        np.array([+0.10, +0.10], dtype=np.float32),  # louder, slightly higher pitch
    ]),
    "calm": np.concatenate([
        -0.02 * np.ones(12, dtype=np.float32),  # slightly relaxed articulation
        np.array([-0.05, -0.05], dtype=np.float32),  # slightly lower/quieter
    ]),
}


def build_handcrafted(vocab_path: Path) -> dict:
    with open(vocab_path) as f:
        vocab = json.load(f)
    # vocab is dict phoneme_symbol -> int index
    # For handcrafted mode, same delta vector for every phoneme index
    phonemes = [p for p, _ in sorted(vocab.items(), key=lambda x: x[1])]
    V = len(phonemes)
    E = len(EMOTIONS)
    deltas = np.zeros((V, E, FEATURE_DIMS), dtype=np.float32)
    for e_idx, emo in enumerate(EMOTIONS):
        delta_vec = HANDCRAFTED_DELTAS[emo]
        for p_idx in range(V):
            # Skip special tokens — don't modify silence/pad/bos/eos
            phoneme = phonemes[p_idx]
            if phoneme in ("<pad>", "<bos>", "<eos>", "<sil>"):
                continue
            deltas[p_idx, e_idx] = delta_vec
    return {
        "deltas":   deltas,
        "phonemes": phonemes,
        "emotions": EMOTIONS,
    }


def build_esd(esd_features_dir: Path, vocab_path: Path) -> dict:
    """Build deltas from pre-processed ESD frame tuples."""
    raise NotImplementedError(
        "ESD-mode deltas require scripts/emotion/build_esd_features.py to be run first. "
        "That script will emit per-(phoneme, emotion) feature mean files in esd_features_dir, "
        "which this function will consume. Not yet implemented — start with handcrafted mode."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["handcrafted", "esd"], default="handcrafted")
    p.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    p.add_argument("--esd-features-dir", type=str, default="data/esd_features")
    p.add_argument("--out-path", type=str, default="data/emotion_deltas.npz")
    args = p.parse_args()

    vocab_path = Path(args.vocab_path)
    if args.mode == "handcrafted":
        data = build_handcrafted(vocab_path)
    else:
        data = build_esd(Path(args.esd_features_dir), vocab_path)

    np.savez(
        args.out_path,
        deltas=data["deltas"],
        phonemes=np.array(data["phonemes"]),
        emotions=np.array(data["emotions"]),
    )
    print(f"Wrote {args.out_path}")
    print(f"  shape: deltas={data['deltas'].shape}  phonemes={len(data['phonemes'])}  emotions={data['emotions']}")
    print(f"  per-phoneme variation: {'yes' if args.mode == 'esd' else 'no (handcrafted: same delta for all phonemes)'}")


if __name__ == "__main__":
    main()
