"""
Build the training dataset for the emotion-delta predictor.

For each paired (Neutral[S,i], Emotional[S,E,i]) in ESD:
  1) Run g2p on the text → phoneme sequence
  2) Compute proportional per-phoneme durations for BOTH clips
     (we don't have MFA on ESD; proportional alignment is consistent with how
      we built the existing mean-delta table and with build_esd_deltas.py)
  3) Linear-warp neutral to match emotional's frame count within each phoneme,
     so delta = emo[t] - neu_aligned[t] lands on matching phonetic units
  4) Stack 14-dim features (ema(12), log_pitch, loudness). Log-pitch keeps us
     in our main TTS's feature space.
  5) Save (phoneme_ids, per-phoneme-durations, speaker_emb, emotion_idx, delta)
     per utterance as .npz.

Output dir: data/emotion_delta_dataset/
Train/val split: 90/10 held out by text_idx (so no text appears in both).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


EMOTIONS = ["Happy", "Sad", "Angry", "Surprise"]  # excludes Neutral
EMO_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def _feat14(npz) -> np.ndarray:
    ema = npz["ema"]; pitch = npz["pitch"]; loud = npz["loudness"]
    T = min(ema.shape[0], pitch.shape[0], loud.shape[0])
    log_pitch = np.log(pitch[:T] + 1.0)
    return np.concatenate([ema[:T], log_pitch[:, None], loud[:T, None]],
                          axis=1).astype(np.float32)


def _phoneme_ids(text: str, g2p: G2p, vocab: PhonemeVocab) -> list[int]:
    """Same phoneme-extraction rule as our synth pipeline: single leading <sil>,
    drop interior spaces/punct, map to vocab ids without BOS/EOS."""
    raw = g2p(text)
    phonemes = ["<sil>"] + [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
    # Use encode with add_bos_eos=False — this dataset has no sequence-end tokens;
    # the downstream model handles sequence boundaries itself.
    return vocab.encode(phonemes, add_bos_eos=False)


def _proportional_durations(n_phonemes: int, T: int) -> np.ndarray:
    """Split T frames across n_phonemes as evenly as possible, ensuring each
    phoneme has at least 1 frame. Returns int array of length n_phonemes summing
    to T (may be slightly adjusted if T < n_phonemes, we guarantee sum <= T)."""
    if n_phonemes == 0:
        return np.array([], dtype=np.int64)
    base = T // n_phonemes
    rem = T - base * n_phonemes
    durs = np.full(n_phonemes, base, dtype=np.int64)
    durs[:rem] += 1
    # Guarantee minimum 1 by stealing from longest bins
    zeros = np.where(durs == 0)[0]
    for z in zeros:
        donor = int(np.argmax(durs))
        if durs[donor] > 1:
            durs[donor] -= 1
            durs[z] += 1
    return durs


def _align_within_phonemes(src: np.ndarray, src_durs: np.ndarray,
                           tgt_durs: np.ndarray) -> np.ndarray:
    """Linear-warp `src` (T_src, D) onto a grid defined by `tgt_durs`, where
    both cover the same phoneme sequence. For each phoneme p, take src's
    frames [cumsum(src_durs)[p-1]:cumsum(src_durs)[p]] and linearly
    interpolate to length tgt_durs[p]. Concatenate.

    Returns shape (sum(tgt_durs), D).
    """
    D = src.shape[1]
    out_T = int(tgt_durs.sum())
    out = np.zeros((out_T, D), dtype=src.dtype)
    src_starts = np.concatenate([[0], np.cumsum(src_durs)])
    tgt_starts = np.concatenate([[0], np.cumsum(tgt_durs)])
    for p in range(len(tgt_durs)):
        s0, s1 = int(src_starts[p]), int(src_starts[p + 1])
        t0, t1 = int(tgt_starts[p]), int(tgt_starts[p + 1])
        n_src = max(1, s1 - s0)
        n_tgt = t1 - t0
        if n_tgt == 0:
            continue
        if n_src == 1:
            out[t0:t1] = src[s0:s0 + 1]
            continue
        x_old = np.arange(n_src)
        x_new = np.linspace(0, n_src - 1, n_tgt)
        chunk = src[s0:s1]
        for c in range(D):
            out[t0:t1, c] = np.interp(x_new, x_old, chunk[:, c])
    return out


def iter_speakers(features_dir: Path):
    speakers = sorted({f.name.split("_")[0]
                       for f in features_dir.glob("*_Neutral_*_*.npz")})
    return speakers


def build_one_pair(neu_path: Path, emo_path: Path, emotion: str,
                   g2p: G2p, vocab: PhonemeVocab) -> Optional[dict]:
    try:
        neu = np.load(neu_path, allow_pickle=True)
        emo = np.load(emo_path, allow_pickle=True)
    except (EOFError, OSError, ValueError):
        return None

    # Texts should match; take emo's text (emotional reading is what we model)
    text = str(emo["text"])
    if not text or not text.strip():
        return None

    phon_ids = _phoneme_ids(text, g2p, vocab)
    n_phon = len(phon_ids)
    if n_phon < 2:
        return None

    nf = _feat14(neu)   # (T_n, 14)
    ef = _feat14(emo)   # (T_e, 14)
    T_n, T_e = nf.shape[0], ef.shape[0]
    if T_n < n_phon or T_e < n_phon:
        return None

    # Proportional durations for both (same phoneme sequence, different frame counts)
    n_durs = _proportional_durations(n_phon, T_n)
    e_durs = _proportional_durations(n_phon, T_e)

    # Align neutral → emotional's phoneme-frame grid
    nf_aligned = _align_within_phonemes(nf, n_durs, e_durs)
    # Paranoia: truncate to common length
    T_common = min(nf_aligned.shape[0], ef.shape[0])
    delta = ef[:T_common] - nf_aligned[:T_common]   # (T_common, 14)

    return {
        "phoneme_ids": np.asarray(phon_ids, dtype=np.int64),
        "phoneme_durations": e_durs.astype(np.int64),
        "emotion_idx": np.int64(EMO_TO_IDX[emotion]),
        "speaker_emb": emo["spk_emb"].astype(np.float32),
        "speaker": str(emo["speaker"]),
        "text": text,
        "delta": delta.astype(np.float32),
        "T": np.int64(T_common),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=str, default="data/esd_features")
    ap.add_argument("--out-dir", type=str, default="data/emotion_delta_dataset")
    ap.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of texts held out for validation (per speaker).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    vocab = PhonemeVocab(args.vocab)
    g2p = G2p()

    speakers = iter_speakers(feat_dir)
    print(f"Speakers: {speakers}")

    rng = np.random.default_rng(args.seed)

    total, skipped, corrupt = 0, 0, 0
    per_emotion = {e: 0 for e in EMOTIONS}

    for spk in speakers:
        neu_files = sorted(feat_dir.glob(f"{spk}_Neutral_{spk}_*.npz"))
        n_neu = len(neu_files)
        if n_neu == 0:
            continue
        # Deterministic val split on text_idx per speaker
        idx = np.arange(n_neu)
        rng.shuffle(idx)
        val_n = max(1, int(n_neu * args.val_frac))
        val_idx = set(int(x) for x in idx[:val_n])

        for emotion in EMOTIONS:
            emo_files = sorted(feat_dir.glob(f"{spk}_{emotion}_{spk}_*.npz"))
            for i in range(min(len(neu_files), len(emo_files))):
                rec = build_one_pair(neu_files[i], emo_files[i], emotion, g2p, vocab)
                if rec is None:
                    corrupt += 1
                    continue
                split = "val" if i in val_idx else "train"
                name = f"{spk}_{emotion}_{i:04d}.npz"
                np.savez(out_dir / split / name, **rec)
                total += 1
                per_emotion[emotion] += 1
        print(f"  {spk}: done. total={total}")

    meta = {
        "features_dir": str(feat_dir),
        "total_utterances": total,
        "per_emotion": per_emotion,
        "emotions": EMOTIONS,
        "emotion_to_idx": EMO_TO_IDX,
        "speakers": speakers,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "corrupt_skipped": corrupt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {total} utterances ({corrupt} corrupt) → {out_dir}")
    print(f"Per emotion: {per_emotion}")


if __name__ == "__main__":
    main()
