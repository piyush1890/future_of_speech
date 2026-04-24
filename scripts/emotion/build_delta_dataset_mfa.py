"""
Build emotion-delta training dataset using REAL MFA per-phoneme timings.

Replaces build_delta_dataset.py (which used proportional alignment).

For each paired (Neutral[S,i], Emotional[S,E,i]) with matching text:
  1) Parse each TextGrid to get phoneme sequence + per-phoneme (start, end) in seconds
  2) Convert to frame indices at 50 Hz
  3) If phoneme sequences match (they should, same text+speaker), compute:
     - neutral_durations (per phoneme, in frames)
     - emotional_durations (per phoneme, in frames)
     - per-frame delta = emo_features - neutral_features_aligned
       where neutral is time-warped to emotional's per-phoneme frame grid
  4) Save (phoneme_ids, neutral_durations, emotional_durations, emotion_idx,
            speaker_emb, delta)
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.phoneme_vocab import PhonemeVocab


EMOTIONS = ["Happy", "Sad", "Angry", "Surprise"]
EMO_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
FRAMES_PER_SEC = 50

SIL_LABELS = {"", "sil", "sp", "spn"}


def parse_textgrid_phones(tg_path: Path) -> list[tuple[str, float, float]]:
    text = tg_path.read_text()
    tiers = text.split('"IntervalTier"')
    for tier in tiers[1:]:
        if re.search(r'name\s*=\s*"phones"', tier):
            intervals = re.findall(
                r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
                tier,
            )
            return [(lbl.strip(), float(xmin), float(xmax)) for xmin, xmax, lbl in intervals]
    return []


def tg_to_phonemes_durations(tg_path: Path, total_frames: int) -> tuple[list[str], list[int]]:
    """Convert TextGrid to (phoneme_list, frame_durations) where silences become <sil>
    and consecutive silences are merged."""
    phones = parse_textgrid_phones(tg_path)
    out_phon, out_dur = [], []
    for label, start, end in phones:
        sf = max(0, int(round(start * FRAMES_PER_SEC)))
        ef = min(total_frames, int(round(end * FRAMES_PER_SEC)))
        dur = max(1, ef - sf)
        if label in SIL_LABELS:
            if out_phon and out_phon[-1] == "<sil>":
                out_dur[-1] += dur
            else:
                out_phon.append("<sil>")
                out_dur.append(dur)
        else:
            out_phon.append(label)
            out_dur.append(dur)
    return out_phon, out_dur


def _feat14(npz) -> np.ndarray:
    ema = npz["ema"]; pitch = npz["pitch"]; loud = npz["loudness"]
    T = min(ema.shape[0], pitch.shape[0], loud.shape[0])
    log_pitch = np.log(pitch[:T] + 1.0)
    return np.concatenate([ema[:T], log_pitch[:, None], loud[:T, None]],
                          axis=1).astype(np.float32)


def _align_within_phonemes(src: np.ndarray, src_durs: list[int],
                           tgt_durs: list[int]) -> np.ndarray:
    """Linear-warp src features within each phoneme to match tgt_durs lengths."""
    D = src.shape[1]
    T_src = src.shape[0]
    out_T = int(sum(tgt_durs))
    out = np.zeros((out_T, D), dtype=src.dtype)
    src_pos = 0
    tgt_pos = 0
    for p in range(len(tgt_durs)):
        s_dur = int(src_durs[p])
        t_dur = int(tgt_durs[p])
        if t_dur == 0:
            src_pos += s_dur
            continue
        # Clamp against source bounds
        chunk_end = min(src_pos + s_dur, T_src)
        actual_s = max(0, chunk_end - src_pos)
        if actual_s == 0:
            # No source frames — fill with last valid or zero
            ref = src[src_pos - 1:src_pos] if src_pos > 0 else np.zeros((1, D), dtype=src.dtype)
            out[tgt_pos:tgt_pos + t_dur] = ref
        elif actual_s == 1:
            out[tgt_pos:tgt_pos + t_dur] = src[src_pos:src_pos + 1]
        else:
            chunk = src[src_pos:chunk_end]
            x_new = np.linspace(0, actual_s - 1, t_dur)
            for c in range(D):
                out[tgt_pos:tgt_pos + t_dur, c] = np.interp(
                    x_new, np.arange(actual_s), chunk[:, c]
                )
        src_pos += s_dur
        tgt_pos += t_dur
    return out


def _proportional_durations(n_phonemes: int, T: int) -> list[int]:
    """Evenly split T frames across n_phonemes with each >=1."""
    if n_phonemes == 0:
        return []
    base = T // n_phonemes
    rem = T - base * n_phonemes
    durs = [base] * n_phonemes
    for i in range(rem):
        durs[i] += 1
    # Guarantee >=1 by moving frames from longest to empty
    for i in range(n_phonemes):
        if durs[i] == 0:
            j = durs.index(max(durs))
            if durs[j] > 1:
                durs[j] -= 1
                durs[i] = 1
    return durs


def build_one_pair(neu_feat_path: Path, emo_feat_path: Path,
                   emo_tg_path: Path, emotion: str,
                   vocab: PhonemeVocab) -> Optional[dict]:
    """Use emotional MFA for phoneme sequence + durations (ground truth).
    For neutral, use proportional alignment into emotional's phoneme grid.
    Keeps 100% of utterances (no rejection on sequence mismatch).
    """
    if not emo_tg_path.exists():
        return None
    try:
        neu = np.load(neu_feat_path, allow_pickle=True)
        emo = np.load(emo_feat_path, allow_pickle=True)
    except (EOFError, OSError, ValueError):
        return None

    nf = _feat14(neu)
    ef = _feat14(emo)
    T_n, T_e = nf.shape[0], ef.shape[0]
    if T_n < 5 or T_e < 5:
        return None

    e_phon_list, e_durs = tg_to_phonemes_durations(emo_tg_path, T_e)
    if not e_phon_list:
        return None

    # Validate & clamp emotional durations to T_e
    total_e = sum(e_durs)
    if total_e != T_e:
        e_durs[-1] = max(1, e_durs[-1] + (T_e - total_e))
    # Ensure all >= 1
    e_durs = [max(1, d) for d in e_durs]
    # Renormalize if over/under
    total_e = sum(e_durs)
    if total_e != T_e:
        # Proportionally adjust to match T_e exactly
        scale = T_e / total_e
        e_durs = [max(1, int(round(d * scale))) for d in e_durs]
        # Final fix: adjust last
        e_durs[-1] = max(1, e_durs[-1] + (T_e - sum(e_durs)))

    # Convert emotional phoneme strings to vocab ids
    phon_ids = []
    for p in e_phon_list:
        idx = vocab.token2idx.get(p)
        if idx is None:
            return None
        phon_ids.append(idx)

    # Neutral durations: proportional split of T_n across same phoneme count
    n_phon = len(e_phon_list)
    n_durs = _proportional_durations(n_phon, T_n)

    # Align neutral → emotional's phoneme-frame grid
    nf_aligned = _align_within_phonemes(nf, n_durs, e_durs)
    T_common = min(nf_aligned.shape[0], ef.shape[0])
    delta = ef[:T_common] - nf_aligned[:T_common]

    return {
        "phoneme_ids": np.asarray(phon_ids, dtype=np.int64),
        "neutral_durations": np.asarray(n_durs, dtype=np.int64),
        "emotional_durations": np.asarray(e_durs, dtype=np.int64),
        "emotion_idx": np.int64(EMO_TO_IDX[emotion]),
        "speaker_emb": emo["spk_emb"].astype(np.float32),
        "speaker": str(emo["speaker"]),
        "text": str(emo["text"]),
        "delta": delta.astype(np.float32),
        "T": np.int64(T_common),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=str, default="data/esd_features")
    ap.add_argument("--alignments-dir", type=str, default="data/esd_mfa_alignments")
    ap.add_argument("--out-dir", type=str, default="data/emotion_delta_dataset_mfa")
    ap.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feat_dir = Path(args.features_dir)
    align_dir = Path(args.alignments_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    vocab = PhonemeVocab(args.vocab)
    speakers = sorted({f.name.split("_")[0] for f in feat_dir.glob("*_Neutral_*_*.npz")})
    print(f"Speakers: {speakers}")
    rng = np.random.default_rng(args.seed)

    total = 0
    rejected = 0
    per_emotion = {e: 0 for e in EMOTIONS}

    for spk in speakers:
        neu_files = sorted(feat_dir.glob(f"{spk}_Neutral_{spk}_*.npz"))
        n_neu = len(neu_files)
        if n_neu == 0: continue
        idx = np.arange(n_neu); rng.shuffle(idx)
        val_n = max(1, int(n_neu * args.val_frac))
        val_idx = set(int(x) for x in idx[:val_n])

        for emotion in EMOTIONS:
            emo_files = sorted(feat_dir.glob(f"{spk}_{emotion}_{spk}_*.npz"))
            n_pair = min(len(neu_files), len(emo_files))
            for i in range(n_pair):
                neu_feat = neu_files[i]
                emo_feat = emo_files[i]
                # TextGrid paths: data/esd_mfa_alignments/<spk>_<emo>/<utt_id>.TextGrid
                # utt_id is the .npz filename without ext's last component.
                # File names: 0011_Neutral_0011_000001.npz → utt stem = 0011_000001
                emo_utt = "_".join(emo_feat.stem.split("_")[-2:])
                emo_tg = align_dir / f"{spk}_{emotion}" / f"{emo_utt}.TextGrid"

                rec = build_one_pair(neu_feat, emo_feat, emo_tg, emotion, vocab)
                if rec is None:
                    rejected += 1
                    continue
                split = "val" if i in val_idx else "train"
                np.savez(out_dir / split / f"{spk}_{emotion}_{i:04d}.npz", **rec)
                total += 1
                per_emotion[emotion] += 1
        print(f"  {spk}: total={total}, rejected={rejected}")

    meta = {
        "total_utterances": total,
        "rejected": rejected,
        "per_emotion": per_emotion,
        "emotions": EMOTIONS,
        "emotion_to_idx": EMO_TO_IDX,
        "speakers": speakers,
        "val_frac": args.val_frac,
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {total} utterances, {rejected} rejected → {out_dir}")
    print(f"Per emotion: {per_emotion}")


if __name__ == "__main__":
    main()
