"""
Overnight pipeline: SPARC encode all audio → compute norm stats → run alignment.
Run this, go to sleep, train in the morning.

Usage:
    eval "$(~/miniconda3/bin/conda shell.zsh hook)" && conda activate arttts
    cd ~/projects/articulatory-tts
    python run_overnight.py
"""
import glob
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: SPARC encode all audio ──────────────────────────────
    print("=" * 60)
    print("STEP 1: SPARC encoding (this is the slow part)")
    print("=" * 60)

    from sparc import load_model
    coder = load_model("en", device="mps")

    flacs = sorted(glob.glob("data/LibriSpeech/dev-clean/**/*.flac", recursive=True))
    print(f"Found {len(flacs)} .flac files")

    speaker_embs = {}
    success, failed, skipped = 0, 0, 0
    t_start = time.time()

    for i, f in enumerate(tqdm(flacs, desc="Encoding")):
        utt_id = Path(f).stem
        spk_id = utt_id.split("-")[0]
        out_path = out_dir / f"{utt_id}.npz"

        if out_path.exists():
            # Still collect speaker emb from existing file
            d = np.load(out_path)
            if spk_id not in speaker_embs:
                speaker_embs[spk_id] = []
            speaker_embs[spk_id].append(d["spk_emb"])
            skipped += 1
            continue

        try:
            code = coder.encode(f)
            ema = np.asarray(code["ema"], dtype=np.float32)
            pitch = np.asarray(code["pitch"], dtype=np.float32).squeeze(-1)
            loudness = np.asarray(code["loudness"], dtype=np.float32).squeeze(-1)
            spk_emb = np.asarray(code["spk_emb"], dtype=np.float32)

            # Align lengths (EMA and pitch/loudness may differ by 1 frame)
            min_len = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
            ema, pitch, loudness = ema[:min_len], pitch[:min_len], loudness[:min_len]

            np.savez_compressed(out_path, ema=ema, pitch=pitch, loudness=loudness, spk_emb=spk_emb)

            if spk_id not in speaker_embs:
                speaker_embs[spk_id] = []
            speaker_embs[spk_id].append(spk_emb)
            success += 1
        except Exception as e:
            print(f"\nFAILED {utt_id}: {e}")
            failed += 1

        # Progress checkpoint every 100 files
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (success + skipped) / elapsed
            remaining = (len(flacs) - i - 1) / max(rate, 0.01)
            print(f"\n  [{success + skipped}/{len(flacs)}] {elapsed/60:.1f}min elapsed, ~{remaining/60:.1f}min remaining")

    encoding_time = time.time() - t_start
    print(f"\nEncoding done in {encoding_time/60:.1f} min: {success} new, {skipped} skipped, {failed} failed")

    # ── Step 2: Save speaker embeddings ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Saving speaker embeddings")
    print("=" * 60)

    avg_spk = {k: np.mean(v, axis=0).tolist() for k, v in speaker_embs.items()}
    with open(out_dir / "speaker_embeddings.json", "w") as f:
        json.dump(avg_spk, f)
    print(f"Saved {len(avg_spk)} speaker embeddings")

    # ── Step 3: Compute normalization stats ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Computing normalization stats")
    print("=" * 60)

    all_feat = []
    for p in sorted(out_dir.glob("*.npz")):
        if p.stem == "norm_stats":
            continue
        d = np.load(p)
        pitch_col = d["pitch"][:, None] if d["pitch"].ndim == 1 else d["pitch"]
        loud_col = d["loudness"][:, None] if d["loudness"].ndim == 1 else d["loudness"]
        all_feat.append(np.concatenate([d["ema"], pitch_col, loud_col], axis=-1))

    all_feat = np.concatenate(all_feat, axis=0)
    mean, std = all_feat.mean(0), all_feat.std(0)
    std[std < 1e-6] = 1.0
    np.savez(out_dir / "norm_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Norm stats: mean shape={mean.shape}, channels: {['TD_x','TD_y','TB_x','TB_y','TT_x','TT_y','UL_x','UL_y','LI_x','LI_y','LL_x','LL_y','pitch','loudness']}")
    print(f"Total frames: {all_feat.shape[0]:,} ({all_feat.shape[0]/50/3600:.1f} hours at 50Hz)")

    # ── Step 4: Duration alignment ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Phoneme-to-frame duration alignment")
    print("=" * 60)

    # Filter phonemes to only those with features
    phonemes_path = Path("data/processed/phonemes.json")
    with open(phonemes_path) as f:
        phoneme_data = json.load(f)

    available = {p.stem for p in out_dir.glob("*.npz") if p.stem != "norm_stats"}
    filtered = {k: v for k, v in phoneme_data.items() if k in available}
    print(f"Utterances with both features and phonemes: {len(filtered)}")

    # Relative phoneme durations (vowels longer, stops shorter)
    REL_DUR = {
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
        "<sil>": 1.5,
    }

    alignments = {}
    for utt_id, pdata in filtered.items():
        npz_path = out_dir / f"{utt_id}.npz"
        total_frames = np.load(npz_path)["ema"].shape[0]
        phonemes = pdata["phonemes"]
        if not phonemes:
            continue

        weights = [REL_DUR.get(p, 1.0) for p in phonemes]
        total_w = sum(weights)
        raw = [w / total_w * total_frames for w in weights]
        durations = [max(1, int(round(d))) for d in raw]

        # Fix rounding to match total
        diff = total_frames - sum(durations)
        indices = sorted(range(len(durations)), key=lambda i: -raw[i])
        for i in range(abs(diff)):
            idx = indices[i % len(indices)]
            durations[idx] += 1 if diff > 0 else (-1 if durations[idx] > 1 else 0)

        alignments[utt_id] = {"durations": durations, "total_frames": total_frames, "num_phonemes": len(phonemes)}

    align_path = Path("data/processed/alignments.json")
    with open(align_path, "w") as f:
        json.dump(alignments, f)

    avg_frames = np.mean([a["total_frames"] for a in alignments.values()])
    avg_phon = np.mean([a["num_phonemes"] for a in alignments.values()])
    print(f"Aligned {len(alignments)} utterances")
    print(f"Avg: {avg_frames:.0f} frames/utt ({avg_frames/50:.1f}s), {avg_phon:.0f} phonemes/utt, {avg_frames/avg_phon:.1f} frames/phoneme")

    # ── Done ────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"ALL DONE in {total_time/60:.1f} minutes!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  data/features/*.npz          — {len(available)} utterance features")
    print(f"  data/features/norm_stats.npz — normalization stats")
    print(f"  data/features/speaker_embeddings.json — {len(avg_spk)} speakers")
    print(f"  data/processed/alignments.json — {len(alignments)} alignments")
    print(f"\nNext morning, run:")
    print(f"  cd ~/projects/articulatory-tts")
    print(f'  eval "$(~/miniconda3/bin/conda shell.zsh hook)" && conda activate arttts')
    print(f"  python training/train_vq.py --device mps")


if __name__ == "__main__":
    main()
