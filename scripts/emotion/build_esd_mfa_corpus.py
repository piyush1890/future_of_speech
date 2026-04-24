"""
Organize raw ESD audio into an MFA-compatible corpus directory.

Input:  data/esd_raw/Emotion Speech Dataset/<speaker>/<emotion>/*.wav
        Plus per-speaker transcripts in <speaker>/<speaker>.txt
        (format: utt_id<TAB>text<TAB>emotion per line)

Output: data/esd_mfa_corpus/<speaker>_<emotion>/<utt_id>.wav
                                                /<utt_id>.lab  (text)

We only keep English speakers (0011-0020).
MFA expects same-basename audio + label files in one directory.

Usage: python scripts/emotion/build_esd_mfa_corpus.py
"""
import argparse
import shutil
import sys
from pathlib import Path


ENGLISH_SPEAKERS = [f"{i:04d}" for i in range(11, 21)]    # 0011-0020


def parse_speaker_transcript(txt_path: Path) -> dict[str, tuple[str, str]]:
    """Each line: utt_id <TAB> text <TAB> emotion"""
    utt = {}
    for raw in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.strip().split("\t")
        if len(parts) < 3:
            continue
        utt_id, text, emotion = parts[0], parts[1], parts[2]
        utt[utt_id] = (text, emotion)
    return utt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str,
                    default="data/esd_raw/Emotion Speech Dataset")
    ap.add_argument("--out-dir", type=str, default="data/esd_mfa_corpus")
    ap.add_argument("--link", action="store_true",
                    help="Create hardlinks instead of copying WAVs (saves disk).")
    args = ap.parse_args()

    root = Path(args.raw_root)
    if not root.exists():
        print(f"ERROR: {root} does not exist. Extract ESD zip first.", file=sys.stderr)
        sys.exit(1)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    total_wavs = 0
    total_missing_txt = 0
    per_spk = {}

    for spk in ENGLISH_SPEAKERS:
        spk_dir = root / spk
        if not spk_dir.exists():
            print(f"  skip {spk}: no dir")
            continue
        txt_path = spk_dir / f"{spk}.txt"
        if not txt_path.exists():
            print(f"  skip {spk}: no transcript {txt_path}")
            continue
        utt_meta = parse_speaker_transcript(txt_path)

        # Walk emotion subdirs (Neutral, Happy, Sad, Angry, Surprise)
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir():
                continue
            emotion = emo_dir.name
            target = out / f"{spk}_{emotion}"
            target.mkdir(parents=True, exist_ok=True)

            wav_count = 0
            for wav_file in sorted(emo_dir.glob("*.wav")):
                utt_id = wav_file.stem
                if utt_id not in utt_meta:
                    total_missing_txt += 1
                    continue
                text, _ = utt_meta[utt_id]

                # Skip if already done
                tgt_wav = target / f"{utt_id}.wav"
                tgt_lab = target / f"{utt_id}.lab"
                if tgt_wav.exists() and tgt_lab.exists():
                    wav_count += 1
                    continue

                if args.link:
                    try:
                        if not tgt_wav.exists():
                            tgt_wav.hardlink_to(wav_file)
                    except OSError:
                        shutil.copyfile(wav_file, tgt_wav)
                else:
                    if not tgt_wav.exists():
                        shutil.copyfile(wav_file, tgt_wav)
                tgt_lab.write_text(text, encoding="utf-8")
                wav_count += 1
                total_wavs += 1

            per_spk[f"{spk}_{emotion}"] = wav_count
            print(f"  {spk}_{emotion}: {wav_count} utterances")

    print(f"\nTotal: {total_wavs} wavs organized, {total_missing_txt} missing txt")
    print(f"Corpus ready at: {out}")
    print(f"Run MFA align next.")


if __name__ == "__main__":
    main()
