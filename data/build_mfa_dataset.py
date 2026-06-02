"""
Build training dataset directly from MFA alignments.
Instead of using g2p phonemes and trying to match them to MFA,
we build the phoneme sequence FROM MFA's phone tier, adding <sil>
for silence markers. This guarantees perfect alignment.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

ARTICULATORY_RATE = 50  # SPARC = 50 Hz


def parse_textgrid_phones(textgrid_path: str) -> list[tuple[str, float, float]]:
    """Parse phone tier from TextGrid."""
    text = Path(textgrid_path).read_text()
    phones = []
    tiers = text.split('"IntervalTier"')
    for tier in tiers[1:]:
        # Match tier NAME attribute, not arbitrary text containing "phones"
        # (utterances saying the word "phones" would otherwise trip us up)
        if re.search(r'name\s*=\s*"phones"', tier):
            intervals = re.findall(
                r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
                tier,
            )
            for xmin, xmax, label in intervals:
                phones.append((label.strip(), float(xmin), float(xmax)))
    return phones


def mfa_to_phonemes_and_durations(
    mfa_phones: list[tuple[str, float, float]],
    total_frames: int,
) -> tuple[list[str], list[int]]:
    """
    Convert MFA phone sequence directly to phonemes + durations.

    Returns:
        phonemes: list of phoneme strings (using our vocab: <sil> for silence,
                  ARPABET with stress for speech phones)
        durations: list of frame counts per phoneme
    """
    phonemes = []
    durations = []

    for phone, start, end in mfa_phones:
        sf = max(0, int(round(start * ARTICULATORY_RATE)))
        ef = min(total_frames, int(round(end * ARTICULATORY_RATE)))
        dur = max(1, ef - sf)

        if phone in ("", "sil", "sp", "spn"):
            # Merge consecutive silences; skip very short ones unless meaningful
            if dur < 1:
                continue
            phonemes.append("<sil>")
            durations.append(dur)
        else:
            # MFA uses ARPABET with stress (e.g., "IH1", "M", "ER0")
            phonemes.append(phone.upper())
            durations.append(dur)

    # Merge consecutive <sil> tokens
    merged_phonemes = []
    merged_durations = []
    for p, d in zip(phonemes, durations):
        if p == "<sil>" and merged_phonemes and merged_phonemes[-1] == "<sil>":
            merged_durations[-1] += d
        else:
            merged_phonemes.append(p)
            merged_durations.append(d)

    # Adjust to match total_frames
    diff = total_frames - sum(merged_durations)
    if diff != 0:
        indices = sorted(range(len(merged_durations)), key=lambda i: -merged_durations[i])
        for j in range(abs(diff)):
            idx = indices[j % len(indices)]
            if diff > 0:
                merged_durations[idx] += 1
            elif merged_durations[idx] > 1:
                merged_durations[idx] -= 1

    assert sum(merged_durations) == total_frames
    assert len(merged_phonemes) == len(merged_durations)
    return merged_phonemes, merged_durations


def build_vocab(all_phoneme_seqs: list[list[str]]) -> dict[str, int]:
    """Build vocabulary from all phoneme sequences."""
    PAD, BOS, EOS, SIL = "<pad>", "<bos>", "<eos>", "<sil>"
    unique = set()
    for seq in all_phoneme_seqs:
        unique.update(seq)

    vocab = {PAD: 0, BOS: 1, EOS: 2, SIL: 3}
    for p in sorted(unique):
        if p not in vocab:
            vocab[p] = len(vocab)
    return vocab


def main(args):
    features_dir = Path(args.features_dir)
    textgrid_dir = Path(args.textgrid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TextGrids
    tg_lookup = {}
    for tg in textgrid_dir.rglob("*.TextGrid"):
        tg_lookup[tg.stem] = str(tg)

    # Load text transcripts from original phonemes file AND LibriSpeech transcript files
    text_lookup = {}
    # From original phonemes file
    if Path(args.original_phonemes_path).exists():
        with open(args.original_phonemes_path) as f:
            for utt_id, data in json.load(f).items():
                text_lookup[utt_id] = data.get("text", "")

    # Also scan LibriSpeech transcript files for any missing texts
    for trans_file in Path(args.features_dir).parent.rglob("*.trans.txt"):
        for line in trans_file.read_text().strip().split("\n"):
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                text_lookup[parts[0]] = parts[1]

    # Also scan the LibriSpeech directories
    for libri_dir in ["data/LibriSpeech/dev-clean", "data/LibriSpeech/train-clean-100"]:
        for trans_file in Path(libri_dir).rglob("*.trans.txt"):
            for line in trans_file.read_text().strip().split("\n"):
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    text_lookup[parts[0]] = parts[1]

    print(f"Text transcripts available: {len(text_lookup)}")

    # Iterate over ALL feature files (not just original phoneme data)
    all_feature_files = sorted([
        p.stem for p in features_dir.glob("*.npz") if p.stem != "norm_stats"
    ])
    print(f"Feature files: {len(all_feature_files)}")
    print(f"TextGrid files: {len(tg_lookup)}")

    phoneme_dataset = {}
    alignments = {}
    all_seqs = []
    success = 0
    skipped = 0

    for utt_id in tqdm(all_feature_files, desc="Building MFA dataset"):
        feat_path = features_dir / f"{utt_id}.npz"
        if utt_id not in tg_lookup:
            skipped += 1
            continue

        d = np.load(feat_path)
        total_frames = d["ema"].shape[0]

        try:
            mfa_phones = parse_textgrid_phones(tg_lookup[utt_id])
            phonemes, durations = mfa_to_phonemes_and_durations(mfa_phones, total_frames)
        except Exception as e:
            if skipped < 5:
                print(f"  Error {utt_id}: {e}")
            skipped += 1
            continue

        if not phonemes:
            skipped += 1
            continue

        text = text_lookup.get(utt_id, "")
        speaker_id = utt_id.split("-")[0]

        phoneme_dataset[utt_id] = {
            "text": text,
            "phonemes": phonemes,
            "speaker_id": speaker_id,
        }

        alignments[utt_id] = {
            "durations": durations,
            "total_frames": total_frames,
            "num_phonemes": len(phonemes),
        }

        all_seqs.append(phonemes)
        success += 1

    # Build vocabulary (or load existing to ensure index consistency with pretrained model)
    if args.existing_vocab and Path(args.existing_vocab).exists():
        print(f"Loading existing vocab from {args.existing_vocab}")
        with open(args.existing_vocab) as f:
            vocab = json.load(f)

        # Verify full coverage — FAIL LOUDLY if any phoneme is missing.
        # Silent <pad> mapping would corrupt training.
        all_new_phonemes = set()
        for seq in all_seqs:
            all_new_phonemes.update(seq)
        missing = all_new_phonemes - set(vocab.keys())
        if missing:
            raise RuntimeError(
                f"FATAL: New dataset contains {len(missing)} phonemes NOT in existing vocab.\n"
                f"Missing phonemes: {sorted(missing)}\n"
                f"Using --existing-vocab would silently map these to <pad>, corrupting training.\n"
                f"Options:\n"
                f"  1. Use --allow-missing-phonemes to proceed with <pad> mapping (NOT RECOMMENDED).\n"
                f"  2. Drop the --existing-vocab flag to build a fresh vocab (but then pretrained\n"
                f"     transformer weights won't load — you'd need to train from scratch).\n"
                f"  3. Exclude utterances containing these phonemes from your dataset."
            )
        print(f"  Vocab coverage OK: all {len(all_new_phonemes)} phonemes in data found in existing vocab")
    else:
        vocab = build_vocab(all_seqs)

    # Add indices to phoneme dataset — strict by default (missing phoneme is a bug)
    allow_missing = getattr(args, 'allow_missing_phonemes', False)
    for utt_id, pdata in phoneme_dataset.items():
        seq_indices = []
        for p in pdata["phonemes"]:
            if p in vocab:
                seq_indices.append(vocab[p])
            elif allow_missing:
                seq_indices.append(vocab["<pad>"])
            else:
                raise RuntimeError(f"Phoneme {p!r} in {utt_id} not in vocab. "
                                   f"(Use --allow-missing-phonemes to force <pad> substitution.)")
        indices = [vocab["<bos>"]] + seq_indices + [vocab["<eos>"]]
        pdata["indices"] = indices

    # Save everything
    with open(output_dir / "phonemes_mfa.json", "w") as f:
        json.dump(phoneme_dataset, f)

    with open(output_dir / "alignments_mfa.json", "w") as f:
        json.dump(alignments, f)

    with open(output_dir / "vocab_mfa.json", "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nDone: {success} utterances, {skipped} skipped")
    print(f"Vocabulary: {len(vocab)} tokens")
    print(f"Saved to {output_dir}")

    # Verify: check phoneme-feature consistency
    print("\nVerifying alignment quality...")
    phoneme_features = {}
    for utt_id in list(phoneme_dataset.keys())[:500]:
        d = np.load(features_dir / f"{utt_id}.npz")
        features = np.concatenate([d["ema"], d["pitch"][:, None], d["loudness"][:, None]], axis=-1)
        phonemes = phoneme_dataset[utt_id]["phonemes"]
        durs = alignments[utt_id]["durations"]

        pos = 0
        for phon, dur in zip(phonemes, durs):
            if dur > 0 and pos + dur <= len(features):
                phoneme_features.setdefault(phon, []).append(features[pos:pos+dur].mean(0))
            pos += dur

    print(f"{'Phoneme':>8s} | {'Count':>6s} | EMA std | Top1 VQ%")
    print("-" * 50)
    for phon in ["<sil>", "AH1", "IY1", "AA1", "EH1", "S", "T", "M", "N"]:
        if phon in phoneme_features and len(phoneme_features[phon]) > 5:
            feats = np.array(phoneme_features[phon])
            ema_std = feats[:, :12].std(axis=0).mean()
            print(f"{phon:>8s} | {len(feats):>6d} | {ema_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features_combined")
    parser.add_argument("--textgrid-dir", type=str, default="data/mfa_alignments")
    parser.add_argument("--original-phonemes-path", type=str, default="data/processed_combined/phonemes.json")
    parser.add_argument("--output-dir", type=str, default="data/processed_mfa")
    parser.add_argument("--existing-vocab", type=str, default=None,
                        help="Optional: path to existing vocab_mfa.json to ensure index consistency with a pretrained model")
    parser.add_argument("--allow-missing-phonemes", action="store_true",
                        help="Force missing phonemes to map to <pad> instead of failing. NOT RECOMMENDED — corrupts training.")
    args = parser.parse_args()

    main(args)
