"""
Convert LibriSpeech transcripts to phoneme sequences using g2p-en.
Builds vocabulary and saves per-utterance phoneme sequences.
"""
import argparse
import json
from pathlib import Path

from g2p_en import G2p
from tqdm import tqdm


# Special tokens
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
SIL = "<sil>"  # silence (space between words)

SPECIAL_TOKENS = [PAD, BOS, EOS, SIL]


def find_transcript_files(data_dir: Path) -> list[Path]:
    """Find all .trans.txt files in LibriSpeech directory structure."""
    return sorted(data_dir.rglob("*.trans.txt"))


def parse_transcripts(trans_files: list[Path]) -> dict[str, str]:
    """Parse LibriSpeech transcript files into {utt_id: text} mapping."""
    transcripts = {}
    for tf in trans_files:
        for line in tf.read_text().strip().split("\n"):
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                utt_id, text = parts
                transcripts[utt_id] = text.strip()
    return transcripts


def build_vocab(all_phonemes: list[list[str]]) -> dict[str, int]:
    """Build phoneme vocabulary from all sequences."""
    unique_phonemes = set()
    for seq in all_phonemes:
        unique_phonemes.update(seq)

    # Sort for deterministic ordering
    sorted_phonemes = sorted(unique_phonemes)

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for p in sorted_phonemes:
        if p not in vocab:
            vocab[p] = len(vocab)

    return vocab


def phonemize(data_dir: str, output_dir: str, features_dir: str = None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g2p = G2p()

    # Parse all transcripts
    trans_files = find_transcript_files(data_dir)
    print(f"Found {len(trans_files)} transcript files")
    transcripts = parse_transcripts(trans_files)
    print(f"Total utterances: {len(transcripts)}")

    # If features_dir provided, only process utterances that have features
    if features_dir:
        features_dir = Path(features_dir)
        available_utts = {p.stem for p in features_dir.glob("*.npz") if p.stem != "norm_stats"}
        transcripts = {k: v for k, v in transcripts.items() if k in available_utts}
        print(f"Filtered to {len(transcripts)} utterances with features")

    # Convert to phonemes
    all_phoneme_seqs = {}
    for utt_id, text in tqdm(transcripts.items(), desc="Phonemizing"):
        raw_phonemes = g2p(text)

        # Convert spaces to SIL token, filter empty strings
        phonemes = []
        for p in raw_phonemes:
            if p == " ":
                phonemes.append(SIL)
            elif p.strip():
                phonemes.append(p)

        all_phoneme_seqs[utt_id] = phonemes

    # Build vocabulary
    vocab = build_vocab(list(all_phoneme_seqs.values()))
    print(f"Vocabulary size: {len(vocab)} tokens")

    # Save vocabulary
    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)

    # Save per-utterance phoneme sequences (as token indices)
    dataset = {}
    for utt_id, phonemes in all_phoneme_seqs.items():
        speaker_id = utt_id.split("-")[0]
        indices = [vocab[BOS]] + [vocab.get(p, vocab[PAD]) for p in phonemes] + [vocab[EOS]]
        dataset[utt_id] = {
            "text": transcripts[utt_id],
            "phonemes": phonemes,
            "indices": indices,
            "speaker_id": speaker_id,
        }

    with open(output_dir / "phonemes.json", "w") as f:
        json.dump(dataset, f)

    print(f"Saved vocabulary ({len(vocab)} tokens) and {len(dataset)} phoneme sequences to {output_dir}")

    # Print some examples
    print("\nExamples:")
    for utt_id in list(dataset.keys())[:3]:
        d = dataset[utt_id]
        print(f"  {utt_id}: \"{d['text']}\"")
        print(f"    -> {d['phonemes'][:15]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to LibriSpeech dir")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--features-dir", type=str, default=None, help="Filter to utterances with features")
    args = parser.parse_args()

    phonemize(args.data_dir, args.output_dir, args.features_dir)
