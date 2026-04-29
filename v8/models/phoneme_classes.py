"""
Phoneme-class taxonomy for class-aware rendering.

Two render classes:
  PLATEAU (=0): sustained sounds — render as 1×start + (d-2)×mid + 1×end (HMM-style).
                Vowels (monophthongs), nasals, fricatives, approximants L/R, silences.
  LINEAR  (=1): transient / continuous-motion sounds — render with linear interpolation
                start→mid→end over the duration. Diphthongs, glides, stops, affricates.

Phonemes the model has never seen (e.g. unknown punct) default to PLATEAU.
"""
import torch

PLATEAU = 0
LINEAR = 1

# ARPABET phonemes that get HMM plateau rendering (sustained / steady-state)
PLATEAU_PHONEMES = {
    # Monophthong vowels (no glide)
    "AA", "AA0", "AA1", "AA2",
    "AE", "AE0", "AE1", "AE2",
    "AH", "AH0", "AH1", "AH2",
    "AO", "AO0", "AO1", "AO2",
    "EH", "EH0", "EH1", "EH2",
    "ER", "ER0", "ER1", "ER2",
    "IH", "IH0", "IH1", "IH2",
    "IY", "IY0", "IY1", "IY2",
    "UH", "UH0", "UH1", "UH2",
    "UW", "UW0", "UW1", "UW2",
    # Nasals
    "M", "N", "NG",
    # Fricatives
    "F", "V", "S", "Z", "SH", "ZH", "TH", "DH", "HH",
    # Liquid approximants (sustained, distinct from glides)
    "L", "R",
    # Boundary tokens (silences)
    "<sil>", "<bos>", "<eos>", "<pad>",
}

# ARPABET phonemes that get linear interpolation (transient / continuous motion)
LINEAR_PHONEMES = {
    # Diphthongs — continuous vowel-quality motion across the phoneme
    "AY", "AY0", "AY1", "AY2",
    "AW", "AW0", "AW1", "AW2",
    "OW", "OW0", "OW1", "OW2",
    "OY", "OY0", "OY1", "OY2",
    "EY", "EY0", "EY1", "EY2",
    # Glides
    "W", "Y",
    # Stops — closure, burst, release happen in order
    "B", "P", "T", "D", "K", "G",
    # Affricates — stop + fricative
    "CH", "JH",
}


def build_render_class_table(vocab) -> torch.LongTensor:
    """Build a (vocab_size,) lookup tensor mapping phoneme idx → render class.

    Defaults to PLATEAU for any phoneme not explicitly listed (safer choice).
    """
    table = torch.full((len(vocab),), PLATEAU, dtype=torch.long)
    for tok, idx in vocab.token2idx.items():
        if tok in LINEAR_PHONEMES:
            table[idx] = LINEAR
        elif tok in PLATEAU_PHONEMES:
            table[idx] = PLATEAU
        # Unknown tokens default to PLATEAU
    return table
