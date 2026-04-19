"""
Phoneme vocabulary management for ARPAbet tokens from g2p-en.
"""
import json
from pathlib import Path


class PhonemeVocab:
    """Manages the phoneme vocabulary for the TTS model."""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    SIL = "<sil>"

    def __init__(self, vocab_path: str | Path = None):
        if vocab_path and Path(vocab_path).exists():
            with open(vocab_path) as f:
                self.token2idx = json.load(f)
        else:
            # Default ARPAbet vocab
            self.token2idx = self._build_default_vocab()

        self.idx2token = {v: k for k, v in self.token2idx.items()}

    def _build_default_vocab(self) -> dict[str, int]:
        """Build default ARPAbet vocabulary."""
        special = [self.PAD, self.BOS, self.EOS, self.SIL]
        # Standard ARPAbet phonemes (with stress markers)
        arpabet = [
            "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2",
            "AH", "AH0", "AH1", "AH2", "AO", "AO0", "AO1", "AO2",
            "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2",
            "B", "CH", "D", "DH",
            "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2",
            "EY", "EY0", "EY1", "EY2",
            "F", "G", "HH",
            "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2",
            "JH", "K", "L", "M", "N", "NG",
            "OW", "OW0", "OW1", "OW2", "OY", "OY0", "OY1", "OY2",
            "P", "R", "S", "SH", "T", "TH",
            "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2",
            "V", "W", "Y", "Z", "ZH",
        ]
        all_tokens = special + arpabet
        return {tok: i for i, tok in enumerate(all_tokens)}

    def __len__(self) -> int:
        return len(self.token2idx)

    def __getitem__(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.PAD])

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD]

    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.BOS]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS]

    def encode(self, phonemes: list[str], add_bos_eos: bool = True) -> list[int]:
        """Convert phoneme strings to indices."""
        indices = [self[p] for p in phonemes]
        if add_bos_eos:
            indices = [self.bos_idx] + indices + [self.eos_idx]
        return indices

    def decode(self, indices: list[int]) -> list[str]:
        """Convert indices back to phoneme strings."""
        return [self.idx2token.get(i, self.PAD) for i in indices]
