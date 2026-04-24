"""
PyTorch datasets for VQ tokenizer and TTS transformer training.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VQDataset(Dataset):
    """Dataset for VQ tokenizer training. Returns normalized articulatory feature chunks."""

    def __init__(
        self,
        features_dir: str,
        norm_stats_path: str = None,
        chunk_frames: int = 200,
        stride_frames: int = 100,
    ):
        self.features_dir = Path(features_dir)
        self.chunk_frames = chunk_frames

        # Load normalization stats
        if norm_stats_path is None:
            norm_stats_path = self.features_dir / "norm_stats.npz"
        stats = np.load(norm_stats_path)
        self.mean = stats["mean"]  # (14,)
        self.std = stats["std"]    # (14,)

        # Build index of (file, start_frame) pairs
        self.chunks = []
        npz_files = sorted(self.features_dir.glob("*.npz"))
        for npz_path in npz_files:
            if npz_path.stem == "norm_stats":
                continue
            data = np.load(npz_path)
            T = data["ema"].shape[0]
            for start in range(0, max(1, T - chunk_frames + 1), stride_frames):
                self.chunks.append((str(npz_path), start))

        print(f"VQDataset: {len(self.chunks)} chunks from {len(npz_files)-1} utterances")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        npz_path, start = self.chunks[idx]
        data = np.load(npz_path)

        ema = data["ema"]       # (T, 12)
        pitch = data["pitch"].squeeze()   # ensure 1D
        loudness = data["loudness"].squeeze()  # ensure 1D
        T = min(ema.shape[0], pitch.shape[0], loudness.shape[0])

        # Concatenate to 14-dim
        features = np.concatenate([ema[:T], pitch[:T, None], loudness[:T, None]], axis=-1)  # (T, 14)

        # Extract chunk
        end = min(start + self.chunk_frames, features.shape[0])
        chunk = features[start:end]

        # Pad if needed
        if chunk.shape[0] < self.chunk_frames:
            pad = np.zeros((self.chunk_frames - chunk.shape[0], 14), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        # Normalize
        chunk = (chunk - self.mean) / self.std

        return torch.from_numpy(chunk.astype(np.float32))


class TTSDataset(Dataset):
    """Dataset for TTS transformer training. Returns phonemes, durations, VQ tokens, speaker embeddings."""

    def __init__(
        self,
        features_dir: str,
        phonemes_path: str,
        alignments_path: str,
        vq_tokens_dir: str,
        vocab_path: str = None,
        max_frames: int = 800,
    ):
        self.features_dir = Path(features_dir)
        self.max_frames = max_frames

        # Load phoneme data
        with open(phonemes_path) as f:
            self.phoneme_data = json.load(f)

        # Load alignments
        with open(alignments_path) as f:
            self.alignments = json.load(f)

        # Load speaker embeddings
        spk_emb_path = Path(features_dir) / "speaker_embeddings.json"
        with open(spk_emb_path) as f:
            self.speaker_embs = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

        # VQ tokens directory
        self.vq_tokens_dir = Path(vq_tokens_dir)

        # Build sample list (only utterances present in all sources, under max_frames)
        available_tokens = {p.stem for p in self.vq_tokens_dir.glob("*.npy")}
        self.utt_ids = [
            uid for uid in self.phoneme_data
            if uid in self.alignments and uid in available_tokens
            and self.alignments[uid]["total_frames"] <= self.max_frames
        ]
        total = sum(1 for uid in self.phoneme_data if uid in self.alignments and uid in available_tokens)
        print(f"TTSDataset: {len(self.utt_ids)} utterances (filtered from {total}, max_frames={max_frames})")

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]

        # Phoneme indices (already includes BOS/EOS)
        phoneme_indices = self.phoneme_data[utt_id]["indices"]

        # Durations (for phonemes without BOS/EOS)
        durations = self.alignments[utt_id]["durations"]

        # Add duration=0 for BOS and EOS
        durations = [0] + durations + [0]

        # VQ token indices
        vq_tokens = np.load(self.vq_tokens_dir / f"{utt_id}.npy")  # (T,)

        # Speaker embedding
        speaker_id = utt_id.split("-")[0]
        spk_emb = self.speaker_embs[speaker_id]

        return {
            "phoneme_ids": torch.tensor(phoneme_indices, dtype=torch.long),
            "durations": torch.tensor(durations, dtype=torch.float32),
            "vq_tokens": torch.tensor(vq_tokens, dtype=torch.long),
            "speaker_emb": torch.from_numpy(spk_emb),
            "utt_id": utt_id,
        }


def collate_tts(batch: list[dict]) -> dict:
    """Collate function for TTSDataset with padding."""
    max_phoneme_len = max(b["phoneme_ids"].size(0) for b in batch)
    max_frame_len = max(b["vq_tokens"].size(0) for b in batch)

    B = len(batch)
    phoneme_ids = torch.zeros(B, max_phoneme_len, dtype=torch.long)
    phoneme_mask = torch.zeros(B, max_phoneme_len, dtype=torch.bool)
    durations = torch.zeros(B, max_phoneme_len, dtype=torch.float32)
    vq_tokens = torch.zeros(B, max_frame_len, dtype=torch.long)
    frame_mask = torch.zeros(B, max_frame_len, dtype=torch.bool)
    speaker_embs = torch.stack([b["speaker_emb"] for b in batch])

    for i, b in enumerate(batch):
        plen = b["phoneme_ids"].size(0)
        flen = b["vq_tokens"].size(0)
        phoneme_ids[i, :plen] = b["phoneme_ids"]
        phoneme_mask[i, :plen] = True
        durations[i, :plen] = b["durations"]
        vq_tokens[i, :flen] = b["vq_tokens"]
        frame_mask[i, :flen] = True

    return {
        "phoneme_ids": phoneme_ids,
        "phoneme_mask": phoneme_mask,
        "durations": durations,
        "vq_tokens": vq_tokens,
        "frame_mask": frame_mask,
        "speaker_embs": speaker_embs,
    }
