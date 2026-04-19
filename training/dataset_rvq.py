"""
Dataset for TTS transformer with multi-codebook RVQ tokens.
Token shape per utterance: (T, num_quantizers)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TTSDatasetRVQ(Dataset):
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

        with open(phonemes_path) as f:
            self.phoneme_data = json.load(f)
        with open(alignments_path) as f:
            self.alignments = json.load(f)

        spk_emb_path = Path(features_dir) / "speaker_embeddings.json"
        with open(spk_emb_path) as f:
            self.speaker_embs = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

        self.vq_tokens_dir = Path(vq_tokens_dir)
        available_tokens = {p.stem for p in self.vq_tokens_dir.glob("*.npy")}
        self.utt_ids = [
            uid for uid in self.phoneme_data
            if uid in self.alignments and uid in available_tokens
            and self.alignments[uid]["total_frames"] <= self.max_frames
        ]
        total = sum(1 for uid in self.phoneme_data if uid in self.alignments and uid in available_tokens)
        print(f"TTSDatasetRVQ: {len(self.utt_ids)} utterances (filtered from {total}, max_frames={max_frames})")

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]

        phoneme_indices = self.phoneme_data[utt_id]["indices"]
        durations = self.alignments[utt_id]["durations"]
        durations = [0] + durations + [0]  # BOS/EOS get 0 duration

        vq_tokens = np.load(self.vq_tokens_dir / f"{utt_id}.npy")  # (T, K)
        if vq_tokens.ndim == 1:
            vq_tokens = vq_tokens[:, None]

        # Validate: sum of durations must match the number of frames in vq_tokens
        total_dur = sum(durations)
        if total_dur != vq_tokens.shape[0]:
            # Reconcile: trim durations or pad if mismatch (off-by-one tolerable)
            diff = vq_tokens.shape[0] - total_dur
            if abs(diff) > 2:
                raise ValueError(
                    f"{utt_id}: sum(durations)={total_dur} but vq_tokens has {vq_tokens.shape[0]} frames"
                )
            # Adjust last non-zero duration
            if diff > 0:
                # Find last speech phoneme and extend it
                for i in range(len(durations) - 1, -1, -1):
                    if durations[i] > 0:
                        durations[i] += diff
                        break
            else:
                # diff < 0: trim excess
                for i in range(len(durations) - 1, -1, -1):
                    if durations[i] + diff >= 1:
                        durations[i] += diff
                        break

        speaker_id = utt_id.split("-")[0]
        spk_emb = self.speaker_embs[speaker_id]

        return {
            "phoneme_ids": torch.tensor(phoneme_indices, dtype=torch.long),
            "durations": torch.tensor(durations, dtype=torch.float32),
            "vq_tokens": torch.tensor(vq_tokens, dtype=torch.long),  # (T, K)
            "speaker_emb": torch.from_numpy(spk_emb),
            "utt_id": utt_id,
        }


def collate_tts_rvq(batch):
    max_phoneme_len = max(b["phoneme_ids"].size(0) for b in batch)
    max_frame_len = max(b["vq_tokens"].size(0) for b in batch)
    K = batch[0]["vq_tokens"].size(1)

    B = len(batch)
    phoneme_ids = torch.zeros(B, max_phoneme_len, dtype=torch.long)
    phoneme_mask = torch.zeros(B, max_phoneme_len, dtype=torch.bool)
    durations = torch.zeros(B, max_phoneme_len, dtype=torch.float32)
    vq_tokens = torch.zeros(B, max_frame_len, K, dtype=torch.long)
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
        "vq_tokens": vq_tokens,  # (B, T, K)
        "frame_mask": frame_mask,
        "speaker_embs": speaker_embs,
    }
