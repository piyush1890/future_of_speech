"""
Dataset for TTS transformer with multi-codebook RVQ tokens.
Token shape per utterance: (T, num_quantizers)

v5 addition: optional `metadata_path` surfaces (emotion_id, style_id, intensity)
per utterance for the planner. Stage 1 loads them but doesn't need them; stage 2
trains the planner against them. Defaults to (neutral, default, 1.0) when
metadata is missing — keeps the dataset backwards-compatible with v3/v4 callers.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


# Canonical mappings — keep stable across stage 1 logging and stage 2 training.
# Emotion ids cover ESD's 5-way + LibriSpeech HuBERT-tagged neu fallback.
EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}
N_EMOTIONS = len(EMOTION_TO_ID)

# Style ids cover Expresso's 7-way + "default" fallback for non-Expresso.
STYLE_TO_ID = {
    "default": 0, "narrative": 1, "enunciated": 2, "whisper": 3,
    "laughing": 4, "sad": 5, "happy": 6,
}
ID_TO_STYLE = {v: k for k, v in STYLE_TO_ID.items()}
N_STYLES = len(STYLE_TO_ID)


def _label_to_id(label: str, table: dict, default_id: int = 0) -> int:
    """Map a label string to its id; unknown labels → default_id."""
    if not label: return default_id
    return table.get(label.lower().strip(), default_id)


class TTSDatasetRVQ(Dataset):
    def __init__(
        self,
        features_dir: str,
        phonemes_path: str,
        alignments_path: str,
        vq_tokens_dir: str,
        vocab_path: str = None,
        max_frames: int = 800,
        preload: bool = False,
        metadata_path: str = None,
    ):
        self.features_dir = Path(features_dir)
        self.max_frames = max_frames
        self.preload = preload
        self._cache = {}

        # Optional v5 metadata (emotion/style/intensity per utterance)
        self.metadata = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata for {len(self.metadata)} utterances from {metadata_path}")
        elif metadata_path:
            print(f"WARNING: metadata_path={metadata_path} not found — using defaults "
                  f"(neutral/default/intensity=1.0) for all utterances")

        with open(phonemes_path) as f:
            self.phoneme_data = json.load(f)
        with open(alignments_path) as f:
            self.alignments = json.load(f)

        # Per-utterance speaker embeddings come directly from each utterance's .npz
        # (SPARC stored spk_emb there during encoding). Avoids the off-manifold
        # average-across-utterances problem of loading a pre-averaged JSON.
        # Kept as fallback: averaged embeddings from speaker_embeddings.json, used
        # only when a .npz has no spk_emb field (shouldn't happen in practice).
        self._averaged_embs = {}
        spk_emb_path = Path(features_dir) / "speaker_embeddings.json"
        if spk_emb_path.exists():
            with open(spk_emb_path) as f:
                self._averaged_embs = {
                    k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()
                }

        self.vq_tokens_dir = Path(vq_tokens_dir)
        available_tokens = {p.stem for p in self.vq_tokens_dir.glob("*.npy")}
        self.utt_ids = [
            uid for uid in self.phoneme_data
            if uid in self.alignments and uid in available_tokens
            and self.alignments[uid]["total_frames"] <= self.max_frames
        ]
        total = sum(1 for uid in self.phoneme_data if uid in self.alignments and uid in available_tokens)
        print(f"TTSDatasetRVQ: {len(self.utt_ids)} utterances (filtered from {total}, max_frames={max_frames})")

        if preload:
            print(f"Preloading {len(self.utt_ids)} utterances into RAM...")
            for i, uid in enumerate(self.utt_ids):
                feat_path = self.features_dir / f"{uid}.npz"
                with np.load(feat_path) as f:
                    spk_emb = f["spk_emb"].astype(np.float32) if "spk_emb" in f else self._averaged_embs.get(uid.split("-")[0])
                    ema = f["ema"].astype(np.float32)
                    pitch = f["pitch"].astype(np.float32).squeeze()
                    loudness = f["loudness"].astype(np.float32).squeeze()
                    T_feat = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
                    style_features = np.concatenate([ema[:T_feat], pitch[:T_feat, None], loudness[:T_feat, None]], axis=1)
                tokens = np.load(self.vq_tokens_dir / f"{uid}.npy")
                self._cache[uid] = (spk_emb, style_features, tokens)
                if (i + 1) % 10000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")
            print(f"Preloaded {len(self._cache)} utterances ({sum(v[1].nbytes + v[2].nbytes for v in self._cache.values()) / 1e9:.1f} GB)")

    def __len__(self):
        return len(self.utt_ids)

    def _metadata_for(self, utt_id: str):
        """Return (emotion_id, style_id, intensity) for an utterance, with defaults."""
        m = self.metadata.get(utt_id, {})
        emo_id   = _label_to_id(m.get("emotion_label", "neutral"), EMOTION_TO_ID, 0)
        sty_id   = _label_to_id(m.get("style_label",   "default"), STYLE_TO_ID,   0)
        intensity = float(m.get("intensity", 1.0))
        return emo_id, sty_id, intensity

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]

        phoneme_indices = self.phoneme_data[utt_id]["indices"]
        durations = self.alignments[utt_id]["durations"]
        durations = [0] + durations + [0]  # BOS/EOS get 0 duration

        if self.preload and utt_id in self._cache:
            spk_emb, style_features, vq_tokens = self._cache[utt_id]
        else:
            vq_tokens = np.load(self.vq_tokens_dir / f"{utt_id}.npy")  # (T, K)
            feat_path = self.features_dir / f"{utt_id}.npz"
            with np.load(feat_path) as f:
                if "spk_emb" in f:
                    spk_emb = f["spk_emb"].astype(np.float32)
                else:
                    speaker_id = utt_id.split("-")[0]
                    spk_emb = self._averaged_embs[speaker_id]
                ema = f["ema"].astype(np.float32)
                pitch = f["pitch"].astype(np.float32).squeeze()
                loudness = f["loudness"].astype(np.float32).squeeze()
                T_feat = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
                style_features = np.concatenate([
                    ema[:T_feat], pitch[:T_feat, None], loudness[:T_feat, None]
                ], axis=1)

        if vq_tokens.ndim == 1:
            vq_tokens = vq_tokens[:, None]

        # Validate: sum of durations must match the number of frames in vq_tokens
        total_dur = sum(durations)
        if total_dur != vq_tokens.shape[0]:
            diff = vq_tokens.shape[0] - total_dur
            if abs(diff) > 2:
                raise ValueError(
                    f"{utt_id}: sum(durations)={total_dur} but vq_tokens has {vq_tokens.shape[0]} frames"
                )
            if diff > 0:
                for i in range(len(durations) - 1, -1, -1):
                    if durations[i] > 0:
                        durations[i] += diff
                        break
            else:
                for i in range(len(durations) - 1, -1, -1):
                    if durations[i] + diff >= 1:
                        durations[i] += diff
                        break

        emo_id, sty_id, intensity = self._metadata_for(utt_id)

        return {
            "phoneme_ids": torch.tensor(phoneme_indices, dtype=torch.long),
            "durations": torch.tensor(durations, dtype=torch.float32),
            "vq_tokens": torch.tensor(vq_tokens, dtype=torch.long),  # (T, K)
            "speaker_emb": torch.from_numpy(spk_emb),
            "style_features": torch.from_numpy(style_features),  # (T, 14)
            "emotion_id": torch.tensor(emo_id, dtype=torch.long),
            "style_id":   torch.tensor(sty_id, dtype=torch.long),
            "intensity":  torch.tensor(intensity, dtype=torch.float32),
            "utt_id": utt_id,
        }


class BucketBatchSampler(Sampler):
    """Groups similar-length utterances into batches to minimize padding waste.

    Sorts by frame count, chunks into buckets, shuffles buckets each epoch.
    Within each bucket, utterances have similar length → padding is minimal.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Get frame counts for sorting
        lengths = []
        for i in range(len(dataset)):
            utt_id = dataset.dataset.utt_ids[dataset.indices[i]] if hasattr(dataset, 'indices') else dataset.utt_ids[i]
            T = dataset.dataset.alignments[utt_id]["total_frames"] if hasattr(dataset, 'dataset') else dataset.alignments[utt_id]["total_frames"]
            lengths.append((i, T))
        # Sort by length
        lengths.sort(key=lambda x: x[1])
        # Chunk into batches
        self.batches = []
        for i in range(0, len(lengths), batch_size):
            batch_indices = [idx for idx, _ in lengths[i:i+batch_size]]
            self.batches.append(batch_indices)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def _round_up(n, bucket=100):
    """Round up to nearest bucket boundary so MPS sees few distinct tensor shapes."""
    return ((n + bucket - 1) // bucket) * bucket


def collate_tts_rvq(batch):
    max_phoneme_len = _round_up(max(b["phoneme_ids"].size(0) for b in batch), 16)
    max_frame_len = _round_up(max(b["vq_tokens"].size(0) for b in batch), 100)
    K = batch[0]["vq_tokens"].size(1)

    B = len(batch)
    phoneme_ids = torch.zeros(B, max_phoneme_len, dtype=torch.long)
    phoneme_mask = torch.zeros(B, max_phoneme_len, dtype=torch.bool)
    durations = torch.zeros(B, max_phoneme_len, dtype=torch.float32)
    vq_tokens = torch.zeros(B, max_frame_len, K, dtype=torch.long)
    frame_mask = torch.zeros(B, max_frame_len, dtype=torch.bool)
    speaker_embs = torch.stack([b["speaker_emb"] for b in batch])

    # Style features: pad to same bucketed frame length
    max_style_len = _round_up(max(b["style_features"].size(0) for b in batch), 100)
    style_features = torch.zeros(B, max_style_len, 14, dtype=torch.float32)

    for i, b in enumerate(batch):
        plen = b["phoneme_ids"].size(0)
        flen = b["vq_tokens"].size(0)
        slen = b["style_features"].size(0)
        phoneme_ids[i, :plen] = b["phoneme_ids"]
        phoneme_mask[i, :plen] = True
        durations[i, :plen] = b["durations"]
        vq_tokens[i, :flen] = b["vq_tokens"]
        frame_mask[i, :flen] = True
        style_features[i, :slen] = b["style_features"]

    # v5 metadata: emotion/style/intensity stacks (defaults present even if metadata absent)
    emotion_ids = torch.stack([b["emotion_id"] for b in batch]) \
        if "emotion_id" in batch[0] else torch.zeros(B, dtype=torch.long)
    style_ids   = torch.stack([b["style_id"]   for b in batch]) \
        if "style_id"   in batch[0] else torch.zeros(B, dtype=torch.long)
    intensities = torch.stack([b["intensity"]  for b in batch]) \
        if "intensity"  in batch[0] else torch.ones(B, dtype=torch.float32)

    return {
        "phoneme_ids": phoneme_ids,
        "phoneme_mask": phoneme_mask,
        "durations": durations,
        "vq_tokens": vq_tokens,  # (B, T, K)
        "frame_mask": frame_mask,
        "speaker_embs": speaker_embs,
        "style_features": style_features,  # (B, T_max, 14)
        "emotion_ids": emotion_ids,        # (B,)
        "style_ids":   style_ids,          # (B,)
        "intensities": intensities,        # (B,)
    }
