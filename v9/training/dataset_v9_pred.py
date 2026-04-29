"""
v9 predictor dataset: per-utterance items with GT RVQ tokens (extracted by
the trained tokenizer), durations, phoneme_ids, spk_emb, and optional knobs.

Reads from v9/data/phoneme_tokens/<uid>.npz which has:
  phoneme_ids: (N+2,) — BOS + body + EOS
  start_idx:   (N, K) — body-only RVQ tokens for start half
  end_idx:     (N, K)
  durations:   (N,)
  spk_emb:     (64,)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import random


EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


class V9PredictorDataset(Dataset):
    def __init__(
        self,
        tokens_dir: str = "v9/data/phoneme_tokens",
        knob_source: str = "none",                # "none" | "vad" | "emotion"
        vad_paths: list = None,
        metadata_path: str = "data/utterance_metadata_v5.json",
        max_phonemes: int = 200,
        preload: bool = False,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.max_phonemes = max_phonemes
        self.preload = preload
        self.knob_source = knob_source

        # Knobs
        self.knobs = {}
        self.knob_dim = 0
        if knob_source == "vad":
            self.knob_dim = 3
            for p in (vad_paths or []):
                if not Path(p).exists(): continue
                d = json.load(open(p))
                for k, r in d.items():
                    self.knobs[k] = (r.get("valence", 0.5),
                                     r.get("arousal", 0.5),
                                     r.get("dominance", 0.5))
        elif knob_source == "emotion":
            self.knob_dim = 6
            if Path(metadata_path).exists():
                meta = json.load(open(metadata_path))
                for k, r in meta.items():
                    eid = EMOTION_TO_ID.get(r.get("emotion_label", "neutral"), 0)
                    intensity = float(r.get("intensity", 0.5))
                    one_hot = [0.0] * len(EMOTION_TO_ID)
                    one_hot[eid] = 1.0
                    self.knobs[k] = tuple(one_hot + [intensity])

        # Filter UIDs: must have token file + valid phoneme count
        self.utt_ids = []
        for p in sorted(self.tokens_dir.glob("*.npz")):
            uid = p.stem
            try:
                f = np.load(p, allow_pickle=False)
                n_full = len(f["phoneme_ids"])
                if 4 < n_full <= max_phonemes + 2:
                    self.utt_ids.append(uid)
            except Exception:
                continue

        self._cache = {}
        if preload:
            print(f"Preloading {len(self.utt_ids)} predictor items ...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load(uid)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")
        else:
            print(f"V9PredictorDataset: {len(self.utt_ids)} utterances")

    def _load(self, uid: str) -> dict:
        f = np.load(self.tokens_dir / f"{uid}.npz", allow_pickle=False)
        phoneme_ids = f["phoneme_ids"].astype(np.int64)
        start_idx   = f["start_idx"].astype(np.int64)        # (N_body, K)
        end_idx     = f["end_idx"].astype(np.int64)
        durations   = f["durations"].astype(np.int64)         # (N_body,)
        spk_emb     = f["spk_emb"].astype(np.float32)

        if self.knob_source == "none":
            knobs = np.zeros(0, dtype=np.float32)
        else:
            default = ((0.5, 0.5, 0.5) if self.knob_source == "vad"
                       else tuple([0.0]*5 + [0.5]))
            knobs = np.asarray(self.knobs.get(uid, default), dtype=np.float32)

        return {
            "uid": uid,
            "phoneme_ids": phoneme_ids,
            "start_idx":   start_idx,
            "end_idx":     end_idx,
            "durations":   durations,
            "spk_emb":     spk_emb,
            "knobs":       knobs,
        }

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        return self._cache.get(uid) if self.preload else self._load(uid)


def _round_up(n: int, bucket: int) -> int:
    return ((n + bucket - 1) // bucket) * bucket


def collate_v9_pred(batch, pad_phoneme_id=0, pad_token_id=0):
    """Pad to max_phonemes (rounded up to multiple of 16) per batch.

    Output tensors:
      phoneme_ids:    (B, N_pad) long
      phoneme_mask:   (B, N_pad) bool — valid phoneme positions
      body_mask:      (B, N_pad) bool — True only at body positions (excl. BOS/EOS)
      start_idx:      (B, N_pad, K) long  — body slots filled, BOS/EOS = 0 (masked from loss)
      end_idx:        (B, N_pad, K) long
      log_durations:  (B, N_pad) float
      spk_emb:        (B, 64)
      knobs:          (B, knob_dim) or empty
    """
    B = len(batch)
    K = batch[0]["start_idx"].shape[1]
    N_max = _round_up(max(b["phoneme_ids"].shape[0] for b in batch), 16)

    phoneme_ids   = np.full((B, N_max), pad_phoneme_id, dtype=np.int64)
    phoneme_mask  = np.zeros((B, N_max), dtype=bool)
    body_mask     = np.zeros((B, N_max), dtype=bool)
    start_idx     = np.full((B, N_max, K), pad_token_id, dtype=np.int64)
    end_idx       = np.full((B, N_max, K), pad_token_id, dtype=np.int64)
    log_durations = np.zeros((B, N_max), dtype=np.float32)
    spk_emb       = np.stack([b["spk_emb"] for b in batch])
    knob_dim      = batch[0]["knobs"].shape[0]
    knobs         = np.zeros((B, knob_dim), dtype=np.float32) if knob_dim > 0 else None

    for i, b in enumerate(batch):
        n_full = len(b["phoneme_ids"])
        n_body = n_full - 2
        phoneme_ids[i, :n_full] = b["phoneme_ids"]
        phoneme_mask[i, :n_full] = True
        # Body positions = [1, 1+n_body)
        body_mask[i, 1:1+n_body] = True
        start_idx[i, 1:1+n_body] = b["start_idx"]
        end_idx[i,   1:1+n_body] = b["end_idx"]
        log_durations[i, 1:1+n_body] = np.log(b["durations"].clip(min=1).astype(np.float32))
        if knob_dim > 0:
            knobs[i] = b["knobs"]

    out = {
        "phoneme_ids":  torch.from_numpy(phoneme_ids),
        "phoneme_mask": torch.from_numpy(phoneme_mask),
        "body_mask":    torch.from_numpy(body_mask),
        "start_idx":    torch.from_numpy(start_idx),
        "end_idx":      torch.from_numpy(end_idx),
        "log_durations": torch.from_numpy(log_durations),
        "spk_emb":      torch.from_numpy(spk_emb),
    }
    if knob_dim > 0:
        out["knobs"] = torch.from_numpy(knobs)
    return out


class BucketBatchSampler(Sampler):
    """Group similar-length utterances into batches to minimize padding waste.

    Sorts by phoneme count, chunks into fixed-size batches, shuffles batches each epoch.
    Adapted from training/dataset_rvq.py:BucketBatchSampler.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Get phoneme counts for sorting (handles both Subset and direct dataset)
        underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset
        indices = list(dataset.indices) if hasattr(dataset, "indices") else list(range(len(dataset)))
        lengths = []
        for local_i, ds_idx in enumerate(indices):
            uid = underlying.utt_ids[ds_idx]
            n = len(underlying._load(uid)["phoneme_ids"]) if not underlying.preload \
                else len(underlying._cache[uid]["phoneme_ids"])
            lengths.append((local_i, n))
        lengths.sort(key=lambda x: x[1])
        self.batches = [
            [idx for idx, _ in lengths[i:i+batch_size]]
            for i in range(0, len(lengths), batch_size)
        ]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
