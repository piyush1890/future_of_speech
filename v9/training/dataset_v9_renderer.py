"""
v9 renderer dataset: per-utterance items combining
  - GT frames (for the style encoder to extract per-phoneme z from),
  - GT RVQ tokens (the renderer's prediction target),
  - phoneme IDs, durations, speaker emb, emotion knobs.

Loads from:
  v9/data/phoneme_tokens/<uid>.npz   — RVQ tokens + spk_emb + durations
  data/features_merged_logpitch_v2/<uid>.npz — raw frame features
  data/utterance_metadata_v5.json    — emotion+intensity knobs
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import random


EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}
PAD_STYLE_CODE = 64    # padding index for style code positions (BOS/EOS) — must equal style_codebook_size


class V9RendererDataset(Dataset):
    def __init__(
        self,
        tokens_dir: str = "v9/data/phoneme_tokens",
        features_dir: str = "data/features_merged_logpitch_v2",
        norm_stats_path: str = "data/features_merged_logpitch_v2/norm_stats.npz",
        knob_source: str = "emotion",
        metadata_path: str = "data/utterance_metadata_v5.json",
        max_phonemes: int = 200,
        f_pad_per_phoneme: int = 32,
        preload: bool = False,
    ):
        self.tokens_dir = Path(tokens_dir)
        self.features_dir = Path(features_dir)
        self.max_phonemes = max_phonemes
        self.f_pad = f_pad_per_phoneme
        self.preload = preload
        self.knob_source = knob_source

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"].astype(np.float32)
        self.feat_std  = stats["std"].astype(np.float32)

        self.knobs = {}
        self.knob_dim = 0
        if knob_source == "emotion":
            self.knob_dim = 6
            if Path(metadata_path).exists():
                meta = json.load(open(metadata_path))
                for k, r in meta.items():
                    eid = EMOTION_TO_ID.get(r.get("emotion_label", "neutral"), 0)
                    intensity = float(r.get("intensity", 0.5))
                    one_hot = [0.0] * len(EMOTION_TO_ID); one_hot[eid] = 1.0
                    self.knobs[k] = tuple(one_hot + [intensity])

        # Filter UIDs: token file + features file must exist
        self.utt_ids = []
        for p in sorted(self.tokens_dir.glob("*.npz")):
            uid = p.stem
            if not (self.features_dir / f"{uid}.npz").exists():
                continue
            try:
                f = np.load(p, allow_pickle=False)
                n_full = len(f["phoneme_ids"])
                if 4 < n_full <= max_phonemes + 2:
                    self.utt_ids.append(uid)
            except Exception:
                continue

        self._cache = {}
        if preload:
            print(f"V9RendererDataset: preloading {len(self.utt_ids)} items ...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load(uid)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")
        else:
            print(f"V9RendererDataset: {len(self.utt_ids)} utterances")

    def _load(self, uid: str) -> dict:
        tok = np.load(self.tokens_dir / f"{uid}.npz", allow_pickle=False)
        phoneme_ids = tok["phoneme_ids"].astype(np.int64)            # (N+2,)
        start_idx   = tok["start_idx"].astype(np.int64)              # (N_body, K)
        end_idx     = tok["end_idx"].astype(np.int64)
        durations   = tok["durations"].astype(np.int64)              # (N_body,)
        spk_emb     = tok["spk_emb"].astype(np.float32)

        # Frames (raw) → normalize
        f = np.load(self.features_dir / f"{uid}.npz", allow_pickle=False)
        body_T = int(durations.sum())
        T_full = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
        body_T = min(body_T, T_full)
        feats = np.concatenate([
            f["ema"][:body_T].astype(np.float32),
            f["pitch"][:body_T, None].astype(np.float32),
            f["loudness"][:body_T, None].astype(np.float32),
        ], axis=1)
        feats_norm = (feats - self.feat_mean) / (self.feat_std + 1e-8)

        # Knobs
        if self.knob_dim == 0:
            knobs = np.zeros(0, dtype=np.float32)
        else:
            default = tuple([0.0] * 5 + [0.5])
            knobs = np.asarray(self.knobs.get(uid, default), dtype=np.float32)

        # Build per-phoneme frame blocks padded to F_PAD (truncate longer)
        n_body = len(durations)
        phoneme_blocks = np.zeros((n_body, self.f_pad, 14), dtype=np.float32)
        clipped_lens = np.zeros(n_body, dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(durations)])
        for i in range(n_body):
            s = int(offsets[i]); e = int(offsets[i + 1])
            if e > body_T: e = body_T
            L = max(0, min(e - s, self.f_pad))
            if L > 0:
                phoneme_blocks[i, :L] = feats_norm[s:s + L]
            clipped_lens[i] = max(L, 1)                              # pad-1 for empty

        return {
            "uid": uid,
            "phoneme_ids":   phoneme_ids,                            # (N+2,)
            "start_idx":     start_idx,                              # (N_body, K)
            "end_idx":       end_idx,                                # (N_body, K)
            "durations":     durations,                              # (N_body,)
            "spk_emb":       spk_emb,                                # (64,)
            "knobs":         knobs,
            "phoneme_blocks": phoneme_blocks,                        # (N_body, F_PAD, 14)
            "block_lens":    clipped_lens,                           # (N_body,)
        }

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        return self._cache.get(uid) if self.preload else self._load(uid)


def _round_up(n, b):
    return ((n + b - 1) // b) * b


def collate_v9_renderer(batch, pad_phoneme_id=0, pad_token_id=0):
    """Pad to N_max (multiple of 16) phonemes and B_total per-phoneme blocks.

    Returns dict with:
      phoneme_ids:    (B, N_pad) long
      phoneme_mask:   (B, N_pad) bool
      body_mask:      (B, N_pad) bool
      start_idx:      (B, N_pad, K) long  — body slots filled, BOS/EOS=0
      end_idx:        (B, N_pad, K) long
      log_durations:  (B, N_pad) float
      spk_emb:        (B, 64)
      knobs:          (B, knob_dim)

      phoneme_blocks: (B*max_body, F_PAD, 14)
      block_lens:     (B*max_body,)
      block_ph_ids:   (B*max_body,) long  — phoneme symbol per block
      block_to_pos:   (B*max_body, 2) long — (utt_idx, phoneme_pos_in_utt)
      block_valid:    (B*max_body,) bool — True if this block is real (not padding)
    """
    B = len(batch)
    K = batch[0]["start_idx"].shape[1]
    N_max_full = _round_up(max(b["phoneme_ids"].shape[0] for b in batch), 16)
    F_PAD = batch[0]["phoneme_blocks"].shape[1]
    body_max = max(len(b["durations"]) for b in batch)

    phoneme_ids   = np.full((B, N_max_full), pad_phoneme_id, dtype=np.int64)
    phoneme_mask  = np.zeros((B, N_max_full), dtype=bool)
    body_mask     = np.zeros((B, N_max_full), dtype=bool)
    start_idx     = np.full((B, N_max_full, K), pad_token_id, dtype=np.int64)
    end_idx       = np.full((B, N_max_full, K), pad_token_id, dtype=np.int64)
    log_durations = np.zeros((B, N_max_full), dtype=np.float32)
    spk_emb       = np.stack([b["spk_emb"] for b in batch])
    knob_dim      = batch[0]["knobs"].shape[0]
    knobs         = np.zeros((B, knob_dim), dtype=np.float32) if knob_dim > 0 else None

    # Block-level tensors: pad to body_max per-utterance, total B*body_max.
    blocks_arr   = np.zeros((B * body_max, F_PAD, 14), dtype=np.float32)
    block_lens   = np.ones (B * body_max, dtype=np.int64)
    block_ph_ids = np.zeros(B * body_max, dtype=np.int64)
    block_valid  = np.zeros(B * body_max, dtype=bool)
    block_to_pos = np.zeros((B * body_max, 2), dtype=np.int64)

    for i, b in enumerate(batch):
        n_full = len(b["phoneme_ids"])
        n_body = n_full - 2
        phoneme_ids[i, :n_full] = b["phoneme_ids"]
        phoneme_mask[i, :n_full] = True
        body_mask[i, 1:1+n_body] = True
        start_idx[i, 1:1+n_body] = b["start_idx"]
        end_idx[i,   1:1+n_body] = b["end_idx"]
        log_durations[i, 1:1+n_body] = np.log(b["durations"].clip(min=1).astype(np.float32))
        if knob_dim > 0:
            knobs[i] = b["knobs"]
        # Per-block fields
        for j in range(n_body):
            slot = i * body_max + j
            blocks_arr[slot] = b["phoneme_blocks"][j]
            block_lens[slot] = max(int(b["block_lens"][j]), 1)
            block_ph_ids[slot] = int(b["phoneme_ids"][1 + j])
            block_valid[slot] = True
            block_to_pos[slot, 0] = i
            block_to_pos[slot, 1] = 1 + j

    out = {
        "phoneme_ids":   torch.from_numpy(phoneme_ids),
        "phoneme_mask":  torch.from_numpy(phoneme_mask),
        "body_mask":     torch.from_numpy(body_mask),
        "start_idx":     torch.from_numpy(start_idx),
        "end_idx":       torch.from_numpy(end_idx),
        "log_durations": torch.from_numpy(log_durations),
        "spk_emb":       torch.from_numpy(spk_emb),
        "phoneme_blocks": torch.from_numpy(blocks_arr),
        "block_lens":    torch.from_numpy(block_lens),
        "block_ph_ids":  torch.from_numpy(block_ph_ids),
        "block_to_pos":  torch.from_numpy(block_to_pos),
        "block_valid":   torch.from_numpy(block_valid),
        "B": B, "N_max": N_max_full, "body_max": body_max,
    }
    if knob_dim > 0:
        out["knobs"] = torch.from_numpy(knobs)
    return out


class BucketBatchSampler(Sampler):
    """Sort by phoneme count, chunk into batches, shuffle batches each epoch."""
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset
        indices = list(dataset.indices) if hasattr(dataset, "indices") else list(range(len(dataset)))
        lengths = []
        for local_i, ds_idx in enumerate(indices):
            uid = underlying.utt_ids[ds_idx]
            n = len(underlying._cache[uid]["phoneme_ids"]) if underlying.preload \
                else len(underlying._load(uid)["phoneme_ids"])
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
