"""
v10 dataset — frame-level, with per-frame phoneme_index and EOP.

SPARC features are 50 Hz native (verified empirically: SPARC.decode of 100
frames produces exactly 2.0 sec at 16 kHz = 20 ms hop = 50 Hz).
By default `frame_stride=1` (keep native rate). If `frame_stride>1`,
downsamples per-phoneme via even-spaced indices.

At inference, the tokenizer reconstructs frames at the same rate the dataset
served at training — caller upsamples by frame_stride to 50 Hz for SPARC.

Each item yields:
  uid                   - str
  phoneme_ids           - (N+2,) int64  — BOS + body + EOS (encoder input)
  spk_emb               - (64,)  float32
  knobs                 - (knob_dim,) float32 (optional)
  frames                - (T_body, 14) float32 — body frames only, normalized,
                                                 at 50 Hz after downsample
  body_durations        - (N,) int64 — durations at 50 Hz
  frame_to_enc_pos      - (T_body,) int64 — encoder position to attend to
                                            (= body_phoneme_idx + 1, since BOS is at 0)
  eop                   - (T_body,) uint8 — 1 at last frame of each body phoneme

Tokenizer training uses `frames` only (no phoneme info needed).
Renderer training uses everything.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


def _downsample_blocks(feats: np.ndarray, body_durs: np.ndarray, stride: int):
    """Per-phoneme even-spaced subsample. Each nonzero phoneme → max(1, d//stride) frames.
    Returns (new_feats, new_body_durs) with new_feats.shape[0] == new_body_durs.sum()."""
    if stride <= 1:
        return feats, body_durs
    new_blocks = []
    new_durs = np.zeros_like(body_durs)
    cursor = 0
    for i, d in enumerate(body_durs):
        d = int(d)
        block = feats[cursor:cursor + d]
        cursor += d
        if d == 0:
            new_durs[i] = 0
            continue
        new_d = max(1, d // stride)
        if new_d == 1:
            idx = np.array([d // 2], dtype=np.int64)
        else:
            idx = np.linspace(0, d - 1, new_d).round().astype(np.int64)
        new_blocks.append(block[idx])
        new_durs[i] = new_d
    if not new_blocks:
        return feats[:0], new_durs
    return np.concatenate(new_blocks, axis=0), new_durs


class V10Dataset(Dataset):
    def __init__(
        self,
        features_dir: str = "data/features_merged_logpitch_v2",
        phonemes_path: str = "data/processed_merged_v3/phonemes_mfa.json",
        alignments_path: str = "data/processed_merged_v3/alignments_mfa.json",
        spk_emb_dir: str = "v8/data/phoneme_anchors",
        norm_stats_path: str = "data/features_merged_logpitch_v2/norm_stats.npz",
        knob_source: str = "none",
        metadata_path: str = "data/utterance_metadata_v5.json",
        max_phonemes: int = 200,
        max_frames: int = 800,
        normalize: bool = True,
        preload: bool = False,
        frame_codes_dir: str = None,
        style_codes_dir: str = None,
        frame_stride: int = 1,
    ):
        self.features_dir = Path(features_dir)
        self.spk_emb_dir = Path(spk_emb_dir)
        self.frame_codes_dir = Path(frame_codes_dir) if frame_codes_dir else None
        self.style_codes_dir = Path(style_codes_dir) if style_codes_dir else None
        self.frame_stride = int(frame_stride)
        self.phon_data = json.load(open(phonemes_path))
        self.align_data = json.load(open(alignments_path))
        self.max_phonemes = max_phonemes
        self.max_frames = max_frames
        self.knob_source = knob_source
        self.preload = preload
        self.normalize = normalize

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"].astype(np.float32)
        self.feat_std = stats["std"].astype(np.float32)

        self.knobs = {}
        self.knob_dim = 0
        if knob_source == "emotion":
            self.knob_dim = 6
            if Path(metadata_path).exists():
                meta = json.load(open(metadata_path))
                for k, r in meta.items():
                    eid = EMOTION_TO_ID.get(r.get("emotion_label", "neutral"), 0)
                    intensity = float(r.get("intensity", 0.5))
                    one_hot = [0.0] * len(EMOTION_TO_ID)
                    one_hot[eid] = 1.0
                    self.knobs[k] = tuple(one_hot + [intensity])

        # Filter UIDs
        uids_full = sorted(self.phon_data.keys())
        self.utt_ids = []
        for uid in uids_full:
            if uid not in self.align_data:
                continue
            n = len(self.phon_data[uid].get("indices", []))
            if not (4 < n <= max_phonemes + 2):
                continue
            total_frames = int(np.asarray(self.align_data[uid]["durations"]).sum())
            if total_frames == 0 or total_frames > max_frames:
                continue
            if not (self.features_dir / f"{uid}.npz").exists():
                continue
            if not (self.spk_emb_dir / f"{uid}.npz").exists():
                continue
            if self.frame_codes_dir is not None and not (self.frame_codes_dir / f"{uid}.npz").exists():
                continue
            if self.style_codes_dir is not None and not (self.style_codes_dir / f"{uid}.npz").exists():
                continue
            self.utt_ids.append(uid)

        self._cache = {}
        if preload:
            print(f"Preloading {len(self.utt_ids)} utterances ...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load(uid)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")

    def _load(self, uid: str) -> dict:
        phoneme_ids = np.asarray(self.phon_data[uid]["indices"], dtype=np.int64)
        body_durs = np.asarray(self.align_data[uid]["durations"], dtype=np.int64)
        n_body = len(body_durs)
        if len(phoneme_ids) != n_body + 2:
            raise ValueError(
                f"{uid}: phoneme_ids len {len(phoneme_ids)} != body_dur len {n_body} + 2"
            )

        f = np.load(self.features_dir / f"{uid}.npz", allow_pickle=False)
        T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
        feats = np.concatenate([
            f["ema"][:T].astype(np.float32),
            f["pitch"][:T, None].astype(np.float32),
            f["loudness"][:T, None].astype(np.float32),
        ], axis=1).astype(np.float32)

        if self.normalize:
            feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)

        # Reconcile body durations with available frames (defensive trim)
        body_T = int(body_durs.sum())
        if body_T > T:
            running, new_durs = 0, body_durs.copy()
            for i, d in enumerate(body_durs):
                if running + d > T:
                    new_durs[i] = max(0, T - running)
                    new_durs[i+1:] = 0
                    break
                running += d
            body_durs = new_durs
            body_T = int(body_durs.sum())
        feats = feats[:body_T]

        # Downsample to target frame rate (default 50 Hz from 100 Hz native).
        # Each nonzero phoneme keeps min 1 frame. body_durs adjusted to match.
        if self.frame_stride > 1:
            feats, body_durs = _downsample_blocks(feats, body_durs, self.frame_stride)
            body_T = int(body_durs.sum())

        # Build per-frame phoneme index (encoder positions: BOS=0, body=1..N, EOS=N+1)
        # For body frames, encoder pos = body_idx + 1
        frame_to_enc_pos = np.empty(body_T, dtype=np.int64)
        eop = np.zeros(body_T, dtype=np.uint8)
        cursor = 0
        for i, d in enumerate(body_durs):
            if d <= 0:
                continue
            frame_to_enc_pos[cursor:cursor + d] = i + 1
            eop[cursor + d - 1] = 1
            cursor += d

        spk_npz = np.load(self.spk_emb_dir / f"{uid}.npz", allow_pickle=False)
        spk_emb = spk_npz["spk_emb"].astype(np.float32)

        if self.knob_source == "none":
            knobs = np.zeros(0, dtype=np.float32)
        else:
            default = tuple([0.0] * 5 + [0.5])
            knobs = np.asarray(self.knobs.get(uid, default), dtype=np.float32)

        out = {
            "uid": uid,
            "phoneme_ids": phoneme_ids,
            "spk_emb": spk_emb,
            "knobs": knobs,
            "frames": feats,
            "body_durations": body_durs.astype(np.int64),
            "frame_to_enc_pos": frame_to_enc_pos,
            "eop": eop,
        }
        if self.frame_codes_dir is not None:
            fc = np.load(self.frame_codes_dir / f"{uid}.npz", allow_pickle=False)
            codes = fc["idx"].astype(np.int64)            # (T, K)
            # Trim to body_T (defensive — should already match)
            out["frame_codes"] = codes[:body_T]
        if self.style_codes_dir is not None:
            sc = np.load(self.style_codes_dir / f"{uid}.npz", allow_pickle=False)
            out["style_codes"] = sc["codes"].astype(np.int64)  # (N+2,)
        return out

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> dict:
        uid = self.utt_ids[idx]
        return self._cache.get(uid) if self.preload else self._load(uid)


def _round_up(n: int, block: int) -> int:
    """Round n up to the nearest multiple of block. Stabilizes batch shapes so
    MPS doesn't recompile Metal kernels per batch (prevents shader-cache disk
    blow-up and 20-40 MB/s phantom writes during training)."""
    return ((n + block - 1) // block) * block


def collate_v10_tokenizer(batch):
    """Pad frames to T_max (rounded up to multiple of 16) for tokenizer training."""
    T_max = _round_up(max(b["frames"].shape[0] for b in batch), 16)
    B = len(batch)
    feat_dim = batch[0]["frames"].shape[1]
    frames = np.zeros((B, T_max, feat_dim), dtype=np.float32)
    frame_mask = np.zeros((B, T_max), dtype=bool)
    for i, b in enumerate(batch):
        T = b["frames"].shape[0]
        frames[i, :T] = b["frames"]
        frame_mask[i, :T] = True
    return {
        "frames": torch.from_numpy(frames),
        "frame_mask": torch.from_numpy(frame_mask),
        "uids": [b["uid"] for b in batch],
    }


def collate_v10_renderer(batch):
    """Pad phonemes (N_max) and frames (T_max). Surfaces all per-frame fields.
    Both dims rounded up to multiples of 16 so MPS reuses compiled kernels."""
    N_max = _round_up(max(b["phoneme_ids"].shape[0] for b in batch), 16)
    T_max = _round_up(max(b["frames"].shape[0] for b in batch), 16)
    B = len(batch)
    feat_dim = batch[0]["frames"].shape[1]
    knob_dim = batch[0]["knobs"].shape[0]

    phoneme_ids = np.zeros((B, N_max), dtype=np.int64)
    phoneme_mask = np.zeros((B, N_max), dtype=bool)
    body_durations = np.zeros((B, N_max), dtype=np.int64)  # padded to N_max for convenience
    spk_emb = np.zeros((B, batch[0]["spk_emb"].shape[0]), dtype=np.float32)
    knobs = np.zeros((B, knob_dim), dtype=np.float32)
    frames = np.zeros((B, T_max, feat_dim), dtype=np.float32)
    frame_mask = np.zeros((B, T_max), dtype=bool)
    frame_to_enc_pos = np.zeros((B, T_max), dtype=np.int64)
    eop = np.zeros((B, T_max), dtype=np.uint8)

    for i, b in enumerate(batch):
        N = b["phoneme_ids"].shape[0]
        T = b["frames"].shape[0]
        phoneme_ids[i, :N] = b["phoneme_ids"]
        phoneme_mask[i, :N] = True
        n_body = b["body_durations"].shape[0]
        body_durations[i, :n_body] = b["body_durations"]
        spk_emb[i] = b["spk_emb"]
        if knob_dim > 0:
            knobs[i] = b["knobs"]
        frames[i, :T] = b["frames"]
        frame_mask[i, :T] = True
        frame_to_enc_pos[i, :T] = b["frame_to_enc_pos"]
        eop[i, :T] = b["eop"]

    out = {
        "uids": [b["uid"] for b in batch],
        "phoneme_ids": torch.from_numpy(phoneme_ids),
        "phoneme_mask": torch.from_numpy(phoneme_mask),
        "body_durations": torch.from_numpy(body_durations),
        "spk_emb": torch.from_numpy(spk_emb),
        "knobs": torch.from_numpy(knobs),
        "frames": torch.from_numpy(frames),
        "frame_mask": torch.from_numpy(frame_mask),
        "frame_to_enc_pos": torch.from_numpy(frame_to_enc_pos),
        "eop": torch.from_numpy(eop).float(),
    }
    if "frame_codes" in batch[0]:
        K = batch[0]["frame_codes"].shape[1]
        frame_codes = np.zeros((B, T_max, K), dtype=np.int64)
        for i, b in enumerate(batch):
            T = b["frame_codes"].shape[0]
            frame_codes[i, :T] = b["frame_codes"]
        out["frame_codes"] = torch.from_numpy(frame_codes)
    if "style_codes" in batch[0]:
        style_codes = np.zeros((B, N_max), dtype=np.int64)
        # Default fill should be PAD. Caller responsible for picking PAD value
        # consistent with the style codebook size; we leave zeros for now and
        # fill from data where available — body positions get real codes,
        # padded N positions get 0 (which the model treats as PAD only if
        # the PAD code = 0; otherwise caller must remap). Style code at BOS/EOS
        # is also encoded as PAD by the style encoder, so this is fine when
        # the style encoder writes PAD_CODE at BOS/EOS positions in the .npz.
        for i, b in enumerate(batch):
            sc = b["style_codes"]
            style_codes[i, :sc.shape[0]] = sc
        out["style_codes"] = torch.from_numpy(style_codes)
    return out
