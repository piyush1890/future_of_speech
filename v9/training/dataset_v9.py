"""
v9 dataset: per-utterance items with frame features sliced by phoneme.

Each item yields:
  uid           - str
  phoneme_ids   - (N+2,) int64 — BOS + body phonemes (incl. <sil>) + EOS
  durations     - (N+2,) int64 — frame counts per phoneme; BOS/EOS = 1 (sentinel)
  spk_emb       - (64,) float32
  frames        - (T, 14) float32 — full feature stream (logpitch_v2 normalized)
  frame_offsets - (N+2+1,) int64 — cumulative durations; frames[offsets[i]:offsets[i+1]]
                                   gives phoneme i's frame block
  knobs         - (3 or 6,) float32 — V/A/D or emotion+intensity (optional, used by predictor)

Body slice: indices [1, 1+N) are real phonemes. BOS at 0, EOS at -1 are
synthetic sentinels (1 frame each, contributing trivial blocks).
The tokenizer is trained on body phoneme blocks only; BOS/EOS are skipped.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Mirror v8/training/dataset_v8.py defaults
EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


class V9PhonemeBlocksDataset(Dataset):
    def __init__(
        self,
        features_dir: str = "data/features_merged_logpitch_v2",
        phonemes_path: str = "data/processed_merged_v3/phonemes_mfa.json",
        alignments_path: str = "data/processed_merged_v3/alignments_mfa.json",
        spk_emb_dir: str = "v8/data/phoneme_anchors",
        norm_stats_path: str = "data/features_merged_logpitch_v2/norm_stats.npz",
        knob_source: str = "none",                  # "none" | "vad" | "emotion"
        vad_paths: list = None,
        metadata_path: str = "data/utterance_metadata_v5.json",
        max_phonemes: int = 200,
        normalize: bool = True,
        preload: bool = False,
    ):
        self.features_dir   = Path(features_dir)
        self.spk_emb_dir    = Path(spk_emb_dir)
        self.phon_data      = json.load(open(phonemes_path))
        self.align_data     = json.load(open(alignments_path))
        self.max_phonemes   = max_phonemes
        self.knob_source    = knob_source
        self.preload        = preload
        self.normalize      = normalize

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"].astype(np.float32)
        self.feat_std  = stats["std"].astype(np.float32)

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

        # Filter UIDs: must have phonemes, alignments, features, spk_emb
        uids_full = sorted(self.phon_data.keys())
        self.utt_ids = []
        for uid in uids_full:
            if uid not in self.align_data: continue
            n = len(self.phon_data[uid].get("indices", []))
            if not (4 < n <= max_phonemes + 2): continue   # body length 3+ inside BOS/EOS
            if not (self.features_dir / f"{uid}.npz").exists(): continue
            if not (self.spk_emb_dir / f"{uid}.npz").exists(): continue
            self.utt_ids.append(uid)

        self._cache = {}
        if preload:
            print(f"Preloading {len(self.utt_ids)} utterances ...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load(uid)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")

    def _load(self, uid: str) -> dict:
        # phoneme ids: include BOS + body + EOS (the "indices" field already has these wrapped)
        phoneme_ids = np.asarray(self.phon_data[uid]["indices"], dtype=np.int64)

        # Body durations from MFA (length = body, no BOS/EOS sentinels in alignment file)
        body_durs = np.asarray(self.align_data[uid]["durations"], dtype=np.int64)

        # Build full-length durations: BOS=1, body, EOS=1 — matches phoneme_ids length
        n_body = len(body_durs)
        if len(phoneme_ids) != n_body + 2:
            raise ValueError(
                f"{uid}: phoneme_ids len {len(phoneme_ids)} != body_dur len {n_body} + 2"
            )
        durations = np.zeros(n_body + 2, dtype=np.int64)
        durations[0]  = 1
        durations[-1] = 1
        durations[1:1+n_body] = body_durs

        # Features
        f = np.load(self.features_dir / f"{uid}.npz", allow_pickle=False)
        T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
        feats = np.concatenate([
            f["ema"][:T].astype(np.float32),                    # (T, 12)
            f["pitch"][:T, None].astype(np.float32),            # (T, 1) — already log(p+1)
            f["loudness"][:T, None].astype(np.float32),         # (T, 1)
        ], axis=1).astype(np.float32)                           # (T, 14)

        if self.normalize:
            feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)

        # Body phoneme frame coverage = sum(body_durs) — pad with zeros for BOS/EOS sentinels
        body_T = int(body_durs.sum())
        if body_T > T:
            # Trim body durs to fit available frames (defensive; rarely triggers if MFA matches features)
            running, idx = 0, 0
            new_durs = body_durs.copy()
            for i, d in enumerate(body_durs):
                if running + d > T:
                    new_durs[i]  = max(0, T - running)
                    new_durs[i+1:] = 0
                    break
                running += d
            durations[1:1+n_body] = new_durs
            body_T = int(new_durs.sum())
        elif body_T < T:
            feats = feats[:body_T]

        # Build a 1-frame BOS/EOS sentinel block (zeros)
        feat_dim = feats.shape[1]
        bos_block = np.zeros((1, feat_dim), dtype=np.float32)
        eos_block = np.zeros((1, feat_dim), dtype=np.float32)
        full_feats = np.concatenate([bos_block, feats, eos_block], axis=0)
        # cumulative offsets: shape (N+2+1,)
        frame_offsets = np.concatenate([[0], np.cumsum(durations)]).astype(np.int64)
        assert frame_offsets[-1] == full_feats.shape[0], (
            f"{uid}: offsets[-1]={frame_offsets[-1]} != full_feats len {full_feats.shape[0]}"
        )

        # spk_emb from v8 anchors archive
        spk_npz = np.load(self.spk_emb_dir / f"{uid}.npz", allow_pickle=False)
        spk_emb = spk_npz["spk_emb"].astype(np.float32)

        # Knobs (None if not configured)
        if self.knob_source == "none":
            knobs = np.zeros(0, dtype=np.float32)
        else:
            default = ((0.5, 0.5, 0.5) if self.knob_source == "vad"
                       else tuple([0.0]*5 + [0.5]))
            knobs = np.asarray(self.knobs.get(uid, default), dtype=np.float32)

        return {
            "uid":          uid,
            "phoneme_ids":  phoneme_ids,
            "durations":    durations,
            "frames":       full_feats,
            "frame_offsets": frame_offsets,
            "spk_emb":      spk_emb,
            "knobs":        knobs,
        }

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx: int) -> dict:
        uid = self.utt_ids[idx]
        return self._cache.get(uid) if self.preload else self._load(uid)


def collate_v9(batch):
    """Variable-length per-phoneme block batching is awkward — instead we pass
    everything as a list and let the trainer pull blocks out per phoneme.
    This keeps the data structure simple; tokenizer training operates one
    utterance at a time (loop over phonemes, sum reconstruction loss).
    Predictor training will use a separate collate that pads to N_max."""
    return batch
