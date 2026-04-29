"""
v8 dataset: returns per-phoneme (start, mid, end, durations) anchors plus phoneme IDs
and speaker emb. Reads from preprocessed .npz files in v8/data/phoneme_anchors/.

Important: phoneme IDs from `indices` already include BOS at [0] and EOS at [-1].
Anchors and durations are body-only (length N). We pad them to length N+2 with zeros
for BOS/EOS. Duration for BOS/EOS = 1 (so length regulator outputs 1 frame each).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PhonemeAnchorsDataset(Dataset):
    def __init__(
        self,
        anchors_dir: str = "v8/data/phoneme_anchors",
        z_dir: str = "v8/data/phoneme_z",
        codes_dir: str = None,                # if set, load z_q + code_id
        phonemes_path: str = "data/processed_merged_v3/phonemes_mfa.json",
        vad_paths: list = None,
        knob_source: str = "vad",             # "vad" (3-d continuous) | "emotion" (6-d one-hot+intensity)
        metadata_path: str = "data/utterance_metadata_v5.json",
        max_phonemes: int = 200,
        preload: bool = False,
    ):
        self.anchors_dir = Path(anchors_dir)
        self.z_dir = Path(z_dir) if z_dir else None
        self.codes_dir = Path(codes_dir) if codes_dir else None
        self.phon_data = json.load(open(phonemes_path))
        self.max_phonemes = max_phonemes
        self.preload = preload
        self.knob_source = knob_source

        # Load knobs depending on source
        self.knobs = {}     # uid → tuple of floats (3-d for vad, 6-d for emotion)
        self.knob_dim = 3 if knob_source == "vad" else 6

        if knob_source == "vad" and vad_paths:
            for p in vad_paths:
                if not Path(p).exists():
                    print(f"  WARNING: V/A/D file not found: {p}")
                    continue
                d = json.load(open(p))
                for k, r in d.items():
                    self.knobs[k] = (r.get("valence", 0.5),
                                     r.get("arousal", 0.5),
                                     r.get("dominance", 0.5))
            print(f"  V/A/D loaded for {len(self.knobs)} utterances (knob_dim=3)")

        elif knob_source == "emotion":
            EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}
            n_emotions = len(EMOTION_TO_ID)
            if not Path(metadata_path).exists():
                print(f"  WARNING: metadata file not found: {metadata_path}")
            else:
                meta = json.load(open(metadata_path))
                for k, r in meta.items():
                    eid = EMOTION_TO_ID.get(r.get("emotion_label", "neutral"), 0)
                    intensity = float(r.get("intensity", 0.5))
                    one_hot = [0.0] * n_emotions
                    one_hot[eid] = 1.0
                    self.knobs[k] = tuple(one_hot + [intensity])    # 6-d
                print(f"  emotion+intensity loaded for {len(self.knobs)} utterances (knob_dim=6)")

        all_uts = sorted([f.stem for f in self.anchors_dir.glob("*.npz")])
        self.utt_ids = []
        require_z = self.z_dir is not None
        require_codes = self.codes_dir is not None
        for uid in all_uts:
            if uid not in self.phon_data:
                continue
            n = len(self.phon_data[uid].get("indices", []))
            if not (2 < n <= max_phonemes):
                continue
            if require_z and not (self.z_dir / f"{uid}.npy").exists():
                continue
            if require_codes and not (self.codes_dir / f"{uid}.npz").exists():
                continue
            self.utt_ids.append(uid)

        self._cache = {}
        if preload:
            print(f"Preloading {len(self.utt_ids)} anchor sets...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load_anchors(uid)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")

    def _load_anchors(self, uid):
        f = np.load(self.anchors_dir / f"{uid}.npz", allow_pickle=False)
        out = {
            "start": f["start"].astype(np.float32),
            "mid":   f["mid"].astype(np.float32),
            "end":   f["end"].astype(np.float32),
            "durations": f["durations"].astype(np.int64),
            "spk_emb":   f["spk_emb"].astype(np.float32),
        }
        if self.z_dir is not None:
            z = np.load(self.z_dir / f"{uid}.npy")
            out["z"] = z.astype(np.float32)
        if self.codes_dir is not None:
            cf = np.load(self.codes_dir / f"{uid}.npz", allow_pickle=False)
            out["z_q"] = cf["z_q"].astype(np.float32)         # (N, 256)
            out["code_id"] = cf["code_id"].astype(np.int64)   # (N,)
        return out

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        anch = self._cache[uid] if self.preload else self._load_anchors(uid)
        phoneme_ids = np.asarray(self.phon_data[uid]["indices"], dtype=np.int64)
        # Default knob: zero vector for "emotion" mode, mid V/A/D for "vad" mode
        default_knob = (0.5, 0.5, 0.5) if self.knob_source == "vad" else tuple([0.0]*5 + [0.5])
        knobs = np.asarray(self.knobs.get(uid, default_knob), dtype=np.float32)
        out = {
            "uid": uid,
            "phoneme_ids": phoneme_ids,
            "start":      anch["start"],
            "mid":        anch["mid"],
            "end":        anch["end"],
            "durations":  anch["durations"],
            "spk_emb":    anch["spk_emb"],
            "knobs":      knobs,
        }
        if "z" in anch:
            out["z"] = anch["z"]                 # (N, 256) body-only
        if "z_q" in anch:
            out["z_q"] = anch["z_q"]             # (N, 256) codebook-quantized
            out["code_id"] = anch["code_id"]     # (N,)
        return out


def collate_v8(batch, pad_id=0):
    """Pad to max-(N+2) in batch.
    All arrays end up shape (B, N_max+2, ...) where +2 is BOS+EOS positions.
    BOS/EOS positions: anchors=zeros (no supervision), durations=1, phoneme_mask=True."""
    B = len(batch)
    Ns_full = [len(b["phoneme_ids"]) for b in batch]   # = body_len + 2
    N_max = max(Ns_full)
    F = batch[0]["start"].shape[1]

    has_z = "z" in batch[0]
    has_codes = "z_q" in batch[0]
    Z = batch[0]["z"].shape[1] if has_z else (batch[0]["z_q"].shape[1] if has_codes else 0)

    phoneme_ids = np.full((B, N_max), pad_id, dtype=np.int64)
    start = np.zeros((B, N_max, F), dtype=np.float32)
    mid   = np.zeros((B, N_max, F), dtype=np.float32)
    end   = np.zeros((B, N_max, F), dtype=np.float32)
    durations = np.zeros((B, N_max), dtype=np.int64)
    phoneme_mask = np.zeros((B, N_max), dtype=bool)
    spk_emb = np.zeros((B, batch[0]["spk_emb"].shape[0]), dtype=np.float32)
    K_dim = batch[0]["knobs"].shape[0]
    knobs = np.zeros((B, K_dim), dtype=np.float32)
    z = np.zeros((B, N_max, Z), dtype=np.float32) if has_z else None
    z_q = np.zeros((B, N_max, Z), dtype=np.float32) if has_codes else None
    # code_id: pad with PAD_CODE (-1) at non-body positions; CE loss masks them out
    code_id = np.full((B, N_max), -1, dtype=np.int64) if has_codes else None

    for i, b in enumerate(batch):
        n_full = len(b["phoneme_ids"])         # body + 2
        n_body = n_full - 2
        phoneme_ids[i, :n_full] = b["phoneme_ids"]
        # Anchors at body positions: indices [1, 1+n_body)
        if n_body > 0:
            start[i, 1:1+n_body] = b["start"]
            mid[i,   1:1+n_body] = b["mid"]
            end[i,   1:1+n_body] = b["end"]
            durations[i, 1:1+n_body] = b["durations"]
        # BOS at idx 0, EOS at idx 1+n_body. Both get duration=1.
        durations[i, 0] = 1
        durations[i, 1 + n_body] = 1
        phoneme_mask[i, :n_full] = True
        spk_emb[i] = b["spk_emb"]
        knobs[i] = b["knobs"]
        if has_z:
            z[i, 1:1+n_body] = b["z"]
        if has_codes:
            z_q[i, 1:1+n_body] = b["z_q"]
            code_id[i, 1:1+n_body] = b["code_id"]

    out = {
        "phoneme_ids":  torch.from_numpy(phoneme_ids),
        "phoneme_mask": torch.from_numpy(phoneme_mask),
        "start":        torch.from_numpy(start),
        "mid":          torch.from_numpy(mid),
        "end":          torch.from_numpy(end),
        "durations":    torch.from_numpy(durations),
        "spk_emb":      torch.from_numpy(spk_emb),
        "knobs":        torch.from_numpy(knobs),
        "uids":         [b["uid"] for b in batch],
    }
    if has_z:
        out["z"] = torch.from_numpy(z)
    if has_codes:
        out["z_q"] = torch.from_numpy(z_q)
        out["code_id"] = torch.from_numpy(code_id)
    return out
