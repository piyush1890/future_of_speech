"""
Tag utterances with audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
(continuous valence/arousal/dominance, trained on MSP-Podcast — natural speech).

Each utterance gets 3 continuous values in roughly [0, 1]:
  valence    — pleasantness (0=very negative, 0.5=neutral, 1=very positive)
  arousal    — activation/intensity (0=calm, 1=highly excited)
  dominance  — speaker dominance (0=submissive, 1=dominant)

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python -u scripts/tag_emotions_vad.py \\
      --dataset librispeech --device mps
  KMP_DUPLICATE_LIB_OK=TRUE python -u scripts/tag_emotions_vad.py \\
      --dataset esd --device mps
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)


MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Audeering's V/A/D regression model — exact architecture from their model card."""
    # Newer transformers versions expect this attribute; the custom class doesn't tie weights.
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0]                  # (B, T, H)
        pooled = torch.mean(hidden, dim=1)   # (B, H)
        logits = self.classifier(pooled)     # (B, 3): valence, arousal, dominance
        return logits


def find_libri_audio(utt_id: str, libri_root: Path) -> str | None:
    parts = utt_id.split("-")
    if len(parts) != 3:
        return None
    speaker, chapter, _ = parts
    for split in ["train-clean-100", "train-clean-360", "dev-clean",
                  "dev-other", "test-clean", "test-other"]:
        p = libri_root / split / speaker / chapter / f"{utt_id}.flac"
        if p.exists():
            return str(p)
    return None


def list_librispeech(features_dir: Path, libri_root: Path):
    """Yield (key, path)."""
    for f in sorted(features_dir.glob("*.npz")):
        if f.stem == "norm_stats" or "-" not in f.stem:
            continue
        ap = find_libri_audio(f.stem, libri_root)
        if ap:
            yield f.stem, ap


def list_esd(esd_root: Path):
    """Yield (key, path) — English speakers only."""
    for spk_dir in sorted(esd_root.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name.startswith("__"):
            continue
        try:
            spk_n = int(spk_dir.name)
        except ValueError:
            continue
        if not (11 <= spk_n <= 20):
            continue
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir() or emo_dir.name.startswith("__"):
                continue
            for wav in sorted(emo_dir.glob("*.wav")):
                key = f"{spk_dir.name}_{emo_dir.name}_{wav.stem}"
                yield key, str(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["librispeech", "esd"], required=True)
    ap.add_argument("--features-dir", default="data/features_merged_logpitch_v2")
    ap.add_argument("--libri-root", default="data/LibriSpeech")
    ap.add_argument("--esd-root", default="data/esd_raw/Emotion Speech Dataset")
    ap.add_argument("--out", default=None,
                    help="Defaults to data/<dataset>_emotion_vad.json")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    out_path = Path(args.out or f"data/{args.dataset}_emotion_vad.json")

    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing tags")
    else:
        results = {}

    # Build pending list
    if args.dataset == "librispeech":
        pairs = list(list_librispeech(Path(args.features_dir), Path(args.libri_root)))
    else:
        pairs = list(list_esd(Path(args.esd_root)))
    remaining = [(k, p) for k, p in pairs if k not in results]
    print(f"Total: {len(pairs)}, already tagged: {len(results)}, "
          f"remaining: {len(remaining)}")
    if not remaining:
        print("All done!")
        return

    print(f"Loading {MODEL_ID} on {args.device}...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    # audeering's config has vocab_size=None which fails the new HF dataclass validation.
    # Patch the config dict before instantiation.
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg_dict["vocab_size"] = cfg_dict.get("vocab_size") or 32
    from transformers import Wav2Vec2Config
    config = Wav2Vec2Config(**cfg_dict)
    model = EmotionModel.from_pretrained(MODEL_ID, config=config)
    device = torch.device(args.device)
    model = model.to(device).eval()
    print("  loaded; output is (valence, arousal, dominance) in roughly [0, 1]")

    t0 = time.time()
    tagged = 0
    skipped = 0
    val_acc, aro_acc, dom_acc = [], [], []

    with torch.no_grad():
        for key, wav_path in remaining:
            try:
                wav, sr = sf.read(wav_path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import scipy.signal as sps
                    n_new = int(round(len(wav) * 16000 / sr))
                    wav = sps.resample(wav, n_new).astype(np.float32)
                    sr = 16000
                inputs = fe(wav, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                vad = model(input_values).squeeze(0).cpu().numpy()
                v, a, d = float(vad[0]), float(vad[1]), float(vad[2])
                results[key] = {
                    "valence": round(v, 4),
                    "arousal": round(a, 4),
                    "dominance": round(d, 4),
                }
                val_acc.append(v); aro_acc.append(a); dom_acc.append(d)
                tagged += 1
            except Exception:
                skipped += 1
                continue

            if (tagged % args.save_every) == 0 and tagged > 0:
                elapsed = time.time() - t0
                rate = tagged / elapsed
                eta_sec = (len(remaining) - tagged) / rate
                v_arr = np.array(val_acc); a_arr = np.array(aro_acc); d_arr = np.array(dom_acc)
                print(f"[{tagged}/{len(remaining)}] {rate:.1f} files/s, "
                      f"ETA: {eta_sec/60:.1f} min, skipped: {skipped} | "
                      f"val={v_arr.mean():.2f}±{v_arr.std():.2f} "
                      f"aro={a_arr.mean():.2f}±{a_arr.std():.2f} "
                      f"dom={d_arr.mean():.2f}±{d_arr.std():.2f}")
                with open(out_path, "w") as f:
                    json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone! Tagged {tagged}, skipped {skipped}, time: {elapsed/60:.1f} min")
    if val_acc:
        v = np.array(val_acc); a = np.array(aro_acc); d = np.array(dom_acc)
        print(f"\nDistribution stats:")
        print(f"  valence:    mean={v.mean():.3f}  std={v.std():.3f}  "
              f"p10={np.percentile(v,10):.3f}  p90={np.percentile(v,90):.3f}")
        print(f"  arousal:    mean={a.mean():.3f}  std={a.std():.3f}  "
              f"p10={np.percentile(a,10):.3f}  p90={np.percentile(a,90):.3f}")
        print(f"  dominance:  mean={d.mean():.3f}  std={d.std():.3f}  "
              f"p10={np.percentile(d,10):.3f}  p90={np.percentile(d,90):.3f}")


if __name__ == "__main__":
    main()
