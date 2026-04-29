"""
Tag LibriSpeech utterances with ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
(8-class RAVDESS-trained classifier). Same model used for ESD tagging — keeps the
two datasets on a consistent emotion-label vocabulary.

Output: data/librispeech_emotion_tags_w2v2.json
  { "utt_id": {"label": "neutral", "confidence": 0.86,
               "scores": {"angry": ..., "calm": ..., "disgust": ..., ...}}, ... }

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE python -u scripts/tag_librispeech_emotions_w2v2.py --device mps
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def remap_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if k == "classifier.dense.weight":
            out["projector.weight"] = v
        elif k == "classifier.dense.bias":
            out["projector.bias"] = v
        elif k == "classifier.output.weight":
            out["classifier.weight"] = v
        elif k == "classifier.output.bias":
            out["classifier.bias"] = v
        else:
            out[k] = v
    return out


def find_audio_path(utt_id: str, libri_root: Path) -> str | None:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/features_merged_logpitch_v2")
    ap.add_argument("--libri-root", default="data/LibriSpeech")
    ap.add_argument("--out", default="data/librispeech_emotion_tags_w2v2.json")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing tags")
    else:
        results = {}

    # All LibriSpeech utt_ids: features have "<spk>-<chapter>-<utt>" pattern (with "-")
    # ESD utt_ids look like "0011_Happy_..." (no "-") so they're naturally excluded.
    features_dir = Path(args.features_dir)
    utt_ids = sorted([f.stem for f in features_dir.glob("*.npz")
                      if f.stem != "norm_stats" and "-" in f.stem])
    remaining = [u for u in utt_ids if u not in results]
    print(f"Total LibriSpeech: {len(utt_ids)}, already tagged: {len(results)}, "
          f"remaining: {len(remaining)}")
    if not remaining:
        print("All done!")
        return

    print(f"Loading {MODEL_ID} on {args.device} (with key remapping)...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.num_labels = 8
    cfg.classifier_proj_size = cfg.hidden_size
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, config=cfg, ignore_mismatched_sizes=True,
    )
    bin_path = hf_hub_download(repo_id=MODEL_ID, filename="pytorch_model.bin")
    raw_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(remap_state_dict(raw_sd), strict=False)
    head_keys = ["projector.weight", "projector.bias",
                 "classifier.weight", "classifier.bias"]
    for hk in head_keys:
        if hk in missing:
            print(f"WARNING: head key {hk} not loaded — predictions would be random!")
            return
    print(f"  missing keys: {len(missing)} (e.g. {missing[:3]})")
    print(f"  unexpected keys: {len(unexpected)}")
    print(f"  head keys all present: ok")

    device = torch.device(args.device)
    model = model.to(device).eval()

    libri_root = Path(args.libri_root)
    t0 = time.time()
    tagged = 0
    skipped = 0
    not_found = 0

    with torch.no_grad():
        for utt_id in remaining:
            path = find_audio_path(utt_id, libri_root)
            if path is None:
                not_found += 1
                skipped += 1
                continue

            try:
                wav, sr = sf.read(path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import scipy.signal as sps
                    n_new = int(round(len(wav) * 16000 / sr))
                    wav = sps.resample(wav, n_new).astype(np.float32)
                    sr = 16000
                inputs = fe(wav, sampling_rate=sr, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                logits = model(input_values).logits.squeeze(0)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                idx = int(probs.argmax())
                scores = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
                results[utt_id] = {
                    "label": LABELS[idx],
                    "confidence": round(float(probs[idx]), 4),
                    "scores": scores,
                }
                tagged += 1
            except Exception:
                skipped += 1
                continue

            if (tagged % args.save_every) == 0 and tagged > 0:
                elapsed = time.time() - t0
                rate = tagged / elapsed
                eta_sec = (len(remaining) - tagged) / rate
                print(f"[{tagged}/{len(remaining)}] {rate:.1f} files/s, "
                      f"ETA: {eta_sec/60:.1f} min, skipped: {skipped} "
                      f"(audio not found: {not_found})")
                with open(out_path, "w") as f:
                    json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone! Tagged {tagged}, skipped {skipped} "
          f"(not_found={not_found}), time: {elapsed/60:.1f} min")

    labs = Counter(r["label"] for r in results.values())
    print(f"\nLibriSpeech wav2vec2 label distribution ({len(results)}):")
    for lab, cnt in labs.most_common():
        print(f"  {lab:>10}: {cnt:>6}  ({cnt/len(results):.1%})")


if __name__ == "__main__":
    main()
