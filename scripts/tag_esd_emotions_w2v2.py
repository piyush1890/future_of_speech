"""
Tag ESD utterances with ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
(8-class RAVDESS-trained classifier).

The model uses a custom Wav2Vec2ForSpeechClassification head (classifier.dense + classifier.output)
that doesn't directly load into transformers' standard Wav2Vec2ForSequenceClassification (projector + classifier).
We load the state dict and rename keys to match the standard class.

Output: data/esd_emotion_tags_w2v2.json (same format as HuBERT version)
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
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
LABELS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def remap_state_dict(sd):
    """Rename ehcalabres keys to standard Wav2Vec2ForSequenceClassification keys."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esd-root", type=str,
                    default="data/esd_raw/Emotion Speech Dataset")
    ap.add_argument("--out", type=str, default="data/esd_emotion_tags_w2v2.json")
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    esd_root = Path(args.esd_root)
    out_path = Path(args.out)

    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing tags")
    else:
        results = {}

    all_wavs = []
    for spk_dir in sorted(esd_root.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name.startswith("__"):
            continue
        if not (11 <= int(spk_dir.name) <= 20):
            continue
        for emo_dir in sorted(spk_dir.iterdir()):
            if not emo_dir.is_dir() or emo_dir.name.startswith("__"):
                continue
            for wav in sorted(emo_dir.glob("*.wav")):
                key = f"{spk_dir.name}_{emo_dir.name}_{wav.stem}"
                all_wavs.append((key, str(wav), spk_dir.name, emo_dir.name, wav.stem))

    remaining = [w for w in all_wavs if w[0] not in results]
    print(f"Total: {len(all_wavs)}, already tagged: {len(results)}, "
          f"remaining: {len(remaining)}")
    if not remaining:
        print("All done!")
        return

    print(f"Loading {MODEL_ID} on {args.device} (with key remapping)...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    # ehcalabres uses classifier_proj_size = hidden_size (1024 for xlsr-large) — no bottleneck.
    # Standard config has classifier_proj_size=256. Override to match the checkpoint.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.num_labels = 8
    cfg.classifier_proj_size = cfg.hidden_size   # 1024
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, config=cfg, ignore_mismatched_sizes=True,
    )
    # Then download the original state dict and load with renamed keys
    bin_path = hf_hub_download(repo_id=MODEL_ID, filename="pytorch_model.bin")
    raw_sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    remapped = remap_state_dict(raw_sd)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # We expect 'wav2vec2.masked_spec_embed' to be MISSING (not used at inference)
    # Anything else is suspicious
    head_keys = ["projector.weight", "projector.bias",
                 "classifier.weight", "classifier.bias"]
    for hk in head_keys:
        if hk in missing:
            print(f"WARNING: head key {hk} not loaded — predictions will be random!")
            return
    print(f"  missing keys: {len(missing)} (e.g. {missing[:3]})")
    print(f"  unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    print(f"  head keys all present: {[hk for hk in head_keys if hk not in missing]}")

    device = torch.device(args.device)
    model = model.to(device).eval()

    t0 = time.time()
    tagged = 0
    skipped = 0

    with torch.no_grad():
        for key, wav_path, spk, emo, utt in remaining:
            try:
                wav, sr = sf.read(wav_path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    # quick resample via numpy if not already 16k
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
                results[key] = {
                    "label": LABELS[idx],
                    "confidence": round(float(probs[idx]), 4),
                    "scores": scores,
                    "esd_speaker": spk,
                    "esd_emotion": emo,
                    "esd_utt": utt,
                }
                tagged += 1
            except Exception as e:
                skipped += 1
                continue

            if (tagged % args.save_every) == 0 and tagged > 0:
                elapsed = time.time() - t0
                rate = tagged / elapsed
                eta_sec = (len(remaining) - tagged) / rate
                print(f"[{tagged}/{len(remaining)}] {rate:.1f} files/s, "
                      f"ETA: {eta_sec/60:.1f} min, skipped: {skipped}")
                with open(out_path, "w") as f:
                    json.dump(results, f)

    with open(out_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone! Tagged {tagged}, skipped {skipped}, time: {elapsed/60:.1f} min")

    labs = Counter(r["label"] for r in results.values())
    print(f"\nLabel distribution:")
    for lab, cnt in labs.most_common():
        print(f"  {lab:>10}: {cnt:>5}")

    matrix = {}
    for r in results.values():
        matrix.setdefault(r["esd_emotion"], Counter())[r["label"]] += 1
    print(f"\nESD acted → wav2vec2 label confusion:")
    cols = sorted(set(LABELS))
    print(f"  {'ESD':<12}", "  ".join(f"{l[:6]:>6}" for l in cols))
    for esd, c in sorted(matrix.items()):
        print(f"  {esd:<12}", "  ".join(f"{c.get(l, 0):>6}" for l in cols))


if __name__ == "__main__":
    main()
