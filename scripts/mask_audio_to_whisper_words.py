#!/usr/bin/env python3
"""
Use Whisper word timestamps to keep only recognized words from an audio file.

Best used after source separation + VAD tightening. It writes:
  - word_masked_timeline.wav: same timeline, everything outside word spans muted
  - word_only_compact.wav: only recognized word spans concatenated
  - whisper_words.json: accepted word spans and transcript snippets
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
import torch


DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def valid_word(text: str, min_devanagari_ratio: float) -> bool:
    clean = text.strip().strip("।.!?,;:'\"()[]{}-–—…")
    if not clean:
        return False
    chars = [ch for ch in clean if not ch.isspace()]
    if not chars:
        return False
    deva = sum(1 for ch in chars if "\u0900" <= ch <= "\u097f")
    latin_ok = clean.lower() in {"wikipedia", "citation", "citations"}
    return latin_ok or (deva / len(chars) >= min_devanagari_ratio)


def fade_edges(mask: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    fade = max(1, int(sr * fade_ms / 1000))
    out = mask.astype(np.float32).copy()
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s in starts:
        e = min(len(out), s + fade)
        out[s:e] *= np.linspace(0.0, 1.0, e - s, dtype=np.float32)
    for e in ends:
        s = max(0, e - fade)
        out[s:e] *= np.linspace(1.0, 0.0, e - s, dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--language", default="hi")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--transcript-json", default=None)
    ap.add_argument("--initial-prompt", default="यह हिंदी संवाद है। केवल बोले गए शब्द देवनागरी में लिखें।")
    ap.add_argument("--min-word-prob", type=float, default=0.35)
    ap.add_argument("--min-devanagari-ratio", type=float, default=0.60)
    ap.add_argument("--pad-ms", type=int, default=25)
    ap.add_argument("--fade-ms", type=int, default=8)
    ap.add_argument("--compact-gap-ms", type=int, default=12)
    args = ap.parse_args()

    src = Path(args.input_wav).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    if args.transcript_json:
        with open(args.transcript_json, encoding="utf-8") as f:
            result = json.load(f)
    else:
        import whisper

        model = whisper.load_model(args.model, device=device)
        result = model.transcribe(
            str(src),
            language=args.language,
            task="transcribe",
            word_timestamps=True,
            temperature=0.0,
            beam_size=5,
            fp16=device == "cuda",
            condition_on_previous_text=False,
            initial_prompt=args.initial_prompt,
            verbose=False,
        )
        with open(out_dir / "whisper_raw.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    pad = int(sr * args.pad_ms / 1000)
    intervals: list[dict] = []
    for seg in result.get("segments", []):
        for word in seg.get("words") or []:
            text = str(word.get("word", "")).strip()
            prob = word.get("probability")
            if prob is not None and float(prob) < args.min_word_prob:
                continue
            if not valid_word(text, args.min_devanagari_ratio):
                continue
            if word.get("start") is None or word.get("end") is None:
                continue
            start = max(0, int(round(float(word["start"]) * sr)) - pad)
            end = min(len(wav), int(round(float(word["end"]) * sr)) + pad)
            if end <= start:
                continue
            intervals.append(
                {
                    "word": text,
                    "start": round(start / sr, 3),
                    "end": round(end / sr, 3),
                    "probability": float(prob) if prob is not None else None,
                    "_start_sample": start,
                    "_end_sample": end,
                }
            )

    intervals.sort(key=lambda x: x["_start_sample"])
    merged: list[dict] = []
    for item in intervals:
        if not merged or item["_start_sample"] > merged[-1]["_end_sample"]:
            merged.append(item)
        else:
            merged[-1]["_end_sample"] = max(merged[-1]["_end_sample"], item["_end_sample"])
            merged[-1]["end"] = round(merged[-1]["_end_sample"] / sr, 3)
            merged[-1]["word"] += " " + item["word"]

    mask = np.zeros(len(wav), dtype=bool)
    for item in merged:
        mask[item["_start_sample"]:item["_end_sample"]] = True
    soft_mask = fade_edges(mask, sr, args.fade_ms)
    masked = wav * soft_mask

    compact_parts: list[np.ndarray] = []
    gap = np.zeros(int(sr * args.compact_gap_ms / 1000), dtype=np.float32)
    for idx, item in enumerate(merged):
        part = masked[item["_start_sample"]:item["_end_sample"]]
        if len(part):
            if idx:
                compact_parts.append(gap)
            compact_parts.append(part)
    compact = np.concatenate(compact_parts) if compact_parts else np.zeros(0, dtype=np.float32)

    sf.write(out_dir / "word_masked_timeline.wav", masked, sr)
    sf.write(out_dir / "word_only_compact.wav", compact, sr)
    public_intervals = [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in merged
    ]
    with open(out_dir / "whisper_words.json", "w", encoding="utf-8") as f:
        json.dump(public_intervals, f, ensure_ascii=False, indent=2)
    summary = {
        "input_seconds": round(len(wav) / sr, 3),
        "word_regions": len(public_intervals),
        "word_seconds": round(len(compact) / sr, 3),
        "kept_ratio": round((len(compact) / sr) / max(len(wav) / sr, 1e-6), 4),
        "masked_timeline": str(out_dir / "word_masked_timeline.wav"),
        "compact": str(out_dir / "word_only_compact.wav"),
    }
    with open(out_dir / "word_only_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
