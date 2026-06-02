#!/usr/bin/env python3
"""
Create stricter dialogue-only audio from a separated dialogue/vocals stem.

This is the second-stage cleanup after source separation:
  1) ffmpeg denoise/noise-gate conditioning
  2) Silero VAD speech detection
  3) write same-timeline audio with non-speech muted
  4) write compact audio containing only speech regions
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def condition_audio(src: Path, out: Path, sr: int) -> None:
    # Conservative speech cleanup before VAD. Demucs removes most accompaniment;
    # this pass suppresses broadband noise and low-level ambience.
    filters = ",".join(
        [
            "highpass=f=90",
            "lowpass=f=7200",
            "afftdn=nf=-32",
            "anlmdn=s=0.00001:p=0.002:r=0.002",
            "agate=threshold=0.015:ratio=8:attack=5:release=80",
            "dynaudnorm=f=151:g=12",
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-af",
            filters,
            str(out),
        ]
    )


def merge_timestamps(
    stamps: list[dict[str, int]],
    sr: int,
    min_gap_ms: int,
    pad_ms: int,
    total_samples: int,
) -> list[dict[str, int]]:
    if not stamps:
        return []
    pad = int(sr * pad_ms / 1000)
    min_gap = int(sr * min_gap_ms / 1000)
    expanded = [
        {
            "start": max(0, int(s["start"]) - pad),
            "end": min(total_samples, int(s["end"]) + pad),
        }
        for s in stamps
    ]
    merged = [expanded[0]]
    for s in expanded[1:]:
        prev = merged[-1]
        if s["start"] - prev["end"] <= min_gap:
            prev["end"] = max(prev["end"], s["end"])
        else:
            merged.append(s)
    return merged


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


def write_outputs(
    conditioned: Path,
    out_dir: Path,
    timestamps: list[dict[str, int]],
    sr: int,
    fade_ms: int,
    compact_gap_ms: int,
) -> dict:
    wav, file_sr = sf.read(conditioned, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        raise ValueError(f"expected {sr} Hz, got {file_sr}")

    mask = np.zeros(len(wav), dtype=bool)
    for s in timestamps:
        mask[s["start"]:s["end"]] = True
    soft_mask = fade_edges(mask, sr, fade_ms)
    masked = wav * soft_mask

    compact_parts: list[np.ndarray] = []
    gap = np.zeros(int(sr * compact_gap_ms / 1000), dtype=np.float32)
    for idx, s in enumerate(timestamps):
        part = masked[s["start"]:s["end"]]
        if len(part):
            if idx:
                compact_parts.append(gap)
            compact_parts.append(part)
    compact = np.concatenate(compact_parts) if compact_parts else np.zeros(0, dtype=np.float32)

    masked_path = out_dir / "dialogue_words_only_masked.wav"
    compact_path = out_dir / "dialogue_words_only_compact.wav"
    sf.write(masked_path, masked, sr)
    sf.write(compact_path, compact, sr)

    segments = [
        {
            "start": round(s["start"] / sr, 3),
            "end": round(s["end"] / sr, 3),
            "duration": round((s["end"] - s["start"]) / sr, 3),
        }
        for s in timestamps
    ]
    with open(out_dir / "dialogue_words_only_segments.json", "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    speech_seconds = sum(s["end"] - s["start"] for s in timestamps) / sr
    return {
        "masked_path": str(masked_path),
        "compact_path": str(compact_path),
        "segments_path": str(out_dir / "dialogue_words_only_segments.json"),
        "segments": len(timestamps),
        "source_seconds": round(len(wav) / sr, 3),
        "speech_seconds": round(speech_seconds, 3),
        "kept_ratio": round(speech_seconds / max(len(wav) / sr, 1e-6), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--min-speech-ms", type=int, default=140)
    ap.add_argument("--min-silence-ms", type=int, default=160)
    ap.add_argument("--merge-gap-ms", type=int, default=120)
    ap.add_argument("--speech-pad-ms", type=int, default=35)
    ap.add_argument("--fade-ms", type=int, default=12)
    ap.add_argument("--compact-gap-ms", type=int, default=80)
    args = ap.parse_args()

    src = Path(args.input_wav).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditioned = out_dir / "dialogue_tight_denoised.wav"
    condition_audio(src, conditioned, args.sr)

    model = load_silero_vad()
    wav = read_audio(str(conditioned), sampling_rate=args.sr)
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav)
    stamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=args.sr,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=0,
        return_seconds=False,
    )
    merged = merge_timestamps(
        stamps,
        args.sr,
        min_gap_ms=args.merge_gap_ms,
        pad_ms=args.speech_pad_ms,
        total_samples=wav.numel(),
    )
    summary = write_outputs(
        conditioned,
        out_dir,
        merged,
        sr=args.sr,
        fade_ms=args.fade_ms,
        compact_gap_ms=args.compact_gap_ms,
    )
    summary.update(
        {
            "threshold": args.threshold,
            "min_speech_ms": args.min_speech_ms,
            "min_silence_ms": args.min_silence_ms,
            "merge_gap_ms": args.merge_gap_ms,
            "speech_pad_ms": args.speech_pad_ms,
        }
    )
    with open(out_dir / "dialogue_words_only_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
