#!/usr/bin/env python3
"""
Make a natural dialogue-only version from a separated dialogue/vocals stem.

Unlike word-level masking, this keeps whole speech phrases with generous padding
and soft fades, so words are not chopped and prosody is not damaged. A lightly
conditioned copy is used only for speech detection; the output audio is cut from
the original input to avoid denoise/gate artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
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


def resample_for_detection(src: Path, dst: Path, sr: int) -> None:
    # Mild conditioning only for VAD robustness. This file is not used as output.
    filters = "highpass=f=80,lowpass=f=7600,afftdn=nf=-24"
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
            str(dst),
        ]
    )


def load_output_audio(src: Path, sr: int) -> np.ndarray:
    tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
    try:
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
                str(tmp),
            ]
        )
        wav, file_sr = sf.read(tmp, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if file_sr != sr:
            raise ValueError(f"expected {sr}, got {file_sr}")
        return wav
    finally:
        tmp.unlink(missing_ok=True)


def merge_timestamps(
    stamps: list[dict[str, int]],
    sr: int,
    total_samples: int,
    pad_ms: int,
    merge_gap_ms: int,
) -> list[dict[str, int]]:
    pad = int(sr * pad_ms / 1000)
    merge_gap = int(sr * merge_gap_ms / 1000)
    expanded = [
        {
            "start": max(0, int(s["start"]) - pad),
            "end": min(total_samples, int(s["end"]) + pad),
        }
        for s in stamps
    ]
    if not expanded:
        return []
    merged = [expanded[0]]
    for s in expanded[1:]:
        prev = merged[-1]
        if s["start"] - prev["end"] <= merge_gap:
            prev["end"] = max(prev["end"], s["end"])
        else:
            merged.append(s)
    return merged


def cosine_fade_mask(mask: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    fade = max(1, int(sr * fade_ms / 1000))
    out = mask.astype(np.float32)
    changes = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start in starts:
        end = min(len(out), start + fade)
        n = end - start
        if n > 0:
            out[start:end] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    for end in ends:
        start = max(0, end - fade)
        n = end - start
        if n > 0:
            out[start:end] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    return out


def compact_segments(
    masked: np.ndarray,
    segments: list[dict[str, int]],
    sr: int,
    gap_ms: int,
) -> np.ndarray:
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        part = masked[seg["start"]:seg["end"]]
        if len(part) == 0:
            continue
        if i:
            parts.append(gap)
        parts.append(part.astype(np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def complement_segments(
    kept_segments: list[dict[str, int]],
    total_samples: int,
    *,
    min_duration_ms: int,
    sr: int,
) -> list[dict[str, int]]:
    rejected: list[dict[str, int]] = []
    cursor = 0
    min_samples = max(1, int(sr * min_duration_ms / 1000))
    for seg in kept_segments:
        start = max(0, int(seg["start"]))
        end = min(total_samples, int(seg["end"]))
        if start - cursor >= min_samples:
            rejected.append({"start": cursor, "end": start})
        cursor = max(cursor, end)
    if total_samples - cursor >= min_samples:
        rejected.append({"start": cursor, "end": total_samples})
    return rejected


def export_rejected_segments(
    source_audio: np.ndarray,
    segments: list[dict[str, int]],
    out_dir: Path,
    *,
    sr: int,
) -> list[dict[str, object]]:
    review_dir = out_dir / "rejected_non_dialogue"
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for idx, seg in enumerate(segments):
        start = int(seg["start"])
        end = int(seg["end"])
        path = review_dir / f"{idx:06d}_non_dialogue.wav"
        sf.write(path, source_audio[start:end], sr)
        manifest.append(
            {
                "clip_id": f"non_dialogue_{idx:06d}",
                "start": round(start / sr, 3),
                "end": round(end / sr, 3),
                "duration": round((end - start) / sr, 3),
                "audio": str(path),
                "reject_stage": "00_dialogue_vad",
                "reject_reason": "non_dialogue_or_silence",
            }
        )
    with (review_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--threshold", type=float, default=0.48)
    ap.add_argument("--min-speech-ms", type=int, default=160)
    ap.add_argument("--min-silence-ms", type=int, default=420)
    ap.add_argument("--speech-pad-ms", type=int, default=220)
    ap.add_argument("--merge-gap-ms", type=int, default=520)
    ap.add_argument("--fade-ms", type=int, default=90)
    ap.add_argument("--compact-gap-ms", type=int, default=140)
    ap.add_argument(
        "--export-rejected-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export VAD-rejected non-dialogue spans for review.",
    )
    ap.add_argument("--rejected-min-duration-ms", type=int, default=500)
    args = ap.parse_args()

    src = Path(args.input_wav).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detection_wav = out_dir / "vad_detection_mild.wav"
    resample_for_detection(src, detection_wav, args.sr)

    model = load_silero_vad()
    vad_audio = read_audio(str(detection_wav), sampling_rate=args.sr)
    if not isinstance(vad_audio, torch.Tensor):
        vad_audio = torch.tensor(vad_audio)

    stamps = get_speech_timestamps(
        vad_audio,
        model,
        sampling_rate=args.sr,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=0,
        return_seconds=False,
    )

    source_audio = load_output_audio(src, args.sr)
    segments = merge_timestamps(
        stamps,
        sr=args.sr,
        total_samples=len(source_audio),
        pad_ms=args.speech_pad_ms,
        merge_gap_ms=args.merge_gap_ms,
    )

    mask = np.zeros(len(source_audio), dtype=bool)
    for seg in segments:
        mask[seg["start"]:seg["end"]] = True
    soft_mask = cosine_fade_mask(mask, args.sr, args.fade_ms)
    masked = source_audio * soft_mask
    compact = compact_segments(masked, segments, args.sr, args.compact_gap_ms)

    masked_path = out_dir / "dialogue_phrase_masked_timeline.wav"
    compact_path = out_dir / "dialogue_phrase_compact_natural.wav"
    sf.write(masked_path, masked, args.sr)
    sf.write(compact_path, compact, args.sr)

    public_segments = [
        {
            "start": round(seg["start"] / args.sr, 3),
            "end": round(seg["end"] / args.sr, 3),
            "duration": round((seg["end"] - seg["start"]) / args.sr, 3),
        }
        for seg in segments
    ]
    with open(out_dir / "dialogue_phrase_segments.json", "w", encoding="utf-8") as f:
        json.dump(public_segments, f, indent=2)

    rejected_segments = complement_segments(
        segments,
        len(source_audio),
        min_duration_ms=args.rejected_min_duration_ms,
        sr=args.sr,
    )
    rejected_manifest: list[dict[str, object]] = []
    if args.export_rejected_segments:
        rejected_manifest = export_rejected_segments(
            source_audio,
            rejected_segments,
            out_dir,
            sr=args.sr,
        )

    speech_seconds = sum(seg["end"] - seg["start"] for seg in segments) / args.sr
    rejected_seconds = sum(seg["end"] - seg["start"] for seg in rejected_segments) / args.sr
    summary = {
        "source_seconds": round(len(source_audio) / args.sr, 3),
        "speech_seconds": round(speech_seconds, 3),
        "rejected_non_dialogue_seconds": round(rejected_seconds, 3),
        "rejected_non_dialogue_segments": len(rejected_segments),
        "compact_seconds": round(len(compact) / args.sr, 3),
        "kept_ratio": round(speech_seconds / max(len(source_audio) / args.sr, 1e-6), 4),
        "segments": len(segments),
        "masked_timeline": str(masked_path),
        "compact": str(compact_path),
        "rejected_non_dialogue_dir": (
            str(out_dir / "rejected_non_dialogue") if args.export_rejected_segments else None
        ),
        "rejected_non_dialogue_manifest": (
            str(out_dir / "rejected_non_dialogue" / "manifest.jsonl")
            if args.export_rejected_segments
            else None
        ),
        "rejected_non_dialogue_exported": len(rejected_manifest),
        "threshold": args.threshold,
        "speech_pad_ms": args.speech_pad_ms,
        "merge_gap_ms": args.merge_gap_ms,
        "fade_ms": args.fade_ms,
    }
    with open(out_dir / "dialogue_phrase_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
