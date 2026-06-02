#!/usr/bin/env python3
"""
Boost quiet speech without reducing loud speech.

This is intentionally not a compressor/loudnorm pass. It estimates short-time
RMS/peak, applies gain only where the local RMS is below a target, caps gain by
available peak headroom, and smooths the gain envelope to avoid pumping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def frame_stats(wav: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    rms = []
    peak = []
    for start in range(0, max(1, len(wav) - frame + 1), hop):
        chunk = wav[start:start + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
        peak.append(float(np.max(np.abs(chunk)) + 1e-12))
    if not rms:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    return np.asarray(rms, dtype=np.float32), np.asarray(peak, dtype=np.float32)


def smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) <= 2:
        return values
    n = radius * 2 + 1
    x = np.linspace(-2.5, 2.5, n)
    kernel = np.exp(-0.5 * x * x)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def interpolate_gain(frame_gain: np.ndarray, total: int, frame: int, hop: int) -> np.ndarray:
    centers = np.arange(len(frame_gain), dtype=np.float32) * hop + frame / 2
    if len(centers) == 1:
        return np.full(total, frame_gain[0], dtype=np.float32)
    samples = np.arange(total, dtype=np.float32)
    return np.interp(samples, centers, frame_gain, left=frame_gain[0], right=frame_gain[-1]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("output_wav")
    ap.add_argument("--target-rms-db", type=float, default=-25.0)
    ap.add_argument("--noise-floor-db", type=float, default=-48.0)
    ap.add_argument("--max-gain-db", type=float, default=9.0)
    ap.add_argument("--headroom", type=float, default=0.94)
    ap.add_argument("--frame-ms", type=float, default=80.0)
    ap.add_argument("--hop-ms", type=float, default=10.0)
    ap.add_argument("--smooth-ms", type=float, default=280.0)
    args = ap.parse_args()

    src = Path(args.input_wav)
    dst = Path(args.output_wav)
    dst.parent.mkdir(parents=True, exist_ok=True)

    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    frame = max(1, int(sr * args.frame_ms / 1000.0))
    hop = max(1, int(sr * args.hop_ms / 1000.0))
    rms, peak = frame_stats(wav, frame, hop)

    target = db_to_amp(args.target_rms_db)
    floor = db_to_amp(args.noise_floor_db)
    max_gain = db_to_amp(args.max_gain_db)

    desired = np.ones_like(rms)
    active = rms > floor
    quiet = active & (rms < target)
    desired[quiet] = target / np.maximum(rms[quiet], 1e-8)
    desired = np.minimum(desired, max_gain)

    headroom_gain = args.headroom / np.maximum(peak, 1e-8)
    desired = np.minimum(desired, np.maximum(headroom_gain, 1.0))
    desired = np.maximum(desired, 1.0)

    smooth_radius = max(1, int((args.smooth_ms / 1000.0) / (args.hop_ms / 1000.0) / 2))
    gain_frames = smooth(desired, smooth_radius)
    # Guarantee this remains upward-only even after smoothing.
    gain_frames = np.maximum(gain_frames, 1.0)

    gain = interpolate_gain(gain_frames, len(wav), frame, hop)
    out = wav * gain
    # Safety only: if a boosted region still clips due interpolation, scale just
    # the offending samples via hard limit. Original loud dialogue is untouched
    # unless it was already beyond the headroom.
    out = np.clip(out, -args.headroom, args.headroom)

    sf.write(dst, out, sr)

    summary = {
        "input": str(src),
        "output": str(dst),
        "sample_rate": sr,
        "duration": round(len(wav) / sr, 3),
        "target_rms_db": args.target_rms_db,
        "noise_floor_db": args.noise_floor_db,
        "max_gain_db": args.max_gain_db,
        "gain_min": round(float(gain.min()), 4),
        "gain_mean": round(float(gain.mean()), 4),
        "gain_p90": round(float(np.percentile(gain, 90)), 4),
        "gain_max": round(float(gain.max()), 4),
    }
    with open(dst.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
