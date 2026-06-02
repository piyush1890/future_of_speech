#!/usr/bin/env python3
from __future__ import annotations

import argparse
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def require_mlx_gpu(mx: Any) -> None:
    default_device = mx.default_device()
    if "gpu" not in str(default_device).lower():
        raise RuntimeError(
            f"CPU fallback is disabled; MLX RoFormer must run on GPU, got {default_device}."
        )


def select_mono_channel(audio: np.ndarray, mode: str) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.shape[1] == 1:
        return audio[:, 0].astype(np.float32, copy=False)
    if mode == "left":
        return audio[:, 0].astype(np.float32, copy=False)
    if mode == "right":
        return audio[:, min(1, audio.shape[1] - 1)].astype(np.float32, copy=False)
    if mode == "average":
        return audio.mean(axis=1).astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(np.square(audio), axis=0))
    if mode == "quietest":
        channel = int(np.argmin(rms))
    else:
        channel = int(np.argmax(rms))
    return audio[:, channel].astype(np.float32, copy=False)


def roformer_output_for(output_dir: Path, input_path: Path, model_filename: str) -> Path:
    model_stem = Path(model_filename).stem
    return output_dir / f"{input_path.stem}_(vocals)_{model_stem}.wav"


def write_input_chunks(
    input_wav: Path,
    chunk_dir: Path,
    *,
    chunk_duration: float,
) -> tuple[list[Path], int]:
    audio, sr = sf.read(input_wav, dtype="float32", always_2d=False)
    if audio.ndim == 1:
        total_samples = audio.shape[0]
    else:
        total_samples = audio.shape[0]
    chunk_samples = int(round(sr * chunk_duration))
    if chunk_duration <= 0 or chunk_samples <= 0 or total_samples <= chunk_samples:
        return [input_wav], sr

    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(total_samples, start + chunk_samples)
        path = chunk_dir / f"{input_wav.stem}_chunk_{idx:05d}.wav"
        sf.write(path, audio[start:end], sr)
        paths.append(path)
    return paths, sr


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate vocals with MLX RoFormer on Apple GPU.")
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("--out-wav", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-filename", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--chunk-duration", type=float, default=120.0)
    parser.add_argument("--segment-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--mono-channel",
        choices=["loudest", "left", "right", "quietest", "average"],
        default="left",
    )
    args = parser.parse_args()

    import librosa
    import mlx.core as mx
    from mlx_audio_separator import Separator

    require_mlx_gpu(mx)

    input_wav = args.input_wav.expanduser().resolve()
    out_wav = args.out_wav.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    model_dir = (
        args.model_dir.expanduser().resolve()
        if args.model_dir
        else project_root / "models" / "mlx-audio-separator"
    )
    work_dir = out_wav.parent / "roformer_work"
    model_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=str(model_dir),
        output_dir=str(work_dir),
        output_format="WAV",
        output_single_stem="vocals",
        sample_rate=args.sample_rate,
        # Avoid mlx_audio_io for file probing; it may be compiled against a
        # different MLX patch version. We do explicit soundfile chunking below.
        chunk_duration=None,
        mdxc_params={
            "segment_size": args.segment_size,
            "override_model_segment_size": False,
            "batch_size": args.batch_size,
            "overlap": args.overlap,
            "pitch_shift": 0,
        },
    )
    require_mlx_gpu(mx)
    separator._skip_auto_tune = True
    separator._strict_separation_errors = True
    separator.load_model(args.model_filename)
    model_instance = separator.model_instance

    def prepare_mix_soundfile(self: Any, mix: str | Path) -> np.ndarray:
        wav, sr = sf.read(str(mix), dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        self.input_encoding = "pcm16"
        return np.stack([wav, wav], axis=0).astype(np.float32)

    def write_audio_soundfile(self: Any, stem_path: str, stem_source: Any) -> None:
        audio = np.asarray(stem_source, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            audio = audio.T
        path = Path(self.output_dir) / stem_path if self.output_dir else Path(stem_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, self.sample_rate)

    model_instance.prepare_mix = types.MethodType(prepare_mix_soundfile, model_instance)
    model_instance.write_audio = types.MethodType(write_audio_soundfile, model_instance)

    chunk_paths, _input_sr = write_input_chunks(
        input_wav,
        work_dir / "input_chunks",
        chunk_duration=args.chunk_duration,
    )
    mono_parts: list[np.ndarray] = []
    output_sr: int | None = None
    stem_paths: list[str] = []
    for chunk_path in chunk_paths:
        outputs = [Path(path) for path in separator.separate(str(chunk_path))]
        expected = roformer_output_for(work_dir, chunk_path, args.model_filename)
        vocals = expected if expected.exists() else next(
            (path for path in outputs if "vocals" in path.name.lower()),
            None,
        )
        if vocals is None or not vocals.exists():
            raise FileNotFoundError(f"MLX RoFormer did not produce a vocals stem for {chunk_path}")
        audio, sr = sf.read(vocals, dtype="float32", always_2d=False)
        if output_sr is None:
            output_sr = sr
        elif sr != output_sr:
            raise ValueError(f"Unexpected RoFormer sample rate change: {sr} != {output_sr}")
        mono_parts.append(select_mono_channel(audio, args.mono_channel))
        stem_paths.append(str(vocals))

    if output_sr is None:
        output_sr = args.sample_rate
    mono = np.concatenate(mono_parts) if mono_parts else np.zeros(0, dtype=np.float32)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, mono, output_sr)
    print(
        {
            "input": str(input_wav),
            "output": str(out_wav),
            "roformer_stems": stem_paths,
            "sample_rate": output_sr,
            "mono_channel": args.mono_channel,
            "chunks": len(chunk_paths),
        }
    )


if __name__ == "__main__":
    main()
