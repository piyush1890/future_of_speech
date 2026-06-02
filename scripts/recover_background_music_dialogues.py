#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}
DEFAULT_PROMPT = (
    "यह हिंदी फिल्म या सीरीज का संवाद है। केवल साफ़ सुनाई देने वाले बोले गए "
    "संवाद को देवनागरी में सही शब्द और विराम चिन्ह के साथ लिखें।"
)


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def slugify(value: str) -> str:
    value = value.strip().lower() or "unknown"
    value = re.sub(r"[^a-z0-9_+.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:120] or "unknown"


def clean_token(text: str) -> str:
    return text.strip().strip("।.!?,;:'\"()[]{}-–—…").lower()


def text_tokens(text: str) -> list[str]:
    return [token for token in (clean_token(part) for part in text.split()) if token]


def repeated_ngram_fraction(tokens: list[str]) -> float:
    best = 0.0
    for n in (2, 3, 4):
        if len(tokens) < n * 2:
            continue
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        if counts:
            best = max(best, max(counts.values()) * n / max(1, len(tokens)))
    return float(best)


def max_consecutive_repeat(tokens: list[str]) -> int:
    best = 0
    current = 0
    previous = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        best = max(best, current)
    return best


def clipped01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def residual_music_metrics(path: Path, row: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    import torch

    if args.post_recovery_song_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for post-recovery acoustic gate; refusing CPU fallback.")

    wav, sr = load_wav(path)
    n_fft = args.post_recovery_song_n_fft
    hop = max(1, int(sr * args.post_recovery_song_hop_ms / 1000.0))
    device = torch.device(args.post_recovery_song_device)
    x = torch.from_numpy(wav.astype(np.float32, copy=False)).to(device)
    window = torch.hann_window(n_fft, device=device)
    with torch.inference_mode():
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        )
        power = spec.abs().square()
        freq = torch.linspace(0.0, sr / 2.0, n_fft // 2 + 1, device=device)
        band = (freq >= args.post_recovery_song_band_low_hz) & (
            freq <= args.post_recovery_song_band_high_hz
        )
        low_voice = (freq >= args.post_recovery_song_voice_low_hz) & (
            freq <= args.post_recovery_song_voice_high_hz
        )
        band_power = power[band].sum(dim=0).clamp_min(1e-9)
        low_voice_ratio = power[low_voice].sum(dim=0) / band_power
        k = min(args.post_recovery_song_tonal_top_k, int(band.sum().item()))
        tonal_ratio = torch.topk(power[band], k=k, dim=0).values.sum(dim=0) / band_power
        if args.post_recovery_song_device == "mps":
            torch.mps.synchronize()

    band_power_np = band_power.detach().cpu().numpy()
    tonal_np = tonal_ratio.detach().cpu().numpy()
    low_voice_np = low_voice_ratio.detach().cpu().numpy()
    active_floor = max(
        float(np.percentile(band_power_np, args.post_recovery_song_active_percentile)),
        1e-9,
    )
    active = band_power_np >= active_floor
    voiced = active & (
        (tonal_np >= args.post_recovery_song_tonal_voiced_threshold)
        | (low_voice_np >= args.post_recovery_song_low_voice_threshold)
    )
    max_voiced_run = longest_true_run(voiced) * hop / sr if len(voiced) else 0.0
    harmonic_ratio = float(np.mean(tonal_np)) if len(tonal_np) else 0.0

    word_durations = [
        max(0.0, float(word["end"]) - float(word["start"]))
        for word in row.get("words", [])
        if word.get("start") is not None and word.get("end") is not None
    ]
    duration = float(row.get("duration") or max(len(wav) / max(sr, 1), 0.1))
    word_count = int(row.get("word_count") or len(text_tokens(str(row.get("text") or ""))))
    speech_rate = float(word_count / max(duration, 0.1))
    max_word_duration = float(max(word_durations) if word_durations else 0.0)
    repeated_fraction = float(row.get("repeated_ngram_fraction") or 0.0)
    compression_ratio = float(row.get("compression_ratio") or 0.0)
    song_score = (
        0.30 * clipped01((max_word_duration - 0.90) / 1.40)
        + 0.20 * clipped01((1.90 - speech_rate) / 0.80)
        + 0.18 * clipped01((max_voiced_run - 1.00) / 1.80)
        + 0.12 * clipped01((repeated_fraction - 0.16) / 0.20)
        + 0.10 * clipped01((compression_ratio - 2.10) / 0.90)
        + 0.10 * clipped01((harmonic_ratio - 0.18) / 0.22)
    )
    return {
        "song_like_score": float(song_score),
        "harmonic_ratio": float(harmonic_ratio),
        "max_voiced_run_seconds": float(max_voiced_run),
        "speech_rate_wps": float(speech_rate),
        "max_word_duration": float(max_word_duration),
        "repeated_ngram_fraction": float(repeated_fraction),
        "compression_ratio": float(compression_ratio),
    }


def post_recovery_acoustic_reject_reasons(
    path: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, float] | None]:
    if not args.post_recovery_acoustic_gate:
        return [], None
    metrics = residual_music_metrics(path, row, args)
    residual_tonal_music = (
        metrics["song_like_score"] >= args.post_recovery_song_score_threshold
        and (
            metrics["harmonic_ratio"] >= args.post_recovery_min_harmonic_ratio
            or metrics["max_voiced_run_seconds"] >= args.post_recovery_min_voiced_run
        )
    )
    if residual_tonal_music:
        return ["recovered_residual_background_music"], metrics
    return [], metrics


def post_recovery_reject_reasons(
    row: dict[str, Any],
    args: argparse.Namespace,
    *,
    text_override: str | None = None,
) -> list[str]:
    if not args.post_recovery_quality_gate:
        return []
    text = str(text_override if text_override is not None else row.get("text") or "")
    tokens = text_tokens(text)
    reasons: list[str] = []
    if not tokens:
        reasons.append("recovered_empty_text")
        return reasons
    consecutive = max_consecutive_repeat(tokens)
    repeat_fraction = repeated_ngram_fraction(tokens)
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if consecutive > args.post_max_consecutive_repeat:
        reasons.append("recovered_consecutive_repetition")
    if repeat_fraction > args.post_max_repeated_ngram_fraction:
        reasons.append("recovered_repetitive_text")
    if unique_ratio < args.post_min_unique_word_ratio:
        reasons.append("recovered_low_unique_word_ratio")
    if "\ufffd" in text:
        reasons.append("recovered_bad_unicode")
    return reasons


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False), sr


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


def export_slice(
    source_wav: np.ndarray,
    sr: int,
    row: dict[str, Any],
    out_path: Path,
    *,
    pad_seconds: float,
) -> bool:
    start = row.get("start")
    end = row.get("end")
    if start is None or end is None:
        return False
    start_i = max(0, int(round((float(start) - pad_seconds) * sr)))
    end_i = min(len(source_wav), int(round((float(end) + pad_seconds) * sr)))
    if end_i <= start_i:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, source_wav[start_i:end_i], sr)
    return True


def reason_parts(row: dict[str, Any]) -> set[str]:
    return {part for part in str(row.get("reject_reason") or "").split(";") if part}


def is_recoverable_background_music(row: dict[str, Any]) -> bool:
    parts = reason_parts(row)
    return "background_music_or_noise" in parts and "song_like" not in parts


def existing_sentence_index(rows: list[dict[str, Any]]) -> int:
    best = -1
    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        match = re.fullmatch(r"sentence_(\d+)", clip_id)
        if match:
            best = max(best, int(match.group(1)))
    return best + 1


def resample_linear(wav: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr or len(wav) == 0:
        return wav.astype(np.float32, copy=False)
    new_len = max(1, int(round(len(wav) * target_sr / src_sr)))
    old_x = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, wav).astype(np.float32)


def rebuild_compact(rows: list[dict[str, Any]], out_path: Path, *, gap_seconds: float) -> None:
    parts: list[np.ndarray] = []
    target_sr: int | None = None
    for row in rows:
        audio = row.get("audio")
        if not audio:
            continue
        path = Path(str(audio))
        if not path.exists():
            continue
        wav, sr = load_wav(path)
        if target_sr is None:
            target_sr = sr
        wav = resample_linear(wav, sr, target_sr)
        if parts:
            parts.append(np.zeros(int(target_sr * gap_seconds), dtype=np.float32))
        parts.append(wav)
    if target_sr is None:
        target_sr = 16000
    compact = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    sf.write(out_path, compact, target_sr)


def recalc_filter_summary(
    sentence_dir: Path,
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    recovery_summary: dict[str, Any],
) -> None:
    summary_path = sentence_dir / "summary.json"
    summary = read_json(summary_path)
    reason_counts: dict[str, int] = {}
    for row in rejected_rows:
        parts = reason_parts(row) or {"unknown"}
        for reason in parts:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary.update(
        {
            "accepted_sentences": len(accepted_rows),
            "rejected_sentences": len(rejected_rows),
            "accepted_seconds": round(
                sum(float(row.get("duration") or 0.0) for row in accepted_rows), 3
            ),
            "rejected_seconds": round(
                sum(float(row.get("duration") or 0.0) for row in rejected_rows), 3
            ),
            "reason_counts": dict(sorted(reason_counts.items())),
            "background_music_recovery": recovery_summary,
        }
    )
    write_json(summary_path, summary)


def demucs_output_for(recovery_dir: Path, stem: str) -> Path:
    return recovery_dir / "demucs" / "htdemucs" / stem / "vocals.wav"


def roformer_output_for(recovery_dir: Path, input_path: Path, model_filename: str) -> Path:
    model_stem = Path(model_filename).stem
    return recovery_dir / "roformer" / f"{input_path.stem}_(vocals)_{model_stem}.wav"


def separate_with_demucs(
    input_paths: list[Path],
    recovery_dir: Path,
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    if not input_paths:
        return {}
    if args.demucs_device == "cpu":
        raise RuntimeError("CPU fallback is disabled; Demucs recovery must run on mps/cuda.")
    run(
        [
            args.demucs,
            "--two-stems=vocals",
            "-n",
            "htdemucs",
            "-d",
            args.demucs_device,
            "-o",
            str(recovery_dir / "demucs"),
            *[str(path) for path in input_paths],
        ],
        cwd=project_root,
    )
    return {path.stem: demucs_output_for(recovery_dir, path.stem) for path in input_paths}


def require_mlx_gpu(mx: Any) -> None:
    default_device = mx.default_device()
    if "gpu" not in str(default_device).lower():
        raise RuntimeError(
            f"CPU fallback is disabled; MLX RoFormer must run on GPU, got {default_device}."
        )


def transcribe_recovered_audio(path: Path, args: argparse.Namespace, project_root: Path) -> str | None:
    if not args.validate_recovered_asr:
        return None
    import mlx.core as mx
    import mlx_whisper

    require_mlx_gpu(mx)
    mlx_model = args.mlx_model
    if mlx_model is None:
        mlx_model = str(project_root / "models" / "mlx-whisper-large-v3-turbo")
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=mlx_model,
        language=args.language,
        task="transcribe",
        word_timestamps=True,
        verbose=False,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=args.initial_prompt,
    )
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in result.get("segments", [])
        if str(segment.get("text") or "").strip()
    ).strip()


def separate_with_mlx_roformer(
    input_paths: list[Path],
    recovery_dir: Path,
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    if not input_paths:
        return {}

    import librosa
    import mlx.core as mx
    from mlx_audio_separator import Separator

    require_mlx_gpu(mx)

    output_dir = recovery_dir / "roformer"
    model_dir = (
        Path(args.roformer_model_dir).expanduser()
        if args.roformer_model_dir
        else project_root / "models" / "mlx-audio-separator"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        model_file_dir=str(model_dir.resolve()),
        output_dir=str(output_dir),
        output_format="WAV",
        output_single_stem="vocals",
        sample_rate=args.roformer_sample_rate,
        chunk_duration=None,
        mdxc_params={
            "segment_size": args.roformer_segment_size,
            "override_model_segment_size": False,
            "batch_size": args.roformer_batch_size,
            "overlap": args.roformer_overlap,
            "pitch_shift": 0,
        },
    )
    require_mlx_gpu(mx)
    separator._skip_auto_tune = True
    separator._strict_separation_errors = True
    separator.load_model(args.roformer_model_filename)
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

    separated: dict[str, Path] = {}
    for input_path in input_paths:
        outputs = [Path(path) for path in separator.separate(str(input_path))]
        expected = roformer_output_for(recovery_dir, input_path, args.roformer_model_filename)
        if expected.exists():
            separated[input_path.stem] = expected
            continue
        vocals = next((path for path in outputs if "vocals" in path.name.lower()), None)
        if vocals is not None:
            separated[input_path.stem] = vocals
    return separated


def separate_recoverable_audio(
    input_paths: list[Path],
    recovery_dir: Path,
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Path]:
    if args.recovery_backend == "demucs":
        return separate_with_demucs(input_paths, recovery_dir, args, project_root)
    if args.recovery_backend == "mlx-roformer":
        return separate_with_mlx_roformer(input_paths, recovery_dir, args, project_root)
    raise ValueError(f"Unsupported recovery backend: {args.recovery_backend}")


def convert_vocals(
    vocals: Path,
    out_path: Path,
    *,
    ffmpeg: str,
    mono_channel: str,
    project_root: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(vocals, dtype="float32", always_2d=False)
    mono = select_mono_channel(audio, mono_channel)
    temp_mono = out_path.with_name(f"{out_path.stem}.mono_tmp.wav")
    sf.write(temp_mono, mono, sr)
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(temp_mono),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            "highpass=f=100,lowpass=f=7600,equalizer=f=180:t=q:w=1.0:g=-2",
            str(out_path),
        ],
        cwd=project_root,
    )
    try:
        temp_mono.unlink()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover dialogue rejected only for background music by running a second vocal separation pass."
    )
    parser.add_argument("sentence_dir", type=Path)
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--language", default="hi")
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--initial-prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--validate-recovered-asr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-transcribe recovered audio on MLX GPU and reject clips whose cleaned audio still looks repetitive/corrupt.",
    )
    parser.add_argument(
        "--recovery-backend",
        choices=["mlx-roformer", "demucs"],
        default="mlx-roformer",
        help="Separator for music-under-dialogue recovery. mlx-roformer uses a trained UVR/MelBand RoFormer model on Apple Silicon GPU.",
    )
    parser.add_argument(
        "--roformer-model-filename",
        default="vocals_mel_band_roformer.ckpt",
        help="MLX audio-separator model filename to download/use for vocal isolation.",
    )
    parser.add_argument("--roformer-model-dir", type=Path, default=None)
    parser.add_argument("--roformer-sample-rate", type=int, default=44100)
    parser.add_argument("--roformer-segment-size", type=int, default=256)
    parser.add_argument("--roformer-overlap", type=int, default=8)
    parser.add_argument("--roformer-batch-size", type=int, default=1)
    parser.add_argument("--demucs", default="demucs")
    parser.add_argument("--demucs-device", default="mps")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--mono-channel",
        choices=["loudest", "left", "right", "quietest", "average"],
        default="left",
        help="How to collapse recovered stereo stems to mono. RoFormer can emit a time-wrapped right channel on mono dialogue, so left is the default.",
    )
    parser.add_argument("--pad-seconds", type=float, default=0.04)
    parser.add_argument("--compact-gap-seconds", type=float, default=0.18)
    parser.add_argument(
        "--post-recovery-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After music separation, keep only recovered clips whose existing transcript does not look repetitive/corrupt.",
    )
    parser.add_argument(
        "--post-recovery-acoustic-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject recovered clips that still contain tonal background music after separation.",
    )
    parser.add_argument(
        "--post-recovery-song-device",
        choices=["mps"],
        default="mps",
        help="Device for the post-recovery acoustic gate. CPU fallback is intentionally disabled.",
    )
    parser.add_argument("--post-recovery-song-score-threshold", type=float, default=0.50)
    parser.add_argument("--post-recovery-min-harmonic-ratio", type=float, default=0.58)
    parser.add_argument("--post-recovery-min-voiced-run", type=float, default=2.50)
    parser.add_argument("--post-recovery-song-n-fft", type=int, default=1024)
    parser.add_argument("--post-recovery-song-hop-ms", type=float, default=10.0)
    parser.add_argument("--post-recovery-song-band-low-hz", type=float, default=80.0)
    parser.add_argument("--post-recovery-song-band-high-hz", type=float, default=4000.0)
    parser.add_argument("--post-recovery-song-voice-low-hz", type=float, default=80.0)
    parser.add_argument("--post-recovery-song-voice-high-hz", type=float, default=500.0)
    parser.add_argument("--post-recovery-song-tonal-top-k", type=int, default=5)
    parser.add_argument("--post-recovery-song-active-percentile", type=float, default=30.0)
    parser.add_argument("--post-recovery-song-tonal-voiced-threshold", type=float, default=0.12)
    parser.add_argument("--post-recovery-song-low-voice-threshold", type=float, default=0.08)
    parser.add_argument("--post-max-consecutive-repeat", type=int, default=2)
    parser.add_argument("--post-max-repeated-ngram-fraction", type=float, default=0.20)
    parser.add_argument("--post-min-unique-word-ratio", type=float, default=0.55)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sentence_dir = args.sentence_dir.resolve()
    source_wav_path = args.source_wav.resolve()
    project_root = args.project_root.resolve()
    recovery_dir = sentence_dir / "background_music_recovery"
    summary_path = recovery_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(encoding="utf-8"))
        return

    accepted_path = sentence_dir / "accepted_sentences.jsonl"
    rejected_path = sentence_dir / "rejected_sentences.jsonl"
    accepted_rows = read_jsonl(accepted_path)
    rejected_rows = read_jsonl(rejected_path)
    recoverable_rows = [
        row for row in rejected_rows if is_recoverable_background_music(row)
    ]
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if not recoverable_rows:
        summary = {
            "recoverable_background_music_clips": 0,
            "recovered_clips": 0,
            "failed_clips": 0,
            "recovered_seconds": 0.0,
            "recovery_dir": str(recovery_dir),
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not source_wav_path.exists():
        raise FileNotFoundError(f"source wav does not exist: {source_wav_path}")

    source_wav, sr = load_wav(source_wav_path)
    original_dir = recovery_dir / "original"
    cleaned_dir = recovery_dir / "cleaned"
    clips_dir = sentence_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    input_paths: list[Path] = []
    row_by_stem: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(recoverable_rows):
        old_id = slugify(str(row.get("clip_id") or f"background_music_{idx:06d}"))
        stem = f"{idx:06d}_{old_id}"
        out_path = original_dir / f"{stem}.wav"
        if export_slice(source_wav, sr, row, out_path, pad_seconds=args.pad_seconds):
            input_paths.append(out_path)
            row_by_stem[stem] = row

    separated_paths = separate_recoverable_audio(
        input_paths,
        recovery_dir,
        args,
        project_root,
    )

    next_idx = existing_sentence_index(accepted_rows)
    recovered_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    recovered_old_ids: set[str] = set()
    for input_path in input_paths:
        stem = input_path.stem
        row = row_by_stem[stem]
        vocals = separated_paths.get(stem)
        if vocals is None:
            if args.recovery_backend == "demucs":
                vocals = demucs_output_for(recovery_dir, stem)
            else:
                vocals = roformer_output_for(
                    recovery_dir, input_path, args.roformer_model_filename
                )
        if not vocals.exists():
            failed = dict(row)
            failed["background_music_recovery_error"] = f"missing_{args.recovery_backend}_vocals"
            failed_rows.append(failed)
            continue

        new_clip_id = f"sentence_{next_idx:06d}"
        next_idx += 1
        cleaned_review = cleaned_dir / f"{stem}_dialogue.wav"
        clip_path = clips_dir / f"{new_clip_id}.wav"
        convert_vocals(
            vocals,
            cleaned_review,
            ffmpeg=args.ffmpeg,
            mono_channel=args.mono_channel,
            project_root=project_root,
        )
        recovered_asr_text = transcribe_recovered_audio(cleaned_review, args, project_root)
        recovered_reject_reasons = post_recovery_reject_reasons(
            row,
            args,
            text_override=recovered_asr_text,
        )
        acoustic_reject_reasons, acoustic_metrics = post_recovery_acoustic_reject_reasons(
            cleaned_review,
            row,
            args,
        )
        recovered_reject_reasons.extend(acoustic_reject_reasons)
        if recovered_reject_reasons:
            failed = dict(row)
            failed["background_music_recovery_error"] = ";".join(recovered_reject_reasons)
            failed["background_music_original_audio"] = str(input_path)
            failed["background_music_cleaned_audio"] = str(cleaned_review)
            failed["background_music_recovery_method"] = args.recovery_backend
            failed["background_music_mono_channel"] = args.mono_channel
            if recovered_asr_text is not None:
                failed["background_music_recovered_asr_text"] = recovered_asr_text
            if acoustic_metrics is not None:
                failed["background_music_recovered_acoustic_metrics"] = {
                    key: round(value, 4) for key, value in acoustic_metrics.items()
                }
            failed_rows.append(failed)
            continue
        shutil.copy2(cleaned_review, clip_path)

        recovered = dict(row)
        recovered["source_clip_id"] = row.get("clip_id")
        recovered["clip_id"] = new_clip_id
        recovered["audio"] = str(clip_path)
        recovered["accepted"] = True
        recovered["reject_reason"] = None
        recovered["background_music_recovered"] = True
        recovered["background_music_original_audio"] = str(input_path)
        recovered["background_music_cleaned_audio"] = str(cleaned_review)
        recovered["background_music_recovery_method"] = args.recovery_backend
        recovered["background_music_mono_channel"] = args.mono_channel
        if recovered_asr_text is not None:
            recovered["background_music_recovered_asr_text"] = recovered_asr_text
        if acoustic_metrics is not None:
            recovered["background_music_recovered_acoustic_metrics"] = {
                key: round(value, 4) for key, value in acoustic_metrics.items()
            }
        if args.recovery_backend == "mlx-roformer":
            recovered["background_music_recovery_model"] = args.roformer_model_filename
        recovered_rows.append(recovered)
        recovered_old_ids.add(str(row.get("clip_id") or ""))

    if recovered_rows:
        backup_path = sentence_dir / "rejected_sentences_before_background_music_recovery.jsonl"
        if not backup_path.exists():
            shutil.copy2(rejected_path, backup_path)
        accepted_rows = [*accepted_rows, *recovered_rows]
        rejected_rows = [
            row
            for row in rejected_rows
            if str(row.get("clip_id") or "") not in recovered_old_ids
        ]
        write_jsonl(accepted_path, accepted_rows)
        write_jsonl(rejected_path, rejected_rows)
        rebuild_compact(
            accepted_rows,
            sentence_dir / "accepted_sentences_compact.wav",
            gap_seconds=args.compact_gap_seconds,
        )

    write_jsonl(recovery_dir / "recovered_sentences.jsonl", recovered_rows)
    write_jsonl(recovery_dir / "failed_sentences.jsonl", failed_rows)
    summary = {
        "recoverable_background_music_clips": len(recoverable_rows),
        "exported_for_recovery": len(input_paths),
        "recovered_clips": len(recovered_rows),
        "failed_clips": len(failed_rows),
        "post_recovery_quality_gate": bool(args.post_recovery_quality_gate),
        "recovered_seconds": round(
            sum(float(row.get("duration") or 0.0) for row in recovered_rows), 3
        ),
        "recovery_backend": args.recovery_backend,
        "recovery_model": (
            args.roformer_model_filename
            if args.recovery_backend == "mlx-roformer"
            else "htdemucs"
        ),
        "mono_channel": args.mono_channel,
        "validate_recovered_asr": bool(args.validate_recovered_asr),
        "recovery_dir": str(recovery_dir),
        "original_dir": str(original_dir),
        "cleaned_dir": str(cleaned_dir),
        "recovered_manifest": str(recovery_dir / "recovered_sentences.jsonl"),
    }
    write_json(summary_path, summary)
    recalc_filter_summary(sentence_dir, accepted_rows, rejected_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
