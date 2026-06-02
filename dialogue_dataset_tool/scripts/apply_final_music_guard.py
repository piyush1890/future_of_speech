#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized.*",
    category=UserWarning,
)

from clean_accepted_dialogues_roformer import transcribe_audio, validate_cleaned_clip
from recover_background_music_dialogues import (
    DEFAULT_PROMPT,
    convert_vocals,
    read_jsonl,
    residual_music_metrics,
    separate_with_mlx_roformer,
    write_json,
    write_jsonl,
)


class SpeechMusicClassifier:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.classifier_device != "mps":
            raise RuntimeError("CPU fallback is disabled; final music classifier must run on mps.")

        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required for final music classifier; refusing CPU fallback.")

        model_ref = str(args.classifier_model_dir or args.classifier_model)
        self.torch = torch
        self.device = torch.device("mps")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_ref)
        self.model = AutoModelForAudioClassification.from_pretrained(model_ref).to(self.device)
        self.model.eval()
        self.labels = {int(key): value for key, value in self.model.config.id2label.items()}
        self.sample_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000) or 16000)
        self.window_samples = int(args.classifier_window_seconds * self.sample_rate)
        self.hop_samples = int(args.classifier_hop_seconds * self.sample_rate)

    def load_audio(self, path: Path) -> np.ndarray:
        import librosa

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        return wav.astype(np.float32, copy=False)

    def iter_windows(self, wav: np.ndarray) -> list[np.ndarray]:
        if len(wav) <= self.window_samples:
            return [wav]
        windows: list[np.ndarray] = []
        stop = max(1, len(wav) - self.window_samples + 1)
        for start in range(0, stop, self.hop_samples):
            windows.append(wav[start : start + self.window_samples])
        if (len(wav) - self.window_samples) % self.hop_samples:
            windows.append(wav[-self.window_samples :])
        return windows

    def classify(self, path: Path) -> dict[str, Any]:
        wav = self.load_audio(path)
        best_scores: dict[str, float] = {}
        best_music_score = -1.0
        best_top: list[dict[str, Any]] = []
        for window in self.iter_windows(wav):
            inputs = self.feature_extractor(
                window,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits[0]
                probs = self.torch.softmax(logits, dim=-1).detach().cpu().numpy()
            scores = {self.labels[idx].lower(): float(probs[idx]) for idx in range(len(probs))}
            music_score = max(
                scores.get("music", 0.0),
                scores.get("speech_music", 0.0),
                scores.get("speech music", 0.0),
            )
            if music_score > best_music_score:
                best_music_score = music_score
                best_scores = scores
                top_indices = np.argsort(probs)[-5:][::-1]
                best_top = [
                    {
                        "label": self.labels[int(idx)],
                        "score": round(float(probs[int(idx)]), 4),
                    }
                    for idx in top_indices
                ]
        return {
            "classifier_music_prob": round(float(best_scores.get("music", 0.0)), 4),
            "classifier_speech_music_prob": round(
                float(max(best_scores.get("speech_music", 0.0), best_scores.get("speech music", 0.0))),
                4,
            ),
            "classifier_speech_prob": round(float(best_scores.get("speech", 0.0)), 4),
            "classifier_top": best_top,
        }


def resolve_path(value: Any, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def copy_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_prompt(row: dict[str, Any], prompts_dir: Path, cache: dict[Path, Path]) -> str | None:
    prompt_value = row.get("voice_prompt")
    if not prompt_value:
        return None
    prompt = Path(str(prompt_value)).expanduser()
    if not prompt.exists():
        return str(prompt_value)
    prompt = prompt.resolve()
    if prompt not in cache:
        target = prompts_dir / prompt.name
        if target.exists() and target.resolve() != prompt:
            target = prompts_dir / f"{prompt.stem}_{len(cache):06d}{prompt.suffix}"
        copy_audio(prompt, target)
        cache[prompt] = target
    return str(cache[prompt])


def score_clip(
    path: Path,
    row: dict[str, Any],
    args: argparse.Namespace,
    classifier: SpeechMusicClassifier | None,
) -> dict[str, Any]:
    metrics = residual_music_metrics(path, row, args)
    output: dict[str, Any] = {key: round(float(value), 4) for key, value in metrics.items()}
    if classifier is not None:
        output.update(classifier.classify(path))
    return output


def classifier_suspect(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(metrics.get("classifier_speech_music_prob") or 0.0) >= args.classifier_speech_music_suspect_threshold
        or float(metrics.get("classifier_music_prob") or 0.0) >= args.classifier_music_suspect_threshold
    )


def classifier_high_confidence(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(metrics.get("classifier_speech_music_prob") or 0.0) >= args.classifier_speech_music_high_threshold
        or float(metrics.get("classifier_music_prob") or 0.0) >= args.classifier_music_high_threshold
    )


def classifier_supported_short_acoustic_high_confidence(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(metrics.get("classifier_speech_music_prob") or 0.0)
        >= args.classifier_speech_music_short_acoustic_threshold
        and metrics["song_like_score"] >= args.short_acoustic_score_threshold
        and metrics["harmonic_ratio"] >= args.short_acoustic_min_harmonic_ratio
    )


def acoustic_raw_suspect(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        metrics["song_like_score"] >= args.raw_suspect_score_threshold
        and (
            metrics["harmonic_ratio"] >= args.raw_suspect_min_harmonic_ratio
            or metrics["max_voiced_run_seconds"] >= args.raw_suspect_min_voiced_run
        )
    )


def is_raw_suspect(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return acoustic_raw_suspect(metrics, args) or classifier_suspect(metrics, args)


def acoustic_raw_high_confidence(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        metrics["song_like_score"] >= args.raw_high_score_threshold
        and (
            metrics["harmonic_ratio"] >= args.raw_high_min_harmonic_ratio
            or metrics["max_voiced_run_seconds"] >= args.raw_high_min_voiced_run
        )
    )


def is_raw_high_confidence(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        acoustic_raw_high_confidence(metrics, args)
        or classifier_high_confidence(metrics, args)
        or classifier_supported_short_acoustic_high_confidence(metrics, args)
    )


def acoustic_cleaned_high_confidence(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        metrics["song_like_score"] >= args.cleaned_reject_score_threshold
        and (
            metrics["harmonic_ratio"] >= args.cleaned_reject_min_harmonic_ratio
            or metrics["max_voiced_run_seconds"] >= args.cleaned_reject_min_voiced_run
        )
    )


def is_cleaned_high_confidence(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        acoustic_cleaned_high_confidence(metrics, args)
        or classifier_high_confidence(metrics, args)
        or classifier_supported_short_acoustic_high_confidence(metrics, args)
    )


def keep_row(
    row: dict[str, Any],
    *,
    audio_path: Path,
    prompts_dir: Path,
    prompt_cache: dict[Path, Path],
    status: str,
    raw_metrics: dict[str, Any],
    cleaned_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kept = dict(row)
    kept["audio"] = str(audio_path)
    prompt = copy_prompt(kept, prompts_dir, prompt_cache)
    if prompt is not None:
        kept["voice_prompt"] = prompt
    kept["music_guard_status"] = status
    kept["music_guard_raw_metrics"] = raw_metrics
    if cleaned_metrics is not None:
        kept["music_guard_cleaned_metrics"] = cleaned_metrics
    kept["accepted"] = True
    kept["train_allowed"] = True
    return kept


def copy_review_pair(
    original: Path,
    cleaned: Path | None,
    review_dir: Path,
    clip_id: str,
) -> tuple[Path, Path | None]:
    original_out = review_dir / "original" / f"{clip_id}_original.wav"
    copy_audio(original, original_out)
    cleaned_out = None
    if cleaned is not None and cleaned.exists():
        cleaned_out = review_dir / "cleaned" / f"{clip_id}_cleaned.wav"
        copy_audio(cleaned, cleaned_out)
    return original_out, cleaned_out


def manifest_row(
    row: dict[str, Any],
    *,
    status: str,
    raw_metrics: dict[str, Any],
    cleaned_metrics: dict[str, Any] | None = None,
    reject_reason: str | None = None,
    original_review: Path | None = None,
    cleaned_review: Path | None = None,
    validation_metrics: dict[str, Any] | None = None,
    validation_reasons: list[str] | None = None,
) -> dict[str, Any]:
    item = {
        "clip_id": row.get("clip_id"),
        "text": row.get("text"),
        "duration": row.get("duration"),
        "status": status,
        "reject_reason": reject_reason,
        "raw_audio": row.get("audio"),
        "raw_metrics": raw_metrics,
    }
    if cleaned_metrics is not None:
        item["cleaned_metrics"] = cleaned_metrics
    if original_review is not None:
        item["original_review_audio"] = str(original_review)
    if cleaned_review is not None:
        item["cleaned_review_audio"] = str(cleaned_review)
    if validation_metrics is not None:
        item["validation_metrics"] = validation_metrics
    if validation_reasons:
        item["validation_reasons"] = validation_reasons
    return item


def apply_music_guard(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    prompt_dir = args.prompt_dir.resolve()
    manifest_path = prompt_dir / "train_with_prompts.jsonl"
    rows = read_jsonl(manifest_path)
    if not rows:
        raise FileNotFoundError(f"missing or empty train manifest: {manifest_path}")

    out_dir = args.out_dir.resolve() if args.out_dir else prompt_dir.parent / args.out_dir_name
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = out_dir / "clips"
    prompts_dir = out_dir / "prompts"
    review_dir = out_dir / "music_guard_review"
    rejected_review_dir = review_dir / "rejected_music_high_confidence"
    uncertain_status = "uncertain_music_rejected" if args.exclude_uncertain else "uncertain_music_kept"
    uncertain_review_dir = review_dir / uncertain_status
    repaired_review_dir = review_dir / "accepted_music_removed"
    for directory in (clips_dir, prompts_dir, rejected_review_dir, uncertain_review_dir, repaired_review_dir):
        directory.mkdir(parents=True, exist_ok=True)

    classifier = None if args.classifier_backend == "none" else SpeechMusicClassifier(args)

    raw_metrics_by_clip: dict[str, dict[str, Any]] = {}
    suspicious: list[tuple[dict[str, Any], Path]] = []
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    uncertain_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prompt_cache: dict[Path, Path] = {}

    for row in rows:
        clip_id = str(row.get("clip_id") or f"clip_{len(raw_metrics_by_clip):06d}")
        audio = resolve_path(row.get("audio"), prompt_dir)
        if audio is None or not audio.exists():
            audit_rows.append(
                manifest_row(
                    row,
                    status="missing_audio",
                    raw_metrics={},
                    reject_reason="music_guard_missing_audio",
                )
            )
            continue
        metrics = score_clip(audio, row, args, classifier)
        raw_metrics_by_clip[clip_id] = metrics
        if is_raw_suspect(metrics, args):
            suspicious.append((row, audio))
        else:
            out_audio = clips_dir / f"{clip_id}.wav"
            copy_audio(audio, out_audio)
            kept = keep_row(
                row,
                audio_path=out_audio,
                prompts_dir=prompts_dir,
                prompt_cache=prompt_cache,
                status="accepted_no_music",
                raw_metrics=metrics,
            )
            kept_rows.append(kept)
            audit_rows.append(
                manifest_row(row, status="accepted_no_music", raw_metrics=metrics)
            )

    separated = separate_with_mlx_roformer([audio for _, audio in suspicious], out_dir / "roformer_cleanup", args, project_root)

    for row, audio in suspicious:
        clip_id = str(row.get("clip_id") or audio.stem)
        raw_metrics = raw_metrics_by_clip.get(clip_id) or score_clip(audio, row, args, classifier)
        vocals = separated.get(audio.stem)
        cleaned = None
        cleaned_metrics = None
        validation_reasons: list[str] = []
        validation_metrics: dict[str, Any] = {}

        if vocals is not None and vocals.exists():
            cleaned = out_dir / "roformer_cleanup" / "cleaned" / f"{clip_id}_music_guard.wav"
            convert_vocals(
                vocals,
                cleaned,
                ffmpeg=args.ffmpeg,
                mono_channel=args.mono_channel,
                project_root=project_root,
            )
            cleaned_metrics = score_clip(cleaned, row, args, classifier)
            cleaned_text = transcribe_audio(cleaned, args, project_root) if args.validate_asr else None
            validation_reasons, validation_metrics = validate_cleaned_clip(
                row,
                audio,
                cleaned,
                cleaned_text,
                args,
            )

        raw_high = is_raw_high_confidence(raw_metrics, args)
        cleaned_high = cleaned_metrics is not None and is_cleaned_high_confidence(cleaned_metrics, args)
        if raw_high and cleaned_high:
            original_review, cleaned_review = copy_review_pair(audio, cleaned, rejected_review_dir, clip_id)
            rejected = dict(row)
            rejected["accepted"] = False
            rejected["train_allowed"] = False
            rejected["reject_reason"] = "final_music_guard_high_confidence"
            rejected["music_guard_status"] = "rejected_music_high_confidence"
            rejected["music_guard_original_audio"] = str(original_review)
            if cleaned_review is not None:
                rejected["music_guard_cleaned_audio"] = str(cleaned_review)
            rejected["music_guard_raw_metrics"] = raw_metrics
            rejected["music_guard_cleaned_metrics"] = cleaned_metrics
            rejected["music_guard_validation_metrics"] = validation_metrics
            rejected_rows.append(rejected)
            audit_rows.append(
                manifest_row(
                    row,
                    status="rejected_music_high_confidence",
                    raw_metrics=raw_metrics,
                    cleaned_metrics=cleaned_metrics,
                    reject_reason="final_music_guard_high_confidence",
                    original_review=original_review,
                    cleaned_review=cleaned_review,
                    validation_metrics=validation_metrics,
                    validation_reasons=validation_reasons,
                )
            )
            continue

        if cleaned is not None and cleaned_metrics is not None and not validation_reasons:
            original_review, cleaned_review = copy_review_pair(audio, cleaned, repaired_review_dir, clip_id)
            out_audio = clips_dir / f"{clip_id}.wav"
            copy_audio(cleaned, out_audio)
            kept = keep_row(
                row,
                audio_path=out_audio,
                prompts_dir=prompts_dir,
                prompt_cache=prompt_cache,
                status="accepted_music_removed",
                raw_metrics=raw_metrics,
                cleaned_metrics=cleaned_metrics,
            )
            kept["music_guard_original_audio"] = str(original_review)
            if cleaned_review is not None:
                kept["music_guard_cleaned_audio"] = str(cleaned_review)
            kept["music_guard_validation_metrics"] = validation_metrics
            kept_rows.append(kept)
            audit_rows.append(
                manifest_row(
                    row,
                    status="accepted_music_removed",
                    raw_metrics=raw_metrics,
                    cleaned_metrics=cleaned_metrics,
                    original_review=original_review,
                    cleaned_review=cleaned_review,
                    validation_metrics=validation_metrics,
                )
            )
            continue

        original_review, cleaned_review = copy_review_pair(audio, cleaned, uncertain_review_dir, clip_id)
        uncertain = dict(row)
        uncertain["music_guard_status"] = uncertain_status
        uncertain["music_guard_original_audio"] = str(original_review)
        if cleaned_review is not None:
            uncertain["music_guard_cleaned_audio"] = str(cleaned_review)
        uncertain["music_guard_raw_metrics"] = raw_metrics
        if cleaned_metrics is not None:
            uncertain["music_guard_cleaned_metrics"] = cleaned_metrics
        uncertain["music_guard_validation_metrics"] = validation_metrics
        uncertain["music_guard_validation_reasons"] = validation_reasons
        if args.exclude_uncertain:
            uncertain["accepted"] = False
            uncertain["train_allowed"] = False
            uncertain["reject_reason"] = "final_music_guard_uncertain"
        uncertain_rows.append(uncertain)
        audit_rows.append(
            manifest_row(
                row,
                status=uncertain_status,
                raw_metrics=raw_metrics,
                cleaned_metrics=cleaned_metrics,
                reject_reason=";".join(validation_reasons) if validation_reasons else "music_guard_uncertain",
                original_review=original_review,
                cleaned_review=cleaned_review,
                validation_metrics=validation_metrics,
                validation_reasons=validation_reasons,
            )
        )
        if not args.exclude_uncertain:
            out_audio = clips_dir / f"{clip_id}.wav"
            copy_audio(audio, out_audio)
            kept = keep_row(
                row,
                audio_path=out_audio,
                prompts_dir=prompts_dir,
                prompt_cache=prompt_cache,
                status="uncertain_music_kept",
                raw_metrics=raw_metrics,
                cleaned_metrics=cleaned_metrics,
            )
            kept_rows.append(kept)

    write_jsonl(out_dir / "train_with_prompts.jsonl", kept_rows)
    write_jsonl(out_dir / "paired_with_prompts.jsonl", kept_rows)
    write_jsonl(out_dir / "voice_prompt_pairs.jsonl", kept_rows)
    write_jsonl(out_dir / "music_guard_audit.jsonl", audit_rows)
    write_jsonl(out_dir / "rejected_music_high_confidence.jsonl", rejected_rows)
    write_jsonl(out_dir / "uncertain_music.jsonl", uncertain_rows)

    input_seconds = sum(float(row.get("duration") or 0.0) for row in rows)
    kept_seconds = sum(float(row.get("duration") or 0.0) for row in kept_rows)
    rejected_seconds = sum(float(row.get("duration") or 0.0) for row in rejected_rows)
    uncertain_seconds = sum(float(row.get("duration") or 0.0) for row in uncertain_rows)
    status_counts: dict[str, int] = {}
    for row in audit_rows:
        status = str(row.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "source_prompt_dir": str(prompt_dir),
        "input_clips": len(rows),
        "input_seconds": round(input_seconds, 2),
        "raw_suspicious_clips": len(suspicious),
        "raw_suspicious_seconds": round(
            sum(float(row.get("duration") or 0.0) for row, _ in suspicious), 2
        ),
        "rejected_music_high_confidence_clips": len(rejected_rows),
        "rejected_music_high_confidence_seconds": round(rejected_seconds, 2),
        "uncertain_music_clips": len(uncertain_rows),
        "uncertain_music_seconds": round(uncertain_seconds, 2),
        "exclude_uncertain": bool(args.exclude_uncertain),
        "paired_clips": len(kept_rows),
        "paired_seconds": round(kept_seconds, 2),
        "train_clips": len(kept_rows),
        "train_seconds": round(kept_seconds, 2),
        "status_counts": dict(sorted(status_counts.items())),
        "settings": {
            "raw_suspect_score_threshold": args.raw_suspect_score_threshold,
            "raw_suspect_min_harmonic_ratio": args.raw_suspect_min_harmonic_ratio,
            "raw_suspect_min_voiced_run": args.raw_suspect_min_voiced_run,
            "raw_high_score_threshold": args.raw_high_score_threshold,
            "raw_high_min_harmonic_ratio": args.raw_high_min_harmonic_ratio,
            "raw_high_min_voiced_run": args.raw_high_min_voiced_run,
            "cleaned_reject_score_threshold": args.cleaned_reject_score_threshold,
            "cleaned_reject_min_harmonic_ratio": args.cleaned_reject_min_harmonic_ratio,
            "cleaned_reject_min_voiced_run": args.cleaned_reject_min_voiced_run,
            "classifier_backend": args.classifier_backend,
            "classifier_model": args.classifier_model,
            "classifier_speech_music_suspect_threshold": args.classifier_speech_music_suspect_threshold,
            "classifier_music_suspect_threshold": args.classifier_music_suspect_threshold,
            "classifier_speech_music_high_threshold": args.classifier_speech_music_high_threshold,
            "classifier_music_high_threshold": args.classifier_music_high_threshold,
            "classifier_speech_music_short_acoustic_threshold": args.classifier_speech_music_short_acoustic_threshold,
            "short_acoustic_score_threshold": args.short_acoustic_score_threshold,
            "short_acoustic_min_harmonic_ratio": args.short_acoustic_min_harmonic_ratio,
        },
        "outputs": {
            "train_with_prompts": str(out_dir / "train_with_prompts.jsonl"),
            "paired_with_prompts": str(out_dir / "paired_with_prompts.jsonl"),
            "voice_prompt_pairs": str(out_dir / "voice_prompt_pairs.jsonl"),
            "clips": str(clips_dir),
            "prompts": str(prompts_dir),
            "audit_manifest": str(out_dir / "music_guard_audit.jsonl"),
            "rejected_manifest": str(out_dir / "rejected_music_high_confidence.jsonl"),
            "rejected_music_high_confidence": str(rejected_review_dir),
            "uncertain_manifest": str(out_dir / "uncertain_music.jsonl"),
            "uncertain_music": str(uncertain_review_dir),
            "accepted_music_removed": str(repaired_review_dir),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a high-precision final music guard to paired training clips.")
    parser.add_argument("prompt_dir", type=Path, help="Final prompt directory with train_with_prompts.jsonl.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out-dir-name", default="strict_voice_prompts_t034_p034_training_guarded_global_join_music_guard")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--initial-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--validate-asr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-uncertain", action="store_true")
    parser.add_argument("--roformer-model-filename", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--roformer-model-dir", type=Path, default=None)
    parser.add_argument("--roformer-sample-rate", type=int, default=44100)
    parser.add_argument("--roformer-segment-size", type=int, default=256)
    parser.add_argument("--roformer-overlap", type=int, default=8)
    parser.add_argument("--roformer-batch-size", type=int, default=1)
    parser.add_argument("--mono-channel", choices=["loudest", "left", "right", "quietest", "average"], default="left")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--classifier-backend",
        choices=["wav2vec2", "none"],
        default="wav2vec2",
        help="Optional trained speech/music classifier. wav2vec2 uses labels: music, speech, speech_music.",
    )
    parser.add_argument(
        "--classifier-model",
        default="FerhatDk/wav2vec2-base_music_speech_both_classification",
    )
    parser.add_argument("--classifier-model-dir", type=Path, default=None)
    parser.add_argument("--classifier-device", choices=["mps"], default="mps")
    parser.add_argument("--classifier-window-seconds", type=float, default=10.0)
    parser.add_argument("--classifier-hop-seconds", type=float, default=5.0)
    parser.add_argument("--classifier-speech-music-suspect-threshold", type=float, default=0.65)
    parser.add_argument("--classifier-music-suspect-threshold", type=float, default=0.85)
    parser.add_argument("--classifier-speech-music-high-threshold", type=float, default=0.9925)
    parser.add_argument("--classifier-music-high-threshold", type=float, default=0.95)
    parser.add_argument("--classifier-speech-music-short-acoustic-threshold", type=float, default=0.992)
    parser.add_argument("--short-acoustic-score-threshold", type=float, default=0.14)
    parser.add_argument("--short-acoustic-min-harmonic-ratio", type=float, default=0.47)
    parser.add_argument("--post-recovery-song-device", choices=["mps"], default="mps")
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
    parser.add_argument("--raw-suspect-score-threshold", type=float, default=0.46)
    parser.add_argument("--raw-suspect-min-harmonic-ratio", type=float, default=0.38)
    parser.add_argument("--raw-suspect-min-voiced-run", type=float, default=2.2)
    parser.add_argument("--raw-high-score-threshold", type=float, default=0.62)
    parser.add_argument("--raw-high-min-harmonic-ratio", type=float, default=0.48)
    parser.add_argument("--raw-high-min-voiced-run", type=float, default=3.2)
    parser.add_argument("--cleaned-reject-score-threshold", type=float, default=0.50)
    parser.add_argument("--cleaned-reject-min-harmonic-ratio", type=float, default=0.58)
    parser.add_argument("--cleaned-reject-min-voiced-run", type=float, default=2.5)
    parser.add_argument("--min-duration-ratio", type=float, default=0.80)
    parser.add_argument("--min-token-f1", type=float, default=0.45)
    parser.add_argument("--min-char-similarity", type=float, default=0.72)
    parser.add_argument("--min-tokens-for-f1", type=int, default=4)
    parser.add_argument("--min-tokens-for-length-check", type=int, default=5)
    parser.add_argument("--min-cleaned-tokens", type=int, default=2)
    parser.add_argument("--min-token-count-ratio", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mlx_model is None:
        args.mlx_model = str(args.project_root.resolve() / "models" / "mlx-whisper-large-v3-turbo")
    summary = apply_music_guard(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
