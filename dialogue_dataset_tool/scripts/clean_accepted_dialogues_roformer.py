#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from recover_background_music_dialogues import (
    DEFAULT_PROMPT,
    convert_vocals,
    read_json,
    read_jsonl,
    rebuild_compact,
    reason_parts,
    require_mlx_gpu,
    separate_with_mlx_roformer,
    text_tokens,
    write_json,
    write_jsonl,
)


def token_f1(reference: list[str], candidate: list[str]) -> float:
    if not reference or not candidate:
        return 0.0
    ref_counts = Counter(reference)
    cand_counts = Counter(candidate)
    common = sum((ref_counts & cand_counts).values())
    precision = common / max(1, len(candidate))
    recall = common / max(1, len(reference))
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def char_similarity(reference: list[str], candidate: list[str]) -> float:
    return float(difflib.SequenceMatcher(None, "".join(reference), "".join(candidate)).ratio())


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


def repeated_ngram_fraction(tokens: list[str]) -> float:
    best = 0.0
    for n in (2, 3, 4):
        if len(tokens) < n * 2:
            continue
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            counts[gram] = counts.get(gram, 0) + 1
        if counts:
            best = max(best, max(counts.values()) * n / max(1, len(tokens)))
    return float(best)


def audio_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames / max(1, info.samplerate))


def transcribe_audio(path: Path, args: argparse.Namespace, project_root: Path) -> str | None:
    if not args.validate_asr:
        return None
    import mlx.core as mx
    import mlx_whisper

    require_mlx_gpu(mx)
    mlx_model = args.mlx_model or str(project_root / "models" / "mlx-whisper-large-v3-turbo")
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=mlx_model,
        language=args.language,
        task="transcribe",
        word_timestamps=False,
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


def validate_cleaned_clip(
    original_row: dict[str, Any],
    original_path: Path,
    cleaned_path: Path,
    cleaned_text: str | None,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    reasons: list[str] = []

    original_duration = audio_duration(original_path)
    cleaned_duration = audio_duration(cleaned_path)
    metrics["original_duration_seconds"] = round(original_duration, 3)
    metrics["cleaned_duration_seconds"] = round(cleaned_duration, 3)
    duration_ratio = cleaned_duration / max(original_duration, 0.001)
    metrics["duration_ratio"] = round(duration_ratio, 4)
    if duration_ratio < args.min_duration_ratio:
        reasons.append("cleanup_audio_too_short")

    if cleaned_text is None:
        return reasons, metrics

    reference_tokens = text_tokens(str(original_row.get("text") or ""))
    cleaned_tokens = text_tokens(cleaned_text)
    metrics["cleaned_asr_text"] = cleaned_text
    metrics["reference_token_count"] = len(reference_tokens)
    metrics["cleaned_token_count"] = len(cleaned_tokens)
    metrics["token_f1"] = round(token_f1(reference_tokens, cleaned_tokens), 4)
    metrics["char_similarity"] = round(char_similarity(reference_tokens, cleaned_tokens), 4)
    # Repetition is useful to inspect because separation bugs can duplicate
    # phrases, but natural Hindi dialogue often repeats words intentionally.
    # Do not reject on this metric alone.
    metrics["max_consecutive_repeat"] = max_consecutive_repeat(cleaned_tokens)
    metrics["repeated_ngram_fraction"] = round(repeated_ngram_fraction(cleaned_tokens), 4)

    if not cleaned_tokens:
        reasons.append("cleanup_empty_asr")
        return reasons, metrics
    if (
        len(reference_tokens) >= args.min_tokens_for_f1
        and metrics["token_f1"] < args.min_token_f1
        and metrics["char_similarity"] < args.min_char_similarity
    ):
        reasons.append("cleanup_low_token_f1")
    if (
        len(reference_tokens) >= args.min_tokens_for_length_check
        and len(cleaned_tokens) < max(args.min_cleaned_tokens, int(len(reference_tokens) * args.min_token_count_ratio))
    ):
        reasons.append("cleanup_text_too_short")
    if "\ufffd" in cleaned_text:
        reasons.append("cleanup_bad_unicode")
    return reasons, metrics


def recalc_summary(
    sentence_dir: Path,
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    cleanup_summary: dict[str, Any],
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
            "accepted_roformer_cleanup": cleanup_summary,
        }
    )
    write_json(summary_path, summary)


def reject_row(
    row: dict[str, Any],
    reasons: list[str],
    *,
    original_review: Path,
    cleaned_review: Path | None,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rejected = dict(row)
    prior = [part for part in reason_parts(rejected) if part]
    all_reasons = [*prior, "roformer_cleanup_failed", *reasons]
    rejected["accepted"] = False
    rejected["reject_reason"] = ";".join(dict.fromkeys(all_reasons))
    rejected["roformer_cleanup_original_audio"] = str(original_review)
    if cleaned_review is not None:
        rejected["roformer_cleanup_cleaned_audio"] = str(cleaned_review)
    rejected["roformer_cleanup_method"] = "mlx-roformer"
    rejected["roformer_cleanup_model"] = args.roformer_model_filename
    rejected["roformer_cleanup_metrics"] = metrics
    return rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLX RoFormer on already accepted dialogue clips and reject clips whose cleaned ASR is damaged."
    )
    parser.add_argument("sentence_dir", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--language", default="hi")
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--initial-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--validate-asr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roformer-model-filename", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--roformer-model-dir", type=Path, default=None)
    parser.add_argument("--roformer-sample-rate", type=int, default=44100)
    parser.add_argument("--roformer-segment-size", type=int, default=256)
    parser.add_argument("--roformer-overlap", type=int, default=8)
    parser.add_argument("--roformer-batch-size", type=int, default=1)
    parser.add_argument(
        "--mono-channel",
        choices=["loudest", "left", "right", "quietest", "average"],
        default="left",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--compact-gap-seconds", type=float, default=0.18)
    parser.add_argument("--min-duration-ratio", type=float, default=0.80)
    parser.add_argument("--min-token-f1", type=float, default=0.45)
    parser.add_argument("--min-char-similarity", type=float, default=0.72)
    parser.add_argument("--min-tokens-for-f1", type=int, default=4)
    parser.add_argument("--min-tokens-for-length-check", type=int, default=5)
    parser.add_argument("--min-cleaned-tokens", type=int, default=2)
    parser.add_argument("--min-token-count-ratio", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sentence_dir = args.sentence_dir.resolve()
    project_root = args.project_root.resolve()
    cleanup_dir = sentence_dir / "accepted_roformer_cleanup"
    summary_path = cleanup_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(encoding="utf-8"))
        return

    accepted_path = sentence_dir / "accepted_sentences.jsonl"
    rejected_path = sentence_dir / "rejected_sentences.jsonl"
    accepted_rows = read_jsonl(accepted_path)
    rejected_rows = read_jsonl(rejected_path)
    cleanup_dir.mkdir(parents=True, exist_ok=True)

    if not accepted_rows:
        summary = {
            "input_accepted_clips": 0,
            "cleaned_kept_clips": 0,
            "rejected_after_cleanup": 0,
            "updated_clips": 0,
            "cleanup_dir": str(cleanup_dir),
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    backup_accepted = sentence_dir / "accepted_sentences_before_roformer_cleanup.jsonl"
    backup_rejected = sentence_dir / "rejected_sentences_before_roformer_cleanup.jsonl"
    if not backup_accepted.exists():
        shutil.copy2(accepted_path, backup_accepted)
    if rejected_path.exists() and not backup_rejected.exists():
        shutil.copy2(rejected_path, backup_rejected)

    original_dir = cleanup_dir / "original"
    cleaned_dir = cleanup_dir / "cleaned"
    rejected_dir = cleanup_dir / "rejected"
    original_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    input_paths: list[Path] = []
    row_by_stem: dict[str, dict[str, Any]] = {}
    original_review_by_stem: dict[str, Path] = {}
    missing_rows: list[dict[str, Any]] = []
    for row in accepted_rows:
        clip_id = str(row.get("clip_id") or Path(str(row.get("audio") or "clip")).stem)
        audio = row.get("audio")
        if not audio:
            missing_rows.append(
                reject_row(
                    row,
                    ["cleanup_missing_audio_path"],
                    original_review=original_dir / f"{clip_id}_missing.wav",
                    cleaned_review=None,
                    metrics={},
                    args=args,
                )
            )
            continue
        audio_path = Path(str(audio))
        if not audio_path.is_absolute():
            audio_path = (sentence_dir / audio_path).resolve()
        if not audio_path.exists():
            missing_rows.append(
                reject_row(
                    row,
                    ["cleanup_missing_audio_file"],
                    original_review=original_dir / f"{clip_id}_missing.wav",
                    cleaned_review=None,
                    metrics={"missing_audio": str(audio_path)},
                    args=args,
                )
            )
            continue
        original_review = original_dir / f"{clip_id}_original.wav"
        if not original_review.exists():
            shutil.copy2(audio_path, original_review)
        input_paths.append(audio_path)
        row_by_stem[audio_path.stem] = row
        original_review_by_stem[audio_path.stem] = original_review

    separated_paths = separate_with_mlx_roformer(input_paths, cleanup_dir, args, project_root)

    kept_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = [*missing_rows]
    cleaned_rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        stem = input_path.stem
        row = row_by_stem[stem]
        clip_id = str(row.get("clip_id") or stem)
        original_review = original_review_by_stem[stem]
        vocals = separated_paths.get(stem)
        if vocals is None or not vocals.exists():
            failed = reject_row(
                row,
                ["cleanup_missing_roformer_vocals"],
                original_review=original_review,
                cleaned_review=None,
                metrics={},
                args=args,
            )
            failed_rows.append(failed)
            try:
                input_path.unlink()
            except OSError:
                pass
            continue

        cleaned_review = cleaned_dir / f"{clip_id}_roformer_dialogue.wav"
        convert_vocals(
            vocals,
            cleaned_review,
            ffmpeg=args.ffmpeg,
            mono_channel=args.mono_channel,
            project_root=project_root,
        )
        cleaned_text = transcribe_audio(cleaned_review, args, project_root)
        reject_reasons, metrics = validate_cleaned_clip(
            row,
            original_review,
            cleaned_review,
            cleaned_text,
            args,
        )
        if reject_reasons:
            reason_dir = rejected_dir / reject_reasons[0]
            reason_dir.mkdir(parents=True, exist_ok=True)
            rejected_original = reason_dir / f"{clip_id}_original.wav"
            rejected_cleaned = reason_dir / f"{clip_id}_roformer_dialogue.wav"
            shutil.copy2(original_review, rejected_original)
            shutil.copy2(cleaned_review, rejected_cleaned)
            failed = reject_row(
                row,
                reject_reasons,
                original_review=rejected_original,
                cleaned_review=rejected_cleaned,
                metrics=metrics,
                args=args,
            )
            failed_rows.append(failed)
            try:
                input_path.unlink()
            except OSError:
                pass
            continue

        shutil.copy2(cleaned_review, input_path)
        kept = dict(row)
        kept["audio"] = str(input_path)
        kept["roformer_cleanup_applied"] = True
        kept["roformer_cleanup_original_audio"] = str(original_review)
        kept["roformer_cleanup_cleaned_audio"] = str(cleaned_review)
        kept["roformer_cleanup_method"] = "mlx-roformer"
        kept["roformer_cleanup_model"] = args.roformer_model_filename
        kept["roformer_cleanup_mono_channel"] = args.mono_channel
        kept["roformer_cleanup_metrics"] = metrics
        kept_rows.append(kept)
        cleaned_rows.append(kept)

    rejected_rows = [*rejected_rows, *failed_rows]
    write_jsonl(accepted_path, kept_rows)
    write_jsonl(rejected_path, rejected_rows)
    write_jsonl(cleanup_dir / "cleaned_sentences.jsonl", cleaned_rows)
    write_jsonl(cleanup_dir / "failed_sentences.jsonl", failed_rows)
    rebuild_compact(
        kept_rows,
        sentence_dir / "accepted_sentences_compact.wav",
        gap_seconds=args.compact_gap_seconds,
    )

    summary = {
        "input_accepted_clips": len(accepted_rows),
        "exported_for_cleanup": len(input_paths),
        "cleaned_kept_clips": len(kept_rows),
        "rejected_after_cleanup": len(failed_rows),
        "updated_clips": len(cleaned_rows),
        "cleaned_kept_seconds": round(
            sum(float(row.get("duration") or 0.0) for row in kept_rows), 3
        ),
        "rejected_after_cleanup_seconds": round(
            sum(float(row.get("duration") or 0.0) for row in failed_rows), 3
        ),
        "validate_asr": bool(args.validate_asr),
        "cleanup_backend": "mlx-roformer",
        "cleanup_model": args.roformer_model_filename,
        "mono_channel": args.mono_channel,
        "cleanup_dir": str(cleanup_dir),
        "original_dir": str(original_dir),
        "cleaned_dir": str(cleaned_dir),
        "rejected_dir": str(rejected_dir),
        "cleaned_manifest": str(cleanup_dir / "cleaned_sentences.jsonl"),
        "failed_manifest": str(cleanup_dir / "failed_sentences.jsonl"),
    }
    write_json(summary_path, summary)
    recalc_summary(sentence_dir, kept_rows, rejected_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
