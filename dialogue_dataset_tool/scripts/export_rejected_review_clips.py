#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


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


def slugify(value: str) -> str:
    value = value.strip().lower() or "unknown"
    value = re.sub(r"[^a-z0-9_+.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:120] or "unknown"


def resolve_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def rejection_step(reason: str) -> tuple[str, str]:
    parts = {part for part in reason.split(";") if part}
    if "song_like" in parts:
        return "01_song_filter", "song_like"
    if "background_music_or_noise" in parts:
        return "01_background_music_or_noise", "background_music_or_noise"
    if "roformer_cleanup_failed" in parts:
        cleanup_reasons = [
            part for part in parts if part.startswith("cleanup_") and part != "cleanup_missing_audio_path"
        ]
        return "02_roformer_cleanup", slugify(";".join(sorted(cleanup_reasons)) or reason)
    return "02_sentence_quality", slugify(reason)


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


def export_filter_rejections(
    sentence_dir: Path,
    review_dir: Path,
    source_wav_path: Path | None,
    project_root: Path,
    *,
    pad_seconds: float,
) -> dict[str, Any]:
    manifest_path = sentence_dir / "rejected_sentences_before_background_music_recovery.jsonl"
    if not manifest_path.exists():
        manifest_path = sentence_dir / "rejected_sentences.jsonl"
    rejected_rows = read_jsonl(manifest_path)
    if not rejected_rows:
        return {
            "rows": 0,
            "exported": 0,
            "manifest": str(manifest_path),
            "reason": "no_rejected_rows",
        }

    source_wav: np.ndarray | None = None
    sr: int | None = None
    if source_wav_path is not None and source_wav_path.exists():
        source_wav, sr = load_wav(source_wav_path)
    manifests: dict[Path, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, float]] = {}
    exported = 0
    for idx, row in enumerate(rejected_rows):
        reason = str(row.get("reject_reason") or "unknown")
        step, bucket = rejection_step(reason)
        clip_id = slugify(str(row.get("clip_id") or f"rejected_{idx:06d}"))
        out_path = review_dir / step / bucket / f"{idx:06d}_{clip_id}.wav"
        copied_paths: list[Path] = []
        direct_audio = resolve_path(str(row.get("audio") or ""), project_root)
        cleaned_audio = resolve_path(
            str(
                row.get("roformer_cleanup_cleaned_audio")
                or row.get("background_music_cleaned_audio")
                or ""
            ),
            project_root,
        )
        original_audio = resolve_path(
            str(
                row.get("roformer_cleanup_original_audio")
                or row.get("background_music_original_audio")
                or ""
            ),
            project_root,
        )

        if cleaned_audio is not None and cleaned_audio.exists():
            cleaned_out = out_path.with_name(f"{out_path.stem}_cleaned.wav")
            cleaned_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cleaned_audio, cleaned_out)
            copied_paths.append(cleaned_out)
        if original_audio is not None and original_audio.exists():
            original_out = out_path.with_name(f"{out_path.stem}_original.wav")
            original_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_audio, original_out)
            copied_paths.append(original_out)
        if not copied_paths and direct_audio is not None and direct_audio.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(direct_audio, out_path)
            copied_paths.append(out_path)
        if (
            not copied_paths
            and source_wav is not None
            and sr is not None
            and export_slice(source_wav, sr, row, out_path, pad_seconds=pad_seconds)
        ):
            copied_paths.append(out_path)
        if not copied_paths:
            continue
        record = dict(row)
        record["review_audio"] = str(copied_paths[0])
        if len(copied_paths) > 1:
            record["review_audio_alternates"] = [str(path) for path in copied_paths[1:]]
        manifests.setdefault(out_path.parent / "manifest.jsonl", []).append(record)
        key = f"{step}/{bucket}"
        counts.setdefault(key, {"clips": 0, "seconds": 0.0})
        counts[key]["clips"] += 1
        counts[key]["seconds"] += float(row.get("duration") or 0.0)
        exported += 1

    for manifest, rows in manifests.items():
        write_jsonl(manifest, rows)
    return {
        "rows": len(rejected_rows),
        "exported": exported,
        "manifest": str(manifest_path),
        "buckets": counts,
    }


def export_pairing_rejections(
    prompt_dir: Path | None,
    review_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if prompt_dir is None:
        return {"rows": 0, "exported": 0, "reason": "missing_prompt_dir"}
    pair_rows = read_jsonl(prompt_dir / "voice_prompt_pairs.jsonl")
    if not pair_rows:
        return {"rows": 0, "exported": 0, "reason": "missing_voice_prompt_pairs"}

    manifests: dict[Path, list[dict[str, Any]]] = {}
    counts: dict[str, dict[str, float]] = {}
    exported = 0
    for idx, row in enumerate(pair_rows):
        status = str(row.get("prompt_status") or "unknown")
        if status == "strict_match":
            continue
        audio_path = resolve_path(row.get("audio"), project_root)
        if audio_path is None or not audio_path.exists():
            continue
        bucket = slugify(status)
        clip_id = slugify(str(row.get("clip_id") or f"unpaired_{idx:06d}"))
        out_path = review_dir / "03_prompt_pairing" / bucket / f"{idx:06d}_{clip_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, out_path)
        record = dict(row)
        record["review_audio"] = str(out_path)
        manifests.setdefault(out_path.parent / "manifest.jsonl", []).append(record)
        key = f"03_prompt_pairing/{bucket}"
        counts.setdefault(key, {"clips": 0, "seconds": 0.0})
        counts[key]["clips"] += 1
        counts[key]["seconds"] += float(row.get("duration") or 0.0)
        exported += 1

    for manifest, rows in manifests.items():
        write_jsonl(manifest, rows)
    return {"rows": len(pair_rows), "exported": exported, "buckets": counts}


def round_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: round_summary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_summary(item) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export rejected/intermediate clips for review.")
    parser.add_argument("--sentence-dir", type=Path, required=True)
    parser.add_argument("--source-wav", type=Path, default=None)
    parser.add_argument("--prompt-dir", type=Path, default=None)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--pad-seconds", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentence_dir = args.sentence_dir.resolve()
    project_root = args.project_root.resolve()
    prompt_dir = args.prompt_dir.resolve() if args.prompt_dir else None
    source_wav = args.source_wav.resolve() if args.source_wav else None
    review_dir = (args.review_dir or (sentence_dir / "rejected_review")).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "review_dir": str(review_dir),
        "sentence_dir": str(sentence_dir),
        "source_wav": str(source_wav) if source_wav else None,
        "prompt_dir": str(prompt_dir) if prompt_dir else None,
        "filter_rejections": export_filter_rejections(
            sentence_dir,
            review_dir,
            source_wav,
            project_root,
            pad_seconds=args.pad_seconds,
        ),
        "pairing_rejections": export_pairing_rejections(prompt_dir, review_dir, project_root),
    }
    summary = round_summary(summary)
    with (review_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
