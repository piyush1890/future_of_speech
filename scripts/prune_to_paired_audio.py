#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SENTENCE_DIR_NAME = "audible_sentences_no_songs_mlx_ecapa_mps_sentence_t050_nosnap_rejectcont_max30"
DEFAULT_PROMPT_DIR_NAME = "strict_voice_prompts_t034_p034_training"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def resolve_data_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def audio_path_for_row(row: dict[str, Any], project_root: Path) -> Path | None:
    for key in ("audio", "audio_filepath", "path"):
        path = resolve_data_path(row.get(key), project_root)
        if path is not None:
            return path
    return None


def discover_sentence_dirs(paths: list[Path], prompt_dir_name: str) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        candidates: list[Path] = []
        if (path / "accepted_sentences.jsonl").exists() or (path / prompt_dir_name / "train_with_prompts.jsonl").exists():
            candidates.append(path)
        if (path / DEFAULT_SENTENCE_DIR_NAME).exists():
            candidates.append(path / DEFAULT_SENTENCE_DIR_NAME)
        candidates.extend(
            train_manifest.parent.parent
            for train_manifest in path.glob(f"*/{DEFAULT_SENTENCE_DIR_NAME}/{prompt_dir_name}/train_with_prompts.jsonl")
        )
        candidates.extend(
            train_manifest.parent.parent
            for train_manifest in path.glob(f"**/{DEFAULT_SENTENCE_DIR_NAME}/{prompt_dir_name}/train_with_prompts.jsonl")
        )
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in seen and (candidate / prompt_dir_name / "train_with_prompts.jsonl").exists():
                found.append(candidate)
                seen.add(candidate)
    return found


def unlink_if_exists(path: Path, removed: list[str]) -> None:
    if path.exists() and path.is_file():
        path.unlink()
        removed.append(str(path))


def prune_sentence_dir(
    sentence_dir: Path,
    *,
    prompt_dir_name: str,
    project_root: Path,
    remove_other_prompt_dirs: bool,
) -> dict[str, Any]:
    prompt_dir = sentence_dir / prompt_dir_name
    train_manifest = prompt_dir / "train_with_prompts.jsonl"
    if not train_manifest.exists():
        return {"sentence_dir": str(sentence_dir), "skipped": True, "reason": "missing_train_manifest"}

    rows = read_jsonl(train_manifest)
    keep_audio_paths = {
        path for row in rows if (path := audio_path_for_row(row, project_root)) is not None
    }
    keep_prompt_paths = {
        path for row in rows if (path := resolve_data_path(row.get("voice_prompt"), project_root)) is not None
    }

    removed_files: list[str] = []
    clips_dir = sentence_dir / "clips"
    removed_clip_count = 0
    if clips_dir.exists():
        for wav in sorted(clips_dir.glob("*.wav")):
            if wav.resolve() not in keep_audio_paths:
                wav.unlink()
                removed_files.append(str(wav))
                removed_clip_count += 1

    prompts_dir = prompt_dir / "prompts"
    removed_prompt_count = 0
    if prompts_dir.exists():
        for wav in sorted(prompts_dir.glob("*.wav")):
            if wav.resolve() not in keep_prompt_paths:
                wav.unlink()
                removed_files.append(str(wav))
                removed_prompt_count += 1

    accepted_manifest = sentence_dir / "accepted_sentences.jsonl"
    if accepted_manifest.exists():
        write_jsonl(accepted_manifest, rows)
    write_jsonl(sentence_dir / "paired_only.jsonl", rows)
    write_jsonl(train_manifest, rows)

    for stale_file in [
        sentence_dir / "accepted_sentences_compact.wav",
        sentence_dir / "rejected_sentences.jsonl",
        prompt_dir / "voice_prompt_pairs.jsonl",
        prompt_dir / "paired_with_prompts.jsonl",
        prompt_dir / "ecapa_embeddings.npy",
        prompt_dir / "ecapa_distances.npy",
    ]:
        unlink_if_exists(stale_file, removed_files)

    removed_prompt_dirs: list[str] = []
    if remove_other_prompt_dirs and sentence_dir.exists():
        for sibling in sorted(sentence_dir.iterdir()):
            if sibling == prompt_dir:
                continue
            if sibling.is_dir() and sibling.name.startswith("strict_voice_prompts"):
                shutil.rmtree(sibling)
                removed_prompt_dirs.append(str(sibling))

    retained_seconds = round(sum(float(row.get("duration") or 0.0) for row in rows), 2)
    retained_clip_count = len(rows)

    summary_path = sentence_dir / "summary.json"
    summary: dict[str, Any] = read_json(summary_path) if summary_path.exists() else {}
    summary.setdefault("pre_prune_accepted_sentences", summary.get("accepted_sentences", retained_clip_count))
    summary.setdefault("pre_prune_accepted_seconds", summary.get("accepted_seconds", retained_seconds))
    summary["accepted_sentences"] = retained_clip_count
    summary["accepted_seconds"] = retained_seconds
    summary["retained_paired_sentences"] = retained_clip_count
    summary["retained_paired_seconds"] = retained_seconds
    summary["pruned_unpaired_clip_files"] = removed_clip_count
    summary["paired_only_manifest"] = str(sentence_dir / "paired_only.jsonl")
    write_json(summary_path, summary)

    prune_summary = {
        "sentence_dir": str(sentence_dir),
        "prompt_dir": str(prompt_dir),
        "retained_target_clips": retained_clip_count,
        "retained_prompt_clips": len(keep_prompt_paths),
        "retained_seconds": retained_seconds,
        "removed_unpaired_target_clips": removed_clip_count,
        "removed_unreferenced_prompt_clips": removed_prompt_count,
        "removed_files": len(removed_files),
        "removed_prompt_dirs": removed_prompt_dirs,
        "outputs": {
            "paired_only": str(sentence_dir / "paired_only.jsonl"),
            "train_with_prompts": str(train_manifest),
            "accepted_sentences": str(accepted_manifest),
            "clips": str(clips_dir),
            "prompts": str(prompts_dir),
        },
    }
    write_json(prompt_dir / "paired_only_prune_summary.json", prune_summary)
    return prune_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep only strict prompt-paired audio and JSONL rows.")
    parser.add_argument("paths", nargs="+", type=Path, help="Sentence dirs, episode dirs, or roots to scan.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--prompt-dir-name", default=DEFAULT_PROMPT_DIR_NAME)
    parser.add_argument("--remove-other-prompt-dirs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    sentence_dirs = discover_sentence_dirs(args.paths, args.prompt_dir_name)
    summaries = [
        prune_sentence_dir(
            sentence_dir,
            prompt_dir_name=args.prompt_dir_name,
            project_root=project_root,
            remove_other_prompt_dirs=args.remove_other_prompt_dirs,
        )
        for sentence_dir in sentence_dirs
    ]
    total = {
        "sentence_dirs": len(summaries),
        "retained_target_clips": sum(int(item.get("retained_target_clips") or 0) for item in summaries),
        "retained_seconds": round(sum(float(item.get("retained_seconds") or 0.0) for item in summaries), 2),
        "removed_unpaired_target_clips": sum(int(item.get("removed_unpaired_target_clips") or 0) for item in summaries),
        "details": summaries,
    }
    print(json.dumps(total, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
