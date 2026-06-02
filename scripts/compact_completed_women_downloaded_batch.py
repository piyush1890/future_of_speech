#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/Users/piyush/projects/articulatory-tts")
BATCH_ROOT = PROJECT_ROOT / "data" / "licensed_hindi_women_centric_downloaded_8_full_clean_guarded"
UPLOAD_ROOT = PROJECT_ROOT / "data" / "women_centric_downloaded_training_upload_music_guarded_20260527"
LOG_PATH = PROJECT_ROOT / "data" / "batch_logs" / "women_centric_downloaded_8_full" / "upload_compactor.log"

ABSOLUTE_PATH_KEYS_TO_DROP = {
    "roformer_cleanup_original_audio",
    "roformer_cleanup_cleaned_audio",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def completed_movie_roots() -> list[Path]:
    roots = []
    if not BATCH_ROOT.exists():
        return roots
    for summary_path in sorted(BATCH_ROOT.glob("*/batch_summary.json")):
        root = summary_path.parent
        try:
            summary = read_json(summary_path)
        except Exception as exc:
            append_log(f"Skipping unreadable summary {summary_path}: {exc}")
            continue
        if int(summary.get("failures") or 0) != 0:
            continue
        if float(summary.get("paired_seconds") or 0.0) <= 0:
            continue
        roots.append(root)
    return roots


def final_outputs(movie_root: Path) -> tuple[Path, Path, Path, dict]:
    summary = read_json(movie_root / "batch_summary.json")
    summaries = summary.get("summaries") or []
    if not summaries:
        raise ValueError(f"No per-episode summaries in {movie_root / 'batch_summary.json'}")
    outputs = summaries[0].get("outputs") or {}
    manifest = Path(outputs.get("train_with_prompts") or "")
    clips_dir = Path(outputs.get("clips") or "")
    prompts_dir = Path(outputs.get("prompts") or "")
    if not manifest.exists():
        raise FileNotFoundError(f"Missing train manifest: {manifest}")
    if not clips_dir.exists():
        raise FileNotFoundError(f"Missing clips dir: {clips_dir}")
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Missing prompts dir: {prompts_dir}")
    return manifest, clips_dir, prompts_dir, summary


def copy_used_file(src_value: str, src_base: Path, dst_dir: Path) -> str:
    src = Path(src_value)
    if not src.is_absolute():
        src = src_base / src
    if not src.exists():
        raise FileNotFoundError(f"Missing referenced audio: {src}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)
    return dst.name


def sanitize_row(row: dict, slug: str, index: int, manifest_dir: Path, movie_dir: Path) -> dict:
    out = dict(row)
    for key in ABSOLUTE_PATH_KEYS_TO_DROP:
        out.pop(key, None)

    audio_name = copy_used_file(str(row["audio"]), manifest_dir, movie_dir / "clips")
    prompt_name = copy_used_file(str(row["voice_prompt"]), manifest_dir, movie_dir / "prompts")

    out["movie_id"] = slug
    out["dataset_row_id"] = f"{slug}_{index:06d}"
    out["audio"] = f"movies/{slug}/clips/{audio_name}"
    out["voice_prompt"] = f"movies/{slug}/prompts/{prompt_name}"
    return out


def compact_movie(movie_root: Path) -> dict:
    slug = movie_root.name
    manifest, _clips_dir, _prompts_dir, batch_summary = final_outputs(movie_root)
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError(f"No train rows in {manifest}")

    tmp_movie_dir = UPLOAD_ROOT / "movies" / f".{slug}.tmp"
    movie_dir = UPLOAD_ROOT / "movies" / slug
    if tmp_movie_dir.exists():
        shutil.rmtree(tmp_movie_dir)
    tmp_movie_dir.mkdir(parents=True, exist_ok=True)

    compact_rows = [sanitize_row(row, slug, index, manifest.parent, tmp_movie_dir) for index, row in enumerate(rows)]
    train_seconds = round(sum(float(row.get("duration") or row.get("audio_duration_seconds") or 0.0) for row in compact_rows), 2)
    movie_summary = {
        "slug": slug,
        "title": (batch_summary.get("summaries") or [{}])[0].get("title") or slug,
        "train_clips": len(compact_rows),
        "train_seconds": train_seconds,
        "source_batch_summary": str(movie_root / "batch_summary.json"),
        "compacted_at": now(),
    }
    write_jsonl(tmp_movie_dir / "train_with_prompts.jsonl", compact_rows)
    write_json(tmp_movie_dir / "summary.json", movie_summary)

    expected_clips = {Path(row["audio"]).name for row in compact_rows}
    expected_prompts = {Path(row["voice_prompt"]).name for row in compact_rows}
    copied_clips = {path.name for path in (tmp_movie_dir / "clips").glob("*.wav")}
    copied_prompts = {path.name for path in (tmp_movie_dir / "prompts").glob("*.wav")}
    if expected_clips - copied_clips:
        raise RuntimeError(f"Missing copied clips for {slug}: {sorted(expected_clips - copied_clips)[:5]}")
    if expected_prompts - copied_prompts:
        raise RuntimeError(f"Missing copied prompts for {slug}: {sorted(expected_prompts - copied_prompts)[:5]}")

    if movie_dir.exists():
        shutil.rmtree(movie_dir)
    tmp_movie_dir.rename(movie_dir)
    append_log(f"Compacted {slug}: clips={len(compact_rows)} seconds={train_seconds:.2f}")
    return movie_summary


def rebuild_aggregate() -> dict:
    movies_root = UPLOAD_ROOT / "movies"
    all_rows: list[dict] = []
    summaries: list[dict] = []
    if movies_root.exists():
        for movie_dir in sorted(path for path in movies_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
            train_manifest = movie_dir / "train_with_prompts.jsonl"
            summary_path = movie_dir / "summary.json"
            if not train_manifest.exists() or not summary_path.exists():
                continue
            rows = read_jsonl(train_manifest)
            all_rows.extend(rows)
            summaries.append(read_json(summary_path))
    write_jsonl(UPLOAD_ROOT / "train_all_movies.jsonl", all_rows)
    aggregate = {
        "created_or_updated_at": now(),
        "source_batch_root": str(BATCH_ROOT),
        "movies": summaries,
        "movie_count": len(summaries),
        "train_clips": len(all_rows),
        "train_seconds": round(sum(float(row.get("duration") or row.get("audio_duration_seconds") or 0.0) for row in all_rows), 2),
    }
    write_json(UPLOAD_ROOT / "movie_summary.json", aggregate)
    readme = (
        "# Women-Centric Downloaded Hindi Dialogue Training Upload\n\n"
        "Use `train_all_movies.jsonl` for the full compact dataset. Each row has relative `audio` "
        "and `voice_prompt` paths rooted at this folder.\n\n"
        "Per-movie copies live under `movies/<movie_id>/` with target clips, voice prompts, "
        "and small manifests only. Source videos, rejected clips, Demucs/RoFormer outputs, "
        "and other bulky intermediate folders are intentionally excluded.\n"
    )
    (UPLOAD_ROOT / "README.md").write_text(readme, encoding="utf-8")
    return aggregate


def prune_movie_root(movie_root: Path, min_free_gb: float, force: bool = False) -> bool:
    upload_movie = UPLOAD_ROOT / "movies" / movie_root.name
    if not (upload_movie / "train_with_prompts.jsonl").exists():
        return False
    if not force and free_gib(PROJECT_ROOT) >= min_free_gb:
        return False
    size_before = shutil.disk_usage(PROJECT_ROOT).free
    shutil.rmtree(movie_root)
    size_after = shutil.disk_usage(PROJECT_ROOT).free
    freed_gib = (size_after - size_before) / (1024**3)
    append_log(f"Pruned raw folder {movie_root.name}; freed about {freed_gib:.2f} GiB")
    return True


def run_once(args: argparse.Namespace) -> dict:
    compacted = []
    pruned = []
    errors = []
    for movie_root in completed_movie_roots():
        slug = movie_root.name
        try:
            if not (UPLOAD_ROOT / "movies" / slug / "train_with_prompts.jsonl").exists():
                compact_movie(movie_root)
                compacted.append(slug)
            if prune_movie_root(movie_root, args.prune_below_gb, force=args.force_prune):
                pruned.append(slug)
        except Exception as exc:
            errors.append({"slug": slug, "error": str(exc)})
            append_log(f"Error for {slug}: {exc}")
    aggregate = rebuild_aggregate()
    status = {
        "checked_at": now(),
        "free_gib": round(free_gib(PROJECT_ROOT), 2),
        "compact_root": str(UPLOAD_ROOT),
        "compacted": compacted,
        "pruned": pruned,
        "errors": errors,
        "aggregate": aggregate,
    }
    write_json(UPLOAD_ROOT / "compactor_status.json", status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact completed movie outputs and prune raw intermediates when low on disk.")
    parser.add_argument("--loop", action="store_true", help="Keep checking until stopped.")
    parser.add_argument("--interval-seconds", type=int, default=180)
    parser.add_argument("--prune-below-gb", type=float, default=25.0)
    parser.add_argument("--force-prune", action="store_true", help="Delete raw completed movie folders after compacting regardless of free space.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    append_log(
        f"Compactor start loop={args.loop} interval={args.interval_seconds}s prune_below_gb={args.prune_below_gb}"
    )
    while True:
        status = run_once(args)
        append_log(
            f"Status free={status['free_gib']:.2f}GiB compacted={status['compacted']} pruned={status['pruned']} errors={len(status['errors'])}"
        )
        if not args.loop:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
