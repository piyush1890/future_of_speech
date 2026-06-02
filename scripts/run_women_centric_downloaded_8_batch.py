#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/Users/piyush/projects/articulatory-tts")
DOWNLOADS_DIR = Path("/Users/piyush/Downloads")
OUT_BASE = PROJECT_ROOT / "data" / "licensed_hindi_women_centric_downloaded_8_full_clean_guarded"
LOG_DIR = PROJECT_ROOT / "data" / "batch_logs" / "women_centric_downloaded_8_full"
PROGRESS_TSV = OUT_BASE / "batch_progress.tsv"
SOURCE_INVENTORY_TSV = OUT_BASE / "source_inventory.tsv"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}

MOVIES = [
    ("kahaani_2012", "Kahaani (2012)", ("kahaani full movie",)),
    ("the_dirty_picture_2011", "The Dirty Picture (2011)", ("dirty", "picture")),
    ("mardaani_2014", "Mardaani (2014)", ("mardaani", "2014")),
    ("nh10_2015", "NH10 (2015)", ("nh10",)),
    ("gulaab_gang_2014", "Gulaab Gang (2014)", ("gulaab", "gang")),
    ("revolver_rani_2014", "Revolver Rani (2014)", ("revolver", "rani")),
    ("pataakha_2018", "Pataakha (2018)", ("pataakha",)),
    (
        "dolly_kitty_aur_woh_chamakte_sitare_2020",
        "Dolly Kitty Aur Woh Chamakte Sitare (2020)",
        ("dolly", "kitty"),
    ),
    (
        "lipstick_under_my_burkha_2016",
        "Lipstick Under My Burkha (2016)",
        ("lipstick", "burkha"),
    ),
    (
        "kahaani_2_2016",
        "Kahaani 2: Durga Rani Singh (2016)",
        ("kahaani 2",),
    ),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def video_files() -> list[Path]:
    files: list[Path] = []
    for path in DOWNLOADS_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def find_source(tokens: tuple[str, ...], files: list[Path]) -> Path | None:
    matches = []
    for path in files:
        name = path.name.lower()
        if all(token.lower() in name for token in tokens):
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_size)


def load_summary(out_root: Path) -> dict:
    summary_path = out_root / "batch_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def is_done(out_root: Path) -> bool:
    data = load_summary(out_root)
    return data.get("failures") == 0 and float(data.get("paired_seconds") or 0) > 0


def summary_numbers(out_root: Path) -> tuple[float, int, int]:
    data = load_summary(out_root)
    paired_seconds = float(data.get("paired_seconds") or 0)
    train_clips = sum(int(item.get("train_clips") or 0) for item in data.get("summaries", []))
    music_rejects = 0
    for item in data.get("summaries", []):
        summary = item.get("music_guard_summary") or {}
        music_rejects += int(summary.get("rejected_music_high_confidence_clips") or 0)
        music_rejects += int(summary.get("uncertain_music_clips") or 0)
    return paired_seconds, train_clips, music_rejects


def progress(
    slug: str,
    title: str,
    status: str,
    paired_seconds: float,
    train_clips: int,
    music_rejects: int,
    source: Path | None,
    out_root: Path,
    log: Path,
) -> None:
    source_text = "" if source is None else str(source)
    append(
        PROGRESS_TSV,
        (
            f"{now()}\t{slug}\t{title}\t{status}\t{paired_seconds:.2f}\t"
            f"{train_clips}\t{music_rejects}\t{source_text}\t{out_root}\t{log}\n"
        ),
    )


def is_ac_power() -> bool:
    result = subprocess.run(["pmset", "-g", "batt"], text=True, capture_output=True, check=False)
    first_line = (result.stdout.splitlines() or [""])[0]
    return "AC Power" in first_line


def wait_for_ac_power() -> None:
    while not is_ac_power():
        append(LOG_DIR / "batch.log", f"[{now()}] Waiting for AC power before starting GPU batch.\n")
        time.sleep(60)


def start_caffeinate() -> subprocess.Popen | None:
    caffeinate = shutil.which("caffeinate")
    if not caffeinate or os.environ.get("DISABLE_CAFFEINATE") == "1":
        return None
    log = LOG_DIR / "caffeinate.log"
    return subprocess.Popen(
        [caffeinate, "-dimsu", "-w", str(os.getpid())],
        stdout=log.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential local-file runner for downloaded women-centric Hindi movies, excluding Mardaani 2."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait-for-ac", action="store_true")
    parser.add_argument("--start-slug", help="Skip movies before this slug and continue from there.")
    parser.add_argument("--only-slug", help="Run only this slug.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_TSV.exists():
        append(
            PROGRESS_TSV,
            (
                "finished_at\tslug\ttitle\tstatus\tpaired_seconds\ttrain_clips\t"
                "music_rejects\tsource\tout_root\tlog\n"
            ),
        )

    python = PROJECT_ROOT / ".venv_dialogue_dataset" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)

    files = video_files()
    resolved = [(slug, title, find_source(tokens, files)) for slug, title, tokens in MOVIES]
    inventory_rows = ["slug\ttitle\tstatus\tsource\n"]
    for slug, title, source in resolved:
        inventory_rows.append(f"{slug}\t{title}\t{'found' if source else 'missing'}\t{'' if source is None else source}\n")
    write(SOURCE_INVENTORY_TSV, "".join(inventory_rows))

    append(LOG_DIR / "batch.log", f"Batch start requested: {now()}\n")
    append(LOG_DIR / "batch.log", f"Output base: {OUT_BASE}\n")
    append(LOG_DIR / "batch.log", f"Logs: {LOG_DIR}\n")
    append(LOG_DIR / "batch.log", f"Source inventory: {SOURCE_INVENTORY_TSV}\n")
    append(LOG_DIR / "batch.log", "Excluded by request: Mardaani 2 (2019)\n")
    append(LOG_DIR / "batch.log", "GPU mode: MPS only, PYTORCH_ENABLE_MPS_FALLBACK=0\n")
    append(LOG_DIR / "batch.log", "Final music/noise guard: enabled, uncertain clips excluded from training.\n")

    if not args.dry_run and not args.no_wait_for_ac:
        wait_for_ac_power()

    caffeinate_proc = None if args.dry_run else start_caffeinate()
    if caffeinate_proc:
        append(LOG_DIR / "batch.log", f"Caffeinate pid: {caffeinate_proc.pid}\n")

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

    started = args.start_slug is None
    for slug, title, source in resolved:
        if args.only_slug and slug != args.only_slug:
            continue
        out_root = OUT_BASE / slug
        log = LOG_DIR / f"{slug}.log"
        if not started:
            if slug == args.start_slug:
                started = True
            else:
                continue
        if source is None:
            progress(slug, title, "missing_source", 0.0, 0, 0, None, out_root, log)
            continue
        if args.dry_run:
            progress(slug, title, "dry_run", 0.0, 0, 0, source, out_root, log)
            continue
        if is_done(out_root):
            paired_seconds, train_clips, music_rejects = summary_numbers(out_root)
            progress(slug, title, "skipped_done", paired_seconds, train_clips, music_rejects, source, out_root, log)
            continue

        append(LOG_DIR / "batch.log", f"\n[{now()}] START {title}\nsource={source}\nout={out_root}\nlog={log}\n")
        cmd = [
            str(python),
            str(PROJECT_ROOT / "run_dialogue_dataset.py"),
            str(source),
            "--out-root",
            str(out_root),
            "--slug",
            slug,
            "--title",
            title,
            "--music-guard",
            "--music-guard-exclude-uncertain",
        ]
        with log.open("w", encoding="utf-8") as handle:
            handle.write("+ " + " ".join(cmd) + "\n")
            handle.flush()
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)

        paired_seconds, train_clips, music_rejects = summary_numbers(out_root)
        if result.returncode == 0:
            status = "done" if load_summary(out_root).get("failures") == 0 else "done_with_failures"
        else:
            status = f"failed_{result.returncode}"
        progress(slug, title, status, paired_seconds, train_clips, music_rejects, source, out_root, log)
        append(
            LOG_DIR / "batch.log",
            (
                f"[{now()}] END {title} status={status} paired_seconds={paired_seconds:.2f} "
                f"train_clips={train_clips} music_rejects={music_rejects}\n"
            ),
        )

    if args.start_slug is not None and not started:
        append(LOG_DIR / "batch.log", f"[{now()}] start slug was not found: {args.start_slug}\n")
        return 2
    append(LOG_DIR / "batch.log", f"Batch complete: {now()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
