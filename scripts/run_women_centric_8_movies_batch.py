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
OUT_BASE = PROJECT_ROOT / "data" / "licensed_hindi_women_centric_batch_full_clean_guarded"
LOG_DIR = PROJECT_ROOT / "data" / "batch_logs" / "women_centric_8_full"
PROGRESS_TSV = OUT_BASE / "batch_progress.tsv"

MOVIES = [
    (
        "subah_umbartha_1982",
        "Umbartha / Subah (1982)",
        Path("/Users/piyush/Downloads/vidssave.com Subah (1982) _ Superhit Classic Movie With English Subtitles _ सुबह _ Grishi Karnad, Smita Patil 144P.mp4"),
    ),
    (
        "rudaali_1993",
        "Rudaali (1993)",
        Path("/Users/piyush/Downloads/Rudaali 1993 Full Movie HD _ Dimple Kapadia, Rakhee Gulzar, Raj Babbar, Amjad Khan _ Facts & Review.mp4"),
    ),
    (
        "astitva_2000",
        "Astitva (2000)",
        Path("/Users/piyush/Downloads/vidssave.com Astitva - अस्तित्व (2000) _ Tabu, Sachin Khedekar & Mahesh Manjrekar _ Full Hindi Movie (HD) 144P.mp4"),
    ),
    (
        "page_3_2005",
        "Page 3 (2005)",
        Path("/Users/piyush/Downloads/Page 3 (2005) Full HIndi Movie _ Konkona Sen Sharma, Tara Sharma, Boman Irani, Atul Kulkarni.mp4"),
    ),
    (
        "no_one_killed_jessica_2011",
        "No One Killed Jessica (2011)",
        Path("/Users/piyush/Downloads/vidssave.com No One Killed Jessica 2011 full movie _ Rani Mukerji _ Vidya Balan thriller hindi movie 144P.mp4"),
    ),
    (
        "tanu_weds_manu_returns_2015",
        "Tanu Weds Manu Returns (2015)",
        Path("/Users/piyush/Downloads/vidssave.com Tanu Weds Manu Returns - Full Movie HD _ Kangana Ranaut _ R. Madhavan _ Bollywood Romantic Comedy 144P.mp4"),
    ),
    (
        "raazi_2018",
        "Raazi (2018)",
        Path("/Users/piyush/Downloads/vidssave.com Raazi (2018) _ Full Movie _ Alia Bhatt & Vicky Kaushal _ Bollywood Hindi Movie Spy Thriller _ HD 144P.mp4"),
    ),
    (
        "pagglait_2021",
        "Pagglait (2021)",
        Path("/Users/piyush/Downloads/Pagglait Full Movie _ Sanya Malhotra _ Ashutosh Rana _ Ashlesha Thakur _ Shruti S _ Review & Facts.mp4"),
    ),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def load_summary(out_root: Path) -> dict:
    summary_path = out_root / "batch_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def is_done(out_root: Path) -> bool:
    data = load_summary(out_root)
    return data.get("failures") == 0 and float(data.get("paired_seconds") or 0) > 0


def summary_numbers(out_root: Path) -> tuple[float, int]:
    data = load_summary(out_root)
    paired_seconds = float(data.get("paired_seconds") or 0)
    train_clips = sum(int(item.get("train_clips") or 0) for item in data.get("summaries", []))
    return paired_seconds, train_clips


def progress(slug: str, title: str, status: str, paired_seconds: float, train_clips: int, out_root: Path, log: Path) -> None:
    append(
        PROGRESS_TSV,
        f"{now()}\t{slug}\t{title}\t{status}\t{paired_seconds:.2f}\t{train_clips}\t{out_root}\t{log}\n",
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
    parser = argparse.ArgumentParser(description="Sequential runner for the eight women-centric Hindi movie dialogue batch.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait-for-ac", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_TSV.exists():
        append(PROGRESS_TSV, "finished_at\tslug\ttitle\tstatus\tpaired_seconds\ttrain_clips\tout_root\tlog\n")

    python = PROJECT_ROOT / ".venv_dialogue_dataset" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)

    append(LOG_DIR / "batch.log", f"Batch start requested: {now()}\n")
    append(LOG_DIR / "batch.log", f"Output base: {OUT_BASE}\n")
    append(LOG_DIR / "batch.log", f"Logs: {LOG_DIR}\n")
    append(LOG_DIR / "batch.log", "GPU mode: MPS only, PYTORCH_ENABLE_MPS_FALLBACK=0\n")
    append(LOG_DIR / "batch.log", "Final music guard: enabled, uncertain clips excluded from training.\n")

    if not args.dry_run and not args.no_wait_for_ac:
        wait_for_ac_power()

    caffeinate_proc = None if args.dry_run else start_caffeinate()
    if caffeinate_proc:
        append(LOG_DIR / "batch.log", f"Caffeinate pid: {caffeinate_proc.pid}\n")

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

    for slug, title, source in MOVIES:
        out_root = OUT_BASE / slug
        log = LOG_DIR / f"{slug}.log"
        if not source.exists():
            progress(slug, title, "missing_source", 0.0, 0, out_root, log)
            continue
        if args.dry_run:
            progress(slug, title, "dry_run", 0.0, 0, out_root, log)
            continue
        if is_done(out_root):
            paired_seconds, train_clips = summary_numbers(out_root)
            progress(slug, title, "skipped_done", paired_seconds, train_clips, out_root, log)
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

        paired_seconds, train_clips = summary_numbers(out_root)
        if result.returncode == 0:
            status = "done" if load_summary(out_root).get("failures") == 0 else "done_with_failures"
        else:
            status = f"failed_{result.returncode}"
        progress(slug, title, status, paired_seconds, train_clips, out_root, log)
        append(
            LOG_DIR / "batch.log",
            f"[{now()}] END {title} status={status} paired_seconds={paired_seconds:.2f} train_clips={train_clips}\n",
        )

    append(LOG_DIR / "batch.log", f"Batch complete: {now()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
