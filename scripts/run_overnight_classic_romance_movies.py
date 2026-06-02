#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/Users/piyush/projects/articulatory-tts")
OUT_BASE = PROJECT_ROOT / "data" / "licensed_hindi_classic_romance_batch_full_clean_guarded"
LOG_DIR = PROJECT_ROOT / "data" / "batch_logs" / "classic_romance_overnight"
PROGRESS_TSV = OUT_BASE / "batch_progress.tsv"

MOVIES = [
    (
        "chashme_buddoor_1981",
        "Chashme Buddoor (1981)",
        Path("/Users/piyush/Downloads/vidssave.com Chashme Buddoor - चश्मे बद्दूर  - Full Movie - Farooq Sheikh, Deepti Naval, Rakesh Bedi - HD 144P.mp4"),
    ),
    (
        "gharonda_1977",
        "Gharonda (1977)",
        Path("/Users/piyush/Downloads/vidssave.com Gharaonda (HD) (1977) Full Hindi Movie _ Amol Palekar, Zarina Wahab, Dr. Shreeram Lagoo 144P.mp4"),
    ),
    (
        "saath_saath_1982",
        "Saath Saath (1982)",
        Path("/Users/piyush/Downloads/vidssave.com Saath Saath {HD} Farooque Shaikh _ Deepti Naval _ Satish Shah Hindi Full Movie (With Eng Subtitles) 360P.mp4"),
    ),
    (
        "ahista_ahista_1981",
        "Ahista Ahista (1981)",
        Path("/Users/piyush/Downloads/vidssave.com Ahista Ahista (1981) Full Hindi Movie _ Shammi Kapoor, Nanda, Kunal Kapoor. Padmini Kolhapure 144P.mp4"),
    ),
    (
        "sparsh_1980",
        "Sparsh (1980)",
        Path("/Users/piyush/Downloads/vidssave.com Sparsh (HD & Eng Subs) Hindi Full Movie - Naseeruddin Shah - Shabana Azmi - Bollywood Classic Movies 144P.mp4"),
    ),
    (
        "manzil_1979",
        "Manzil (1979)",
        Path("/Users/piyush/Downloads/vidssave.com Manzil 1979 Full Movie - Amitabh Bachchan, Moushumi Chatterjee _ RD Burman Musical _ Old Hindi Films 144P.mp4"),
    ),
    (
        "ijaazat_1987",
        "Ijaazat (1987)",
        Path("/Users/piyush/Downloads/vidssave.com Ijaazat (1987) Full Hindi Movie _ Naseeruddin Shah, Rekha, Anuradha Patel 144P.mp4"),
    ),
    ("ankahi_1985", "Ankahi (1985)", Path("/Users/piyush/Downloads/Ankahee- 1985.mp4")),
    (
        "gaman_1978",
        "Gaman (1978)",
        Path("/Users/piyush/Downloads/vidssave.com Gaman - 1978 FULL MOVIE  - Farooq Shaikh, Smita Patil (Super Hit Classic GAMAN Bollywood Movie) 144P.mp4"),
    ),
    ("ek_baar_phir_1980", "Ek Baar Phir (1980)", Path("/Users/piyush/Downloads/Ek Baar Phir _ Full Movie _ HD.mp4")),
    (
        "mausam_1975",
        "Mausam (1975)",
        Path("/Users/piyush/Downloads/vidssave.com Mausam Hindi Full HD Movie _ Sanjeev Kumar _ Sharmila Tagore _ 1975 Bollywood Full Movie 144P.mp4"),
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
    parser = argparse.ArgumentParser(description="Sequential overnight runner for the classic Hindi romance movie batch.")
    parser.add_argument("--dry-run", action="store_true")
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

    caffeinate_proc = start_caffeinate()
    append(LOG_DIR / "batch.log", f"Batch start: {now()}\n")
    append(LOG_DIR / "batch.log", f"Output base: {OUT_BASE}\n")
    append(LOG_DIR / "batch.log", f"Logs: {LOG_DIR}\n")
    append(LOG_DIR / "batch.log", "GPU mode: MPS only, PYTORCH_ENABLE_MPS_FALLBACK=0\n")
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
