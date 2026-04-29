"""
Quick health check for v5 stage 1 training. Run any time:

    python3 scripts/monitor_v5_training.py

Prints a compact one-screen summary: process status, current epoch/step,
recent gradient ratios, epoch val_CE trend, codebook usage trend, latest
checkpoint, ETA estimate.

No training-process modification, just reads logs + metrics file.
"""
import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


CKPT_DIR = Path("checkpoints_v5_stage1")
LOG_PATH = Path("train_v5_stage1.log")
METRICS  = CKPT_DIR / "metrics.jsonl"
TARGET_EPOCHS = 50


def ansi(code, s): return f"\x1b[{code}m{s}\x1b[0m"
def b(s):   return ansi("1", s)
def dim(s): return ansi("2", s)
def red(s): return ansi("31", s)
def green(s): return ansi("32", s)
def yellow(s): return ansi("33", s)


def proc_status():
    # Match either the shell wrapper or the python child via training script name
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "train_v5_stage1"], text=True
        ).strip()
        if out:
            pids = out.split("\n")
            return f"{green('RUNNING')} (PIDs {','.join(pids)})"
    except subprocess.CalledProcessError:
        pass
    try:
        out = subprocess.check_output(["pgrep", "-f", "watchdog.sh"], text=True).strip()
        if out:
            return f"{yellow('STOPPED')} (watchdog alive PID {out.split()[0]} — will restart)"
    except subprocess.CalledProcessError:
        pass
    return red("STOPPED (watchdog also dead)")


def parse_metrics():
    if not METRICS.exists(): return [], [], []
    steps, epochs, audits = [], [], []
    with open(METRICS) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") == "step": steps.append(r)
            elif r.get("type") == "epoch": epochs.append(r)
            elif r.get("type") == "audit": audits.append(r)
    return steps, epochs, audits


def latest_tqdm_line():
    if not LOG_PATH.exists(): return None
    # Read tail and find the last \r-terminated tqdm line
    with open(LOG_PATH, "rb") as f:
        f.seek(0, 2); size = f.tell()
        f.seek(max(0, size - 16384))
        chunk = f.read().decode(errors="ignore")
    # Split on either newline or carriage-return
    lines = re.split(r"[\r\n]+", chunk)
    for l in reversed(lines):
        if "Epoch " in l and "/" in l and "%" in l:
            return l.strip()
    return None


def fmt_eta(seconds):
    return str(timedelta(seconds=int(seconds))).replace("days,", "d")


def main():
    print(b("=" * 76))
    print(b("v5 STAGE 1 TRAINING MONITOR"), dim(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(b("=" * 76))

    # Process
    print(f"\n  Process: {proc_status()}")
    if LOG_PATH.exists():
        log_age = time.time() - LOG_PATH.stat().st_mtime
        print(f"  Log last updated: {fmt_eta(log_age)} ago")

    # Latest tqdm line
    tq = latest_tqdm_line()
    if tq:
        print(f"  Latest progress: {tq[:120]}")

    # Metrics
    steps, epochs, audits = parse_metrics()
    print(f"\n  metrics.jsonl: {len(steps)} step records, {len(epochs)} epochs, {len(audits)} audits")

    # Gradient ratio trajectory
    grad_steps = [r for r in steps if "grad_ratios" in r]
    if grad_steps:
        last10 = grad_steps[-10:]
        ratios = [r["grad_ratios"]["smooth/total"] * 100 for r in last10]
        ces    = [r["losses"]["ce"] for r in last10]
        print(f"\n  {b('Gradient ratios')} (last {len(last10)} measurements):")
        for r, ce in zip(last10, ces):
            ratio = r["grad_ratios"]["smooth/total"] * 100
            color = green if 10 <= ratio <= 25 else yellow if ratio <= 35 else red
            print(f"    step {r['step']:5d}  ce={ce:5.2f}  smooth/total={color(f'{ratio:5.1f}%')}")
        avg = sum(ratios) / len(ratios)
        col = green if 10 <= avg <= 25 else yellow if avg <= 35 else red
        print(f"    {b('mean')}: {col(f'{avg:5.1f}%')}  (target 10-25%)")

    # Epoch trend
    if epochs:
        print(f"\n  {b('Epoch summary')} (per-level val_CE, accuracy):")
        for r in epochs[-8:]:
            v = r["val"]; t = r["train"]
            print(
                f"    epoch {r['epoch']:3d}  elapsed={r['elapsed_sec']/60:4.1f}m  "
                f"train_CE={t['ce']:.3f}  val_CE={v['ce_total']:.3f}  "
                f"L0={v.get('ce_L0',0):.2f} L1={v.get('ce_L1',0):.2f} "
                f"L2={v.get('ce_L2',0):.2f} L3={v.get('ce_L3',0):.2f}  "
                f"acc={v.get('acc_L0',0)*100:.0f}/{v.get('acc_L1',0)*100:.0f}/"
                f"{v.get('acc_L2',0)*100:.0f}/{v.get('acc_L3',0)*100:.0f}%"
            )
        v4_baseline = 4.07
        latest_val = epochs[-1]["val"]["ce_total"]
        if latest_val < v4_baseline:
            print(f"    {green(f'  → val_CE {latest_val:.3f} BEATS v4 baseline {v4_baseline:.2f}')}")
        else:
            gap = latest_val - v4_baseline
            print(f"    {dim(f'  → val_CE {latest_val:.3f} vs v4 {v4_baseline:.2f} (gap {gap:+.3f})')}")

    # Codebook usage
    cb_audits = [a for a in audits if a["name"] == "style_codebook_usage"]
    if cb_audits:
        print(f"\n  {b('Style codebook usage')}:")
        for a in cb_audits[-6:]:
            p = a["payload"]
            active = p["active"]
            active_pct = active / 512 * 100
            col = green if active_pct > 50 else yellow if active_pct > 20 else red
            colored_active = col(f"{active}/512")
            print(f"    epoch {a['epoch']:3d}  active={colored_active} ({active_pct:.0f}%)  "
                  f"perplexity={p['perplexity']:5.1f}  top10_cov={p['top10_coverage']*100:.1f}%")

    # Latest checkpoint
    ckpt_path = CKPT_DIR / "transformer_best.pt"
    if ckpt_path.exists():
        try:
            import torch
            c = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            print(f"\n  {b('Latest best checkpoint')}: epoch={c['epoch']}  "
                  f"val_loss={c['val_loss']:.4f}  step={c['global_step']}")
        except Exception as e:
            print(f"\n  Latest checkpoint: (load failed — {e})")
    else:
        print(f"\n  {dim('No checkpoint saved yet (first epoch hasnt finished)')}")

    # ETA
    if epochs:
        e_per_epoch = sum(r["elapsed_sec"] for r in epochs[-3:]) / min(3, len(epochs))
        epochs_left = TARGET_EPOCHS - epochs[-1]["epoch"]
        eta = epochs_left * e_per_epoch
        print(f"\n  ETA: {fmt_eta(eta)}  ({epochs_left} epochs × {e_per_epoch/60:.1f}m/epoch)")

    print()


if __name__ == "__main__":
    main()
