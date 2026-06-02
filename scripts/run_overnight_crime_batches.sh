#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/piyush/projects/articulatory-tts"
PY="/Users/piyush/miniconda3/envs/arttts/bin/python"
YTDLP="/Users/piyush/miniconda3/envs/arttts/bin/yt-dlp"
DEMUCS="/Users/piyush/miniconda3/envs/arttts/bin/demucs"

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

CRIMES_MANIFEST="$ROOT/data/owned_hindi_crimes_aaj_kal/source/crimes_aaj_kal_all_official_canonical.jsonl"
CRIME_PATROL_MANIFEST="$ROOT/data/owned_hindi_crime_patrol/source/newer_women_voice_priority_20.jsonl"
CRIMES_OUT="$ROOT/data/owned_hindi_crimes_aaj_kal"
CRIME_PATROL_OUT="$ROOT/data/owned_hindi_crime_patrol_newer_women"
LOG_DIR="$ROOT/data/batch_logs"
MASTER_LOG="$LOG_DIR/overnight_crimes_then_crime_patrol.log"
SUMMARY="$LOG_DIR/overnight_crimes_then_crime_patrol_summary.json"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "=== Overnight batch started: $(date) ==="
echo "Crimes Aaj Kal manifest: $CRIMES_MANIFEST"
echo "Crime Patrol manifest: $CRIME_PATROL_MANIFEST"

"$PY" "$ROOT/scripts/run_hindi_dialogue_episode_pipeline.py" \
  --url-list "$CRIMES_MANIFEST" \
  --out-root "$CRIMES_OUT" \
  --project-root "$ROOT" \
  --python "$PY" \
  --ytdlp "$YTDLP" \
  --demucs "$DEMUCS" \
  --continue-on-error

echo "=== Crimes Aaj Kal batch finished: $(date) ==="

"$PY" "$ROOT/scripts/run_hindi_dialogue_episode_pipeline.py" \
  --url-list "$CRIME_PATROL_MANIFEST" \
  --out-root "$CRIME_PATROL_OUT" \
  --project-root "$ROOT" \
  --python "$PY" \
  --ytdlp "$YTDLP" \
  --demucs "$DEMUCS" \
  --continue-on-error

echo "=== Crime Patrol newer women-priority batch finished: $(date) ==="

"$PY" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path("/Users/piyush/projects/articulatory-tts")
summary_path = root / "data/batch_logs/overnight_crimes_then_crime_patrol_summary.json"
batches = [
    ("crimes_aaj_kal", root / "data/owned_hindi_crimes_aaj_kal/batch_summary.json"),
    ("crime_patrol_newer_women", root / "data/owned_hindi_crime_patrol_newer_women/batch_summary.json"),
]

summary = {"batches": {}, "totals": {"episodes": 0, "failures": 0, "accepted_seconds": 0.0, "paired_seconds": 0.0, "train_seconds": 0.0}}
for name, path in batches:
    if not path.exists():
        summary["batches"][name] = {"missing": str(path)}
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    summary["batches"][name] = data
    summary["totals"]["episodes"] += int(data.get("episodes") or 0)
    summary["totals"]["failures"] += int(data.get("failures") or 0)
    for key in ("accepted_seconds", "paired_seconds", "train_seconds"):
        summary["totals"][key] += float(data.get(key) or 0.0)

for key in ("accepted_seconds", "paired_seconds", "train_seconds"):
    summary["totals"][key] = round(summary["totals"][key], 2)

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Summary: $SUMMARY"
echo "=== Overnight batch ended: $(date) ==="
