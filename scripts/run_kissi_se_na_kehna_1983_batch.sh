#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/piyush/projects/articulatory-tts"
PY="/Users/piyush/miniconda3/envs/arttts/bin/python"
YTDLP="/Users/piyush/miniconda3/envs/arttts/bin/yt-dlp"
DEMUCS="/Users/piyush/miniconda3/envs/arttts/bin/demucs"

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

MANIFEST="$ROOT/data/licensed_hindi_kissi_se_na_kehna_1983/source/kissi_se_na_kehna_1983_nh_comedy_duniya.jsonl"
OUT_ROOT="$ROOT/data/licensed_hindi_kissi_se_na_kehna_1983"
LOG_DIR="$ROOT/data/batch_logs"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "=== Kissi Se Na Kehna batch started: $(date) ==="
echo "Manifest: $MANIFEST"

"$PY" "$ROOT/scripts/run_hindi_dialogue_episode_pipeline.py" \
  --url-list "$MANIFEST" \
  --out-root "$OUT_ROOT" \
  --project-root "$ROOT" \
  --python "$PY" \
  --ytdlp "$YTDLP" \
  --demucs "$DEMUCS" \
  --continue-on-error

echo "=== Kissi Se Na Kehna batch ended: $(date) ==="
