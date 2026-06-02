#!/bin/zsh
set -euo pipefail

cd /Users/piyush/projects/articulatory-tts
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export PYTORCH_ENABLE_MPS_FALLBACK=0

echo "Starting classic romance batch at $(date -Iseconds)"
echo "Logs: /Users/piyush/projects/articulatory-tts/data/batch_logs/classic_romance_overnight"
echo "Progress: /Users/piyush/projects/articulatory-tts/data/licensed_hindi_classic_romance_batch_full_clean_guarded/batch_progress.tsv"
echo

.venv_dialogue_dataset/bin/python scripts/run_overnight_classic_romance_movies.py
status=$?

echo
echo "CLASSIC_ROMANCE_BATCH_EXITED_WITH_STATUS=$status"
echo "Finished at $(date -Iseconds)"
echo
echo "You can close this Terminal window now."
exec $SHELL
