#!/bin/zsh
# Watchdog: checks every 5 minutes if training is running, restarts if not.
# Run this in a separate terminal: ./watchdog.sh
# Or via launchd: com.articulatory-tts.watchdog.plist

eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

# Configure which pipeline we're training
SCRIPT="train_hier_v3_style_smooth.sh"
LOG="train_hier_v3_style_smooth.log"
CHECKPOINT="checkpoints_rvq_logpitch_hier_v3/transformer_best.pt"
TARGET_EPOCH=50

# Process patterns to detect running training (any of these means "busy, don't restart")
PATTERNS=("$SCRIPT" "train_transformer_rvq_hier.py" "train_transformer_rvq.py" "train_vq_rvq.py" "tokenize_features_rvq.py" "build_mfa_dataset.py")

echo "Watchdog started at $(date)"
echo "  Script:     $SCRIPT"
echo "  Log:        $LOG"
echo "  Checkpoint: $CHECKPOINT"
echo "  Target:     epoch $TARGET_EPOCH"

is_running() {
    for pat in "${PATTERNS[@]}"; do
        if pgrep -f "$pat" > /dev/null 2>&1; then
            return 0
        fi
    done
    return 1
}

get_best_epoch() {
    KMP_DUPLICATE_LIB_OK=TRUE python -c "
import torch
try:
    c = torch.load('$CHECKPOINT', map_location='cpu', weights_only=True)
    print(c['epoch'])
except:
    print(0)
" 2>/dev/null
}

while true; do
    if ! is_running; then
        CURRENT_EPOCH=$(get_best_epoch)

        if [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCH" ] 2>/dev/null; then
            echo "[$(date)] Training complete at epoch $CURRENT_EPOCH. Watchdog exiting."
            break
        fi

        # Safety: if no checkpoint exists at all, something is very wrong - don't restart blindly
        if [ ! -f "$CHECKPOINT" ] && [ "$CURRENT_EPOCH" -eq 0 ]; then
            # First ever run - let the script run to create checkpoint
            echo "[$(date)] No checkpoint yet (first run). Starting $SCRIPT..."
        else
            echo "[$(date)] Training not running. Best epoch: $CURRENT_EPOCH. Restarting $SCRIPT..."
        fi

        nohup ./$SCRIPT > $LOG 2>&1 &
        TRAIN_PID=$!
        echo "[$(date)] Restarted (PID: $TRAIN_PID)"
        sleep 30
    else
        RUNNING_PID=$(pgrep -f "train_transformer_rvq_hier.py" | head -1)
        BEST_EPOCH=$(get_best_epoch)
        if [ -n "$RUNNING_PID" ]; then
            echo "[$(date)] Training running (PID: $RUNNING_PID, best epoch so far: $BEST_EPOCH)"
        else
            echo "[$(date)] Preprocessing running (transformer not yet started)"
        fi
    fi

    sleep 300
done

echo "Watchdog finished at $(date)"
