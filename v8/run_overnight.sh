#!/bin/zsh
# v8 overnight pipeline (two-stage):
#   1. Extract per-phoneme z from full dataset (v5 frozen style encoder)  [~3 min]
#   2. Re-extract phoneme anchors (idempotent)                            [~2 min]
#   3. Train v8 stage 1: encoder + heads, fed GT-z directly  (~1 h MPS)
#   4. Train v8 stage 2: planner predicting z from text+knobs (~1 h MPS)
#   5. Render audio matrix at multiple V/A/D + speakers
#
# Total wall time estimate: ~2.5 h on MPS.
# Run: ./v8/run_overnight.sh

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="v8/logs/overnight_${TS}"
OUT_DIR="v8/outputs/overnight_${TS}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

run_step() {
    local name="$1"; shift
    echo ""
    echo "============================================"
    echo "[$(date)] STEP: $name"
    echo "============================================"
    "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
    echo "[$(date)] DONE: $name"
}

# ─── Step 1: extract per-phoneme z's ─────────────────────
run_step "01_extract_z" \
    python -u v8/scripts/extract_per_phoneme_z.py

# ─── Step 2: anchors (idempotent — skip if exist) ───────
if [[ ! -f "v8/data/phoneme_anchors/_complete" ]]; then
    run_step "02_extract_anchors" \
        python -u v8/scripts/extract_phoneme_anchors.py
    touch "v8/data/phoneme_anchors/_complete"
else
    echo "[$(date)] STEP: 02_extract_anchors — already done, skipping"
fi

# ─── Step 3: train v8 stage 1 (encoder + heads, GT-z fed) ────
run_step "03_train_stage1" \
    python -u v8/training/train_v8_stage1.py \
        --device mps \
        --epochs 15 \
        --batch-size 32 \
        --warmup-steps 500 \
        --preload \
        --log-every 200 \
        --render-mode linear \
        --checkpoint-dir v8/checkpoints/stage1_z

# ─── Step 4: train v8 stage 2 (planner predicts z) ───────
run_step "04_train_stage2" \
    python -u v8/training/train_v8_stage2.py \
        --device mps \
        --epochs 15 \
        --batch-size 32 \
        --warmup-steps 500 \
        --preload \
        --log-every 200 \
        --checkpoint-dir v8/checkpoints/stage2_planner

# ─── Step 5: render A/B audio matrix ─────────────────────
TEXT="You know, I tried that the other day, and it actually made me feel a lot better."
S1_CKPT="v8/checkpoints/stage1_z/best.pt"
S2_CKPT="v8/checkpoints/stage2_planner/best.pt"

render() {
    local label="$1" v="$2" a="$3" d="$4" cfg="$5" spk="$6"
    local args="--stage1-checkpoint $S1_CKPT --planner-checkpoint $S2_CKPT --valence $v --arousal $a --dominance $d --cfg-scale $cfg --device cpu --output ${OUT_DIR}/${label}.wav"
    if [[ -n "$spk" ]]; then args="$args --speaker-emb $spk"; fi
    echo "[$(date)] render: $label  V=$v A=$a D=$d  cfg=$cfg  spk=${spk:-default}"
    python -u v8/inference/synthesize_v8_2stage.py "$TEXT" $=args 2>&1 | tail -3 | tee -a "$LOG_DIR/05_render.log"
}

# Default speaker, varied V/A/D
render "default_neutral_05"     0.5 0.5 0.5 1.0 ""
render "default_high_07"        0.8 0.8 0.5 1.0 ""
render "default_low_03"         0.2 0.2 0.3 1.0 ""

# Sharpened CFG
render "default_high_07_cfg3"   0.8 0.8 0.5 3.0 ""
render "default_low_03_cfg3"    0.2 0.2 0.3 3.0 ""

# Also re-render the diagnostic at the GT-z target (oracle path) to check
# stage 1 is actually using z's properly. Skipped here — needs known utt.

# RDJ speaker (voice cloning)
RDJ_SPK="reference_audio/rdj/rdj_spk_emb.npy"
if [[ -f "$RDJ_SPK" ]]; then
    render "rdj_neutral_05"       0.5 0.5 0.5 1.0 "$RDJ_SPK"
    render "rdj_high_07_cfg3"     0.8 0.8 0.5 3.0 "$RDJ_SPK"
    render "rdj_low_03_cfg3"      0.2 0.2 0.3 3.0 "$RDJ_SPK"
fi

echo ""
echo "============================================"
echo "[$(date)] OVERNIGHT PIPELINE COMPLETE"
echo "  Logs:          $LOG_DIR/"
echo "  Audio:         $OUT_DIR/"
echo "  Stage1 ckpt:   v8/checkpoints/stage1_z/best.pt"
echo "  Stage2 ckpt:   v8/checkpoints/stage2_planner/best.pt"
echo "============================================"
