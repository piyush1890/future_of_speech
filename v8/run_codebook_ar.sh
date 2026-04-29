#!/bin/zsh
# v8 codebook AR pipeline:
#   1. Quantize z's via v5 codebook → code IDs + z_q
#   2. Re-train stage 1 with z_q (quantized z's, matching inference distribution)
#   3. Train stage 2 codebook AR planner (CE on code IDs, AR + sampling)
#   4. Render audio with comma-pauses + AR sampling

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="v8/logs/cb_ar_${TS}"
OUT_DIR="v8/outputs/cb_ar_${TS}"
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

# Step 1: quantize z's (~3 min)
run_step "01_quantize_z" \
    python -u v8/scripts/quantize_z_via_v5_codebook.py

# Step 2: stage 1 retrain with z_q (~1.5h MPS)
run_step "02_stage1_zq" \
    python -u v8/training/train_v8_stage1.py \
        --device mps \
        --epochs 15 \
        --batch-size 32 \
        --warmup-steps 500 \
        --preload \
        --log-every 200 \
        --render-mode linear \
        --use-quantized-z \
        --checkpoint-dir v8/checkpoints/stage1_zq

# Step 3: codebook AR planner (~1h MPS)
run_step "03_stage2_cb_ar" \
    python -u v8/training/train_v8_stage2_codebook.py \
        --device mps \
        --epochs 15 \
        --batch-size 32 \
        --warmup-steps 500 \
        --preload \
        --log-every 200 \
        --checkpoint-dir v8/checkpoints/stage2_cb_ar

# Step 4: renders
TEXT="You know, I tried that the other day, and it actually made me feel a lot better."
S1="v8/checkpoints/stage1_zq/best.pt"
S2="v8/checkpoints/stage2_cb_ar/best.pt"

render() {
    local label="$1" v="$2" a="$3" d="$4" temp="$5" cfg="$6" spk="$7"
    local args="--stage1-checkpoint $S1 --planner-checkpoint $S2 --valence $v --arousal $a --dominance $d --cfg-scale $cfg --device cpu --output ${OUT_DIR}/${label}.wav"
    if [[ -n "$spk" ]]; then args="$args --speaker-emb $spk"; fi
    echo "[$(date)] render: $label  V=$v A=$a D=$d  temp=$temp  cfg=$cfg"
    python -u v8/inference/synthesize_v8_2stage.py "$TEXT" $=args 2>&1 | tail -3 | tee -a "$LOG_DIR/04_render.log"
}

# Default speaker
render "default_neutral_temp1"  0.5 0.5 0.5 1.0 1.0 ""
render "default_high_temp1"     0.8 0.8 0.5 1.0 1.0 ""
render "default_low_temp1"      0.2 0.2 0.3 1.0 1.0 ""
render "default_high_temp07"    0.8 0.8 0.5 0.7 1.0 ""
render "default_low_temp07"     0.2 0.2 0.3 0.7 1.0 ""

RDJ_SPK="reference_audio/rdj/rdj_spk_emb.npy"
if [[ -f "$RDJ_SPK" ]]; then
    render "rdj_neutral_temp1"  0.5 0.5 0.5 1.0 1.0 "$RDJ_SPK"
    render "rdj_high_temp1"     0.8 0.8 0.5 1.0 1.0 "$RDJ_SPK"
fi

echo ""
echo "============================================"
echo "[$(date)] CODEBOOK AR PIPELINE COMPLETE"
echo "  Logs: $LOG_DIR/"
echo "  Audio: $OUT_DIR/"
echo "============================================"
