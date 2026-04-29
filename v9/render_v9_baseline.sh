#!/bin/zsh
# Render 5 v9 baseline samples (single-stage predictor) for comparison vs v10.
set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

OUT_DIR="v9/outputs/baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/render.log"

TOK="v9/checkpoints/phoneme_rvq/best.pt"
PRED="v9/checkpoints/predictor/best.pt"

TEXT='You know, I tried that the other day, and it actually made me feel a lot better.'

render() {
    local label="$1" emo="$2" intensity="$3"
    local out="$OUT_DIR/${label}.wav"
    echo "==> $label  emo=$emo intensity=$intensity" | tee -a "$LOG"
    python -u v9/inference/synthesize_v9.py "$TEXT" \
        --tokenizer-checkpoint "$TOK" \
        --predictor-checkpoint "$PRED" \
        --emotion "$emo" --intensity "$intensity" \
        --temperature 1.0 --cfg-scale 1.0 \
        --device cpu --output "$out" 2>&1 | tee -a "$LOG" | tail -5
    echo "" | tee -a "$LOG"
}

render "v9_neutral_05"   neutral  0.5
render "v9_happy_08"     happy    0.8
render "v9_sad_07"       sad      0.7
render "v9_angry_09"     angry    0.9
render "v9_surprise_08"  surprise 0.8

echo "DONE — $OUT_DIR" | tee -a "$LOG"
ls -la "$OUT_DIR"
