#!/bin/zsh
# v5 planner (stage 2) — predicts per-phoneme style codes from text + knobs.
# GT codes derived from frozen v5 stage 1 archived checkpoint.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

echo "============================================"
echo "v5 planner (stage 2) — start"
echo "Started at: $(date)"
echo "============================================"

python -u training/train_v5_stage2.py \
    --stage1-checkpoint  checkpoints_v5_stage1_archived/transformer_best.pt \
    --metadata-path      data/utterance_metadata_v5.json \
    --checkpoint-dir     checkpoints_v5_planner \
    --device             mps \
    --epochs             10 \
    --batch-size         32 \
    --lr                 5e-4 \
    --weight-decay       0.01 \
    --warmup-steps       500 \
    --d-model            128 \
    --nhead              4 \
    --num-layers         4 \
    --d-ff               512 \
    --dropout            0.1 \
    --knob-dropout       0.1 \
    --preload

echo "Finished at: $(date)"
