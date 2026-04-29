#!/bin/zsh
# v5 stage 1 — joint training of per-phoneme style encoder + style codebook +
# frame decoder + RVQ output heads. From scratch (architecture changed from v4).
#
# Watchdog-friendly: --resume picks up from checkpoints_v5_stage1/ if a checkpoint
# exists; otherwise starts a fresh run.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

echo "============================================"
echo "v5 stage 1 — per-phoneme style codes + planner-ready codebook"
echo "Started at: $(date)"
echo "============================================"

python -u training/train_v5_stage1.py \
    --features-dir       data/features_merged_logpitch_v2 \
    --phonemes-path      data/processed_merged_v3/phonemes_mfa.json \
    --alignments-path    data/processed_merged_v3/alignments_mfa.json \
    --vq-tokens-dir      data/rvq_tokens_logpitch_v3 \
    --vocab-path         data/processed_all/vocab_mfa.json \
    --metadata-path      data/utterance_metadata_v5.json \
    --checkpoint-dir     checkpoints_v5_stage1 \
    --rvq-checkpoint     checkpoints_rvq_logpitch_v2/rvq_best.pt \
    --resume \
    --device       mps \
    --epochs       50 \
    --batch-size   16 \
    --lr           3e-4 \
    --weight-decay 0.01 \
    --warmup-steps 2000 \
    --dur-weight   0.1 \
    --smooth-weight     0.10 \
    --vq-commit-weight  0.25 \
    --codebook-size       512 \
    --num-quantizers      4 \
    --style-codebook-size 512 \
    --d-model    256 \
    --d-ff       1024 \
    --num-layers 4 \
    --dropout    0.1 \
    --max-frames 800 \
    --preload \
    --scalar-every     50 \
    --grad-ratio-every 200

echo "Finished at: $(date)"
