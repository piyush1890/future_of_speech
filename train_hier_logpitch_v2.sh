#!/bin/zsh
# Train log-pitch hier transformer on the expanded 82k dataset (v2).

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Log-pitch Hier Transformer v2 (82k dataset)"
echo "Started at: $(date)"
echo "============================================"

python -u training/train_transformer_rvq_hier.py \
    --features-dir data/features_merged_logpitch_v2 \
    --phonemes-path data/processed_merged_v2/phonemes_mfa.json \
    --alignments-path data/processed_merged_v2/alignments_mfa.json \
    --vq-tokens-dir data/rvq_tokens_logpitch_v2 \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_rvq_logpitch_hier_v2 \
    --resume \
    --device mps \
    --epochs 50 \
    --batch-size 8 \
    --lr 3e-4 \
    --dropout 0.1 \
    --num-layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --weight-decay 0.01 \
    --warmup-steps 2000 \
    --num-quantizers 4 \
    --codebook-size 512

echo "Finished at: $(date)"
