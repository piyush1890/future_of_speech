#!/bin/zsh
# Train hierarchical-RVQ transformer on the merged 54K dataset.
# Preprocessing (features merge, MFA build, RVQ tokenize) is assumed done — see
# train_combined_360.sh for that pipeline.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Hierarchical RVQ Training"
echo "Started at: $(date)"
echo "============================================"

mkdir -p checkpoints_rvq_hier

# First run: init from flat e28 checkpoint. Subsequent runs will --resume from
# the hier checkpoint automatically (flat init skipped when resume_path exists).
INIT_FROM="checkpoints_rvq_merged/transformer_best_flat_e28.pt"

python training/train_transformer_rvq_hier.py \
    --features-dir data/features_merged \
    --phonemes-path data/processed_merged/phonemes_mfa.json \
    --alignments-path data/processed_merged/alignments_mfa.json \
    --vq-tokens-dir data/rvq_tokens_merged \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_rvq_hier \
    --init-from "$INIT_FROM" \
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

echo ""
echo "Done at: $(date)"
