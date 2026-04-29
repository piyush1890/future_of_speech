#!/bin/zsh
# Train hierarchical-RVQ transformer on the pitch-normalized 55K dataset.
# Fresh training — no init-from, no resume on first run. The feature distribution
# is different from un-normalized (pitch is now zero-mean-unit-var per speaker),
# so the flat e28 checkpoint would be a misleading initialization.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Hierarchical RVQ Training — Pitch-Normalized"
echo "Started at: $(date)"
echo "============================================"

mkdir -p checkpoints_rvq_pitchnorm_hier

python training/train_transformer_rvq_hier.py \
    --features-dir data/features_merged_pitchnorm \
    --phonemes-path data/processed_merged/phonemes_mfa.json \
    --alignments-path data/processed_merged/alignments_mfa.json \
    --vq-tokens-dir data/rvq_tokens_pitchnorm \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_rvq_pitchnorm_hier \
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
