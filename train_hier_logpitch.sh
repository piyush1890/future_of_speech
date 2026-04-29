#!/bin/zsh
# Train hierarchical-RVQ transformer on log-pitch features with per-utterance spk_emb.
# Fresh training from scratch — feature distribution differs from prior runs (log-scaled
# pitch channel), so no init-from. Dataset now uses per-utterance speaker embeddings
# loaded directly from each .npz (Concept 5) instead of the averaged speaker JSON.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Hierarchical RVQ Training — Log-pitch + Per-utterance spk_emb"
echo "Started at: $(date)"
echo "============================================"

mkdir -p checkpoints_rvq_logpitch_hier

python training/train_transformer_rvq_hier.py \
    --features-dir data/features_merged_logpitch \
    --phonemes-path data/processed_merged/phonemes_mfa.json \
    --alignments-path data/processed_merged/alignments_mfa.json \
    --vq-tokens-dir data/rvq_tokens_logpitch \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_rvq_logpitch_hier \
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
