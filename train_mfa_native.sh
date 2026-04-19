#!/bin/zsh
# Train with MFA-native phonemes (phonemes built FROM MFA, not matched to g2p)
# This gives perfect alignment by construction.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "MFA-Native Training Pipeline"
echo "Started at: $(date)"
echo "============================================"

# ── Step 1: Tokenize features (VQ already trained) ──
echo "[Step 1/3] Tokenizing features for MFA dataset..."

python training/tokenize_features.py \
    --features-dir data/features_combined \
    --checkpoint checkpoints_combined/vq_best.pt \
    --output-dir data/vq_tokens_combined \
    --device mps

# ── Step 2: Train transformer ──
echo ""
echo "[Step 2/3] Training transformer with MFA-native alignment..."
echo "Started at: $(date)"

python training/train_transformer.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_mfa/phonemes_mfa.json \
    --alignments-path data/processed_mfa/alignments_mfa.json \
    --vq-tokens-dir data/vq_tokens_combined \
    --vocab-path data/processed_mfa/vocab_mfa.json \
    --checkpoint-dir checkpoints_mfa_native \
    --device mps \
    --epochs 300 \
    --batch-size 8 \
    --lr 5e-4 \
    --dropout 0.2 \
    --num-layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --weight-decay 0.02 \
    --warmup-steps 1500

# ── Step 3: Generate audio ──
echo ""
echo "[Step 3/3] Generating test audio..."

python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints_combined/vq_best.pt',
    transformer_checkpoint='checkpoints_mfa_native/transformer_best.pt',
    vocab_path='data/processed_mfa/vocab_mfa.json',
    norm_stats_path='data/features_combined/norm_stats.npz',
    device='cpu',
)

with open('data/features_combined/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

for i, text in enumerate([
    'Hello, how are you today?',
    'The quick brown fox jumps over the lazy dog.',
    'This speech was generated using articulatory tokens.',
]):
    wav, sr = tts.synthesize(text, speaker_emb=spk_emb)
    sf.write(f'outputs/test_mfa_native_{i}.wav', wav, sr)
    print(f'Saved test_mfa_native_{i}.wav ({len(wav)/sr:.2f}s)')

print('\nDone! Listen to outputs/test_mfa_native_*.wav')
"

echo ""
echo "============================================"
echo "ALL DONE at $(date)"
echo "============================================"
