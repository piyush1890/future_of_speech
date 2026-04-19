#!/bin/zsh
# Full training pipeline on 31K utterances with MFA-native alignment.
# Run this and go to sleep.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Full 31K Dataset Training Pipeline"
echo "Started at: $(date)"
echo "============================================"

# ── Step 1: Train VQ (if not already done) ──
if [ ! -f checkpoints_all/vq_best.pt ]; then
    echo ""
    echo "[Step 1/4] Training VQ tokenizer..."
    python training/train_vq.py \
        --features-dir data/features_all \
        --checkpoint-dir checkpoints_all \
        --device mps --epochs 100
else
    echo "[Step 1/4] VQ already trained, skipping."
fi

# ── Step 2: Tokenize features ──
echo ""
echo "[Step 2/4] Tokenizing features..."
python training/tokenize_features.py \
    --features-dir data/features_all \
    --checkpoint checkpoints_all/vq_best.pt \
    --output-dir data/vq_tokens_all \
    --device mps

# ── Step 3: Train transformer ──
echo ""
echo "[Step 3/4] Training transformer on 31K utterances..."
echo "Started at: $(date)"

python training/train_transformer.py \
    --features-dir data/features_all \
    --phonemes-path data/processed_all/phonemes_mfa.json \
    --alignments-path data/processed_all/alignments_mfa.json \
    --vq-tokens-dir data/vq_tokens_all \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_all \
    --device mps \
    --epochs 50 \
    --batch-size 8 \
    --lr 5e-4 \
    --dropout 0.1 \
    --num-layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --weight-decay 0.01 \
    --warmup-steps 2000 \
    --resume

# ── Step 4: Generate test audio ──
echo ""
echo "[Step 4/4] Generating test audio..."

python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints_all/vq_best.pt',
    transformer_checkpoint='checkpoints_all/transformer_best.pt',
    vocab_path='data/processed_all/vocab_mfa.json',
    norm_stats_path='data/features_all/norm_stats.npz',
    device='cpu',
)

with open('data/features_all/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

for i, text in enumerate([
    'Hello, how are you today?',
    'The quick brown fox jumps over the lazy dog.',
    'This speech was generated using articulatory tokens.',
    'Nobody has ever tried this approach before.',
    'Welcome to the future of text to speech technology.',
]):
    wav, sr = tts.synthesize(text, speaker_emb=spk_emb)
    sf.write(f'outputs/test_full_{i}.wav', wav, sr)
    print(f'Saved test_full_{i}.wav ({len(wav)/sr:.2f}s): \"{text}\"')

print('\nDone! Listen to outputs/test_full_*.wav')
"

echo ""
echo "============================================"
echo "ALL DONE at $(date)"
echo "============================================"
