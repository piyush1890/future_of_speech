#!/bin/zsh
# Retrain transformer on dev-clean with stronger regularization.
# The 100h encoding keeps running in the other terminal.
#
# Changes from first attempt:
#   - dropout 0.2 → 0.3
#   - 2 layers instead of 4 (smaller model = less overfitting)
#   - weight decay 0.01 → 0.05
#   - 500 epochs (cosine schedule means later epochs have very low LR)

eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Regularized training on dev-clean (2703 utts)"
echo "============================================"

python training/train_transformer.py \
    --features-dir data/features \
    --phonemes-path data/processed/phonemes.json \
    --alignments-path data/processed/alignments.json \
    --vq-tokens-dir data/vq_tokens \
    --vocab-path data/processed/vocab.json \
    --checkpoint-dir checkpoints_reg \
    --device mps \
    --epochs 500 \
    --batch-size 8 \
    --lr 5e-4 \
    --dropout 0.3 \
    --num-layers 2 \
    --d-ff 512 \
    --weight-decay 0.05 \
    --warmup-steps 500

echo ""
echo "============================================"
echo "Training complete! Testing inference..."
echo "============================================"

python -c "
import sys, json
sys.path.insert(0, '.')
import numpy as np
import soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints/vq_best.pt',
    transformer_checkpoint='checkpoints_reg/transformer_best.pt',
    vocab_path='data/processed/vocab.json',
    norm_stats_path='data/features/norm_stats.npz',
    device='cpu',
)

with open('data/features/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_id = list(spk_embs.keys())[0]
spk_emb = np.array(spk_embs[spk_id], dtype=np.float32)

sentences = [
    'Hello, how are you today?',
    'The quick brown fox jumps over the lazy dog.',
    'This speech was generated using articulatory tokens.',
]

for i, text in enumerate(sentences):
    wav, sr = tts.synthesize(text, speaker_emb=spk_emb)
    out_path = f'outputs/test_reg_{i}.wav'
    sf.write(out_path, wav, sr)
    print(f'Saved {out_path} ({len(wav)/sr:.2f}s): \"{text}\"')

print()
print('Done! Listen to outputs/test_reg_*.wav')
"
