#!/bin/zsh
# Train the transformer overnight. Run from ~/projects/articulatory-tts/
# Usage: ./train_overnight.sh

eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Starting transformer training (200 epochs)"
echo "Estimated time: ~6-7 hours on M4 Pro"
echo "============================================"

python training/train_transformer.py \
    --device mps \
    --epochs 200 \
    --batch-size 8 \
    --lr 1e-3 \
    --warmup-steps 1000

echo ""
echo "============================================"
echo "Training complete! Testing inference..."
echo "============================================"

# Quick test: synthesize a sentence
python -c "
import sys
sys.path.insert(0, '.')
from inference.synthesize import ArticulatoryTTS
import numpy as np
import soundfile as sf

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints/vq_best.pt',
    transformer_checkpoint='checkpoints/transformer_best.pt',
    vocab_path='data/processed/vocab.json',
    norm_stats_path='data/features/norm_stats.npz',
    device='cpu',
)

# Use a speaker embedding from training data
import json
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
    out_path = f'outputs/test_{i}.wav'
    sf.write(out_path, wav, sr)
    print(f'Saved {out_path} ({len(wav)/sr:.2f}s): \"{text}\"')

print()
print('Done! Listen to the outputs/ directory.')
"
