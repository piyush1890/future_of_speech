#!/bin/zsh
# Train autoregressive transformer with Whisper alignments
#
# Usage: cd ~/projects/articulatory-tts && ./train_ar.sh

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Autoregressive TTS Transformer Training"
echo "Using Whisper alignments (our best so far)"
echo "Started at: $(date)"
echo "============================================"

python training/train_ar.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --alignments-path data/processed_combined/alignments_whisper.json \
    --vq-tokens-dir data/vq_tokens_combined \
    --vocab-path data/processed_combined/vocab.json \
    --checkpoint-dir checkpoints_ar \
    --device mps \
    --epochs 200 \
    --batch-size 8 \
    --lr 5e-4 \
    --dropout 0.2 \
    --num-layers 4 \
    --d-model 256 \
    --d-ff 1024 \
    --weight-decay 0.02 \
    --warmup-steps 1500

echo ""
echo "============================================"
echo "Generating test audio..."
echo "============================================"

python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints_combined/vq_best.pt',
    transformer_checkpoint='checkpoints_ar/transformer_best.pt',
    vocab_path='data/processed_combined/vocab.json',
    norm_stats_path='data/features_combined/norm_stats.npz',
    device='cpu',
)

with open('data/features_combined/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

sentences = [
    'Hello, how are you today?',
    'The quick brown fox jumps over the lazy dog.',
    'This speech was generated using articulatory tokens.',
]

for i, text in enumerate(sentences):
    wav, sr = tts.synthesize(text, speaker_emb=spk_emb)
    sf.write(f'outputs/test_ar_{i}.wav', wav, sr)
    print(f'Saved test_ar_{i}.wav ({len(wav)/sr:.2f}s): \"{text}\"')

print('\nDone! Listen to outputs/test_ar_*.wav')
"
