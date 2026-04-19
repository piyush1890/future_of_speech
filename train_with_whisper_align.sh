#!/bin/zsh
# 1. Run Whisper alignment on combined data
# 2. Retrain transformer with proper alignments
#
# Usage: cd ~/projects/articulatory-tts && ./train_with_whisper_align.sh

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Step 1: Whisper alignment (~1-2 hours)"
echo "Started at: $(date)"
echo "============================================"

python data/align_whisper.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --audio-dirs data/LibriSpeech/dev-clean data/LibriSpeech/train-clean-100 \
    --output-path data/processed_combined/alignments_whisper.json \
    --whisper-model tiny

echo ""
echo "============================================"
echo "Step 2: Retrain transformer with Whisper alignment"
echo "Started at: $(date)"
echo "============================================"

python training/train_transformer.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --alignments-path data/processed_combined/alignments_whisper.json \
    --vq-tokens-dir data/vq_tokens_combined \
    --vocab-path data/processed_combined/vocab.json \
    --checkpoint-dir checkpoints_whisper \
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

echo ""
echo "============================================"
echo "Step 3: Generate test audio"
echo "============================================"

python -c "
import sys, json
sys.path.insert(0, '.')
import numpy as np
import soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints_combined/vq_best.pt',
    transformer_checkpoint='checkpoints_whisper/transformer_best.pt',
    vocab_path='data/processed_combined/vocab.json',
    norm_stats_path='data/features_combined/norm_stats.npz',
    device='cpu',
)

with open('data/features_combined/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_id = list(spk_embs.keys())[0]
spk_emb = np.array(spk_embs[spk_id], dtype=np.float32)

sentences = [
    'Hello, how are you today?',
    'The quick brown fox jumps over the lazy dog.',
    'This speech was generated using articulatory tokens.',
    'Nobody has ever tried this approach before.',
]

for i, text in enumerate(sentences):
    wav, sr = tts.synthesize(text, speaker_emb=spk_emb)
    out_path = f'outputs/test_whisper_{i}.wav'
    sf.write(out_path, wav, sr)
    print(f'Saved {out_path} ({len(wav)/sr:.2f}s): \"{text}\"')

print('\\nDone! Listen to outputs/test_whisper_*.wav')
"

echo ""
echo "============================================"
echo "ALL DONE at $(date)"
echo "============================================"
