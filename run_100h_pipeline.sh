#!/bin/zsh
# Full pipeline for train-clean-100 (100 hours).
# Run after download completes.
#
# Usage:
#   cd ~/projects/articulatory-tts
#   ./run_100h_pipeline.sh
#
# Expected total time: ~8-12 hours on M4 Pro

set -e  # Exit on error

eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Articulatory TTS — 100h Pipeline"
echo "Started at: $(date)"
echo "============================================"

# ── Step 0: Check download is complete ──
if [ ! -f data/train-clean-100.tar.gz ]; then
    echo "ERROR: data/train-clean-100.tar.gz not found. Download it first:"
    echo "  cd data && curl -L -o train-clean-100.tar.gz http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    exit 1
fi

# ── Step 1: Extract if not already done ──
if [ ! -d data/LibriSpeech/train-clean-100 ]; then
    echo ""
    echo "[Step 1/6] Extracting train-clean-100..."
    cd data && tar xzf train-clean-100.tar.gz && cd ..
    echo "Extracted. $(find data/LibriSpeech/train-clean-100 -name '*.flac' | wc -l) .flac files"
else
    echo "[Step 1/6] Already extracted."
fi

# ── Step 2: SPARC encode audio (THE SLOW PART) ──
echo ""
echo "[Step 2/6] SPARC encoding 10K files (~14 hours on MPS)..."
echo "Started at: $(date)"

python data/preprocess.py \
    --data-dir data/LibriSpeech/train-clean-100 \
    --output-dir data/features_100h \
    --device mps \
    --max-files 10000

echo "SPARC encoding done at: $(date)"

# ── Step 3: Phonemize transcripts ──
echo ""
echo "[Step 3/6] Phonemizing transcripts..."

python data/phonemize.py \
    --data-dir data/LibriSpeech/train-clean-100 \
    --output-dir data/processed_100h \
    --features-dir data/features_100h

# ── Step 4: Duration alignment ──
echo ""
echo "[Step 4/6] Computing duration alignments..."

python data/align.py \
    --features-dir data/features_100h \
    --phonemes-path data/processed_100h/phonemes.json \
    --output-path data/processed_100h/alignments.json

# ── Step 5: Retrain VQ on larger dataset ──
echo ""
echo "[Step 5/6] Training VQ tokenizer on 100h data..."

python training/train_vq.py \
    --features-dir data/features_100h \
    --checkpoint-dir checkpoints_100h \
    --device mps \
    --epochs 100

# Tokenize all features
echo ""
echo "Tokenizing features..."

python training/tokenize_features.py \
    --features-dir data/features_100h \
    --checkpoint checkpoints_100h/vq_best.pt \
    --output-dir data/vq_tokens_100h \
    --device mps

# ── Step 6: Train transformer (with more regularization) ──
echo ""
echo "[Step 6/6] Training transformer on 100h data..."
echo "Started at: $(date)"

python training/train_transformer.py \
    --features-dir data/features_100h \
    --phonemes-path data/processed_100h/phonemes.json \
    --alignments-path data/processed_100h/alignments.json \
    --vq-tokens-dir data/vq_tokens_100h \
    --vocab-path data/processed_100h/vocab.json \
    --checkpoint-dir checkpoints_100h \
    --device mps \
    --epochs 200 \
    --batch-size 8 \
    --lr 1e-3 \
    --dropout 0.2 \
    --warmup-steps 2000

echo ""
echo "============================================"
echo "Training complete at: $(date)"
echo "============================================"

# ── Test inference ──
echo ""
echo "Generating test audio..."

python -c "
import sys, json
sys.path.insert(0, '.')
import numpy as np
import soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints_100h/vq_best.pt',
    transformer_checkpoint='checkpoints_100h/transformer_best.pt',
    vocab_path='data/processed_100h/vocab.json',
    norm_stats_path='data/features_100h/norm_stats.npz',
    device='cpu',
)

with open('data/features_100h/speaker_embeddings.json') as f:
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
    out_path = f'outputs/test_100h_{i}.wav'
    sf.write(out_path, wav, sr)
    print(f'Saved {out_path} ({len(wav)/sr:.2f}s): \"{text}\"')

print()
print('Done! Listen to the outputs/ directory.')
"

echo ""
echo "============================================"
echo "ALL DONE! Check outputs/ for generated audio."
echo "============================================"
