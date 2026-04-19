#!/bin/zsh
# Train on combined dev-clean + 100h data (~5077 utterances)
#
# Usage:
#   cd ~/projects/articulatory-tts && ./train_combined.sh

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Combined training: dev-clean + 100h subset"
echo "Started at: $(date)"
echo "============================================"

# ── Step 1: Merge features into combined dir ──
echo ""
echo "[Step 1/5] Merging feature files..."

mkdir -p data/features_combined

# Symlink all dev-clean features
for f in data/features/*.npz; do
    base=$(basename "$f")
    if [ "$base" != "norm_stats.npz" ]; then
        ln -sf "$(pwd)/$f" "data/features_combined/$base" 2>/dev/null
    fi
done

# Symlink all 100h features
for f in data/features_100h/*.npz; do
    base=$(basename "$f")
    if [ "$base" != "norm_stats.npz" ]; then
        ln -sf "$(pwd)/$f" "data/features_combined/$base" 2>/dev/null
    fi
done

# Also copy speaker embeddings from both
python -c "
import json, numpy as np
from pathlib import Path

spk = {}
for p in ['data/features/speaker_embeddings.json', 'data/features_100h/speaker_embeddings.json']:
    if Path(p).exists():
        with open(p) as f:
            spk.update(json.load(f))

with open('data/features_combined/speaker_embeddings.json', 'w') as f:
    json.dump(spk, f)
print(f'Combined speakers: {len(spk)}')
"

TOTAL=$(ls data/features_combined/*.npz 2>/dev/null | wc -l)
echo "Total feature files: $TOTAL"

# ── Step 2: Compute combined norm stats ──
echo ""
echo "[Step 2/5] Computing normalization stats..."

python -c "
import numpy as np
from pathlib import Path

all_feat = []
for p in sorted(Path('data/features_combined').glob('*.npz')):
    if p.stem == 'norm_stats': continue
    d = np.load(p)
    pitch = d['pitch'][:, None] if d['pitch'].ndim == 1 else d['pitch']
    loud = d['loudness'][:, None] if d['loudness'].ndim == 1 else d['loudness']
    all_feat.append(np.concatenate([d['ema'], pitch, loud], axis=-1))

all_feat = np.concatenate(all_feat, axis=0)
mean, std = all_feat.mean(0), all_feat.std(0)
std[std < 1e-6] = 1.0
np.savez('data/features_combined/norm_stats.npz', mean=mean.astype(np.float32), std=std.astype(np.float32))
print(f'Frames: {all_feat.shape[0]:,} ({all_feat.shape[0]/50/3600:.1f} hours at 50Hz)')
print(f'Mean: {mean}')
print(f'Std: {std}')
"

# ── Step 3: Phonemize + align for combined data ──
echo ""
echo "[Step 3/5] Phonemizing and aligning..."

mkdir -p data/processed_combined

# Phonemize train-clean-100 transcripts (dev-clean already done)
python data/phonemize.py \
    --data-dir data/LibriSpeech/train-clean-100 \
    --output-dir data/processed_combined \
    --features-dir data/features_combined

# Merge with dev-clean phonemes
python -c "
import json

combined = {}
for p in ['data/processed/phonemes.json', 'data/processed_combined/phonemes.json']:
    with open(p) as f:
        combined.update(json.load(f))

with open('data/processed_combined/phonemes.json', 'w') as f:
    json.dump(combined, f)
print(f'Combined phoneme entries: {len(combined)}')
"

# Also merge vocab (take the larger one)
python -c "
import json

vocabs = []
for p in ['data/processed/vocab.json', 'data/processed_combined/vocab.json']:
    with open(p) as f:
        vocabs.append(json.load(f))

# Merge: keep all tokens from both
merged = dict(vocabs[0])
for k, v in vocabs[1].items():
    if k not in merged:
        merged[k] = len(merged)

with open('data/processed_combined/vocab.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Combined vocab size: {len(merged)}')
"

# Align durations
python data/align.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --output-path data/processed_combined/alignments.json

# ── Step 4: Train VQ on combined data ──
echo ""
echo "[Step 4/5] Training VQ tokenizer on combined data..."

python training/train_vq.py \
    --features-dir data/features_combined \
    --checkpoint-dir checkpoints_combined \
    --device mps \
    --epochs 100

echo ""
echo "Tokenizing features..."

python training/tokenize_features.py \
    --features-dir data/features_combined \
    --checkpoint checkpoints_combined/vq_best.pt \
    --output-dir data/vq_tokens_combined \
    --device mps

# ── Step 5: Train transformer ──
echo ""
echo "[Step 5/5] Training transformer on combined data..."
echo "Started at: $(date)"

python training/train_transformer.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --alignments-path data/processed_combined/alignments.json \
    --vq-tokens-dir data/vq_tokens_combined \
    --vocab-path data/processed_combined/vocab.json \
    --checkpoint-dir checkpoints_combined \
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
echo "Training complete at: $(date)"
echo "============================================"

# ── Generate test audio ──
echo ""
echo "Generating test audio..."

python -c "
import sys, json
sys.path.insert(0, '.')
import numpy as np
import soundfile as sf
from inference.synthesize import ArticulatoryTTS

tts = ArticulatoryTTS(
    vq_checkpoint='checkpoints_combined/vq_best.pt',
    transformer_checkpoint='checkpoints_combined/transformer_best.pt',
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
    out_path = f'outputs/test_combined_{i}.wav'
    sf.write(out_path, wav, sr)
    print(f'Saved {out_path} ({len(wav)/sr:.2f}s): \"{text}\"')

print()
print('Done! Listen to outputs/test_combined_*.wav')
"
