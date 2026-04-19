#!/bin/zsh
# Train on combined 31K + new 23K + any Mac encoded files = ~54K total
# Resumes transformer from previous best checkpoint

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Combined Training Pipeline (31K + 23K + Mac)"
echo "Started at: $(date)"
echo "============================================"

# ── Step 1: Merge features into one directory ──
echo ""
echo "[Step 1/5] Merging features..."

mkdir -p data/features_merged

# Symlink feature files ONLY (exclude norm_stats.npz — handled in Step 3)
echo "Linking features_all (31K)..."
find data/features_all -maxdepth 1 -name '*.npz' ! -name 'norm_stats.npz' \
    -exec ln -sf "$PWD/{}" data/features_merged/ \; 2>/dev/null || true

echo "Linking features_360_colab (23K new)..."
find data/features_360_colab -maxdepth 1 -name '*.npz' ! -name 'norm_stats.npz' \
    -exec ln -sf "$PWD/{}" data/features_merged/ \; 2>/dev/null || true

if [ -d data/features_360_raw ]; then
    echo "Linking features_360_raw (Mac encoded)..."
    find data/features_360_raw -maxdepth 1 -name '*.npz' ! -name 'norm_stats.npz' \
        -exec ln -sf "$PWD/{}" data/features_merged/ \; 2>/dev/null || true
fi

# Clean up any stale norm_stats.npz symlink from previous runs
rm -f data/features_merged/norm_stats.npz

TOTAL=$(ls data/features_merged/ | grep -c '.npz')
echo "Total feature files: $TOTAL"

# ── Step 2: Build MFA dataset ──
echo ""
echo "[Step 2/5] Building MFA dataset..."

python data/build_mfa_dataset.py \
    --features-dir data/features_merged \
    --textgrid-dir data/mfa_alignments \
    --original-phonemes-path data/processed_combined/phonemes.json \
    --output-dir data/processed_merged \
    --existing-vocab data/processed_all/vocab_mfa.json

# ── Step 3: REUSE OLD norm stats (for RVQ/transformer consistency) + rebuild speaker embeddings ──
echo ""
echo "[Step 3/5] Reusing OLD norm stats + building speaker embeddings..."

python -c "
import numpy as np, json, shutil
from pathlib import Path

features_dir = Path('data/features_merged')

# REUSE old norm stats — critical for consistency with pretrained RVQ + transformer
old_stats = Path('data/features_all/norm_stats.npz').resolve()
new_stats = features_dir / 'norm_stats.npz'

# Remove any existing symlink (from the merge step) before copying real file
if new_stats.is_symlink() or new_stats.exists():
    new_stats.unlink()

print(f'Copying old norm stats: {old_stats} -> {new_stats}')
shutil.copy2(str(old_stats), str(new_stats))

stats = np.load(str(new_stats))
print(f'  mean[:4]: {stats[\"mean\"][:4]}')
print(f'  std[:4]:  {stats[\"std\"][:4]}')

# Build speaker embeddings over full merged dataset (doesn't affect tokenization)
print()
print('Building speaker embeddings over merged dataset...')
spk_embs = {}
for p in sorted(features_dir.glob('*.npz')):
    if p.stem == 'norm_stats':
        continue
    spk_id = p.stem.split('-')[0]
    d = np.load(str(p))
    if spk_id not in spk_embs:
        spk_embs[spk_id] = []
    spk_embs[spk_id].append(d['spk_emb'])

avg_spk = {k: np.mean(v, axis=0).tolist() for k, v in spk_embs.items()}
with open(features_dir / 'speaker_embeddings.json', 'w') as f:
    json.dump(avg_spk, f)
print(f'Speakers: {len(avg_spk)}')
"

# ── Step 4: Retokenize features with existing RVQ ──
echo ""
echo "[Step 4/5] Tokenizing features with RVQ..."

python training/tokenize_features_rvq.py \
    --features-dir data/features_merged \
    --checkpoint checkpoints_rvq/rvq_best.pt \
    --output-dir data/rvq_tokens_merged \
    --device mps

# ── Step 5: Resume transformer training from previous best ──
echo ""
echo "[Step 5/5] Resuming transformer training from previous best..."
echo "Started at: $(date)"

# Copy previous best ONLY on first run (empty merged dir).
# If already has a checkpoint (from prior merged training), keep it.
mkdir -p checkpoints_rvq_merged
if [ ! -f checkpoints_rvq_merged/transformer_best.pt ]; then
    echo "First merged run - seeding from checkpoints_rvq/transformer_best.pt"
    cp checkpoints_rvq/transformer_best.pt checkpoints_rvq_merged/transformer_best.pt
else
    echo "Existing merged checkpoint found - preserving it"
    python -c "
import torch
c = torch.load('checkpoints_rvq_merged/transformer_best.pt', map_location='cpu', weights_only=True)
print(f'  Current best: epoch={c[\"epoch\"]}, val_CE={c[\"val_loss\"]:.4f}')
"
fi

python training/train_transformer_rvq.py \
    --features-dir data/features_merged \
    --phonemes-path data/processed_merged/phonemes_mfa.json \
    --alignments-path data/processed_merged/alignments_mfa.json \
    --vq-tokens-dir data/rvq_tokens_merged \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_rvq_merged \
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
    --codebook-size 512 \
    --resume

# ── Generate test audio ──
echo ""
echo "Generating test audio..."

python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from inference.synthesize_rvq import ArticulatoryTTSRVQ

tts = ArticulatoryTTSRVQ(
    rvq_checkpoint='checkpoints_rvq/rvq_best.pt',
    transformer_checkpoint='checkpoints_rvq_merged/transformer_best.pt',
    vocab_path='data/processed_all/vocab_mfa.json',
    norm_stats_path='data/features_merged/norm_stats.npz',
    device='cpu',
)

with open('data/features_merged/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

for i, text in enumerate([
    'Hello, how are you today?',
    'The quick brown fox jumps over the lazy dog.',
    'This speech was generated using articulatory tokens.',
    'Nobody has ever tried this approach before.',
]):
    wav, sr = tts.synthesize(text, speaker_emb=spk_emb)
    sf.write(f'outputs/test_merged_{i}.wav', wav, sr)
    print(f'Saved test_merged_{i}.wav')
"

echo ""
echo "Done at: $(date)"
