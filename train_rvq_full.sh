#!/bin/zsh
# Full pipeline with Residual VQ: train RVQ, verify, tokenize, train transformer, generate.
# Original single-VQ work is preserved in checkpoints_all/ (this uses checkpoints_rvq/)
#
# Usage: cd ~/projects/articulatory-tts && ./train_rvq_full.sh

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "Residual VQ Training Pipeline"
echo "Started at: $(date)"
echo "============================================"

# ── Step 1: Train Residual VQ ──
if [ ! -f checkpoints_rvq/rvq_best.pt ]; then
    echo ""
    echo "[Step 1/5] Training Residual VQ (4 codebooks)..."
    python training/train_vq_rvq.py \
        --features-dir data/features_all \
        --checkpoint-dir checkpoints_rvq \
        --device mps --epochs 100 \
        --num-quantizers 4 --codebook-size 512
else
    echo "[Step 1/5] RVQ already trained, skipping."
fi

# ── Step 2: Verify VQ round-trip quality ──
echo ""
echo "[Step 2/5] Verifying RVQ round-trip quality..."

python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, torch, soundfile as sf
from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from sparc import load_model

ckpt = torch.load('checkpoints_rvq/rvq_best.pt', map_location='cpu', weights_only=True)
rvq = ArticulatoryRVQTokenizer(
    codebook_size=ckpt['args']['codebook_size'],
    num_quantizers=ckpt['args']['num_quantizers'],
    latent_dim=ckpt['args']['latent_dim'],
    hidden_dim=ckpt['args']['hidden_dim'],
)
rvq.load_state_dict(ckpt['model_state_dict'])
rvq.eval()
print(f'Loaded RVQ: val_loss={ckpt[\"val_loss\"]:.4f}')

stats = np.load('data/features_all/norm_stats.npz')
mean, std = stats['mean'], stats['std']

sparc = load_model('en', device='cpu')

d = np.load('data/features_all/1272-128104-0000.npz')
ema, pitch, loudness = d['ema'], d['pitch'], d['loudness']
spk_emb = d['spk_emb']

# RVQ round-trip
features = np.concatenate([ema, pitch[:,None], loudness[:,None]], axis=-1)
features_norm = (features - mean) / std
x = torch.from_numpy(features_norm.astype(np.float32)).unsqueeze(0)
with torch.no_grad():
    result = rvq(x)
    recon_norm = result['reconstructed'].squeeze(0).numpy()
recon = recon_norm * std + mean

wav_rvq = sparc.decode(recon[:, :12], recon[:, 12], recon[:, 13], spk_emb)
sf.write('outputs/groundtruth_rvq_roundtrip.wav', wav_rvq, sparc.sr)
print(f'Saved outputs/groundtruth_rvq_roundtrip.wav ({len(wav_rvq)/sparc.sr:.2f}s)')
print('Compare with outputs/groundtruth_vq_roundtrip.wav (single VQ)')
" 2>&1 | grep -v Warning | grep -v FutureWarning | grep -v weight_norm | grep -v Loading | grep -v UserWarning

# ── Step 3: Tokenize all features with RVQ ──
echo ""
echo "[Step 3/5] Tokenizing features with RVQ..."
python training/tokenize_features_rvq.py \
    --features-dir data/features_all \
    --checkpoint checkpoints_rvq/rvq_best.pt \
    --output-dir data/rvq_tokens_all \
    --device mps

# ── Step 4: Train transformer with RVQ tokens ──
echo ""
echo "[Step 4/5] Training transformer with multi-codebook RVQ tokens..."
echo "Started at: $(date)"

python training/train_transformer_rvq.py \
    --features-dir data/features_all \
    --phonemes-path data/processed_all/phonemes_mfa.json \
    --alignments-path data/processed_all/alignments_mfa.json \
    --vq-tokens-dir data/rvq_tokens_all \
    --vocab-path data/processed_all/vocab_mfa.json \
    --checkpoint-dir checkpoints_rvq \
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
    --num-quantizers 4 \
    --codebook-size 512 \
    --resume

# ── Step 5: Generate test audio ──
echo ""
echo "[Step 5/5] Generating test audio..."

python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from inference.synthesize_rvq import ArticulatoryTTSRVQ

tts = ArticulatoryTTSRVQ(
    rvq_checkpoint='checkpoints_rvq/rvq_best.pt',
    transformer_checkpoint='checkpoints_rvq/transformer_best.pt',
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
    sf.write(f'outputs/test_rvq_{i}.wav', wav, sr)
    print(f'Saved test_rvq_{i}.wav ({len(wav)/sr:.2f}s)')

print('\\nCompare:')
print('  outputs/test_rvq_*.wav      — Residual VQ (this run)')
print('  outputs/test_epoch14_*.wav  — Single VQ baseline')
" 2>&1 | grep -v Warning | grep -v FutureWarning | grep -v weight_norm | grep -v Loading | grep -v UserWarning

echo ""
echo "============================================"
echo "ALL DONE at $(date)"
echo "============================================"
