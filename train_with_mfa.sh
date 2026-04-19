#!/bin/zsh
# Retrain with MFA phoneme-level alignments (the best labels we can get)
#
# Usage: cd ~/projects/articulatory-tts && ./train_with_mfa.sh

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

echo "============================================"
echo "MFA Alignment + Retrain"
echo "Started at: $(date)"
echo "============================================"

# ── Step 1: Convert MFA TextGrids to our alignment format ──
echo ""
echo "[Step 1/3] Converting MFA alignments..."

python data/convert_mfa_alignments.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --textgrid-dir data/mfa_alignments \
    --output-path data/processed_combined/alignments_mfa.json

# ── Step 2: Verify alignment quality ──
echo ""
echo "[Step 2/3] Verifying alignment quality..."

python -c "
import json, numpy as np
from collections import Counter

with open('data/processed_combined/phonemes.json') as f:
    phoneme_data = json.load(f)
with open('data/processed_combined/alignments_mfa.json') as f:
    alignments = json.load(f)

# Check phoneme-token consistency
phoneme_tokens = {}
for utt_id in list(phoneme_data.keys())[:2000]:
    if utt_id not in alignments:
        continue
    try:
        tokens = np.load(f'data/vq_tokens_combined/{utt_id}.npy')
    except:
        continue
    phonemes = phoneme_data[utt_id]['phonemes']
    durations = alignments[utt_id]['durations']
    pos = 0
    for phon, dur in zip(phonemes, durations):
        if phon not in phoneme_tokens:
            phoneme_tokens[phon] = []
        phoneme_tokens[phon].extend(tokens[pos:pos+dur].tolist())
        pos += dur

print('Phoneme -> VQ token consistency (MFA alignment):')
print(f'{\"Phoneme\":>8s} | {\"Frames\":>7s} | {\"Unique\":>6s} | Top1%')
print('-' * 45)
for phon in ['AH1', 'IY1', 'AA1', 'EH1', 'S', 'T', 'M', 'N', '<sil>']:
    if phon not in phoneme_tokens or not phoneme_tokens[phon]:
        continue
    toks = phoneme_tokens[phon]
    counts = Counter(toks)
    unique = len(counts)
    top1_pct = counts.most_common(1)[0][1] / len(toks) * 100
    print(f'{phon:>8s} | {len(toks):>7d} | {unique:>6d} | {top1_pct:.1f}%')
"

# ── Step 3: Train transformer ──
echo ""
echo "[Step 3/3] Training transformer with MFA alignments..."
echo "Started at: $(date)"

python training/train_transformer.py \
    --features-dir data/features_combined \
    --phonemes-path data/processed_combined/phonemes.json \
    --alignments-path data/processed_combined/alignments_mfa.json \
    --vq-tokens-dir data/vq_tokens_combined \
    --vocab-path data/processed_combined/vocab.json \
    --checkpoint-dir checkpoints_mfa \
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
    transformer_checkpoint='checkpoints_mfa/transformer_best.pt',
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
    out_path = f'outputs/test_mfa_{i}.wav'
    sf.write(out_path, wav, sr)
    print(f'Saved {out_path} ({len(wav)/sr:.2f}s): \"{text}\"')

print('\nDone! Listen to outputs/test_mfa_*.wav')
"
