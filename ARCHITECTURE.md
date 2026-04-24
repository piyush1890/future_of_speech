# Articulatory Token TTS — Architecture & Training Documentation

## System Overview

A text-to-speech system using **discrete articulatory tokens** as the intermediate representation. Instead of predicting acoustic features (mel spectrograms) or audio codec tokens, we predict the physical positions of speech articulators (tongue, jaw, lips) + pitch + loudness, then decode through a pretrained articulatory vocoder (SPARC).

```
Text → g2p → Phonemes → [Transformer] → RVQ Token IDs → [RVQ Decode] → 14-dim features → [SPARC Vocoder] → Waveform
                              ↑                                              ↑
                         spk_emb (64d)                                  spk_emb (64d)
                         style_vec (256d)
```

### Why Articulatory Tokens?

1. **Language-independent**: mouth positions are the same physics regardless of language. A Hindi retroflex ट and English alveolar T are just different coordinates in the same 14-dim space.
2. **Compact**: 14 continuous dims vs 80 mel dims. Quantizes well with small codebooks.
3. **Physically grounded**: constraints like "pitch can't jump between adjacent frames" are real physics, not learned heuristics.

---

## Components

### 1. SPARC Vocoder (Pretrained, Frozen)

**Source**: `speech-articulatory-coding` package
**Function**: Encode audio → articulatory features; Decode features → audio
**Features**: 14-dim per frame at 50 Hz

| Channels | What | Range |
|----------|------|-------|
| 0-1 | Tongue tip X, Y | Fastest-moving articulator |
| 2-3 | Tongue body X, Y | Moderate speed |
| 4-5 | Tongue dorsum X, Y | Slow |
| 6-7 | Jaw X, Y | Moderate |
| 8-9 | Upper lip X, Y | Slow |
| 10-11 | Lower lip X, Y | Moderate |
| 12 | Pitch (log Hz) | Stored as log(Hz + 1) |
| 13 | Loudness | Energy envelope |

**Speaker embedding**: 64-dim vector per utterance, captures vocal tract shape/timbre.

**Encode API**: `result = sparc.encode(wav)` → dict with keys: `ema`, `pitch`, `loudness`, `spk_emb`, `periodicity`, `pitch_stats`, `ft_len`

**Decode API**: `wav = sparc.decode(ema, pitch_hz, loudness, spk_emb)`

### 2. RVQ Tokenizer (`models/vq_tokenizer_rvq.py`)

**Purpose**: Compress 14-dim continuous features → discrete tokens for transformer prediction.

**Architecture** (~0.5M params):
```
Input (T, 14) → Linear(14, 128) → LayerNorm → GELU → Linear(128, 64)  [encoder]
             → ResidualVQ(dim=64, 4 codebooks × 512 entries)            [quantizer]
             → Linear(64, 128) → LayerNorm → GELU → Linear(128, 14)    [decoder]
```

**Output**: 4 integer token IDs per frame, each 0-511. Total vocabulary: 512 × 4 = 2048 effective tokens.

**RVQ mechanics**: Level 0 captures coarse structure. Level k captures the residual error after levels 0..k-1. Summing all levels' codebook vectors reconstructs the original latent.

**Training**: MSE reconstruction loss + commitment loss. Trained separately before transformer. Frozen during transformer training.

**Checkpoint**: `checkpoints_rvq_logpitch/rvq_best.pt`

**Codebook access** (for soft-decode smoothness loss):
```python
codebook_vectors = rvq.vq.layers[k]._codebook.embed  # (1, 512, 64)
decoded_features = rvq.decoder(sum_of_codebook_vectors)  # (B, T, 14)
```

### 3. Hierarchical Transformer (`models/transformer_rvq_hier.py`)

**Purpose**: Predict articulatory token sequences from phonemes + speaker + style.

**Architecture** (~7.7M params):
```
Phoneme Encoder:
  Embedding(73, 256) + PositionalEncoding
  + speaker_proj(spk_emb: 64 → 256) broadcast
  + style_proj(style_vec: 256 → 256) broadcast
  4 × TransformerEncoderLayer(d=256, heads=4, ff=1024, dropout=0.1, norm_first=True)

Duration Predictor (on encoder output):
  Conv1d(256,256,k=3) → ReLU → LN → Conv1d(256,256,k=3) → ReLU → LN → Linear(256,1) → Softplus

Length Regulator:
  Expand each phoneme embedding by its predicted duration → (B, T_frames, 256)
  Uses torch.repeat_interleave (optimized, no per-phoneme GPU sync)

Frame Decoder:
  PositionalEncoding
  4 × TransformerEncoderLayer(d=256, heads=4, ff=1024)

Hierarchical Output Heads:
  4 × Linear(256, 512)  — one per RVQ level
  Level k+1 conditions on level k's chosen token via cb_embeds[k]
  Teacher forcing during training (ground-truth tokens fed to next level)
  Greedy argmax during inference
```

**Key design decisions**:
- Non-autoregressive (parallel frame generation) — articulatory features are smooth, no need for AR
- Speaker conditioning via additive bias (not FiLM or cross-attention)
- Style conditioning via additive bias from StyleEncoder output
- cb_embeds zero-initialized → model starts equivalent to flat variant, enabling partial checkpoint loading

### 4. Style Encoder (`models/style_encoder.py`)

**Purpose**: Extract a style vector from a reference utterance's SPARC features. Enables Approach B style conditioning — "speak this text like THAT reference audio."

**Architecture** (~0.42M params):
```
Input: (B, T, 14) SPARC features
  → 4 × Conv1d(stride=2) + BatchNorm + ReLU     [compress time by 16x]
  → Bidirectional GRU(hidden=128)                 [aggregate to fixed length]
  → Linear(256, 256)                               [project to style_dim]
Output: (B, 256) style vector
```

**Training**: jointly with the transformer. Each training utterance's own features are fed to the style encoder — it learns to extract whatever prosodic/stylistic information helps the transformer reconstruct the output.

**Inference**: encode a reference audio clip through SPARC → feed features to StyleEncoder → get style_vec → condition transformer.

---

## Training Data (v3): ~112K utterances

| Source | Files | Hours | Content | Phoneme Source |
|--------|-------|-------|---------|----------------|
| LibriSpeech train-clean-100 | ~28K | ~100h | Audiobook reading | MFA alignment |
| LibriSpeech train-clean-360 (partial) | ~54K | ~180h | Audiobook reading | MFA alignment |
| ESD (Emotional Speech Database) | ~17.5K | ~29h | 5 emotions × 10 speakers | MFA alignment |
| Expresso (Meta/FAIR) | ~11.6K | ~40h | 7 styles × 4 speakers | g2p + proportional |
| **Total** | **~112K** | **~349h** | | |

### Data Pipeline

```
Raw audio (.flac/.wav)
  → SPARC encode → {ema, pitch, loudness, spk_emb}.npz
  → Log-pitch transform → features_merged_logpitch_v2/*.npz
  → RVQ tokenize → rvq_tokens_logpitch_v2/*.npy
  → MFA/g2p alignment → processed_merged_v3/{phonemes,alignments}_mfa.json
```

### Feature File Format (.npz)

```
ema:      (T, 12) float32  — articulatory positions
pitch:    (T,)    float32  — log(Hz + 1)
loudness: (T,)    float32  — energy
spk_emb:  (64,)   float32  — speaker embedding
```

### Normalization

`norm_stats.npz` contains per-channel `mean` (14,) and `std` (14,) over the full training set. Training: `feat_norm = (feat - mean) / std`. Inference: `feat = feat_norm * std + mean`.

---

## Loss Function

### 1. Cross-Entropy (token prediction)

```python
total_ce = sum(level_weights[k] * masked_CE(logits[:,:,k], target[:,:,k]) for k in range(4))
```

Uniform level weights [0.25 × 4]. Treats each frame independently.

### 2. Duration Loss

```python
dur_loss = MSE(log(pred_dur + 1), log(gt_dur + 1))
```

### 3. Temporal Smoothness Loss (NEW)

Differentiable soft-decode through frozen RVQ codebook → per-channel weighted frame-to-frame penalty:

```python
soft_features = soft_decode(softmax(logits/T), codebook) → rvq.decoder → (B, T, 14)
diff = soft_features[:, 1:] - soft_features[:, :-1]
smooth_loss = (diff² × channel_weights).mean()
```

**Channel weights** (from σ² of inference smoothing):

| Channel | Weight | Physics |
|---------|--------|---------|
| Tongue tip (0-1) | 0.12 | Fast moves real |
| Other EMA (2-11) | 0.33 | Moderate |
| **Pitch (12)** | **3.0** | Vocal cords have inertia |
| Loudness (13) | 1.33 | Lungs can't change instantly |

**Total**: `total_loss = ce_loss + 0.1 × dur_loss + 6.0 × smooth_loss`

Smooth loss contributes ~15% of gradient. First time pitch gets dominant training signal (52% of smooth gradient vs ~2-5% of CE gradient).

---

## Inference Post-Processing

Channel-aware Gaussian smoothing matching the training loss physics:

```python
# Tongue tip: very light (real fast movements exist)
features[:, 0:2]  = gaussian_filter1d(sigma=0.3)
# Other EMA: light
features[:, 2:12] = gaussian_filter1d(sigma=0.5)
# Pitch: smooth in LOG space (before Hz conversion)
features[:, 12]   = gaussian_filter1d(sigma=1.5)
pitch_hz = exp(features[:, 12]) - 1.0
# Loudness: smooth
features[:, 13]   = gaussian_filter1d(sigma=1.0)
```

---

## Performance

### Training (M4 Pro, 24GB, MPS)

| Config | Batch | it/s | Samples/s |
|--------|-------|------|-----------|
| Old (B=8, old LR) | 8 | 2.3 | 18 |
| **New (B=24, optimized LR)** | **24** | **~4.8** | **~114** |

**Key optimization**: LengthRegulator `torch.repeat_interleave` replacing per-phoneme `.item()` GPU sync. ~6x throughput improvement.

---

## Bilingual (English + Hindi)

SPARC features are language-independent. Same RVQ codebook serves both (retrain on mixed data for full coverage). Separate transformer per language, initialized from English checkpoint.

**Hindi roundtrip verified**: SPARC roundtrip perfect. RVQ roundtrip: pronunciation correct, minor shakiness from missing retroflex codebook entries.

**Hindi data**: 3.2K files (Hindi SER), 40 hours planned (IndicVoices).

---

## Key Learnings

---

## Adding New Data — Mandatory Checklist

Every time new audio data is added to the training set, ALL steps must be followed in order. Skipping any step causes silent corruption that surfaces as slurred/broken audio.

### Step 1: SPARC Encode
```bash
# Encode raw audio → {ema, pitch, loudness, spk_emb}.npz
# Use Colab GPU for speed. Save to Drive.
# VERIFY: each .npz has keys: ema(T,12), pitch(T,), loudness(T,), spk_emb(64,)
```

### Step 2: Log-Pitch Transform + Merge
```bash
python scripts/merge_features_logpitch_v2.py \
  --sources data/NEW_FEATURES_DIR \
  --out-dir data/features_merged_logpitch_v2 \
  --skip-existing
```
**This recomputes `norm_stats.npz`** over the FULL merged set. The mean/std WILL change. This is expected and necessary.

### Step 3: Retrain RVQ Codebook ⚠️ CRITICAL
```bash
python training/train_vq_rvq.py \
  --features-dir data/features_merged_logpitch_v2 \
  --checkpoint-dir checkpoints_rvq_logpitch_v2 \
  --device mps --epochs 100 --batch-size 32
```
**The RVQ codebook MUST be retrained when norm_stats change.** The codebook entries represent the normalized feature distribution. If the distribution shifts (new data changes mean/std), old codebook entries are misaligned → coarse quantization → slurred output.

**DO NOT skip this step.** It takes ~2-3 hours. Skipping it silently degrades ALL downstream quality.

### Step 4: Re-Tokenize ALL Features
```bash
python training/tokenize_features_rvq.py \
  --features-dir data/features_merged_logpitch_v2 \
  --checkpoint checkpoints_rvq_logpitch_v2/rvq_best.pt \
  --output-dir data/rvq_tokens_logpitch_v3 \
  --device mps
```
**Must use the NEW RVQ checkpoint.** Must tokenize ALL files (not just new ones) because the codebook changed.

### Step 5: Build Phonemes + Alignments
```bash
# For MFA-aligned data (LibriSpeech, ESD):
python data/build_mfa_dataset.py \
  --features-dir data/features_merged_logpitch_v2 \
  --textgrid-dir data/mfa_alignments \
  --output-dir data/processed_merged_vN

# For g2p-aligned data (Expresso, etc.):
# Custom script per dataset, merge JSONs
```

### Step 6: Verify Before Training
```bash
python -c "
import json, numpy as np
# Check counts match
features = len(list(Path('data/features_merged_logpitch_v2').glob('*.npz'))) - 1  # minus norm_stats
tokens = len(list(Path('data/rvq_tokens_logpitch_v3').glob('*.npy')))
phonemes = len(json.load(open('data/processed_merged_vN/phonemes_mfa.json')))
print(f'Features: {features}, Tokens: {tokens}, Phonemes: {phonemes}')
assert abs(features - tokens) < 10, 'Feature/token count mismatch!'
assert abs(features - phonemes) < 1000, 'Feature/phoneme count mismatch!'

# Check norm_stats are reasonable
ns = np.load('data/features_merged_logpitch_v2/norm_stats.npz')
pitch_mean = ns['mean'][12]
pitch_std = ns['std'][12]
print(f'Pitch: mean={pitch_mean:.2f} (expect ~5.0), std={pitch_std:.2f} (expect ~0.4)')
assert 4.5 < pitch_mean < 5.5, f'Pitch mean {pitch_mean} looks wrong!'
assert 0.2 < pitch_std < 1.0, f'Pitch std {pitch_std} looks wrong!'
print('All checks passed')
"
```

### Step 7: Update Training Script
```bash
# Update ALL paths in train_hier_v3_style_smooth.sh:
#   --features-dir    → new merged dir
#   --phonemes-path   → new phonemes JSON
#   --alignments-path → new alignments JSON
#   --vq-tokens-dir   → new tokens dir (from new RVQ)
#   --rvq-checkpoint  → new RVQ checkpoint
```

### Step 8: Train Transformer (from scratch)
Since RVQ changed, old transformer checkpoints are incompatible. Start fresh.

### Common Shape Bug
SPARC can produce ema, pitch, loudness with off-by-one frame counts. EVERY file that reads .npz features must do:
```python
ema = data["ema"]
pitch = data["pitch"].squeeze()      # ensure 1D
loudness = data["loudness"].squeeze() # ensure 1D
T = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
features = np.concatenate([ema[:T], pitch[:T, None], loudness[:T, None]], axis=1)
```
**Files that need this guard**: `dataset.py`, `dataset_rvq.py`, `tokenize_features_rvq.py`, `merge_features_logpitch_v2.py`, any new script that loads .npz features.

---

## Key Learnings

1. **Pitch smoothness is physics**: frame-to-frame jumps physically impossible. Must be in training loss, not just post-processing.
2. **CE undertreats pitch**: 1 dim out of 14 = ~2-5% of gradient. Smoothness loss with weight 3.0 fixes this.
3. **Style ≠ emotion ≠ speaker**: three separate conditioning dimensions (spk_emb, style_vec, future emotion tokens).
4. **LengthRegulator `.item()` sync was 2x bottleneck**: `repeat_interleave` eliminates it.
5. **LibriSpeech has ~26% emotional content**: HuBERT tagging shows it's not all neutral.
6. **Inference σ maps to training loss via σ²**: same physics expressed as either post-processing or gradient signal.
7. **Post-hoc deltas don't work**: adding emotion after generation garbles speech. Conditioning before generation (style encoder) is the correct architecture.
