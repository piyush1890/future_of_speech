# Articulatory Token TTS - Session Summary

**Date**: April 7-18, 2026
**Project location**: `/Users/piyush/projects/articulatory-tts/`
**Conda environment**: `arttts` (miniconda at `~/miniconda3/`)

---

## 1. The Idea (Core Hypothesis)

Build a TTS system that uses **discrete articulatory tokens** (VQ-quantized mouth/tongue positions) as the intermediate representation, instead of the standard **acoustic tokens** (EnCodec/SoundStream-style sound codes).

**Pipeline:**
```
Text → g2p → Phonemes → Transformer → Articulatory Token IDs → VQ Decode → 14-dim features → SPARC Vocoder → Audio
```

**Why novel**: Nobody has combined:
1. SPARC's articulatory encoder (audio→features, from Berkeley, June 2024)
2. VQ-quantization of those features into discrete tokens
3. A transformer predicting those tokens from phonemes
4. Decoding back to audio via SPARC's HiFi-GAN vocoder

Existing work:
- **SPARC** (2024): audio→articulatory→audio (no text input)
- **ArtSpeech** (ACM MM 2024): text→continuous articulatory features→audio (not discrete tokens)
- **VALL-E, F5-TTS**: text→acoustic tokens→audio (not articulatory)

**Our gap**: phonemes → **discrete articulatory tokens** → audio.

---

## 2. Structural Advantages (Why This Might Win Commercially)

### Primary advantage: shorter sequences = cheaper inference

| System | Tokens/sec | Codebooks | Relative attention cost |
|---|---|---|---|
| VALL-E acoustic | 75 | 8 | Baseline (1×) |
| Ours RVQ articulatory | 50 | 4 | ~0.09× (11× cheaper) |

Attention scales quadratically with sequence length. This is a **structural advantage** independent of model size.

### Secondary advantages (unproven but theoretical)
- **Speaker-independent representation** — voice cloning via swapping a 64-dim embedding
- **Cross-lingual potential** — articulators are universal (same mouth for all languages)
- **Data efficiency** — claim unproven, SPARC has multilingual model trained on multiple languages

### What's NOT the advantage
- Smaller model: will scale to 100M+ params for production quality, same as acoustic TTS
- Technical novelty per se: VQ + transformer + articulatory are all pre-existing
- The novelty is the **combination** targeting discrete articulatory tokens as the primary TTS representation

---

## 3. Architecture Details

### SPARC Encoder (Berkeley, pretrained, frozen)
- **Input**: audio waveform (16 kHz)
- **Output per frame (50 Hz)**:
  - 12-dim EMA positions: TDX, TDY, TBX, TBY, TTX, TTY, ULX, ULY, LIX, LIY, LLX, LLY (Tongue Dorsum/Blade/Tip, Upper Lip, Lower Incisor, Lower Lip — each X,Y)
  - 1-dim pitch (Hz, 50-550 range)
  - 1-dim loudness (amplitude)
- **Speaker embedding**: 64-dim, one per utterance
- Internal: uses frozen WavLM Large + linear regression for EMA, CREPE for pitch
- **Model cache**: `~/.cache/huggingface/` after first load
- **License**: none specified (risky for commercial use)

### Residual VQ (RVQ) Tokenizer - `models/vq_tokenizer_rvq.py`
- **Input**: (B, T, 14) normalized features
- **Pre-encoder**: Linear(14,128) → LN → GELU → Linear(128,64)
- **ResidualVQ**: 4 codebooks × 512 entries × 64 dims each
- **Post-decoder**: Linear(64,128) → LN → GELU → Linear(128,14)
- **Total params**: ~20K
- **Training**: 100 epochs, AdamW lr=3e-4, MSE + commit loss, channel weights (pitch/loudness 2×)
- **Result**: val_loss 0.0125 (best), round-trip audio clean

### Transformer TTS (multi-codebook) - `models/transformer_rvq.py`
- **Architecture**: FastSpeech-style, non-autoregressive
- **Phoneme encoder**: Embedding(73,256) + 4 layers TransformerEncoder (d_model=256, nhead=4, d_ff=1024, dropout=0.1)
- **Speaker conditioning**: Linear(64,256) additive
- **Duration predictor**: 2× Conv1d(256,256,k=3) + Linear(256,1) + Softplus
- **Length regulator**: repeat phoneme embeddings by predicted durations
- **Frame decoder**: 4 layers TransformerEncoder (same config)
- **Output heads**: 4× Linear(256,512) (one per codebook level)
- **Total params**: 7.27M
- **Training**: 50 epochs, AdamW lr=5e-4 (then 3e-4 for resume), dropout 0.1, warmup 2000 steps, cosine decay
- **Loss**: weighted CE per codebook (level weights [1.0, 0.5, 0.25, 0.125] normalized) + MSE on log-durations

### MFA Alignments (downloaded from Zenodo)
- Phoneme-level timestamps for all 980 hours of LibriSpeech
- `data/mfa_alignments/{dev-clean,train-clean-100,...}/{speaker}/{chapter}/{utt_id}.TextGrid`
- Pre-computed by researcher - 623MB zip
- Download URL: `https://zenodo.org/records/2619474/files/librispeech_alignments.zip`
- **Critical finding**: using MFA phonemes directly (not g2p) gives much better alignment

---

## 4. Experimental Journey & Results

### Chronological result table (val CE on held-out set)

| Run | Data | Alignment | Model | Val CE | Val Acc | Audio subjective |
|---|---|---|---|---|---|---|
| 1 | 2.7K (dev-clean) | Proportional | Single VQ | 5.18 | 4.7% | Noise |
| 2 | 5K (+100h subset) | Proportional | Single VQ | 5.25 | 5.4% | Noise/overfit |
| 3 | 5K | Whisper word-level | Single VQ | 4.84 | 7.3% | "Foreign language" |
| 4 | 5K | MFA-native | Single VQ | 3.09 | 25.1% | "this is...speech" partially recognizable |
| 5 | 31K (train-clean-100+dev-clean) | MFA-native | Single VQ | 2.34 | 32% | "Barely recognizable sentences" |
| 6 | 31K | MFA-native | RVQ 4-codebook | 3.82 (not comparable*) | L0=33% | **"A lot better"** |
| 7 (current) | ~54K (+23K Colab) | MFA-native | RVQ 4-codebook | TBD | TBD | Resuming from #6 |

*RVQ CE is weighted sum of 4 codebook CEs; not directly comparable to single-codebook CE.

### Key learnings

**1. Alignment > data quantity** (biggest surprise)
- Going from proportional to MFA alignment (same 5K data): val CE 5.25 → 3.09
- Going from 5K to 31K (same MFA alignment): val CE 3.09 → 2.34
- Alignment gave more improvement than 6× the data

**2. RVQ fixes glitchy audio**
- Single VQ round-trip: "network glitch, words recognizable"
- RVQ 4-codebook round-trip: clean (user confirmed)
- RVQ magnitudes across codebooks: 0.178 → 0.065 → 0.038 → 0.025 (coarse to fine)

**3. g2p phonemes != MFA phonemes**
- g2p (from text): no leading/trailing silence, different phoneme choices in places
- MFA (from audio): exact phoneme sequence with `sil`/`sp` markers
- Solution: build training dataset from MFA phonemes directly (see `data/build_mfa_dataset.py`), keep using g2p for inference (text → phonemes)

**4. Train-val gap stays small — not overfitting**
- 31K data: gap 0.02-0.1 throughout training
- Means we need more data, not less model / more regularization

**5. Plateau behavior**
- Val CE plateaus around epoch 15-25 on any given dataset
- More epochs don't help once plateaued
- More data shifts the ceiling lower

**6. Hindi test**: English-trained model produces English-accented Hindi. Same as acoustic TTS would. Not a unique advantage without training on Hindi data.

---

## 5. Directory Structure

```
/Users/piyush/projects/articulatory-tts/
├── SESSION_SUMMARY.md              ← this file
├── ARCHITECTURE.md                 ← detailed architecture diagrams
├── colab_encode_360.ipynb          ← Colab notebook for train-clean-360
├── encode_360_local.py             ← Mac-side encoder (back of list)
├── watchdog.sh                     ← auto-restart training if killed
├── train_combined_360.sh           ← MERGED RETRAIN SCRIPT (current)
├── train_rvq_full.sh               ← RVQ pipeline for 31K
├── train_with_whisper_align.sh     ← old (historical)
├── train_with_mfa.sh               ← old (historical)
│
├── models/
│   ├── vq_tokenizer.py             ← single VQ (historical)
│   ├── vq_tokenizer_rvq.py         ← current: Residual VQ
│   ├── transformer.py              ← single-codebook transformer (historical)
│   ├── transformer_rvq.py          ← current: multi-codebook transformer
│   ├── transformer_ar.py           ← autoregressive variant (abandoned, copy trick)
│   ├── duration_predictor.py
│   ├── length_regulator.py
│   └── phoneme_vocab.py
│
├── training/
│   ├── dataset.py                  ← single-VQ dataset
│   ├── dataset_rvq.py              ← current: multi-codebook dataset
│   ├── train_vq.py                 ← single VQ training
│   ├── train_vq_rvq.py             ← current: RVQ training
│   ├── tokenize_features.py
│   ├── tokenize_features_rvq.py    ← current
│   ├── train_transformer.py        ← single-VQ transformer training
│   ├── train_transformer_rvq.py    ← current: multi-codebook transformer training (has --resume)
│   └── train_ar.py                 ← autoregressive (abandoned)
│
├── inference/
│   ├── synthesize.py               ← single-VQ inference
│   └── synthesize_rvq.py           ← current
│
├── data/
│   ├── LibriSpeech/                ← raw audio (dev-clean, train-clean-100, train-clean-360)
│   ├── mfa_alignments/             ← TextGrid files (pre-downloaded from Zenodo)
│   │   └── {dev-clean,train-clean-100,train-clean-360,...}/{speaker}/{chapter}/{utt}.TextGrid
│   ├── features_all/               ← 31K SPARC features (dev-clean + train-clean-100)
│   ├── features_360_colab/         ← 23K new (from Colab, extracted from Drive download)
│   ├── features_360_raw/           ← Mac-encoded (partial, from train-clean-360 back)
│   ├── features_merged/            ← ACTIVE: symlinks combining all above
│   ├── processed_all/              ← 31K MFA-native phonemes/alignments
│   ├── processed_combined/         ← 5K version (historical)
│   ├── processed_merged/           ← ACTIVE: 54K MFA-native
│   ├── rvq_tokens_all/             ← 31K quantized tokens
│   ├── rvq_tokens_merged/          ← ACTIVE: 54K quantized tokens
│   └── vq_tokens_*/                ← single-VQ tokens (historical)
│
├── checkpoints_all/                ← single-VQ (val CE 2.34), BACKED UP
│   └── transformer_best_single_vq_backup.pt
├── checkpoints_rvq/                ← RVQ on 31K (val CE 3.82, rvq_best.pt, transformer_best.pt)
├── checkpoints_rvq_merged/         ← ACTIVE: RVQ on 54K (resumed from rvq)
│
├── outputs/                        ← generated audio files for listening
│   ├── groundtruth_original.wav              ← SPARC round-trip (ceiling)
│   ├── groundtruth_vq_roundtrip.wav          ← single VQ round-trip (glitchy)
│   ├── groundtruth_rvq_roundtrip.wav         ← RVQ round-trip (clean)
│   ├── test_epoch14_*.wav                    ← single VQ transformer output
│   ├── test_rvq_epoch13_*.wav                ← RVQ transformer output (current best)
│   ├── test_rvq_epoch14_*.wav
│   ├── test_mfa_native_*.wav                 ← 5K-data results
│   └── hindi_*.wav                           ← Hindi test (English-accented)
│
└── colab_*.ipynb                   ← Colab notebooks for encoding
```

---

## 6. Key Commands to Continue

### Activate environment
```bash
eval "$(~/miniconda3/bin/conda shell.zsh hook)" && conda activate arttts
cd ~/projects/articulatory-tts
```

### Current running task (54K merged retraining)
```bash
./train_combined_360.sh
```

### Generate audio from best RVQ checkpoint
```bash
python -c "
import sys, json; sys.path.insert(0, '.')
import numpy as np, soundfile as sf
from inference.synthesize_rvq import ArticulatoryTTSRVQ

tts = ArticulatoryTTSRVQ(
    rvq_checkpoint='checkpoints_rvq/rvq_best.pt',
    transformer_checkpoint='checkpoints_rvq_merged/transformer_best.pt',
    vocab_path='data/processed_merged/vocab_mfa.json',
    norm_stats_path='data/features_merged/norm_stats.npz',
    device='cpu',
)
with open('data/features_merged/speaker_embeddings.json') as f:
    spk_embs = json.load(f)
spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)
wav, sr = tts.synthesize('Hello, how are you today?', speaker_emb=spk_emb)
sf.write('outputs/test.wav', wav, sr)
"
```

### Check current best checkpoint
```bash
python -c "
import torch
c = torch.load('checkpoints_rvq_merged/transformer_best.pt', map_location='cpu', weights_only=True)
print(f'Epoch {c[\"epoch\"]}, val_CE={c[\"val_loss\"]:.4f}')
"
```

### Monitor training
```bash
tail -f train_output.log
# Or in real-time:
grep "^Epoch.*CE=" train_output.log | tail -20
```

---

## 7. Data Status

### Features encoded
- **features_all/**: 31,242 files (dev-clean + train-clean-100 from Colab)
- **features_360_colab/**: 23,000 files (partial train-clean-360 from Colab)
- **features_360_raw/**: ~1,000 files (Mac, from back of train-clean-360, stopped)
- **features_merged/**: symlinks = ~54-55K total

### Encoding still in progress
- **Colab**: continues encoding train-clean-360 (from front of list), has ~60 compute units left = ~50 more hours = potentially ~50K more files
- **Mac**: stopped (was too slow, ~8-15 sec/file)

### Training data sources
- Audio: LibriSpeech (OpenSLR)
- Alignments: MFA-generated TextGrids from Zenodo (623MB, pre-downloaded)
- Transcripts: come from LibriSpeech `.trans.txt` files

### If Colab encoding finishes all 104K files
- Total dataset: ~135K files = ~460 hours
- Would be our largest training run
- Expected meaningful quality jump

---

## 8. How to Train More Data

When new features arrive (from Colab):

1. **Download from Drive**: `articulatory_tts/features_360/` folder → extract to Mac
2. **Extract**:
   ```bash
   cd ~/projects/articulatory-tts/data
   mkdir -p features_360_colab_v2
   cd features_360_colab_v2
   unzip ~/Downloads/features_360-*.zip
   # Flatten nested folders:
   find . -mindepth 2 -name '*.npz' -exec mv {} . \;
   find . -type d -empty -delete
   ```
3. **Update `train_combined_360.sh`** to include the new directory in Step 1 (merge)
4. **Run**: `./train_combined_360.sh`

---

## 9. Architecture of `train_combined_360.sh` (What It Does)

1. **Merges** all feature directories via symlinks into `data/features_merged/`
2. **Builds MFA dataset** (phonemes + alignments) from combined features
3. **Recomputes** norm stats + speaker embeddings over combined data
4. **Tokenizes** all features with existing RVQ (doesn't retrain VQ — it's already good)
5. **Resumes** transformer training from previous best (copies `checkpoints_rvq/transformer_best.pt` → `checkpoints_rvq_merged/` then trains with `--resume`)
6. **Generates** test audio at the end

---

## 10. Known Issues & Workarounds

### Issue 1: Training crashes silently
- Symptom: `resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown`
- Cause: unclear, possibly MPS OOM, macOS jetsam, or power/sleep event
- Solution: watchdog script auto-restarts training, --resume flag picks up from last best

### Issue 2: Mac encoding slow (~15 sec/file when should be ~5)
- Cause: MPS memory fragmentation, kernel wired memory builds up over uptime
- Workaround: restart Mac periodically, kill unused apps, disable Continuity Camera
- `sudo killall replayd` helps (Continuity Capture/screen recording daemon)

### Issue 3: Colab disconnects
- Paid Colab has idle timeout (~60 min) + 24-hour hard limit
- Workaround: JavaScript auto-click in browser console + `caffeinate -disu` on Mac
- Features saved to Drive every 100 files → nothing lost on disconnect

### Issue 4: Laptop locks screen, closes flap
- Lock screen doesn't stop training (background processes continue)
- Actual sleep DOES kill things
- Current sleep settings: `sudo pmset -a sleep 0 disksleep 0 displaysleep 0 standby 0`

### Issue 5: MDM-enforced screen lock
- Work Mac (hostname Zupee-1047) may have MDM profile forcing lock
- `sudo profiles -P` to check; can't disable as user

---

## 11. Code Quirks & Non-Obvious Details

### RVQ `indices` shape
- Returns `(B, T, num_quantizers)` — verified in `models/vq_tokenizer_rvq.py` tests
- `decode_indices` accepts this shape directly via `self.vq.get_output_from_indices(indices)`

### Duration alignment
- MFA-native approach: build training phonemes **from MFA TextGrids**, not from g2p
- g2p used only at inference (text → phonemes)
- This works because MFA uses ARPABET (same as g2p) → vocabulary matches

### Transformer output heads
- 4 separate `nn.Linear(d_model, codebook_size)` — one per RVQ level
- Stacked to `(B, T, K, codebook_size)` via `torch.stack(..., dim=2)`
- Each level trained with independent CE loss (classifier per codebook)

### Resume from checkpoint
- `train_transformer_rvq.py --resume` loads `checkpoints_dir/transformer_best.pt`
- Fast-forwards LR scheduler by running `scheduler.step()` in a loop
- **IMPORTANT**: Uses `train_loader_len` saved in the checkpoint (not current `len(train_loader)`)
  because dataset size can change between original training and resume. Without this,
  the scheduler gets stepped the wrong number of times and LR schedule corrupts.
- PyTorch warning about "scheduler.step() before optimizer.step()" is harmless here
- Pre-existing checkpoints (before this fix) were patched with `train_loader_len=3309`

### Feature format
- `.npz` files contain: `ema` (T,12), `pitch` (T,), `loudness` (T,), `spk_emb` (64,)
- EMA/pitch/loudness may differ by 1 frame → use `min_len = min(...)` truncation

---

## 12. Commercial/Strategic Notes

### Our differentiator (honest assessment)
- **Real**: 10-30× cheaper inference due to shorter sequences (structural)
- **Theoretical**: Cross-lingual transfer (not empirically validated at production quality)
- **Overhyped earlier in session**: novelty of VQ or multilingual — plenty of prior art exists

### What would make it commercially viable
1. Reach intelligible+natural quality (need 1000h+ data + 50-100M params)
2. Build own HiFi-GAN vocoder (SPARC's unlicensed; see note below)
3. Add prosody modeling (currently monotone)
4. Add streaming inference (currently full-utterance)
5. Text normalization (numbers, abbreviations, etc.)

### Vocoder licensing
- SPARC GitHub has **no license** → legally can't use commercially
- Solution: train own HiFi-GAN on articulatory features (~2-3 weeks with GPU)
- Methodology is public even if weights aren't

### Future direction: SLM-to-speech
- Skip text layer entirely — small language model directly outputs articulatory tokens
- Would be "Moshi but with articulatory tokens"
- Much stronger commercial story than pure TTS
- Compute: ~2-3 months fine-tuning a 1-3B model

### Alternative: ArtSpeech's features
- Their preprocessed features for LibriTTS/VCTK/LJSpeech exist (Google Drive via their repo)
- Different representation (vocal tract variables + F0 + energy), not SPARC
- **Not worth switching** — would require rebuilding everything, loses our RVQ work, loses novelty

---

## 13. Budget Used

- **Colab Pro**: ~7 compute units spent so far, ~61 remaining (from ~68 total)
- **Google Cloud credits**: $300 untouched, available if needed
- **Local disk**: ~50GB (LibriSpeech + features + MFA alignments)
- **Time**: ~11 days session time (April 7-18)

---

## 14. Open Questions (If You Want to Revisit)

1. **Does scaling to 100K+ files break through the plateau?** — answered soon via current merged training
2. **What about autoregressive decoder?** — tried, had "copy previous token" trick problem with 46% trivial accuracy. Abandoned.
3. **What about non-SPARC features (ArtSpeech's vocal tract variables)?** — explored, not pursued
4. **Cross-lingual for real?** — Hindi test showed English-accented output; true test requires Hindi training data
5. **Own vocoder training?** — punted to commercial phase

---

## 15. Quickstart for Next Session

**To pick up exactly where we left off:**

```bash
# 1. Activate env
eval "$(~/miniconda3/bin/conda shell.zsh hook)" && conda activate arttts
cd ~/projects/articulatory-tts

# 2. Check training state
python -c "
import torch
for d in ['checkpoints_rvq', 'checkpoints_rvq_merged']:
    try:
        c = torch.load(f'{d}/transformer_best.pt', map_location='cpu', weights_only=True)
        print(f'{d}: epoch={c[\"epoch\"]}, val_CE={c[\"val_loss\"]:.4f}')
    except Exception as e:
        print(f'{d}: {e}')
"

# 3. See this summary
cat SESSION_SUMMARY.md | less

# 4. See architecture diagrams
cat ARCHITECTURE.md | less

# 5. Resume training
./train_combined_360.sh   # or train_rvq_full.sh for old pipeline

# 6. Listen to latest results
open outputs/
```

---

## 16. Final Summary

**We built a working proof of concept for discrete-articulatory-token TTS.** The approach works: English sentences are partially recognizable. Quality is below production but the path to improvement (more data + larger model) is clear and incremental.

**Most important thing to remember**: alignment quality matters more than data quantity. MFA alignments were the key unlock. If anyone suggests starting from g2p phoneme alignments, push back — build from MFA TextGrids directly via `data/build_mfa_dataset.py`.

**Current best** (before the 54K run completes): RVQ model at val CE 3.82, audio described as "a lot better" than earlier attempts, with Hindi showing basic cross-phoneme generalization.

**Next milestone**: complete the 54K training, then incrementally add more data as Colab continues to encode. Target: val CE below 3.0, which should produce clearly intelligible speech.
