# Articulatory Token TTS — Architecture

## The Big Picture

```
                         TRAINING PIPELINE
                         ================

   Audio (.flac)                          Text transcript
        |                                      |
        v                                      v
  ┌───────────┐                          ┌──────────┐
  │   SPARC    │                          │  g2p-en  │
  │  Encoder   │                          │ (text to │
  │(pretrained)│                          │ phonemes)│
  └─────┬─────┘                          └────┬─────┘
        |                                      |
        v                                      v
  14-dim features                      Phoneme sequence
  (50 Hz per frame)                   ["HH","AH0","L","OW1"]
        |                                      |
        v                                      |
  ┌───────────┐                                |
  │    VQ      │                                |
  │ Tokenizer  │                                |
  │ (trained)  │                                |
  └─────┬─────┘                                |
        |                                      |
        v                                      v
  Token IDs ◄──── LOSS ────────── Transformer predicts
  [42, 187, 53...]                these token IDs
  (ground truth)                  from phonemes


                        INFERENCE PIPELINE
                        ==================

   "Hello world"
        |
        v
  ┌──────────┐
  │  g2p-en  │
  └────┬─────┘
        |
        v
  ["HH","AH0","L","OW1"," ","W","ER1","L","D"]
        |
        v
  ┌─────────────────────────────┐
  │       TRANSFORMER           │
  │  phonemes → token IDs       │
  │  + speaker embedding        │
  │  + duration prediction      │
  └──────────┬──────────────────┘
             |
             v
  Token IDs: [42, 187, 53, 53, 201, ...]
             |
             v
  ┌───────────┐
  │    VQ      │
  │  Decoder   │
  │(codebook   │
  │  lookup)   │
  └─────┬─────┘
        |
        v
  14-dim continuous features (50 Hz)
        |
        v
  ┌───────────┐
  │   SPARC    │
  │  Vocoder   │
  │(pretrained)│
  └─────┬─────┘
        |
        v
   Audio waveform (16 kHz)
```

---

## Component 1: SPARC Encoder (Pretrained, not trained by us)

```
  Audio waveform (16 kHz)
        |
        v
  ┌──────────────────────────────────────────┐
  │              SPARC ENCODER               │
  │                                          │
  │  ┌────────────────────┐                  │
  │  │  WavLM Large       │ (frozen)         │
  │  │  (speech SSL model) │                  │
  │  └─────────┬──────────┘                  │
  │            |                              │
  │     ┌──────┴──────────┐                  │
  │     |                  |                  │
  │     v                  v                  │
  │  ┌────────┐    ┌──────────────┐          │
  │  │Linear  │    │   Speaker    │          │
  │  │Regress.│    │   Encoder    │          │
  │  └───┬────┘    │(periodicity- │          │
  │      |         │ weighted     │          │
  │      v         │ pooling +    │          │
  │   12 EMA       │ FFN)         │          │
  │   features     └──────┬───────┘          │
  │   (50 Hz)             |                  │
  │      |                v                  │
  │      |          64-dim speaker           │
  │      |          embedding                │
  │      |          (1 per utterance)        │
  │      |                                   │
  │  Also extracts:                          │
  │  - Pitch (CREPE, 50 Hz)                  │
  │  - Loudness (50 Hz)                      │
  └──────────────────────────────────────────┘

  Output per frame (50 Hz = 50 frames per second):

  ┌──────────────────────────────────────────────────┐
  │  12 EMA channels          │ Pitch │ Loudness     │
  │                            │       │              │
  │  TD_x  TD_y   (tongue back)│       │              │
  │  TB_x  TB_y   (tongue mid) │  Hz   │  amplitude   │
  │  TT_x  TT_y   (tongue tip)│       │              │
  │  UL_x  UL_y   (upper lip) │       │              │
  │  LI_x  LI_y   (jaw)       │       │              │
  │  LL_x  LL_y   (lower lip) │       │              │
  └──────────────────────────────────────────────────┘
       14 dimensions total per frame
```

**What EMA means physically:**

```
  Side view of mouth:

        Soft palate
            \
    TD ●─────\────── Tongue Dorsum (back)
              \
    TB ●───────── Tongue Blade (middle)
              /
    TT ●─────/──── Tongue Tip
            /
    UL ●──/─── Upper Lip
           |
    LL ●───── Lower Lip
           |
    LI ●───── Lower Incisor (jaw)

  Each ● is tracked in X (front-back) and Y (up-down)
  = 6 articulators × 2 coordinates = 12 dimensions
```

---

## Component 2: VQ Tokenizer (Trained by us)

```
  Continuous features ──► Discrete tokens ──► Reconstructed features
     (14-dim, float)         (integer IDs)       (14-dim, float)

  Detailed architecture:

  Input: (batch, time, 14)
         normalized features
              |
              v
  ┌─────────────────────────┐
  │      PRE-ENCODER        │
  │                         │
  │  Linear(14 → 128)       │
  │  LayerNorm              │
  │  GELU activation        │
  │  Linear(128 → 64)       │
  └───────────┬─────────────┘
              |
              v
        (batch, time, 64)
              |
              v
  ┌─────────────────────────┐
  │    VECTOR QUANTIZER     │
  │                         │
  │  Codebook: 512 vectors  │
  │  Each vector: 64-dim    │
  │                         │
  │  For each frame:        │
  │  1. Find nearest        │
  │     codebook entry      │
  │  2. Replace with that   │
  │     entry               │
  │  3. Record the index    │
  │                         │
  │  ┌───┬───┬───┬───┐     │
  │  │ 0 │ 1 │ 2 │...│512  │ ◄── codebook
  │  └───┴───┴───┴───┘     │
  │    ▲                    │
  │    │ nearest neighbor   │
  │    │                    │
  └───────────┬─────────────┘
              |
              ├──► Token index (e.g., 42)
              |
              v
        (batch, time, 64)
        quantized vectors
              |
              v
  ┌─────────────────────────┐
  │      POST-DECODER       │
  │                         │
  │  Linear(64 → 128)       │
  │  LayerNorm              │
  │  GELU activation        │
  │  Linear(128 → 14)       │
  └───────────┬─────────────┘
              |
              v
  Output: (batch, time, 14)
          reconstructed features

  Loss = MSE(input, output) + commitment_loss
         (pitch & loudness get 2× weight)


  WHAT THE CODEBOOK LEARNS:

  Token 42:  tongue high-front, lips rounded    → "oo" position
  Token 187: tongue low, jaw open, lips wide    → "aa" position
  Token 53:  lips together, jaw closed          → "m"/"b" position
  Token 301: tongue tip up, touching ridge      → "t"/"d" position
  ...
  512 discrete mouth positions that can represent all of speech
```

---

## Component 3: TTS Transformer (Trained by us)

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    TTS TRANSFORMER                           │
  │                                                              │
  │                                                              │
  │   Phonemes: [<bos>, HH, AH0, L, OW1, <sil>, W, ..., <eos>] │
  │       |                                                      │
  │       v                                                      │
  │   ┌──────────────┐                                           │
  │   │  Embedding    │  (73 phonemes → 256-dim vectors)         │
  │   └──────┬───────┘                                           │
  │          |                                                    │
  │          v                                                    │
  │   ┌──────────────┐                                           │
  │   │  + Positional │  (sinusoidal encoding)                   │
  │   │  Encoding     │                                           │
  │   └──────┬───────┘                                           │
  │          |                                                    │
  │          |    ┌──────────────────┐                            │
  │          |    │ Speaker Embedding │                           │
  │          |    │ Linear(64 → 256)  │                           │
  │          |    └────────┬─────────┘                            │
  │          |             |                                      │
  │          v             v                                      │
  │   ┌──────────────────────┐                                   │
  │   │    ADD (phonemes +   │                                   │
  │   │    speaker identity) │                                   │
  │   └──────────┬───────────┘                                   │
  │              |                                                │
  │              v                                                │
  │   ┌────────────────────────────────────────┐                 │
  │   │         ENCODER (2 layers)             │                 │
  │   │                                         │                 │
  │   │  ┌───────────────────────────────────┐ │                 │
  │   │  │ Self-Attention (4 heads)          │ │                 │
  │   │  │ "How does each phoneme relate     │ │                 │
  │   │  │  to its neighbors?"               │ │                 │
  │   │  └───────────────┬───────────────────┘ │                 │
  │   │                  |                      │                 │
  │   │  ┌───────────────v───────────────────┐ │                 │
  │   │  │ Feed-Forward (256 → 512 → 256)    │ │                 │
  │   │  └───────────────┬───────────────────┘ │                 │
  │   │                  |                      │                 │
  │   │            (repeat × 2 layers)          │                 │
  │   └──────────────────┬─────────────────────┘                 │
  │                      |                                        │
  │          ┌───────────┴──────────────┐                        │
  │          |                          |                         │
  │          v                          v                         │
  │   ┌──────────────┐          ┌─────────────────┐             │
  │   │   DURATION    │          │  Encoded         │             │
  │   │   PREDICTOR   │          │  phonemes        │             │
  │   │               │          │  (N, 256)        │             │
  │   │  Conv1d → ReLU │          │                  │             │
  │   │  Conv1d → ReLU │          │                  │             │
  │   │  Linear → SP  │          │                  │             │
  │   │               │          │                  │             │
  │   │  Output: how  │          │                  │             │
  │   │  many frames  │          │                  │             │
  │   │  each phoneme │          │                  │             │
  │   │  lasts        │          │                  │             │
  │   └──────┬───────┘          └────────┬────────┘             │
  │          |                           |                        │
  │          |     e.g., [3, 5, 4, 8, 6, 3, ...]                │
  │          |                           |                        │
  │          v                           v                        │
  │   ┌──────────────────────────────────────────┐               │
  │   │          LENGTH REGULATOR                 │               │
  │   │                                           │               │
  │   │  Repeat each phoneme embedding by its     │               │
  │   │  predicted duration:                      │               │
  │   │                                           │               │
  │   │  Phonemes: [HH    ] [AH0        ] [L  ]  │               │
  │   │  Duration:  3 frames  5 frames     4 fr   │               │
  │   │  Expanded: [HH,HH,HH,AH,AH,AH,AH,AH,    │               │
  │   │            L,L,L,L, ...]                  │               │
  │   │                                           │               │
  │   │  Input:  (N phonemes, 256)                │               │
  │   │  Output: (T frames, 256)                  │               │
  │   └──────────────────┬────────────────────────┘               │
  │                      |                                        │
  │                      v                                        │
  │   ┌────────────────────────────────────────┐                 │
  │   │         DECODER (2 layers)             │                 │
  │   │                                         │                 │
  │   │  Same structure as encoder:             │                 │
  │   │  Self-Attention + Feed-Forward          │                 │
  │   │  × 2 layers                             │                 │
  │   │                                         │                 │
  │   │  "What articulatory token should each   │                 │
  │   │   frame produce?"                       │                 │
  │   └──────────────────┬─────────────────────┘                 │
  │                      |                                        │
  │                      v                                        │
  │   ┌──────────────────────────────────┐                       │
  │   │  OUTPUT PROJECTION               │                       │
  │   │  Linear(256 → 512)              │                       │
  │   │                                  │                       │
  │   │  Produces logits over 512        │                       │
  │   │  codebook entries for each frame │                       │
  │   └──────────────────┬───────────────┘                       │
  │                      |                                        │
  │                      v                                        │
  │   Token IDs: [42, 42, 42, 187, 187, 187, 187, 187,          │
  │               53, 53, 53, 53, ...]                           │
  │              one per frame at 50 Hz                           │
  └──────────────────────────────────────────────────────────────┘


  TRAINING: cross-entropy loss between predicted and ground-truth tokens
  INFERENCE: argmax over logits → token IDs
```

---

## Component 4: SPARC Vocoder (Pretrained, not trained by us)

```
  Token IDs → VQ Codebook Lookup → 14-dim features → Vocoder → Audio

  ┌──────────────────────────────────────────────────────────────┐
  │                    SPARC VOCODER                             │
  │                    (HiFi-GAN + FiLM)                        │
  │                                                              │
  │  Input: 14 channels at 50 Hz                                │
  │         + 64-dim speaker embedding                          │
  │                                                              │
  │  ┌────────────────────────────────────────────┐             │
  │  │  Upsampling stages: [8, 8, 2, 2] = 256×   │             │
  │  │  50 Hz × 256 = 12,800 Hz... wait,          │             │
  │  │  actually [8, 5, 4, 2] = 320×              │             │
  │  │  50 Hz × 320 = 16,000 Hz ✓                │             │
  │  │                                             │             │
  │  │  Stage 1: 50 Hz → 400 Hz   (×8)           │             │
  │  │  Stage 2: 400 Hz → 2000 Hz  (×5)          │             │
  │  │  Stage 3: 2000 Hz → 8000 Hz (×4)          │             │
  │  │  Stage 4: 8000 Hz → 16000 Hz (×2)         │             │
  │  │                                             │             │
  │  │  Each stage:                                │             │
  │  │  TransposedConv (upsample)                  │             │
  │  │    → Multi-Receptive-Field block            │             │
  │  │      (3 parallel residual blocks            │             │
  │  │       with dilations [1, 3, 5])             │             │
  │  │    → FiLM conditioning from speaker_emb     │             │
  │  │      (gain & bias per channel)              │             │
  │  └────────────────────────────────────────────┘             │
  │                                                              │
  │  Speaker FiLM conditioning:                                 │
  │  ┌─────────────────────────────────────────┐               │
  │  │  speaker_emb (64-dim)                    │               │
  │  │       |                                  │               │
  │  │       v                                  │               │
  │  │  Linear → gain (γ) and bias (β)         │               │
  │  │       |                                  │               │
  │  │       v                                  │               │
  │  │  output = γ * features + β              │               │
  │  │                                          │               │
  │  │  This makes the SAME articulatory tokens │               │
  │  │  sound like DIFFERENT speakers            │               │
  │  └─────────────────────────────────────────┘               │
  │                                                              │
  │  Output: waveform at 16,000 Hz                              │
  └──────────────────────────────────────────────────────────────┘
```

---

## End-to-End Data Flow Example

```
  Input: "Hello"

  Step 1 — Text to Phonemes (g2p-en):
  ┌─────────────────────────────────────────┐
  │ "Hello" → ["HH", "AH0", "L", "OW1"]   │
  └─────────────────────────────────────────┘

  Step 2 — Phoneme Encoding (Transformer Encoder):
  ┌─────────────────────────────────────────────────┐
  │ [HH, AH0, L, OW1] → 4 × 256-dim vectors       │
  │ + speaker embedding added                        │
  │ Self-attention learns: "L before OW1 = dark L"  │
  └─────────────────────────────────────────────────┘

  Step 3 — Duration Prediction:
  ┌─────────────────────────────────────────────────┐
  │ HH → 3 frames (60ms)                           │
  │ AH0 → 5 frames (100ms)                         │
  │ L → 4 frames (80ms)                            │
  │ OW1 → 8 frames (160ms)                         │
  │ Total: 20 frames = 400ms                        │
  └─────────────────────────────────────────────────┘

  Step 4 — Length Regulation (expand):
  ┌─────────────────────────────────────────────────┐
  │ Frame:  1  2  3  4  5  6  7  8  9 10 11 ... 20 │
  │ Phone: HH HH HH AH AH AH AH AH  L  L  L  OW  │
  └─────────────────────────────────────────────────┘

  Step 5 — Decoder → Token Prediction:
  ┌─────────────────────────────────────────────────┐
  │ Frame 1 → token 301 (tongue tip up, air burst)  │
  │ Frame 2 → token 301                             │
  │ Frame 3 → token 288 (transitioning...)          │
  │ Frame 4 → token 187 (jaw open, tongue low)      │
  │ Frame 5 → token 187                             │
  │ ...                                              │
  │ Frame 20 → token 42 (lips rounded, tongue back) │
  └─────────────────────────────────────────────────┘

  Step 6 — VQ Decode (codebook lookup → continuous):
  ┌─────────────────────────────────────────────────┐
  │ token 301 → [0.2, -0.5, 0.8, ...]  (14-dim)   │
  │ token 187 → [-0.3, 1.1, -0.2, ...]             │
  │ token 42  → [0.7, 0.4, -0.6, ...]              │
  │                                                  │
  │ Denormalize: features * std + mean               │
  └─────────────────────────────────────────────────┘

  Step 7 — SPARC Vocoder:
  ┌─────────────────────────────────────────────────┐
  │ 20 frames of articulatory features              │
  │ + speaker embedding (voice identity)            │
  │           ↓                                      │
  │ HiFi-GAN upsamples 320×                         │
  │ 20 × 320 = 6,400 audio samples                  │
  │           ↓                                      │
  │ 400ms of audio at 16kHz: "Hello"                │
  └─────────────────────────────────────────────────┘
```

---

## What Makes This Novel

```
  TRADITIONAL TTS (VALL-E, F5-TTS):
  ┌────────────────────────────────────────────┐
  │ Text → Acoustic Tokens → Audio             │
  │         (what it SOUNDS like)              │
  │         ~75 tokens/sec                      │
  │         1024+ codebook                      │
  │         speaker-dependent                   │
  └────────────────────────────────────────────┘

  OUR APPROACH:
  ┌────────────────────────────────────────────┐
  │ Text → Articulatory Tokens → Audio         │
  │         (what the MOUTH does)              │
  │         50 tokens/sec                       │
  │         512 codebook                        │
  │         speaker-INDEPENDENT                 │
  │         (speaker added at vocoder stage)    │
  └────────────────────────────────────────────┘

  Key advantages (if it works):
  • Shorter sequences → cheaper transformer attention
  • Smaller codebook → easier to predict
  • Physically meaningful → better cross-lingual transfer
  • Speaker separated → voice cloning is just swapping an embedding
```

---

## Training Losses Explained

```
  VQ TOKENIZER LOSS:
  ┌────────────────────────────────────────────┐
  │ MSE(reconstructed, original)               │
  │   "How well can we reconstruct features    │
  │    from discrete tokens?"                  │
  │                                            │
  │ + Commitment loss                          │
  │   "Stay close to your assigned codebook    │
  │    entry (don't oscillate)"                │
  │                                            │
  │ Our result: val_loss = 0.2075              │
  │ Perplexity: 164 of 512 codes used          │
  └────────────────────────────────────────────┘

  TRANSFORMER LOSS:
  ┌────────────────────────────────────────────┐
  │ Cross-Entropy(predicted_logits, true_token)│
  │   "Did you predict the right codebook      │
  │    entry for each frame?"                  │
  │                                            │
  │ + Duration MSE(predicted_dur, true_dur)    │
  │   "Did you predict how long each           │
  │    phoneme lasts?"                         │
  │                                            │
  │ Random baseline: CE = ln(512) ≈ 6.24      │
  │ Our best so far: val CE ≈ 5.18            │
  │ Goal: val CE < 4.0                         │
  └────────────────────────────────────────────┘
```
