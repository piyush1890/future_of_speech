# v8 — phoneme-level articulatory TTS

A clean architectural rethink. Predict articulator features at phoneme granularity, interpolate to 50Hz frames, hand to SPARC.

## Why

Articulatory features are inherently slow (~10-15 Hz physical bandwidth) but currently predicted at 50Hz frame-level via discrete RVQ tokens. This:
- Wastes capacity on redundant frames within phonemes
- Introduces frame-level token-argmax jitter (esp. on pitch)
- Couples duration prediction with discrete-token machinery

v8 predicts **per-phoneme anchor features** (start + end) and a **duration scalar**, then renders 50Hz frames via smooth interpolation. SPARC consumes the interpolated stream as before.

## Contrast with v5/v6

| | v5/v6 (frame-level + RVQ) | v8 (phoneme-level + regression) |
|---|---|---|
| Prediction unit | Frame (50Hz, ~200/utt) | Phoneme (~30/utt) |
| Output type | Discrete RVQ tokens | Continuous features |
| Loss | CE on tokens + smoothness | MSE on features |
| Param count (stage 1) | ~7.7M | ~3.3M |
| Inference (CPU est) | ~100ms | ~30ms |
| Jitter source | Frame-level token argmax | None (smooth interpolation) |

## Pipeline

```
text → phonemes → encoder → per-phoneme heads:
                              ├── start_features  (14 dims)
                              ├── end_features    (14 dims)
                              └── duration        (1 scalar)
                                      ↓
                          interpolation → 50Hz frame stream
                                      ↓
                                    SPARC → audio
```

## Status

Experimental. Trains fresh — no checkpoint reuse from v5/v6.
