"""
Per-phoneme style encoder for v5.

Reads an utterance's full SPARC features (T, 14) plus its per-phoneme
durations (N,), slices the feature trajectory into N phoneme-aligned chunks,
and encodes each chunk into a continuous style vector. Output: (B, N, style_dim).

This replaces the v3/v4 pooled StyleEncoder which collapsed an entire utterance
to one vector and broadcast it across all phoneme positions. The pooled version
caused the 4× JRE-stretch bug because long references produced style vectors in a
region the duration predictor extrapolated badly on, AND because the broadcast
forced every phoneme's duration to scale together.

With per-phoneme codes:
  - position i's style vector reads ONLY position i's local prosody
  - each position can carry its own emphasis / contour shape independently
  - reference length doesn't matter (we always emit exactly N codes for N phonemes)

Architecture per phoneme:
  features chunk (T_i, 14) → Conv1d×3 (small kernels, no stride) → time mean-pool → Linear → style_dim

Total params: ~0.3M.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PerPhonemeStyleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        style_dim: int = 256,
        hidden: int = 128,
        n_conv_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        convs = []
        in_ch = input_dim
        for _ in range(n_conv_layers):
            convs.extend([
                nn.Conv1d(in_ch, hidden, kernel_size=kernel_size,
                          stride=1, padding=padding),
                nn.GroupNorm(num_groups=8, num_channels=hidden),
                nn.GELU(),
            ])
            in_ch = hidden
        self.convs = nn.Sequential(*convs)
        # Per-phoneme projection
        self.proj = nn.Linear(hidden, style_dim)

    def forward(
        self,
        features: torch.Tensor,        # (B, T, 14) padded
        durations: torch.Tensor,       # (B, N) integer per-phoneme frame counts
        phoneme_mask: torch.Tensor,    # (B, N) True = real phoneme, False = pad
    ) -> torch.Tensor:
        """
        Returns: (B, N, style_dim) — one style vector per phoneme position.
        Padded phoneme positions get zero vectors (still safe to index/mix).

        Implementation: run conv stack on the FULL feature sequence first
        (cheap, single forward pass), then pool the conv output per phoneme
        chunk using durations. This is much more efficient than encoding
        each phoneme's chunk independently.
        """
        B, T, D = features.shape
        N = durations.shape[1]

        # Conv stack on full features
        x = features.transpose(1, 2)            # (B, 14, T)
        x = self.convs(x)                       # (B, hidden, T)
        x = x.transpose(1, 2)                   # (B, T, hidden)

        # Per-phoneme mean-pool using cumulative-sum trick (vectorized, no loop over phonemes)
        durs_long = durations.long()
        # Phoneme starts: cumsum minus duration → start frame index per phoneme
        ends   = durs_long.cumsum(dim=1)        # (B, N)
        starts = ends - durs_long               # (B, N)

        # Per-frame cumulative sum over hidden channels
        # cum_x[b, t, h] = sum_{i<=t} x[b, i, h]; pad with leading zero so we can take diff
        zeros = torch.zeros(B, 1, x.shape[-1], device=x.device, dtype=x.dtype)
        cum_x = torch.cat([zeros, x.cumsum(dim=1)], dim=1)   # (B, T+1, hidden)

        # Gather sums per phoneme: cum_x[end] - cum_x[start]
        # Clamp indices to valid range; padded phonemes (dur=0) get 0 sum
        starts_idx = starts.unsqueeze(-1).clamp(min=0, max=T)             # (B, N, 1)
        ends_idx   = ends.unsqueeze(-1).clamp(min=0, max=T)               # (B, N, 1)
        starts_idx = starts_idx.expand(-1, -1, x.shape[-1])
        ends_idx   = ends_idx.expand(-1, -1, x.shape[-1])
        sums = cum_x.gather(1, ends_idx) - cum_x.gather(1, starts_idx)    # (B, N, hidden)

        # Mean-pool per phoneme — divide by duration (clamped to 1 to avoid /0)
        denom = durs_long.clamp(min=1).unsqueeze(-1).to(sums.dtype)        # (B, N, 1)
        means = sums / denom                                               # (B, N, hidden)

        # Project to style_dim
        z = self.proj(means)                                               # (B, N, style_dim)

        # Zero out padded phoneme positions (they had dur=0 → undefined contribution)
        z = z * phoneme_mask.unsqueeze(-1).to(z.dtype)
        return z


if __name__ == "__main__":
    # Quick shape test
    encoder = PerPhonemeStyleEncoder(input_dim=14, style_dim=256, hidden=128, n_conv_layers=3)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Per-phoneme style encoder: {n_params:,} params")

    B, T, N = 2, 200, 30
    features = torch.randn(B, T, 14)
    durations = torch.randint(2, 12, (B, N))
    # Force durations to sum to T (or less, with mask)
    durations[:, 0] = T - durations[:, 1:].sum(dim=1)
    durations.clamp_(min=1)
    phoneme_mask = torch.ones(B, N, dtype=torch.bool)

    z = encoder(features, durations, phoneme_mask)
    assert z.shape == (B, N, 256), f"Got {z.shape}, expected ({B}, {N}, 256)"
    print(f"Output shape OK: {z.shape}")
    print(f"Output norm range: {z.norm(dim=-1).min().item():.3f} .. {z.norm(dim=-1).max().item():.3f}")
