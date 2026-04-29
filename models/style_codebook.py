"""
Style codebook for v5: discretizes per-phoneme continuous style vectors into
discrete style codes that the planner (stage 2) will learn to predict from
text + knobs.

Wraps `vector_quantize_pytorch.VectorQuantize` (single level, EMA-updated codebook,
straight-through estimator). Adds:
  - phoneme-mask-aware quantization (padded positions get a sentinel index)
  - convenience helpers for getting code embeddings (without going through the
    quantize forward pass), used at inference when the planner emits codes.

Single level (not RVQ) by design — see plan rationale: 2-level adds operational
cost without clear capability gain at our scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize


PAD_CODE = -1   # Sentinel index emitted for padded phoneme positions; the
                # transformer should look at phoneme_mask, not at this index,
                # to decide whether to use a code embedding.


class StyleCodebook(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        codebook_size: int = 512,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: float = 2.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=ema_decay,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

    def forward(
        self,
        z: torch.Tensor,                # (B, N, latent_dim) per-phoneme continuous style
        phoneme_mask: torch.Tensor,     # (B, N) True = real phoneme
    ):
        """
        Returns:
          quantized: (B, N, latent_dim) — straight-through quantized vectors (gradient flows)
          indices:   (B, N) long — codebook indices, with PAD_CODE at padded positions
          commit_loss: scalar
        """
        # vector_quantize_pytorch expects (B, T, D); we pass (B, N, latent_dim) directly
        quantized, indices, commit_loss = self.vq(z)

        # Mask padded positions in the indices (set to sentinel; loss already ignores)
        # Note: quantized at padded positions is non-zero but we'll mask downstream
        indices = torch.where(phoneme_mask, indices, torch.full_like(indices, PAD_CODE))

        # Zero out quantized at padded positions for safety (shouldn't be consumed but defensive)
        quantized = quantized * phoneme_mask.unsqueeze(-1).to(quantized.dtype)

        return quantized, indices, commit_loss

    def embed_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """At inference, the planner emits indices; this looks them up directly
        without going through the quantize forward (no commitment loss, no codebook
        update). Padded positions (PAD_CODE = -1) get zero embeddings.

        indices: (B, N) long
        returns: (B, N, latent_dim)
        """
        # vector_quantize_pytorch's codebook is at self.vq._codebook.embed: (1, codebook_size, dim)
        codebook = self.vq._codebook.embed
        if codebook.dim() == 3:
            codebook = codebook.squeeze(0)   # (codebook_size, dim)

        valid_mask = indices != PAD_CODE
        safe_idx = indices.clamp(min=0)              # (B, N) — replace PAD_CODE with 0 for indexing
        embeds = codebook[safe_idx]                  # (B, N, dim)
        embeds = embeds * valid_mask.unsqueeze(-1).to(embeds.dtype)
        return embeds


if __name__ == "__main__":
    # Quick test
    cb = StyleCodebook(latent_dim=256, codebook_size=512)
    n_params = sum(p.numel() for p in cb.parameters())
    print(f"StyleCodebook: {n_params:,} params (codebook entries: {512 * 256:,})")

    B, N = 2, 30
    z = torch.randn(B, N, 256)
    phoneme_mask = torch.ones(B, N, dtype=torch.bool)
    phoneme_mask[1, 25:] = False   # batch 1 has 25 real phonemes, last 5 are pad

    quantized, indices, commit = cb(z, phoneme_mask)
    print(f"quantized: {quantized.shape}, dtype {quantized.dtype}")
    print(f"indices  : {indices.shape}, dtype {indices.dtype}")
    print(f"  unique codes used: {indices[indices != PAD_CODE].unique().numel()}")
    print(f"  PAD_CODE positions in batch 1: {(indices[1] == PAD_CODE).sum().item()} (expect 5)")
    print(f"commit_loss: {commit.item():.5f}")

    # Test embed_codes path (inference)
    embeds = cb.embed_codes(indices)
    print(f"embed_codes output: {embeds.shape}")
    assert torch.allclose(embeds[1, 25:], torch.zeros_like(embeds[1, 25:])), "padded should be zero"
    print("Padded positions correctly zeroed in embed_codes")
