"""
v9 per-phoneme style encoder + style codebook (Stage-A: joint with renderer).

Mirrors v5's design (PerPhonemeStyleEncoder + StyleCodebook) but trained
on v9 data only — no v5 checkpoints are loaded.

PerPhonemeStyleEncoder:
  Input:  per-utterance frame block (B, T, 14), durations (B, N), mask (B, N)
          + phoneme symbol embedding (so the encoder can produce style-only
          representations rather than mixing in phonetic identity)
  Output: per-phoneme z (B, N, latent_dim)

StyleCodebook:
  Wraps vector_quantize_pytorch.VectorQuantize. Single level, EMA-updated codebook.
  Returns quantized vectors, per-phoneme code IDs, and commitment loss.
  PAD_CODE = codebook_size; reserved for BOS/EOS positions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


PAD_CODE = 64   # default style_codebook_size; padding_idx for the embedding.
# (callers pass `style_codebook_size=N` to V9StyleCodebook + V9Renderer; PAD = N.)


class V9PerPhonemeStyleEncoder(nn.Module):
    """Encode per-phoneme frame block → z (style-only).

    DESIGN CHANGE FROM v1: encoder no longer sees the phoneme symbol. This
    forces it to encode whatever differentiates this realization from the
    "average" of this phoneme — i.e., style. Combined with a smaller codebook
    (so codes must be shared across phonemes by construction), this prevents
    codes from collapsing to phoneme identity (the failure mode we observed).

    Encoder takes only the frame block. Phoneme symbol is consumed downstream
    (the renderer takes phoneme_ids separately, so the planner / renderer
    chain has it; only the style code itself is "phoneme-blind").
    """
    def __init__(
        self,
        vocab_size: int = 73,        # kept for backward-compat in arg lists
        input_dim: int = 14,
        hidden_dim: int = 128,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attn_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.attn_scale = hidden_dim ** -0.5

        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, frames_per_phoneme: torch.Tensor, ph_ids: torch.Tensor,
                frame_mask: torch.Tensor) -> torch.Tensor:
        """
        frames_per_phoneme: (B_phon, F_max, 14)
        ph_ids:             (B_phon,) long  — IGNORED; kept for API compat
        frame_mask:         (B_phon, F_max) bool
        Returns: (B_phon, latent_dim)
        """
        x = self.norm0(self.proj_in(frames_per_phoneme))            # (B, F, H)
        x = self.norm1(x + F.gelu(self.layer1(x)))
        x = self.norm2(x + F.gelu(self.layer2(x)))
        scores = (x @ self.attn_query) * self.attn_scale
        scores = scores.masked_fill(~frame_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return self.out(pooled)


class V9StyleCodebook(nn.Module):
    """Single-level VQ wrapper around per-phoneme z's.

    EMA-updated codebook (no codebook gradients). Returns:
      quantized: (B_phon, latent_dim)
      codes:     (B_phon,) long  in [0, codebook_size)
      commit_loss: scalar
    """
    def __init__(
        self,
        codebook_size: int = 512,
        latent_dim: int = 256,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            kmeans_iters=10,
        )

    def forward(self, z: torch.Tensor):
        """z: (B_phon, latent_dim). Returns (quantized, codes, commit_loss)."""
        # VectorQuantize expects (B, T, D); we set T=1 by unsqueezing
        z3 = z.unsqueeze(1)                                          # (B, 1, D)
        q3, codes, commit = self.vq(z3)
        return q3.squeeze(1), codes.squeeze(1), commit

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (..., ) long in [0, codebook_size). Returns (..., latent_dim)."""
        flat = codes.reshape(-1)
        # PAD_CODE handling: any index == codebook_size returns zero embedding
        valid = flat < self.codebook_size
        out = torch.zeros(flat.shape[0], self.latent_dim, device=codes.device, dtype=torch.float32)
        if valid.any():
            valid_codes = flat[valid].clamp(0, self.codebook_size - 1)
            entry = self.vq._codebook.embed                          # (1, C, D) or (C, D)
            if entry.dim() == 3:
                entry = entry.squeeze(0)
            out[valid] = entry[valid_codes]
        return out.reshape(*codes.shape, self.latent_dim)
