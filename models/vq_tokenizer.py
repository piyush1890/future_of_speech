"""
VQ Tokenizer for articulatory features.
Encodes 14-dim SPARC features into discrete tokens via a learned encoder + VQ codebook + decoder.
"""
import torch
import torch.nn as nn
import numpy as np
from vector_quantize_pytorch import VectorQuantize


class ArticulatoryVQTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,        # 12 EMA + pitch + loudness
        latent_dim: int = 64,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        # Pre-encoder: project 14-dim → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Vector quantizer
        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=ema_decay,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            kmeans_iters=10,
        )

        # Post-decoder: project latent_dim → 14-dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode continuous features to quantized latents.
        Args:
            x: (B, T, 14) normalized articulatory features
        Returns:
            quantized: (B, T, latent_dim) quantized latent vectors
            indices: (B, T) codebook indices
            commit_loss: scalar commitment loss
        """
        z = self.encoder(x)              # (B, T, latent_dim)
        quantized, indices, commit_loss = self.vq(z)
        return quantized, indices, commit_loss

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents back to continuous features.
        Args:
            quantized: (B, T, latent_dim) quantized vectors
        Returns:
            (B, T, 14) reconstructed features
        """
        return self.decoder(quantized)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices to continuous features.
        Args:
            indices: (B, T) codebook indices
        Returns:
            (B, T, 14) reconstructed features
        """
        quantized = self.vq.get_output_from_indices(indices)  # (B, T, latent_dim)
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass: encode → quantize → decode.
        Args:
            x: (B, T, 14) normalized features
        Returns:
            dict with reconstructed, indices, commit_loss, perplexity
        """
        quantized, indices, commit_loss = self.encode(x)
        reconstructed = self.decode(quantized)

        # Compute codebook utilization (perplexity)
        encodings = torch.zeros(indices.shape[0] * indices.shape[1], self.codebook_size, device=x.device)
        encodings.scatter_(1, indices.reshape(-1, 1), 1)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "reconstructed": reconstructed,
            "indices": indices,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
        }
