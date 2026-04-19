"""
Residual VQ Tokenizer for articulatory features.
Uses 4 codebooks with 512 entries each (EnCodec-style).
Each codebook captures residual error from previous codebooks.
"""
import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


class ArticulatoryRVQTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        # Pre-encoder: project 14-dim → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Residual Vector Quantizer
        self.vq = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizers,
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

    def encode(self, x: torch.Tensor):
        """
        Encode continuous features to quantized latents.
        Args:
            x: (B, T, 14) normalized articulatory features
        Returns:
            quantized: (B, T, latent_dim)
            indices: (B, T, num_quantizers) codebook indices per level
            commit_loss: scalar
        """
        z = self.encoder(x)  # (B, T, latent_dim)
        quantized, indices, commit_losses = self.vq(z)
        commit_loss = commit_losses.sum() if commit_losses.dim() > 0 else commit_losses
        return quantized, indices, commit_loss

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents back to 14-dim features."""
        return self.decoder(quantized)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode codebook indices to continuous features.
        Args:
            indices: (B, T, num_quantizers) codebook indices
        Returns:
            (B, T, 14) reconstructed features
        """
        quantized = self.vq.get_output_from_indices(indices)
        return self.decoder(quantized)

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass: encode → quantize → decode."""
        quantized, indices, commit_loss = self.encode(x)
        reconstructed = self.decoder(quantized)

        # Perplexity for first codebook (most important)
        first_indices = indices[..., 0]  # (B, T)
        encodings = torch.zeros(
            first_indices.shape[0] * first_indices.shape[1],
            self.codebook_size,
            device=x.device,
        )
        encodings.scatter_(1, first_indices.reshape(-1, 1), 1)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "reconstructed": reconstructed,
            "indices": indices,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
        }
