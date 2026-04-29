"""
v10 frame-level RVQ tokenizer.

  (B, T, 14) articulator features  →  Transformer encoder
                                   →  ResidualVQ (K levels, codebook_size)
                                   →  Transformer decoder
                                   →  (B, T, 14) reconstruction

Per-frame K token IDs (B, T, K) are the discrete handle.
Token count for a phoneme = number of frames it spans = its duration.
No phoneme conditioning here — the tokenizer is pure articulator → token mapping.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


class _PE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class V10Tokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        codebook_size: int = 1024,
        num_quantizers: int = 4,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        max_frames: int = 1024,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.K = num_quantizers
        self.codebook_size = codebook_size

        self.proj_in = nn.Linear(input_dim, d_model)
        self.enc_pe = _PE(d_model, max_len=max_frames)
        self.dec_pe = _PE(d_model, max_len=max_frames)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.rvq = ResidualVQ(
            dim=d_model, num_quantizers=num_quantizers, codebook_size=codebook_size,
            decay=ema_decay, commitment_weight=commitment_weight,
            kmeans_init=True, kmeans_iters=10,
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_decoder_layers)

        self.proj_out = nn.Linear(d_model, input_dim)

    def encode(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 14), frame_mask: (B, T) bool (True at valid). Returns (B, T, D)."""
        x = self.proj_in(frames)
        x = self.enc_pe(x)
        return self.encoder(x, src_key_padding_mask=~frame_mask)

    def quantize(self, z: torch.Tensor):
        """z: (B, T, D) → (z_q (B,T,D), idx (B,T,K), commit_loss scalar)."""
        z_q, idx, commit = self.rvq(z)
        if commit.dim() > 0:
            commit = commit.sum()
        return z_q, idx, commit

    def decode_z(self, z_q: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """z_q: (B, T, D) → (B, T, 14)."""
        x = self.dec_pe(z_q)
        x = self.decoder(x, src_key_padding_mask=~frame_mask)
        return self.proj_out(x)

    def forward(self, frames: torch.Tensor, frame_mask: torch.Tensor):
        """frames: (B, T, 14); frame_mask: (B, T) bool. Returns dict."""
        z = self.encode(frames, frame_mask)
        z_q, idx, commit = self.quantize(z)
        recon = self.decode_z(z_q, frame_mask)
        return {"recon": recon, "idx": idx, "commit_loss": commit}

    @torch.no_grad()
    def tokens_to_frames(self, idx: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """Inference: (B, T, K) token IDs → (B, T, 14) reconstruction."""
        z_q = self.rvq.get_output_from_indices(idx)            # (B, T, D)
        return self.decode_z(z_q, frame_mask)
