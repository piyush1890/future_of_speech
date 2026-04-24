"""
Emotion-delta predictor.

Given a phoneme sequence + per-phoneme durations + emotion + speaker embedding,
predict a per-frame 14-dim delta that, when added to a neutral feature
trajectory for the same content, makes it sound emotional.

Architecture (~1-2M params):
  - phoneme embedding (V → d_model)
  - emotion embedding (4 → d_model) broadcast across positions
  - speaker projection (64 → d_model) broadcast across positions
  - phoneme-level transformer encoder (4 layers, 4 heads, d=128, ff=512)
  - length regulator (expand by given per-phoneme durations)
  - frame-level transformer encoder (4 layers)
  - linear head → 14
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return x + self.pe[:, :T]


def length_regulate(x: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-phoneme encodings to per-frame sequences.

    x:          (B, N, d)
    durations:  (B, N) integer; durations[b, n] = frames for phoneme n of batch b.
                Pad positions should have duration 0.
    returns:
      x_expanded: (B, T_max, d) zero-padded
      mask:       (B, T_max) True where valid

    Vectorized: uses torch.repeat_interleave instead of per-phoneme .item() syncs.
    """
    B, N, d = x.shape
    durs = durations.long()
    # One CPU sync for total lengths (T_max needed for allocation)
    totals = durs.sum(dim=1)           # (B,)
    T_max = int(totals.max().item())

    out = x.new_zeros(B, T_max, d)
    mask = torch.zeros(B, T_max, dtype=torch.bool, device=x.device)
    arange_N = torch.arange(N, device=x.device)
    for b in range(B):
        # repeat_interleave works on-device with no per-element sync
        idx = torch.repeat_interleave(arange_N, durs[b])    # (T_b,)
        T_b = idx.shape[0]
        if T_b > 0:
            out[b, :T_b] = x[b, idx]
            mask[b, :T_b] = True
    return out, mask


class EmotionDeltaPredictor(nn.Module):
    def __init__(self, vocab_size: int, n_emotions: int = 4,
                 spk_emb_dim: int = 64, d_model: int = 128, nhead: int = 4,
                 num_phon_layers: int = 4, num_frame_layers: int = 4,
                 d_ff: int = 512, dropout: float = 0.1, out_dim: int = 14,
                 predict_duration: bool = False,
                 variational: bool = False, min_sigma: float = 0.05):
        """`variational=True` adds a per-frame log-variance head. Forward returns
        (mu, log_var, mask) instead of (delta, mask). Sample at inference via
        `delta = mu + exp(0.5 * log_var) * randn`. Loss switches to Gaussian NLL.
        `min_sigma` floors sigma to prevent variance collapse."""
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.predict_duration = predict_duration
        self.variational = variational
        self.min_log_var = math.log(min_sigma ** 2)

        self.phon_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.emo_emb = nn.Embedding(n_emotions, d_model)
        self.spk_proj = nn.Linear(spk_emb_dim, d_model)
        self.phon_pe = SinusoidalPE(d_model)
        self.frame_pe = SinusoidalPE(d_model)

        phon_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.phon_encoder = nn.TransformerEncoder(phon_layer, num_layers=num_phon_layers)

        frame_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.frame_encoder = nn.TransformerEncoder(frame_layer, num_layers=num_frame_layers)

        self.head = nn.Linear(d_model, out_dim)
        if variational:
            self.var_head = nn.Linear(d_model, out_dim)
            # Init log-variance bias near min_log_var so model starts conservative
            nn.init.zeros_(self.var_head.weight)
            nn.init.constant_(self.var_head.bias, self.min_log_var + 0.5)
        if predict_duration:
            self.dur_head = nn.Linear(d_model, 1)

    def forward(
        self,
        phoneme_ids: torch.Tensor,     # (B, N)
        durations: torch.Tensor,       # (B, N)
        emotion_idx: torch.Tensor,     # (B,)
        speaker_emb: torch.Tensor,     # (B, 64)
    ):
        """If variational: returns (mu, log_var, frame_mask).
        Else returns (delta, frame_mask)."""
        B, N = phoneme_ids.shape
        phon_mask = phoneme_ids == 0   # pad id

        x = self.phon_emb(phoneme_ids)                  # (B, N, d)
        x = x + self.emo_emb(emotion_idx).unsqueeze(1)  # broadcast emotion
        x = x + self.spk_proj(speaker_emb).unsqueeze(1) # broadcast speaker
        x = self.phon_pe(x)
        x = self.phon_encoder(x, src_key_padding_mask=phon_mask)

        # Length regulator
        x_exp, frame_mask = length_regulate(x, durations)
        x_exp = self.frame_pe(x_exp)
        # Transformer over frames
        x_exp = self.frame_encoder(x_exp, src_key_padding_mask=~frame_mask)

        mu = self.head(x_exp)  # (B, T_max, 14)
        mu = mu * frame_mask.unsqueeze(-1).float()
        if self.variational:
            log_var = self.var_head(x_exp)
            log_var = log_var.clamp(min=self.min_log_var)
            # At pad positions, set log_var very negative so sigma → 0 (not sigma=1)
            pad_mask = (~frame_mask).unsqueeze(-1).float()
            log_var = log_var * frame_mask.unsqueeze(-1).float() + (-20.0) * pad_mask
            return mu, log_var, frame_mask
        return mu, frame_mask

    @torch.no_grad()
    def sample(self, phoneme_ids, durations, emotion_idx, speaker_emb,
               temperature: float = 1.0):
        """Variational sampling: returns delta = mu + temperature * sigma * eps.
        Falls back to deterministic mu when not variational."""
        self.eval()
        out = self.forward(phoneme_ids, durations, emotion_idx, speaker_emb)
        if self.variational:
            mu, log_var, fm = out
            sigma = torch.exp(0.5 * log_var)
            eps = torch.randn_like(mu)
            return mu + temperature * sigma * eps, fm
        else:
            mu, fm = out
            return mu, fm
