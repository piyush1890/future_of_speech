"""
v8 phoneme-level articulatory TTS.

Architecture:
  text → phoneme_emb + speaker_emb (+ optional style_emb)
       → transformer encoder (per-phoneme contextualization)
       → 3 heads:
           start_features:  Linear(d_model, 14)
           end_features:    Linear(d_model, 14)
           duration:        small MLP, scalar per phoneme
       → interpolation (in inference; in training we use GT durations + force-render)
       → 50Hz frame stream → SPARC → audio

Loss (training):
  L = MSE(pred_start, gt_start) + MSE(pred_end, gt_end) + MSE(log(pred_dur), log(gt_dur))

Padding: phoneme_mask (B, N) — True for valid phoneme positions, False for padding.
"""
import math

import torch
import torch.nn as nn

from .interpolator import LinearInterpolator, HMMInterpolator, HybridInterpolator
from .phoneme_classes import build_render_class_table


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PhonemeTTSv8(nn.Module):
    def __init__(
        self,
        vocab_size: int = 73,
        feature_dim: int = 14,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        speaker_emb_dim: int = 64,
        style_dim: int = None,
        render_mode: str = "hybrid",       # "hybrid" | "hmm" | "linear"
        render_class_table: torch.Tensor = None,   # (vocab_size,) for hybrid
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.render_mode = render_mode

        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, dropout=dropout)

        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        self.style_proj = nn.Linear(style_dim if style_dim is not None else d_model, d_model)
        nn.init.zeros_(self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Heads (3 anchors: start, mid, end)
        self.start_head = nn.Linear(d_model, feature_dim)
        self.mid_head   = nn.Linear(d_model, feature_dim)
        self.end_head   = nn.Linear(d_model, feature_dim)
        # Duration: predict log(duration). Small 2-layer MLP.
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        if render_mode == "hmm":
            self.interpolator = HMMInterpolator()
        elif render_mode == "linear":
            self.interpolator = LinearInterpolator()
        elif render_mode == "hybrid":
            self.interpolator = HybridInterpolator()
            if render_class_table is None:
                raise ValueError("hybrid render_mode requires render_class_table")
            self.register_buffer("render_class_table", render_class_table.long(), persistent=True)
        else:
            raise ValueError(f"Unknown render_mode: {render_mode}")

    def encode(self, phoneme_ids, speaker_emb, phoneme_mask, style_emb=None):
        """phoneme_ids (B, N), speaker_emb (B, 64), phoneme_mask (B, N) bool."""
        x = self.phoneme_embedding(phoneme_ids)         # (B, N, D)
        x = self.encoder_pe(x)
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + spk
        if style_emb is not None:
            x = x + style_emb                            # per-phoneme style addition
        # padding_mask: True where padded
        padding_mask = ~phoneme_mask if phoneme_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x                                         # (B, N, D)

    def predict(self, phoneme_ids, speaker_emb, phoneme_mask, style_emb=None):
        """Returns (start, mid, end, log_dur) all (B, N, *)."""
        h = self.encode(phoneme_ids, speaker_emb, phoneme_mask, style_emb)
        start = self.start_head(h)
        mid   = self.mid_head(h)
        end   = self.end_head(h)
        log_dur = self.duration_head(h).squeeze(-1)      # (B, N)
        return start, mid, end, log_dur

    def _render(self, phoneme_ids, start, mid, end, durations):
        if self.render_mode == "hybrid":
            render_class = self.render_class_table[phoneme_ids]
            return self.interpolator(start, mid, end, durations, render_class)
        return self.interpolator(start, mid, end, durations)

    def forward(self, phoneme_ids, speaker_emb, phoneme_mask, style_emb=None,
                gt_durations=None):
        """If gt_durations given (training), use them. Else use predicted durations."""
        start, mid, end, log_dur = self.predict(phoneme_ids, speaker_emb, phoneme_mask, style_emb)
        if gt_durations is None:
            durations = torch.exp(log_dur).round().long().clamp(min=1)
            durations = durations * phoneme_mask.long()
        else:
            durations = gt_durations
        frames, frame_mask = self._render(phoneme_ids, start, mid, end, durations)
        return {
            "start": start, "mid": mid, "end": end,
            "log_dur": log_dur,
            "durations": durations,
            "frames": frames,
            "frame_mask": frame_mask,
        }

    @torch.no_grad()
    def generate(self, phoneme_ids, speaker_emb, style_emb=None, duration_scale: float = 1.0):
        self.eval()
        phoneme_mask = phoneme_ids != 0
        start, mid, end, log_dur = self.predict(phoneme_ids, speaker_emb, phoneme_mask, style_emb)
        durations = (torch.exp(log_dur) * duration_scale).round().long().clamp(min=1)
        durations = durations * phoneme_mask.long()
        frames, frame_mask = self._render(phoneme_ids, start, mid, end, durations)
        return frames, durations, frame_mask
