"""
Reference-audio style encoder.

Takes a reference utterance's SPARC features (T, 14) and compresses them into
a fixed-length style vector (style_dim). The model learns what aspects of the
reference to encode — pitch contour shape, energy dynamics, speaking rate,
articulatory tension — without us specifying rules.

Architecture: 1D CNN stack (compresses time) → GRU (aggregates) → linear projection.
~0.3M params at default settings.

During training: encode the ground-truth utterance's own features → style_vec.
During inference: encode a reference clip's SPARC features → style_vec.
"""
import torch
import torch.nn as nn


class StyleEncoder(nn.Module):
    def __init__(self, input_dim: int = 14, style_dim: int = 256,
                 hidden: int = 128, n_conv_layers: int = 4):
        super().__init__()
        convs = []
        in_ch = input_dim
        for i in range(n_conv_layers):
            out_ch = hidden
            convs.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        self.convs = nn.Sequential(*convs)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, style_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, 14) — SPARC features of the reference utterance.
        Returns:  (B, style_dim)
        """
        x = features.transpose(1, 2)       # (B, 14, T)
        x = self.convs(x)                  # (B, hidden, T // 2^n_conv)
        x = x.transpose(1, 2)              # (B, T_compressed, hidden)
        _, h = self.gru(x)                 # h: (2, B, hidden) — bidirectional
        h = torch.cat([h[0], h[1]], dim=1) # (B, hidden*2)
        return self.proj(h)                # (B, style_dim)
