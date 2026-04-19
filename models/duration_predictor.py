"""
Duration predictor: predicts per-phoneme frame durations from encoder outputs.
FastSpeech-style convolutional predictor.
"""
import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    def __init__(self, d_model: int = 256, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.ln1 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.ln2 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) encoder output
            mask: (B, N) padding mask (True = valid)
        Returns:
            (B, N) predicted durations in frames (positive reals via softplus)
        """
        # Conv layers expect (B, C, T)
        out = x.transpose(1, 2)
        out = self.conv1(out).transpose(1, 2)
        out = self.dropout(self.relu(self.ln1(out)))
        out = out.transpose(1, 2)
        out = self.conv2(out).transpose(1, 2)
        out = self.dropout(self.relu(self.ln2(out)))

        out = self.linear(out).squeeze(-1)  # (B, N)
        out = nn.functional.softplus(out)    # positive durations

        if mask is not None:
            out = out * mask.float()

        return out
