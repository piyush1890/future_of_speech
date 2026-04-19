"""
Length regulator: expands phoneme-level representations to frame-level
by repeating each phoneme embedding according to its duration.
"""
import torch
import torch.nn as nn


class LengthRegulator(nn.Module):
    """Expand phoneme embeddings to frame-level based on durations."""

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        target_len: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) phoneme-level embeddings
            durations: (B, N) integer durations per phoneme
            target_len: if provided, pad/trim to this length
        Returns:
            expanded: (B, T, D) frame-level embeddings
            mask: (B, T) boolean mask (True = valid frame)
        """
        B, N, D = x.shape
        durations_int = durations.long()

        # Compute max output length
        output_lengths = durations_int.sum(dim=1)  # (B,)
        T = target_len if target_len is not None else output_lengths.max().item()

        expanded = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)

        for b in range(B):
            pos = 0
            for n in range(N):
                dur = durations_int[b, n].item()
                if dur <= 0:
                    continue
                end = min(pos + dur, T)
                if pos < T:
                    expanded[b, pos:end] = x[b, n]
                    mask[b, pos:end] = True
                pos = end
                if pos >= T:
                    break

        return expanded, mask
