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

        # Compute max output length (single CPU sync)
        output_lengths = durations_int.sum(dim=1)  # (B,)
        T = target_len if target_len is not None else int(output_lengths.max().item())

        expanded = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)

        # Use repeat_interleave per batch (one sync per batch item, not per phoneme)
        arange_N = torch.arange(N, device=x.device)
        for b in range(B):
            idx = torch.repeat_interleave(arange_N, durations_int[b])
            T_b = min(idx.shape[0], T)
            if T_b > 0:
                expanded[b, :T_b] = x[b, idx[:T_b]]
                mask[b, :T_b] = True

        return expanded, mask
