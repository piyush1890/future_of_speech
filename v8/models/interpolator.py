"""
Two render modes for per-phoneme (start, mid, end) anchors → 50Hz frame stream.

linear (default-old): two linear segments [start→mid] and [mid→end].
                      Mid is a passing-through point. Long durations smear stops.

hmm (default-new): HMM-style begin / sustained-mid / end states.
                   1 frame at start, (d-2) frames at mid (plateau), 1 frame at end.
                   Duration only stretches the mid plateau — stops don't smear.
                   Matches HMM-TTS state-traversal model linguistically:
                     - vowels: 1 onset + N×(sustained core) + 1 offset
                     - stops:  1 closure + 0–1×(brief burst) + 1 release
"""
import torch
import torch.nn as nn


def linear_interpolate_3pt(start: torch.Tensor, mid: torch.Tensor, end: torch.Tensor,
                           durations: torch.Tensor, max_frames: int = None):
    """
    Args:
        start, mid, end: (B, N, F)  three anchor features per phoneme
        durations:        (B, N)     int64, frames per phoneme
        max_frames: optional total length; default = max(sum(durations))

    Returns:
        frames:    (B, T_total, F)
        frame_mask:(B, T_total) bool
    """
    B, N, F = start.shape
    device = start.device
    total_per_b = durations.sum(dim=1)
    if max_frames is None:
        max_frames = int(total_per_b.max().item())

    out = torch.zeros(B, max_frames, F, device=device, dtype=start.dtype)
    mask = torch.zeros(B, max_frames, device=device, dtype=torch.bool)

    for b in range(B):
        cursor = 0
        for i in range(N):
            d = int(durations[b, i].item())
            if d <= 0:
                continue
            if d == 1:
                out[b, cursor] = mid[b, i]
            elif d == 2:
                out[b, cursor]     = start[b, i]
                out[b, cursor + 1] = end[b, i]
            else:
                half = d // 2
                # Segment 1: start → mid over `half` frames
                t1 = torch.linspace(0.0, 1.0, half, device=device).unsqueeze(-1)  # (half, 1)
                out[b, cursor:cursor + half] = start[b, i].unsqueeze(0) * (1 - t1) + mid[b, i].unsqueeze(0) * t1
                # Segment 2: mid → end over `d - half` frames
                rem = d - half
                t2 = torch.linspace(0.0, 1.0, rem, device=device).unsqueeze(-1)
                out[b, cursor + half:cursor + d] = mid[b, i].unsqueeze(0) * (1 - t2) + end[b, i].unsqueeze(0) * t2
            mask[b, cursor:cursor + d] = True
            cursor += d
    return out, mask


class LinearInterpolator(nn.Module):
    """3-anchor linear interpolator. Two linear segments per phoneme: start→mid, mid→end."""
    def forward(self, start, mid, end, durations, max_frames=None):
        return linear_interpolate_3pt(start, mid, end, durations, max_frames)


def hmm_render_3pt(start: torch.Tensor, mid: torch.Tensor, end: torch.Tensor,
                   durations: torch.Tensor, max_frames: int = None):
    """
    HMM-style: 1 frame begin (start), N-2 frames mid plateau, 1 frame end.

    For very short durations:
      d == 1 → mid only
      d == 2 → [start, end]
      d == 3 → [start, mid, end]
      d >= 4 → [start, mid, mid, ..., mid, end]   (plateau dominates)
    """
    B, N, F = start.shape
    device = start.device
    total_per_b = durations.sum(dim=1)
    if max_frames is None:
        max_frames = int(total_per_b.max().item())

    out = torch.zeros(B, max_frames, F, device=device, dtype=start.dtype)
    mask = torch.zeros(B, max_frames, device=device, dtype=torch.bool)

    for b in range(B):
        cursor = 0
        for i in range(N):
            d = int(durations[b, i].item())
            if d <= 0:
                continue
            if d == 1:
                out[b, cursor] = mid[b, i]
            elif d == 2:
                out[b, cursor]     = start[b, i]
                out[b, cursor + 1] = end[b, i]
            elif d == 3:
                out[b, cursor]     = start[b, i]
                out[b, cursor + 1] = mid[b, i]
                out[b, cursor + 2] = end[b, i]
            else:
                out[b, cursor] = start[b, i]
                # Plateau of mid for d-2 frames
                out[b, cursor + 1:cursor + d - 1] = mid[b, i].unsqueeze(0)
                out[b, cursor + d - 1] = end[b, i]
            mask[b, cursor:cursor + d] = True
            cursor += d
    return out, mask


class HMMInterpolator(nn.Module):
    """3-anchor HMM-style renderer. Begin, sustained-mid plateau, end."""
    def forward(self, start, mid, end, durations, max_frames=None):
        return hmm_render_3pt(start, mid, end, durations, max_frames)


def hybrid_render_3pt(start: torch.Tensor, mid: torch.Tensor, end: torch.Tensor,
                      durations: torch.Tensor, render_class: torch.Tensor,
                      max_frames: int = None):
    """
    Class-aware 3-anchor render:
      render_class[b, n] == 0 (PLATEAU)  → 1×start + (d-2)×mid + 1×end
      render_class[b, n] == 1 (LINEAR)   → linear start→mid→end (two segments)

    Args:
        start, mid, end: (B, N, F)
        durations:        (B, N) int64
        render_class:     (B, N) long, per-phoneme dispatch (0=plateau, 1=linear)
    """
    B, N, F = start.shape
    device = start.device
    total_per_b = durations.sum(dim=1)
    if max_frames is None:
        max_frames = int(total_per_b.max().item())

    out = torch.zeros(B, max_frames, F, device=device, dtype=start.dtype)
    mask = torch.zeros(B, max_frames, device=device, dtype=torch.bool)

    for b in range(B):
        cursor = 0
        for i in range(N):
            d = int(durations[b, i].item())
            if d <= 0:
                continue
            cls = int(render_class[b, i].item())
            if d == 1:
                out[b, cursor] = mid[b, i]
            elif d == 2:
                out[b, cursor]     = start[b, i]
                out[b, cursor + 1] = end[b, i]
            elif cls == 0:    # PLATEAU
                if d == 3:
                    out[b, cursor]     = start[b, i]
                    out[b, cursor + 1] = mid[b, i]
                    out[b, cursor + 2] = end[b, i]
                else:
                    out[b, cursor] = start[b, i]
                    out[b, cursor + 1:cursor + d - 1] = mid[b, i].unsqueeze(0)
                    out[b, cursor + d - 1] = end[b, i]
            else:             # LINEAR (continuous motion through the phoneme)
                if d == 3:
                    out[b, cursor]     = start[b, i]
                    out[b, cursor + 1] = mid[b, i]
                    out[b, cursor + 2] = end[b, i]
                else:
                    half = d // 2
                    t1 = torch.linspace(0.0, 1.0, half, device=device).unsqueeze(-1)
                    out[b, cursor:cursor + half] = start[b, i].unsqueeze(0) * (1 - t1) + mid[b, i].unsqueeze(0) * t1
                    rem = d - half
                    t2 = torch.linspace(0.0, 1.0, rem, device=device).unsqueeze(-1)
                    out[b, cursor + half:cursor + d] = mid[b, i].unsqueeze(0) * (1 - t2) + end[b, i].unsqueeze(0) * t2
            mask[b, cursor:cursor + d] = True
            cursor += d
    return out, mask


class HybridInterpolator(nn.Module):
    """Class-aware 3-anchor interpolator: plateau for sustained sounds, linear for transient."""
    def forward(self, start, mid, end, durations, render_class, max_frames=None):
        return hybrid_render_3pt(start, mid, end, durations, render_class, max_frames)
