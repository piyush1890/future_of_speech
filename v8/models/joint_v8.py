"""
Joint v8 model: planner + phoneme TTS in one nn.Module, trained end-to-end.

Forward:
  text + speaker + V/A/D knobs
       → planner produces per-phoneme style embeddings
       → v8 encoder + heads produce (start, mid, end, log_dur) per phoneme
       → interpolator produces 50Hz frame stream
"""
import torch
import torch.nn as nn

from .phoneme_tts import PhonemeTTSv8
from .v8_planner import V8Planner


class JointV8(nn.Module):
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
        # Planner config
        planner_d_model: int = 128,
        planner_layers: int = 4,
        planner_d_ff: int = 512,
        planner_nhead: int = 4,
        knob_dim: int = 3,
        knob_dropout: float = 0.1,
        render_mode: str = "hmm",
    ):
        super().__init__()

        self.planner = V8Planner(
            vocab_size=vocab_size,
            d_model=planner_d_model,
            nhead=planner_nhead,
            num_layers=planner_layers,
            d_ff=planner_d_ff,
            dropout=dropout,
            knob_dim=knob_dim,
            speaker_emb_dim=speaker_emb_dim,
            style_dim=d_model,                # output matches v8 encoder d_model
            knob_dropout=knob_dropout,
        )

        self.tts = PhonemeTTSv8(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            speaker_emb_dim=speaker_emb_dim,
            style_dim=d_model,
            render_mode=render_mode,
        )

    def predict(self, phoneme_ids, speaker_emb, phoneme_mask, knobs,
                force_drop_knobs: bool = False):
        style_emb = self.planner(
            phoneme_ids, speaker_emb, knobs, phoneme_mask,
            force_drop_knobs=force_drop_knobs,
        )                                                # (B, N, d_model)
        start, mid, end, log_dur = self.tts.predict(
            phoneme_ids, speaker_emb, phoneme_mask, style_emb=style_emb,
        )
        return start, mid, end, log_dur, style_emb

    def forward(self, phoneme_ids, speaker_emb, phoneme_mask, knobs,
                gt_durations=None):
        start, mid, end, log_dur, style_emb = self.predict(
            phoneme_ids, speaker_emb, phoneme_mask, knobs,
        )
        if gt_durations is None:
            durations = torch.exp(log_dur).round().long().clamp(min=1) * phoneme_mask.long()
        else:
            durations = gt_durations
        frames, frame_mask = self.tts.interpolator(start, mid, end, durations)
        return {
            "start": start, "mid": mid, "end": end,
            "log_dur": log_dur, "durations": durations,
            "frames": frames, "frame_mask": frame_mask,
            "style_emb": style_emb,
        }

    @torch.no_grad()
    def generate(self, phoneme_ids, speaker_emb, knobs, duration_scale: float = 1.0,
                 cfg_scale: float = 1.0):
        """
        cfg_scale: classifier-free guidance.
            1.0 = use planner with knobs only.
            >1 = sharpen knob effect:
                style = uncond_style + cfg_scale * (cond_style - uncond_style)
        """
        self.eval()
        phoneme_mask = phoneme_ids != 0

        if cfg_scale != 1.0:
            uncond = self.planner(phoneme_ids, speaker_emb, knobs, phoneme_mask, force_drop_knobs=True)
            cond   = self.planner(phoneme_ids, speaker_emb, knobs, phoneme_mask, force_drop_knobs=False)
            style_emb = uncond + cfg_scale * (cond - uncond)
        else:
            style_emb = self.planner(phoneme_ids, speaker_emb, knobs, phoneme_mask)

        start, mid, end, log_dur = self.tts.predict(
            phoneme_ids, speaker_emb, phoneme_mask, style_emb=style_emb,
        )
        durations = (torch.exp(log_dur) * duration_scale).round().long().clamp(min=1)
        durations = durations * phoneme_mask.long()
        frames, frame_mask = self.tts.interpolator(start, mid, end, durations)
        return frames, durations, frame_mask
