"""
v10 renderer (Stage 1) — frame-level AR over RVQ tokens with per-frame EOP.

Encoder (over N+2 phonemes):
  phoneme_emb + style_code_emb + speaker + knobs → bidirectional transformer
                                                    → enc (B, N+2, D)

Decoder (over T frames, causal AR):
  per-frame input:
    shifted prev-frame token emb (sum over K RVQ levels)
    + phoneme_idx_emb (current phoneme position; from frame_to_enc_pos)
    + frame positional encoding
  cross-attention: full attention over encoder positions
  causal self-attention over frames

Heads:
  K hierarchical RVQ heads → frame token logits (B, T, K, C)
  1 EOP head (linear) → (B, T) — sigmoid + BCE during training
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HierarchicalRVQHeads(nn.Module):
    """K logits heads where head_k attends to embedded tokens from levels 0..k-1.

    Same idea as v9. Used at every frame position (not phoneme).
    """
    def __init__(self, d_model: int, num_quantizers: int, codebook_size: int):
        super().__init__()
        self.K = num_quantizers
        self.codebook_size = codebook_size
        self.proj = nn.ModuleList([nn.Linear(d_model, codebook_size) for _ in range(num_quantizers)])
        self.cb_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model)
            for _ in range(num_quantizers)
        ])
        self.SOS_ID = codebook_size

    def forward(self, h: torch.Tensor, gt_tokens: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D); gt_tokens: (B, T, K). Returns (B, T, K, C)."""
        logits_per_level = []
        running = h
        for k in range(self.K):
            logits_per_level.append(self.proj[k](running))
            if k < self.K - 1:
                running = running + self.cb_emb[k](gt_tokens[..., k])
        return torch.stack(logits_per_level, dim=2)

    def step_logits_one(self, h_pos: torch.Tensor, k: int,
                        prev_tokens_for_pos: torch.Tensor) -> torch.Tensor:
        """h_pos: (B, D); prev_tokens_for_pos: (B, k). Returns (B, C)."""
        running = h_pos.clone()
        for j in range(k):
            running = running + self.cb_emb[j](prev_tokens_for_pos[:, j])
        return self.proj[k](running)


class V10Renderer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 73,
        codebook_size: int = 1024,
        num_quantizers: int = 4,
        style_codebook_size: int = 64,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        speaker_emb_dim: int = 64,
        knob_dim: int = 0,
        knob_dropout: float = 0.3,
        max_phonemes: int = 256,
        max_frames: int = 800,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.K = num_quantizers
        self.style_codebook_size = style_codebook_size
        self.d_model = d_model
        self.knob_dim = knob_dim
        self.knob_dropout = knob_dropout
        self.max_phonemes = max_phonemes
        self.max_frames = max_frames

        # Encoder over phonemes
        self.phoneme_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.style_code_emb = nn.Embedding(style_codebook_size + 1, d_model,
                                           padding_idx=style_codebook_size)
        self.encoder_pe = _PE(d_model, max_len=max_phonemes + 8)

        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        if knob_dim > 0:
            self.knob_proj = nn.Sequential(
                nn.Linear(knob_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            )
        else:
            self.knob_proj = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # Decoder over frames
        # Phoneme-index embedding: vocab covers BOS(0), body(1..max_phonemes), EOS(max_phonemes+1)
        self.ph_idx_emb = nn.Embedding(max_phonemes + 2, d_model)
        self.frame_pe = _PE(d_model, max_len=max_frames + 8)

        # Previous-frame token embeddings (one per RVQ level)
        self.prev_token_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_quantizers)
        ])
        self.bos_step = nn.Parameter(torch.randn(d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.heads = HierarchicalRVQHeads(d_model, num_quantizers, codebook_size)
        self.eop_head = nn.Linear(d_model, 1)

    def encode(self, phoneme_ids, style_codes, spk_emb, knobs, phoneme_mask,
               force_drop_knobs: bool = False):
        """Returns enc: (B, N+2, D)."""
        x = self.phoneme_emb(phoneme_ids) + self.style_code_emb(style_codes)
        spk = self.speaker_proj(spk_emb).unsqueeze(1)
        x = x + spk
        if self.knob_proj is not None and knobs is not None and self.knob_dim > 0:
            knob_vec = self.knob_proj(knobs)
            if self.training and self.knob_dropout > 0:
                drop = (torch.rand(knobs.shape[0], device=knobs.device)
                        > self.knob_dropout).float()
                knob_vec = knob_vec * drop.unsqueeze(-1)
            if force_drop_knobs:
                knob_vec = torch.zeros_like(knob_vec)
            x = x + knob_vec.unsqueeze(1)
        x = self.encoder_pe(x)
        return self.encoder(x, src_key_padding_mask=~phoneme_mask)

    def _make_decoder_input(self, frame_codes: torch.Tensor,
                            frame_to_enc_pos: torch.Tensor) -> torch.Tensor:
        """frame_codes: (B, T, K); frame_to_enc_pos: (B, T). Returns (B, T, D)."""
        B, T, K = frame_codes.shape
        D = self.d_model
        device = frame_codes.device

        # Sum prev-frame token embeddings across K levels
        tok_emb = torch.zeros(B, T, D, device=device)
        for k in range(K):
            tok_emb = tok_emb + self.prev_token_emb[k](frame_codes[..., k])

        # Shift right by 1: position 0 is BOS, position t (t>=1) is tok_emb[t-1]
        shifted = torch.zeros(B, T, D, device=device, dtype=tok_emb.dtype)
        shifted[:, 0] = self.bos_step.unsqueeze(0).expand(B, -1)
        if T > 1:
            shifted[:, 1:] = tok_emb[:, :-1]

        ph_emb = self.ph_idx_emb(frame_to_enc_pos)            # (B, T, D)
        dec_inp = shifted + ph_emb
        return self.frame_pe(dec_inp)

    def forward(self, phoneme_ids, style_codes, spk_emb, knobs, phoneme_mask,
                frame_codes, frame_to_enc_pos, frame_mask, force_drop_knobs: bool = False):
        """Training-time forward (teacher-forced).

        phoneme_ids:      (B, N+2) long
        style_codes:      (B, N+2) long (PAD at BOS/EOS)
        spk_emb:          (B, 64) float
        knobs:            (B, knob_dim) float
        phoneme_mask:     (B, N+2) bool
        frame_codes:      (B, T, K) long — GT frame RVQ tokens
        frame_to_enc_pos: (B, T) long — values in [1, N+1] for valid body frames
        frame_mask:       (B, T) bool

        Returns: dict with frame_logits (B,T,K,C), eop_logit (B,T)
        """
        enc = self.encode(phoneme_ids, style_codes, spk_emb, knobs, phoneme_mask,
                          force_drop_knobs)                                          # (B, N+2, D)

        dec_inp = self._make_decoder_input(frame_codes, frame_to_enc_pos)           # (B, T, D)
        T = dec_inp.size(1)
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=dec_inp.device), diagonal=1
        )
        h = self.decoder(
            dec_inp, enc,
            tgt_mask=causal,
            tgt_key_padding_mask=~frame_mask,
            memory_key_padding_mask=~phoneme_mask,
        )                                                                           # (B, T, D)

        frame_logits = self.heads(h, frame_codes)                                   # (B, T, K, C)
        eop_logit = self.eop_head(h).squeeze(-1)                                    # (B, T)
        return {"frame_logits": frame_logits, "eop_logit": eop_logit, "h": h}
