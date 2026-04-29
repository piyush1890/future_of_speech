"""
v9 predictor (Step 3): AR transformer that emits per-phoneme RVQ tokens
+ duration from text + speaker + (optional) knobs.

Per phoneme i, the predictor emits:
  - 4 start-stack RVQ logits  (B, N, 4, codebook_size)
  - 4 end-stack   RVQ logits  (B, N, 4, codebook_size)
  - 1 log-duration scalar     (B, N)

Architecture follows v5 patterns:
  - Bidirectional encoder over phonemes + speaker projection + knob conditioning
    (with knob_dropout for CFG support).
  - AR decoder with causal mask: at each phoneme position, hidden state
    aggregates encoder context (cross-attention) AND previous phonemes'
    teacher-forced token embeddings (self-attention with causal mask).
  - Hierarchical RVQ heads (head_k conditions on tokens 0..k-1) — proven in v5.
  - Duration head off the same AR hidden state (no separate sequential pass).

Inference: greedy or sampled, with optional classifier-free guidance via
running the model twice (knobs zeroed vs full) and blending logits.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class HierarchicalRVQHeads(nn.Module):
    """K logits heads where head_k attends to embedded tokens from levels 0..k-1.

    During training (teacher forcing): we have GT token IDs at all levels, so we
    can compute all K heads in parallel by feeding in true previous-level tokens.

    During inference (greedy/sampling): caller iterates k=0..K-1, samples token,
    feeds back into the next head's input.
    """
    def __init__(self, d_model: int, num_quantizers: int, codebook_size: int):
        super().__init__()
        self.K = num_quantizers
        self.codebook_size = codebook_size
        # Each level has its own logits projection
        self.proj = nn.ModuleList([nn.Linear(d_model, codebook_size) for _ in range(num_quantizers)])
        # Embeddings for previously-emitted tokens at each level
        # (only k-1 embeddings are used by head k; we keep K of them anyway)
        self.cb_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model)             # +1 for "not yet generated"
            for _ in range(num_quantizers)
        ])
        # Special "not yet generated" token id = codebook_size (last index)
        self.SOS_ID = codebook_size

    def forward(self, h: torch.Tensor, gt_tokens: torch.Tensor) -> torch.Tensor:
        """Training-time parallel forward.

        h:         (B, N, d_model) — AR hidden state at each phoneme
        gt_tokens: (B, N, K)       — GT token IDs (teacher forcing)
        Returns logits: (B, N, K, codebook_size)
        """
        B, N, _ = h.shape
        logits_per_level = []
        running = h
        for k in range(self.K):
            logits_per_level.append(self.proj[k](running))
            if k < self.K - 1:
                # Embed GT token at level k and add to running input for next level
                running = running + self.cb_emb[k](gt_tokens[..., k])
        return torch.stack(logits_per_level, dim=2)              # (B, N, K, C)

    def step_logits_one(self, h_pos: torch.Tensor, k: int,
                        prev_tokens_for_pos: torch.Tensor) -> torch.Tensor:
        """Inference helper: compute logits at level k given previously sampled
        tokens at levels 0..k-1 for ONE phoneme position.

        h_pos:               (B, d_model)
        prev_tokens_for_pos: (B, k) long — sampled tokens at levels 0..k-1
        Returns: (B, codebook_size)
        """
        running = h_pos.clone()
        for j in range(k):
            running = running + self.cb_emb[j](prev_tokens_for_pos[:, j])
        return self.proj[k](running)


class V9Predictor(nn.Module):
    def __init__(
        self,
        vocab_size: int = 73,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        speaker_emb_dim: int = 64,
        knob_dim: int = 0,
        knob_dropout: float = 0.1,
        max_phonemes: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.K = num_quantizers
        self.d_model = d_model
        self.knob_dim = knob_dim
        self.knob_dropout = knob_dropout

        self.phoneme_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, max_len=max_phonemes + 8)
        self.decoder_pe = PositionalEncoding(d_model, max_len=max_phonemes + 8)

        # Speaker + knob conditioners (added to encoder input, broadcast over N)
        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        if knob_dim > 0:
            self.knob_proj = nn.Sequential(
                nn.Linear(knob_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            )
        else:
            self.knob_proj = None

        # Bidirectional encoder over phonemes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # AR decoder: causal self-attention over previous-phoneme outputs +
        # cross-attention to encoder context. Each step's input is the previous
        # phoneme's emitted (start, end) token embedding, projected to d_model.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Two hierarchical heads: one for start stack, one for end stack.
        # Each emits K=num_quantizers logits per phoneme.
        self.start_heads = HierarchicalRVQHeads(d_model, num_quantizers, codebook_size)
        self.end_heads   = HierarchicalRVQHeads(d_model, num_quantizers, codebook_size)

        # Duration head (log-space scalar)
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Embeddings used to feed previous-phoneme tokens into the AR decoder
        # (sum of K start-token embeddings + K end-token embeddings + a learnable
        # BOS-step embedding for position 0).
        self.prev_start_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_quantizers)
        ])
        self.prev_end_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_quantizers)
        ])
        self.bos_step = nn.Parameter(torch.randn(d_model) * 0.02)

    def _make_decoder_input(self, start_tokens, end_tokens):
        """Build the AR decoder's input sequence by shifting GT tokens right by 1.

        start_tokens, end_tokens: (B, N, K) long (GT, teacher-forced)
        Returns: (B, N, d_model) — input sequence for decoder
                 position 0 = BOS step embedding
                 position i (i>=1) = sum of prev-(start, end) token embeddings at i-1
        """
        B, N, K = start_tokens.shape
        device = start_tokens.device
        # Per-position embedding: sum across K levels for both start and end
        emb = torch.zeros(B, N, self.d_model, device=device)
        for k in range(K):
            emb = emb + self.prev_start_emb[k](start_tokens[..., k])
            emb = emb + self.prev_end_emb[k](end_tokens[..., k])
        # Right-shift by 1: position 0 gets BOS embedding, position i (i>=1) sees i-1's emb
        shifted = torch.zeros_like(emb)
        shifted[:, 0] = self.bos_step.unsqueeze(0).expand(B, -1)
        shifted[:, 1:] = emb[:, :-1]
        return shifted

    def encode(self, phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=False):
        """Bidirectional encoder forward.

        phoneme_ids:  (B, N) long
        spk_emb:      (B, speaker_emb_dim)
        knobs:        (B, knob_dim) or None
        phoneme_mask: (B, N) bool — True at valid phoneme positions
        """
        x = self.phoneme_emb(phoneme_ids)                                  # (B, N, D)
        spk = self.speaker_proj(spk_emb).unsqueeze(1)                      # (B, 1, D)
        x = x + spk
        if self.knob_proj is not None and knobs is not None:
            knob_vec = self.knob_proj(knobs)                               # (B, D)
            # Knob dropout (training only): randomly zero whole knob vectors
            if self.training and self.knob_dropout > 0:
                drop_mask = (torch.rand(knobs.shape[0], device=knobs.device) > self.knob_dropout).float()
                knob_vec = knob_vec * drop_mask.unsqueeze(-1)
            if force_drop_knobs:
                knob_vec = torch.zeros_like(knob_vec)
            x = x + knob_vec.unsqueeze(1)
        x = self.encoder_pe(x)
        # Encoder: src_key_padding_mask=True at PADDING positions
        enc = self.encoder(x, src_key_padding_mask=~phoneme_mask)
        return enc                                                         # (B, N, D)

    def forward(self, phoneme_ids, spk_emb, knobs, phoneme_mask,
                gt_start_tokens, gt_end_tokens, force_drop_knobs=False):
        """Training-time forward (teacher-forced).

        Returns dict with:
          start_logits: (B, N, K, codebook_size)
          end_logits:   (B, N, K, codebook_size)
          log_dur:      (B, N)
        """
        enc = self.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs)

        # Build AR decoder input from shifted GT tokens
        dec_inp = self._make_decoder_input(gt_start_tokens, gt_end_tokens)
        dec_inp = self.decoder_pe(dec_inp)
        N = dec_inp.size(1)
        causal_mask = torch.triu(
            torch.full((N, N), float("-inf"), device=dec_inp.device), diagonal=1
        )
        # Decoder: tgt_key_padding_mask masks padded queries; memory_key_padding_mask masks padded encoder positions
        h = self.decoder(
            dec_inp, enc,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~phoneme_mask,
            memory_key_padding_mask=~phoneme_mask,
        )                                                                  # (B, N, D)

        start_logits = self.start_heads(h, gt_start_tokens)                 # (B, N, K, C)
        end_logits   = self.end_heads(h, gt_end_tokens)
        log_dur      = self.duration_head(h).squeeze(-1)                    # (B, N)
        return {
            "start_logits": start_logits,
            "end_logits":   end_logits,
            "log_dur":      log_dur,
        }
