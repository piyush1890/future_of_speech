"""
v9 Stage 1 (renderer): predicts per-phoneme RVQ tokens from
  (text, speaker, GT_style_codes, knobs)

Mirrors V9Predictor but adds a per-phoneme style-code embedding to the
encoder input. Style codes come from v5's frozen StyleCodebook (already
extracted to v8/data/phoneme_codes/<uid>.npz, 512-entry vocabulary).

At training time the GT style codes are fed in. At inference time the
V9StylePlanner generates them from text + knobs.

Output: per-phoneme RVQ tokens (4 start + 4 end levels) + log-duration.
This is identical to V9Predictor — only the input set differs.
"""
import math
import torch
import torch.nn as nn

from v9.models.v9_predictor import (
    PositionalEncoding, HierarchicalRVQHeads,
)


# Reserve PAD_CODE = style_codebook_size (one past the last valid code id);
# it gets a learned zero-style embedding via padding_idx.
PAD_CODE = 512


class V9Renderer(nn.Module):
    """v9 Stage 1 — phoneme-level RVQ token predictor with style-code conditioning.

    Identical to V9Predictor in topology, plus a `style_code_emb` lookup added
    to the encoder input.
    """
    def __init__(
        self,
        vocab_size: int = 73,
        codebook_size: int = 512,            # tokenizer's RVQ codebook size
        num_quantizers: int = 4,
        style_codebook_size: int = 512,      # v5 style codebook size
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
        self.style_codebook_size = style_codebook_size
        self.d_model = d_model
        self.knob_dim = knob_dim
        self.knob_dropout = knob_dropout

        self.phoneme_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # +1 for PAD style code (BOS/EOS positions)
        self.style_code_emb = nn.Embedding(style_codebook_size + 1, d_model,
                                           padding_idx=style_codebook_size)
        self.encoder_pe = PositionalEncoding(d_model, max_len=max_phonemes + 8)
        # Decoder runs at HALF-PHONEME rate (2N positions for N phonemes)
        self.decoder_pe = PositionalEncoding(d_model, max_len=2 * max_phonemes + 16)

        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        if knob_dim > 0:
            self.knob_proj = nn.Sequential(
                nn.Linear(knob_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            )
        else:
            self.knob_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.start_heads = HierarchicalRVQHeads(d_model, num_quantizers, codebook_size)
        self.end_heads   = HierarchicalRVQHeads(d_model, num_quantizers, codebook_size)

        self.duration_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # AR-shift embeddings. Decoder runs at HALF-PHONEME RATE: 2N positions
        # for an N-phoneme utterance, alternating (start[i], end[i], start[i+1], ...).
        # At each step the input is the embedding of the previously-emitted half:
        #   pos 0       : BOS
        #   pos 2i+1    : start[i]   (sum across K levels)
        #   pos 2i+2    : end[i]     (sum across K levels)
        # This makes end[i] prediction get start[i] info processed through the FULL
        # decoder transformer — symmetric to how start gets prev half info.
        self.prev_start_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_quantizers)
        ])
        self.prev_end_emb = nn.ModuleList([
            nn.Embedding(codebook_size + 1, d_model) for _ in range(num_quantizers)
        ])
        self.bos_step = nn.Parameter(torch.randn(d_model) * 0.02)

    def _make_decoder_input(self, start_tokens, end_tokens):
        """Build the 2N-length decoder input sequence (half-phoneme AR).

        start_tokens, end_tokens: (B, N, K)
        Returns:                  (B, 2N, d_model)

        Layout (target positions in the decoder output):
          pos 0       → predicts start[0]    (input: BOS)
          pos 1       → predicts end[0]      (input: start[0])
          pos 2       → predicts start[1]    (input: end[0])
          pos 3       → predicts end[1]      (input: start[1])
          ...
          pos 2i      → predicts start[i]    (input: end[i-1])
          pos 2i+1    → predicts end[i]      (input: start[i])

        At inference, end[i] sees start[i] processed through the full decoder
        transformer (deep, symmetric to how start sees previous halves).
        """
        B, N, K = start_tokens.shape
        device = start_tokens.device

        start_emb = torch.zeros(B, N, self.d_model, device=device)
        end_emb   = torch.zeros(B, N, self.d_model, device=device)
        for k in range(K):
            start_emb = start_emb + self.prev_start_emb[k](start_tokens[..., k])
            end_emb   = end_emb   + self.prev_end_emb[k](end_tokens[..., k])

        # Allocate (B, 2N, d_model)
        inputs = torch.zeros(B, 2 * N, self.d_model, device=device, dtype=start_emb.dtype)
        # Position 0: BOS
        inputs[:, 0] = self.bos_step.unsqueeze(0).expand(B, -1)
        # Positions 1, 3, 5, ..., 2N-1: start[0], start[1], ..., start[N-1]
        inputs[:, 1::2] = start_emb
        # Positions 2, 4, ..., 2N-2: end[0], end[1], ..., end[N-2]
        # (end[N-1] would go at position 2N which is past sequence end — not needed)
        if N > 1:
            inputs[:, 2::2] = end_emb[:, :-1]
        return inputs

    def encode(self, phoneme_ids, spk_emb, knobs, phoneme_mask, style_codes,
               force_drop_knobs=False):
        """style_codes: (B, N) long. PAD_CODE at BOS/EOS positions
        (caller sets up — see collate)."""
        x = self.phoneme_emb(phoneme_ids) + self.style_code_emb(style_codes)
        spk = self.speaker_proj(spk_emb).unsqueeze(1)
        x = x + spk
        if self.knob_proj is not None and knobs is not None:
            knob_vec = self.knob_proj(knobs)
            if self.training and self.knob_dropout > 0:
                drop_mask = (torch.rand(knobs.shape[0], device=knobs.device) > self.knob_dropout).float()
                knob_vec = knob_vec * drop_mask.unsqueeze(-1)
            if force_drop_knobs:
                knob_vec = torch.zeros_like(knob_vec)
            x = x + knob_vec.unsqueeze(1)
        x = self.encoder_pe(x)
        return self.encoder(x, src_key_padding_mask=~phoneme_mask)

    def forward(self, phoneme_ids, spk_emb, knobs, phoneme_mask, style_codes,
                gt_start_tokens, gt_end_tokens, force_drop_knobs=False):
        enc = self.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, style_codes,
                          force_drop_knobs)
        # Half-phoneme rate AR: decoder is 2N long.
        dec_inp = self._make_decoder_input(gt_start_tokens, gt_end_tokens)
        dec_inp = self.decoder_pe(dec_inp)
        L = dec_inp.size(1)                                                 # = 2N
        N = L // 2

        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=dec_inp.device), diagonal=1
        )
        # Decoder padding mask: each phoneme contributes 2 decoder positions
        # (start at 2i, end at 2i+1). Both valid iff the phoneme position is.
        dec_pad_mask = (~phoneme_mask).repeat_interleave(2, dim=1)           # (B, 2N)
        h = self.decoder(
            dec_inp, enc, tgt_mask=causal_mask,
            tgt_key_padding_mask=dec_pad_mask,
            memory_key_padding_mask=~phoneme_mask,
        )                                                                    # (B, 2N, D)

        # Split decoder output:
        h_start = h[:, 0::2, :]                                              # (B, N, D)
        h_end   = h[:, 1::2, :]
        start_logits = self.start_heads(h_start, gt_start_tokens)            # (B, N, K, C)
        end_logits   = self.end_heads(h_end, gt_end_tokens)                  # (B, N, K, C)
        log_dur      = self.duration_head(h_start).squeeze(-1)               # (B, N)
        return {
            "start_logits": start_logits, "end_logits": end_logits,
            "log_dur": log_dur,
        }
