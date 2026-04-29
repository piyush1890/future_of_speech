"""
v9 Stage 2 (style planner): predict per-phoneme style code IDs from
(text, speaker, emotion+intensity).

Architecture: bidirectional encoder over phonemes + AR decoder predicting
1 style code per body phoneme. CE loss against extracted GT style codes.

This is the analog of v8's V8CodebookPlanner but predicts style codes
(coarse 1-of-512 per phoneme), not full RVQ tokens. The renderer uses
these codes as input to generate the rich RVQ tokens.
"""
import math
import torch
import torch.nn as nn

from v9.models.v9_predictor import PositionalEncoding


class V9StylePlanner(nn.Module):
    def __init__(
        self,
        vocab_size: int = 73,
        style_codebook_size: int = 512,
        d_model: int = 192,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 768,
        dropout: float = 0.1,
        speaker_emb_dim: int = 64,
        knob_dim: int = 6,
        knob_dropout: float = 0.1,
        max_phonemes: int = 256,
    ):
        super().__init__()
        self.style_codebook_size = style_codebook_size
        self.SOS_ID = style_codebook_size       # special "start" token (same as PAD)
        self.d_model = d_model
        self.knob_dim = knob_dim
        self.knob_dropout = knob_dropout

        self.phoneme_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, max_len=max_phonemes + 8)
        self.decoder_pe = PositionalEncoding(d_model, max_len=max_phonemes + 8)

        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        self.knob_proj = nn.Sequential(
            nn.Linear(knob_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        ) if knob_dim > 0 else None

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

        # Embedding for previous-step style codes (input to AR decoder)
        # +1 for SOS / PAD index (= style_codebook_size)
        self.code_emb = nn.Embedding(style_codebook_size + 1, d_model,
                                     padding_idx=style_codebook_size)

        # Output head
        self.out_proj = nn.Linear(d_model, style_codebook_size)

    def encode(self, phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=False):
        x = self.phoneme_emb(phoneme_ids)
        x = x + self.speaker_proj(spk_emb).unsqueeze(1)
        if self.knob_proj is not None and knobs is not None:
            knob_vec = self.knob_proj(knobs)
            if self.training and self.knob_dropout > 0:
                drop = (torch.rand(knobs.shape[0], device=knobs.device) > self.knob_dropout).float()
                knob_vec = knob_vec * drop.unsqueeze(-1)
            if force_drop_knobs:
                knob_vec = torch.zeros_like(knob_vec)
            x = x + knob_vec.unsqueeze(1)
        x = self.encoder_pe(x)
        return self.encoder(x, src_key_padding_mask=~phoneme_mask)

    def forward(self, phoneme_ids, spk_emb, knobs, phoneme_mask, gt_codes,
                force_drop_knobs=False):
        """gt_codes: (B, N) long. Position 0 ignored from loss; positions 1..N-2
        are body code IDs in [0, style_codebook_size); position N-1 ignored.

        We shift the input: position i sees code at i-1 (or SOS at 0).
        """
        B, N = phoneme_ids.shape
        device = phoneme_ids.device
        enc = self.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs)

        # Build shifted code input: position 0 = SOS, positions 1..N-1 = gt_codes[:N-1]
        sos = torch.full((B, 1), self.SOS_ID, dtype=torch.long, device=device)
        shifted = torch.cat([sos, gt_codes[:, :-1]], dim=1)              # (B, N)
        dec_inp = self.code_emb(shifted)
        dec_inp = self.decoder_pe(dec_inp)
        causal = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
        h = self.decoder(
            dec_inp, enc, tgt_mask=causal,
            tgt_key_padding_mask=~phoneme_mask,
            memory_key_padding_mask=~phoneme_mask,
        )
        logits = self.out_proj(h)                                        # (B, N, C)
        return logits

    @torch.no_grad()
    def generate(self, phoneme_ids, spk_emb, knobs, phoneme_mask,
                 temperature: float = 1.0, top_k: int = 0, cfg_scale: float = 1.0):
        """AR generation. Returns (B, N) code IDs. PAD positions get SOS_ID."""
        B, N = phoneme_ids.shape
        device = phoneme_ids.device
        enc_cond = self.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=False)
        enc_un   = self.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=True) \
                   if cfg_scale != 1.0 else None

        codes = torch.full((B, N), self.SOS_ID, dtype=torch.long, device=device)
        for i in range(N):
            shifted = torch.cat([
                torch.full((B, 1), self.SOS_ID, dtype=torch.long, device=device),
                codes[:, :-1],
            ], dim=1)
            dec_inp = self.decoder_pe(self.code_emb(shifted))
            causal = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
            h = self.decoder(dec_inp, enc_cond, tgt_mask=causal,
                             tgt_key_padding_mask=~phoneme_mask,
                             memory_key_padding_mask=~phoneme_mask)
            logits = self.out_proj(h[:, i, :])
            if enc_un is not None:
                h_un = self.decoder(dec_inp, enc_un, tgt_mask=causal,
                                    tgt_key_padding_mask=~phoneme_mask,
                                    memory_key_padding_mask=~phoneme_mask)
                logits_un = self.out_proj(h_un[:, i, :])
                logits = logits_un + cfg_scale * (logits - logits_un)
            if temperature <= 0:
                tok = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k, dim=-1)
                    thr = v[..., -1:].expand_as(logits)
                    logits = torch.where(logits < thr, torch.full_like(logits, float("-inf")), logits)
                probs = torch.softmax(logits, dim=-1)
                tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            codes[:, i] = tok
        return codes
