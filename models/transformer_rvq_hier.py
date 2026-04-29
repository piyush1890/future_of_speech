"""
Hierarchical RVQ prediction: head_k conditions on tokens from levels 0..k-1.

Rationale: in RVQ, level k's target is the residual *after* level k-1's choice.
Predicting all K heads independently from the same embedding forces each head
to marginalize over unseen prior choices → smeared targets → near-chance accuracy
for deeper codebooks. Feeding earlier tokens in sharpens the target distribution.

At init, `cb_embeds` are zeroed, making this model numerically identical to the
flat variant. Training grows these embeddings as they become useful — enabling
partial-load from a flat checkpoint without performance regression.

Style conditioning supports two modes (caller picks one):
  - `style_vec`: shape (B, d_model). Pooled per-utterance. Broadcast over phoneme
    positions. v3/v4 path; kept for v4 checkpoint inference. Diagnosed in v4 to
    leak utterance length into duration predictor.
  - `style_emb`: shape (B, N, d_model). Per-phoneme. Added directly without
    broadcasting. v5 path. Caller is responsible for the codebook lookup.

Pass exactly one (or neither, for no style) per call.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from .duration_predictor import DurationPredictor
from .length_regulator import LengthRegulator
from .transformer import PositionalEncoding


class ArticulatoryTTSModelRVQHier(nn.Module):
    def __init__(
        self,
        vocab_size: int = 84,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        speaker_emb_dim: int = 64,
        style_dim: int = None,
        tied_output: bool = False,
        codebook_latent_dim: int = 64,
    ):
        """
        tied_output=False (v3/v4/v5 default): output_projs[k] = Linear(d_model, codebook_size).
                                              Independent learned projection per token id.
                                              Probability is feature-aware only via training-data
                                              correlations.
        tied_output=True  (v6): output_projs[k] = Linear(d_model, codebook_latent_dim).
                                Logits computed as h_proj @ codebook[k].T — direct similarity
                                to each codebook entry. Probability is structurally feature-aware:
                                tokens whose codebook entries are close to h_proj get high probability.
                                The frozen codebooks are stored as a buffer; populate via
                                init_tied_codebooks(rvq_model) before training/inference.
        codebook_latent_dim:    dimension of each codebook entry in the frozen RVQ. Defaults to 64
                                matching our standard RVQ. Only used when tied_output=True.
        """
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.tied_output = tied_output
        self.codebook_latent_dim = codebook_latent_dim

        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        self.style_proj = nn.Linear(style_dim if style_dim is not None else d_model, d_model)
        nn.init.zeros_(self.style_proj.weight)
        nn.init.zeros_(self.style_proj.bias)
        self.duration_predictor = DurationPredictor(d_model, dropout=dropout)
        self.length_regulator = LengthRegulator()

        self.decoder_pe = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection: independent Linear per level (untied) or projection
        # into codebook latent space (tied — logit = h_proj · codebook_entry).
        out_dim = codebook_latent_dim if tied_output else codebook_size
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, out_dim) for _ in range(num_quantizers)
        ])

        # When tied: register a buffer for the frozen codebook entries per level.
        # Populate via init_tied_codebooks(rvq_model) BEFORE training/inference.
        # Stays zeros until populated; if the buffer is all-zero at forward, raises.
        if tied_output:
            self.register_buffer(
                "frozen_codebooks",
                torch.zeros(num_quantizers, codebook_size, codebook_latent_dim),
            )
            # Optional: a per-level learnable temperature so the model can scale
            # similarity scores into a useful logit distribution.
            self.tied_log_temps = nn.Parameter(torch.zeros(num_quantizers))

        # cb_embeds[k] embeds the chosen token at level k to condition level k+1.
        # Only K-1 needed (nothing consumes the last level's token).
        # Zero-init → model starts numerically equivalent to flat variant.
        self.cb_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(num_quantizers - 1)
        ])
        for emb in self.cb_embeds:
            nn.init.zeros_(emb.weight)

    def init_tied_codebooks(self, rvq_model):
        """Copy frozen RVQ codebook entries into the model's buffer.
        Call once after constructing the model (and after loading the RVQ),
        before training. Required when tied_output=True."""
        if not self.tied_output:
            return
        for k in range(self.num_quantizers):
            embed = rvq_model.vq.layers[k]._codebook.embed
            if embed.dim() == 3:   # (1, C, D) → (C, D)
                embed = embed.squeeze(0)
            assert embed.shape == (self.codebook_size, self.codebook_latent_dim), \
                f"Codebook level {k} shape {embed.shape} != ({self.codebook_size}, {self.codebook_latent_dim})"
            self.frozen_codebooks[k].copy_(embed.detach())
        print(f"  Tied output: copied {self.num_quantizers} codebooks into model buffer")

    def encode_phonemes(self, phoneme_ids, speaker_emb, phoneme_mask=None,
                        style_vec=None, style_emb=None):
        if style_vec is not None and style_emb is not None:
            raise ValueError("Pass at most one of style_vec (v4 broadcast) or "
                             "style_emb (v5 per-phoneme), not both.")
        x = self.phoneme_embedding(phoneme_ids)
        x = self.encoder_pe(x)
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + spk
        if style_vec is not None:
            x = x + self.style_proj(style_vec).unsqueeze(1)   # (B, 1, D) broadcast
        elif style_emb is not None:
            # Per-phoneme style; no projection (caller's codebook output is already d_model)
            x = x + style_emb                                  # (B, N, D) direct add
        padding_mask = ~phoneme_mask if phoneme_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x

    def _decode_frames(self, enc_out, durations, target_len, phoneme_mask):
        expanded, frame_mask = self.length_regulator(enc_out, durations, target_len)
        expanded = self.decoder_pe(expanded)
        padding_mask = ~frame_mask if frame_mask.any() else None
        decoded = self.decoder(expanded, src_key_padding_mask=padding_mask)
        return decoded, frame_mask

    def _run_hierarchical_heads(
        self,
        decoded: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ):
        """
        decoded:       (B, T, D)
        target_tokens: (B, T, K) or None

        If target_tokens is given (training), teacher-force — head_{k+1} receives
        the ground-truth token at level k. Otherwise (inference), greedily sample.

        Returns logits (B, T, K, codebook_size).
        """
        B, T, D = decoded.shape
        K = self.num_quantizers
        context = decoded
        logits_per_level = []

        for k in range(K):
            if self.tied_output:
                # h_proj: (B, T, latent_dim); codebook[k]: (codebook_size, latent_dim)
                h_proj = self.output_projs[k](context)
                # Logits = similarity between projected hidden state and each codebook entry
                logits_k = h_proj @ self.frozen_codebooks[k].T   # (B, T, codebook_size)
                # Learnable per-level temperature on the similarity scores
                logits_k = logits_k * torch.exp(self.tied_log_temps[k])
            else:
                logits_k = self.output_projs[k](context)  # (B, T, C)
            logits_per_level.append(logits_k)

            if k < K - 1:
                if target_tokens is not None:
                    tok = target_tokens[:, :T, k]
                else:
                    tok = logits_k.argmax(dim=-1)
                context = context + self.cb_embeds[k](tok)

        return torch.stack(logits_per_level, dim=2)  # (B, T, K, C)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        speaker_emb: torch.Tensor,
        durations: torch.Tensor = None,
        target_len: int = None,
        phoneme_mask: torch.Tensor = None,
        target_tokens: torch.Tensor = None,
        style_vec: torch.Tensor = None,
        style_emb: torch.Tensor = None,
    ) -> dict:
        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask,
                                       style_vec=style_vec, style_emb=style_emb)
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)

        use_durations = durations if durations is not None else pred_durations.round()
        decoded, frame_mask = self._decode_frames(enc_out, use_durations, target_len, phoneme_mask)

        logits = self._run_hierarchical_heads(decoded, target_tokens=target_tokens)

        return {
            "logits": logits,
            "pred_durations": pred_durations,
            "frame_mask": frame_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        phoneme_ids: torch.Tensor,
        speaker_emb: torch.Tensor,
        duration_scale: float = 1.0,
        style_vec: torch.Tensor = None,
        style_emb: torch.Tensor = None,
    ):
        self.eval()
        phoneme_mask = phoneme_ids != 0

        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask,
                                       style_vec=style_vec, style_emb=style_emb)
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)
        pred_durations = (pred_durations * duration_scale).round().clamp(min=1)

        decoded, _ = self._decode_frames(enc_out, pred_durations, None, phoneme_mask)
        logits = self._run_hierarchical_heads(decoded, target_tokens=None)
        token_ids = logits.argmax(dim=-1)

        return token_ids, pred_durations
