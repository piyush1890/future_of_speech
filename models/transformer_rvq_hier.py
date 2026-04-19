"""
Hierarchical RVQ prediction: head_k conditions on tokens from levels 0..k-1.

Rationale: in RVQ, level k's target is the residual *after* level k-1's choice.
Predicting all K heads independently from the same embedding forces each head
to marginalize over unseen prior choices → smeared targets → near-chance accuracy
for deeper codebooks. Feeding earlier tokens in sharpens the target distribution.

At init, `cb_embeds` are zeroed, making this model numerically identical to the
flat variant. Training grows these embeddings as they become useful — enabling
partial-load from a flat checkpoint without performance regression.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        self.duration_predictor = DurationPredictor(d_model, dropout=dropout)
        self.length_regulator = LengthRegulator()

        self.decoder_pe = PositionalEncoding(d_model, dropout=dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_quantizers)
        ])

        # cb_embeds[k] embeds the chosen token at level k to condition level k+1.
        # Only K-1 needed (nothing consumes the last level's token).
        # Zero-init → model starts numerically equivalent to flat variant.
        self.cb_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(num_quantizers - 1)
        ])
        for emb in self.cb_embeds:
            nn.init.zeros_(emb.weight)

    def encode_phonemes(self, phoneme_ids, speaker_emb, phoneme_mask=None):
        x = self.phoneme_embedding(phoneme_ids)
        x = self.encoder_pe(x)
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + spk
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
    ) -> dict:
        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)
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
    ):
        self.eval()
        phoneme_mask = phoneme_ids != 0

        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)
        pred_durations = (pred_durations * duration_scale).round().clamp(min=1)

        decoded, _ = self._decode_frames(enc_out, pred_durations, None, phoneme_mask)
        logits = self._run_hierarchical_heads(decoded, target_tokens=None)
        token_ids = logits.argmax(dim=-1)

        return token_ids, pred_durations
