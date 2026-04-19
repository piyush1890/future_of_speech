"""
TTS Transformer for Residual VQ (multi-codebook) prediction.
Predicts K independent codebook indices per frame.
"""
import math

import torch
import torch.nn as nn

from .duration_predictor import DurationPredictor
from .length_regulator import LengthRegulator
from .transformer import PositionalEncoding


class ArticulatoryTTSModelRVQ(nn.Module):
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

        # Phoneme encoder
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

        # Frame decoder
        self.decoder_pe = PositionalEncoding(d_model, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection: one head per codebook level
        self.output_projs = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_quantizers)
        ])

    def encode_phonemes(self, phoneme_ids, speaker_emb, phoneme_mask=None):
        x = self.phoneme_embedding(phoneme_ids)
        x = self.encoder_pe(x)
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + spk
        padding_mask = ~phoneme_mask if phoneme_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        speaker_emb: torch.Tensor,
        durations: torch.Tensor = None,
        target_len: int = None,
        phoneme_mask: torch.Tensor = None,
    ) -> dict:
        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)

        use_durations = durations if durations is not None else pred_durations.round()
        expanded, frame_mask = self.length_regulator(enc_out, use_durations, target_len)

        expanded = self.decoder_pe(expanded)
        padding_mask = ~frame_mask if frame_mask.any() else None
        decoded = self.decoder(expanded, src_key_padding_mask=padding_mask)

        # Predict K codebooks independently
        logits_per_level = [proj(decoded) for proj in self.output_projs]
        # Stack to (B, T, K, codebook_size)
        logits = torch.stack(logits_per_level, dim=2)

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

        expanded, frame_mask = self.length_regulator(enc_out, pred_durations)
        expanded = self.decoder_pe(expanded)
        decoded = self.decoder(expanded)

        logits_per_level = [proj(decoded) for proj in self.output_projs]
        logits = torch.stack(logits_per_level, dim=2)  # (1, T, K, codebook_size)
        token_ids = logits.argmax(dim=-1)  # (1, T, K)

        return token_ids, pred_durations
