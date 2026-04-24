"""
Articulatory TTS Transformer: phonemes → articulatory token IDs.
Encoder-decoder with duration prediction (FastSpeech-style, non-autoregressive).
"""
import math

import torch
import torch.nn as nn

from .duration_predictor import DurationPredictor
from .length_regulator import LengthRegulator


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ArticulatoryTTSModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 84,
        codebook_size: int = 512,
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

        # Phoneme encoder
        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Speaker conditioning
        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)

        # Duration predictor
        self.duration_predictor = DurationPredictor(d_model, dropout=dropout)

        # Length regulator
        self.length_regulator = LengthRegulator()

        # Frame decoder
        self.decoder_pe = PositionalEncoding(d_model, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # Using TransformerEncoder (not Decoder) since this is non-autoregressive
        # — no causal masking needed, each frame can attend to all expanded phonemes
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection to codebook logits
        self.output_proj = nn.Linear(d_model, codebook_size)

    def encode_phonemes(
        self,
        phoneme_ids: torch.Tensor,
        speaker_emb: torch.Tensor,
        phoneme_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode phonemes with speaker conditioning.
        Args:
            phoneme_ids: (B, N) phoneme token indices
            speaker_emb: (B, 64) SPARC speaker embedding
            phoneme_mask: (B, N) True = valid token
        Returns:
            (B, N, d_model) encoder output
        """
        x = self.phoneme_embedding(phoneme_ids)  # (B, N, d_model)
        x = self.encoder_pe(x)

        # Add speaker conditioning
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)  # (B, 1, d_model)
        x = x + spk

        # Create src_key_padding_mask (True = IGNORE for PyTorch transformers)
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
        """
        Full forward pass.
        Args:
            phoneme_ids: (B, N) phoneme indices
            speaker_emb: (B, 64) speaker embedding
            durations: (B, N) ground-truth durations (for training)
            target_len: target output length (for training)
            phoneme_mask: (B, N) True = valid
        Returns:
            dict with logits, pred_durations, frame_mask
        """
        # Encode phonemes
        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)

        # Predict durations
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)

        # Use ground-truth durations for training, predicted for inference
        use_durations = durations if durations is not None else pred_durations.round()

        # Expand to frame-level
        expanded, frame_mask = self.length_regulator(enc_out, use_durations, target_len)

        # Decode
        expanded = self.decoder_pe(expanded)
        padding_mask = ~frame_mask if frame_mask.any() else None
        decoded = self.decoder(expanded, src_key_padding_mask=padding_mask)

        # Project to codebook logits
        logits = self.output_proj(decoded)  # (B, T, codebook_size)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate articulatory token indices from phonemes.
        Args:
            phoneme_ids: (1, N) phoneme indices
            speaker_emb: (1, 64) speaker embedding
            duration_scale: multiply predicted durations (>1 = slower, <1 = faster)
        Returns:
            token_ids: (1, T) articulatory codebook indices
            durations: (1, N) predicted durations
        """
        self.eval()
        phoneme_mask = phoneme_ids != 0  # assume 0 = pad

        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)
        pred_durations = (pred_durations * duration_scale).round().clamp(min=1)

        expanded, frame_mask = self.length_regulator(enc_out, pred_durations)
        expanded = self.decoder_pe(expanded)
        decoded = self.decoder(expanded)
        logits = self.output_proj(decoded)

        token_ids = logits.argmax(dim=-1)  # (1, T)
        return token_ids, pred_durations
