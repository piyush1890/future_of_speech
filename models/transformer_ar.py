"""
Autoregressive Articulatory TTS Transformer.
Same encoder + duration predictor, but the decoder predicts each frame
conditioned on previous frames' tokens (causal/autoregressive).
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
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ArticulatoryTTSModelAR(nn.Module):
    """
    Autoregressive version: decoder predicts token t based on
    phoneme context + tokens 0..t-1.
    """

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

        # ── Phoneme encoder (same as before) ──
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

        # ── Autoregressive decoder ──
        # Token embedding: previous VQ token → d_model
        self.token_embedding = nn.Embedding(codebook_size + 1, d_model)  # +1 for BOS token
        self.bos_token_id = codebook_size  # use last index as BOS

        self.decoder_pe = PositionalEncoding(d_model, dropout=dropout)

        # True cross-attention decoder: attends to expanded phoneme context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, codebook_size)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask: each position can only attend to earlier positions."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask  # True = BLOCK attention

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
        target_tokens: torch.Tensor = None,
        target_len: int = None,
        phoneme_mask: torch.Tensor = None,
        frame_mask: torch.Tensor = None,
    ) -> dict:
        """
        Training forward pass with teacher forcing.

        Args:
            phoneme_ids: (B, N) phoneme indices
            speaker_emb: (B, 64) speaker embedding
            durations: (B, N) ground-truth durations
            target_tokens: (B, T) ground-truth VQ token indices
            target_len: target output length
            phoneme_mask: (B, N) True = valid phoneme
            frame_mask: (B, T) True = valid frame
        """
        B = phoneme_ids.size(0)

        # Encode phonemes
        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)

        # Predict durations
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)

        # Expand phoneme embeddings to frame-level
        use_durations = durations if durations is not None else pred_durations.round()
        expanded, exp_mask = self.length_regulator(enc_out, use_durations, target_len)
        # expanded: (B, T, d_model) — phoneme context per frame

        T = expanded.size(1)

        # Teacher forcing: shift target tokens right, prepend BOS
        if target_tokens is not None:
            # target_tokens: (B, T) — ground truth token at each frame
            bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=phoneme_ids.device)
            shifted = torch.cat([bos, target_tokens[:, :T-1]], dim=1)  # (B, T)
        else:
            shifted = torch.full((B, T), self.bos_token_id, dtype=torch.long, device=phoneme_ids.device)

        # Embed previous tokens + positional encoding
        token_emb = self.token_embedding(shifted)  # (B, T, d_model)
        # Combine with phoneme context
        decoder_input = token_emb + expanded
        decoder_input = self.decoder_pe(decoder_input)

        # Causal mask: frame t can only see frames 0..t
        causal_mask = self._causal_mask(T, phoneme_ids.device)

        # Padding mask for decoder
        tgt_padding_mask = ~exp_mask if exp_mask.any() else None

        # Memory (encoder output for cross-attention)
        memory = expanded  # phoneme context expanded to frame-level
        memory_padding_mask = ~exp_mask if exp_mask.any() else None

        # Decode
        decoded = self.decoder(
            decoder_input,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        logits = self.output_proj(decoded)  # (B, T, codebook_size)

        return {
            "logits": logits,
            "pred_durations": pred_durations,
            "frame_mask": exp_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        phoneme_ids: torch.Tensor,
        speaker_emb: torch.Tensor,
        duration_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive generation: predict one token at a time.
        """
        self.eval()
        phoneme_mask = phoneme_ids != 0

        enc_out = self.encode_phonemes(phoneme_ids, speaker_emb, phoneme_mask)
        pred_durations = self.duration_predictor(enc_out, phoneme_mask)
        pred_durations = (pred_durations * duration_scale).round().clamp(min=1)

        expanded, frame_mask = self.length_regulator(enc_out, pred_durations)
        T = expanded.size(1)

        # Start with BOS
        generated = [self.bos_token_id]

        for t in range(T):
            # Embed all generated tokens so far
            gen_tensor = torch.tensor([generated], dtype=torch.long, device=phoneme_ids.device)
            token_emb = self.token_embedding(gen_tensor)  # (1, t+1, d_model)

            # Add phoneme context for positions 0..t
            context = expanded[:, :t+1]  # (1, t+1, d_model)
            decoder_input = token_emb + context
            decoder_input = self.decoder_pe(decoder_input)

            # Causal mask
            causal_mask = self._causal_mask(t + 1, phoneme_ids.device)

            # Memory
            memory = expanded[:, :t+1]

            decoded = self.decoder(
                decoder_input,
                memory,
                tgt_mask=causal_mask,
            )

            # Predict next token from last position
            logits = self.output_proj(decoded[:, -1:])  # (1, 1, codebook_size)
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)

        # Remove BOS
        token_ids = torch.tensor([generated[1:]], dtype=torch.long, device=phoneme_ids.device)
        return token_ids, pred_durations
