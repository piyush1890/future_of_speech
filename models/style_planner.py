"""
Style planner for v5 stage 2.

Task: predict the per-phoneme style code sequence (s_1, ..., s_N) given
(phoneme_ids, speaker_emb, emotion_id, style_id, intensity).

Architecture: encoder-decoder transformer, both small.
  - Bidirectional phoneme encoder (4 layers, d=128) reads the full text + control token.
  - Causal-masked AR decoder (4 layers, d=128) emits one code at each phoneme position,
    conditioned on the phoneme at that position and on previously-emitted codes.

Inputs:
  phoneme_ids:   (B, N) long
  speaker_emb:   (B, 64) float
  emotion_id:    (B,) long  → embedded
  style_id:      (B,) long  → embedded
  intensity:     (B,) float → linear-projected
The (emotion + style + intensity) signal is summed into a single 'control vector'
that's prepended as the BOS-equivalent of the encoder input. The encoder then
provides context to the decoder via cross-attention.

Output: logits of shape (B, N, style_codebook_size). Loss is CE per phoneme position.

At inference: AR sampling. Sample/argmax position 0, append, predict position 1, etc.
Or run all positions in parallel with greedy argmax (non-AR fast inference, with the
caveat that codes won't condition on previous codes in this mode — practical when
you don't need diversity).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import PositionalEncoding


class StylePlanner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        style_codebook_size: int = 512,
        n_emotions: int = 5,
        n_styles: int = 7,
        speaker_emb_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        knob_dropout: float = 0.1,    # randomly drop knob conditioning at training
                                       # so model can also operate from text alone
    ):
        super().__init__()
        self.d_model = d_model
        self.style_codebook_size = style_codebook_size
        self.knob_dropout = knob_dropout

        # Phoneme + control embeddings
        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.encoder_pe = PositionalEncoding(d_model, dropout=dropout)

        # Knob embeddings — combined into a single control vector
        self.emotion_emb = nn.Embedding(n_emotions, d_model)
        self.style_emb   = nn.Embedding(n_styles, d_model)
        self.intensity_proj = nn.Linear(1, d_model)
        self.speaker_proj   = nn.Linear(speaker_emb_dim, d_model)
        # Special "null" embedding used when knobs are dropped
        self.null_control = nn.Parameter(torch.zeros(d_model))

        # Encoder over (control_token, phoneme_1, ..., phoneme_N)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # AR decoder: causal-masked self-attention over previously-emitted codes
        # + cross-attention to encoder output.
        self.decoder_pe = PositionalEncoding(d_model, dropout=dropout)
        self.code_embedding = nn.Embedding(style_codebook_size + 1, d_model)
        # +1 for the BOS code (start of sequence)
        self.BOS_CODE = style_codebook_size

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(d_model, style_codebook_size)

    def _build_control_vector(
        self,
        emotion_id: torch.Tensor,    # (B,) long
        style_id: torch.Tensor,      # (B,) long
        intensity: torch.Tensor,     # (B,) float
        speaker_emb: torch.Tensor,   # (B, 64)
        drop_knobs: bool = False,
    ) -> torch.Tensor:
        """Returns (B, d_model) control token. With drop_knobs=True, returns
        a learned 'null' embedding (no knob signal) — used at training to encourage
        text-only generalization, and at inference for classifier-free guidance."""
        if drop_knobs:
            B = emotion_id.shape[0]
            # Still include speaker — that's identity, not a knob to drop
            return self.null_control.unsqueeze(0).expand(B, -1) + self.speaker_proj(speaker_emb)
        ctrl = (
            self.emotion_emb(emotion_id) +
            self.style_emb(style_id) +
            self.intensity_proj(intensity.unsqueeze(-1)) +
            self.speaker_proj(speaker_emb)
        )
        return ctrl

    def _encode(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_mask: torch.Tensor,
        control_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory (B, N+1, D), memory_mask (B, N+1))."""
        x = self.phoneme_embedding(phoneme_ids)               # (B, N, D)
        x = self.encoder_pe(x)
        # Prepend control as position 0
        ctrl = control_vec.unsqueeze(1)                       # (B, 1, D)
        x = torch.cat([ctrl, x], dim=1)                       # (B, N+1, D)
        # mask: control is always valid → True
        ctrl_mask = torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)
        memory_mask = torch.cat([ctrl_mask, phoneme_mask], dim=1)
        padding_mask = ~memory_mask
        memory = self.encoder(x, src_key_padding_mask=padding_mask)
        return memory, memory_mask

    def forward(
        self,
        phoneme_ids: torch.Tensor,            # (B, N)
        phoneme_mask: torch.Tensor,           # (B, N) True = real
        speaker_emb: torch.Tensor,            # (B, 64)
        emotion_id: torch.Tensor,             # (B,) long
        style_id: torch.Tensor,               # (B,) long
        intensity: torch.Tensor,              # (B,) float
        target_codes: Optional[torch.Tensor] = None,    # (B, N) long, GT codes (training)
    ) -> dict:
        """Training (teacher-forced) or evaluation forward.
          target_codes: ground-truth code sequence; shifted by one for AR input.
        Returns: {"logits": (B, N, codebook_size)}.
        """
        B, N = phoneme_ids.shape

        # Random knob-dropout (forces model to learn from text alone too)
        drop_knobs = self.training and (torch.rand(1).item() < self.knob_dropout)
        ctrl = self._build_control_vector(emotion_id, style_id, intensity,
                                          speaker_emb, drop_knobs=drop_knobs)

        memory, memory_mask = self._encode(phoneme_ids, phoneme_mask, ctrl)

        # Decoder input: BOS + target_codes[:, :-1]; predict target_codes[:, :].
        # If no target_codes (inference), use BOS-only (caller should run AR loop instead).
        if target_codes is None:
            # Fallback: predict from BOS only (1-step). Caller usually loops.
            dec_in = torch.full((B, 1), self.BOS_CODE, dtype=torch.long, device=phoneme_ids.device)
        else:
            bos = torch.full((B, 1), self.BOS_CODE, dtype=torch.long, device=phoneme_ids.device)
            dec_in = torch.cat([bos, target_codes[:, :-1].clamp(min=0)], dim=1)   # PAD_CODE=-1 → 0

        dec_emb = self.code_embedding(dec_in)
        dec_emb = self.decoder_pe(dec_emb)

        # Causal mask
        T_dec = dec_emb.shape[1]
        causal = torch.triu(torch.ones(T_dec, T_dec, device=dec_emb.device, dtype=torch.bool),
                            diagonal=1)
        # Decoder cross-attention: keys/values come from memory
        memory_key_padding_mask = ~memory_mask

        decoded = self.decoder(
            dec_emb, memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_key_padding_mask,
        )                                                           # (B, T_dec, D)
        logits = self.output_proj(decoded)                          # (B, T_dec, codebook_size)
        return {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_mask: torch.Tensor,
        speaker_emb: torch.Tensor,
        emotion_id: torch.Tensor,
        style_id: torch.Tensor,
        intensity: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        cfg_scale: float = 1.0,         # classifier-free guidance scale (1.0 = off)
    ) -> torch.Tensor:
        """AR sampling. Returns codes (B, N) long.

        cfg_scale > 1: at each step, run a parallel forward with knobs dropped, then push
        the conditional distribution AWAY from the unconditional one. This sharpens the
        knob effect — useful when emotion=happy is too subtle.
        """
        self.eval()
        B, N = phoneme_ids.shape

        ctrl_cond   = self._build_control_vector(emotion_id, style_id, intensity, speaker_emb, False)
        memory_cond, memory_mask = self._encode(phoneme_ids, phoneme_mask, ctrl_cond)
        if cfg_scale != 1.0:
            ctrl_uncond = self._build_control_vector(emotion_id, style_id, intensity, speaker_emb, True)
            memory_uncond, _ = self._encode(phoneme_ids, phoneme_mask, ctrl_uncond)

        # Start with BOS
        codes = torch.full((B, 1), self.BOS_CODE, dtype=torch.long, device=phoneme_ids.device)
        out_codes = []
        for i in range(N):
            dec_emb = self.decoder_pe(self.code_embedding(codes))
            T_dec = dec_emb.shape[1]
            causal = torch.triu(torch.ones(T_dec, T_dec, device=dec_emb.device, dtype=torch.bool),
                                diagonal=1)
            memory_key_padding_mask = ~memory_mask
            decoded = self.decoder(dec_emb, memory_cond, tgt_mask=causal,
                                   memory_key_padding_mask=memory_key_padding_mask)
            logits_cond = self.output_proj(decoded[:, -1])              # (B, codebook_size)

            if cfg_scale != 1.0:
                decoded_u = self.decoder(dec_emb, memory_uncond, tgt_mask=causal,
                                         memory_key_padding_mask=memory_key_padding_mask)
                logits_uncond = self.output_proj(decoded_u[:, -1])
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = logits.topk(top_k, dim=-1)
                cutoff = v[..., -1:].expand_as(logits)
                logits = torch.where(logits < cutoff, torch.full_like(logits, -1e9), logits)
            probs = F.softmax(logits, dim=-1)
            chosen = torch.multinomial(probs, num_samples=1) if temperature > 0 else logits.argmax(dim=-1, keepdim=True)
            out_codes.append(chosen)
            codes = torch.cat([codes, chosen], dim=1)

        # Concat the N predicted codes (drop BOS)
        return torch.cat(out_codes, dim=1)   # (B, N)


if __name__ == "__main__":
    # Quick test
    planner = StylePlanner(vocab_size=73, style_codebook_size=512,
                           n_emotions=5, n_styles=7, d_model=128,
                           num_encoder_layers=4, num_decoder_layers=4)
    n = sum(p.numel() for p in planner.parameters())
    print(f"StylePlanner: {n:,} params")

    B, N = 3, 25
    phoneme_ids = torch.randint(1, 73, (B, N))
    phoneme_mask = torch.ones(B, N, dtype=torch.bool)
    phoneme_mask[2, 20:] = False
    spk = torch.randn(B, 64)
    emo = torch.tensor([0, 1, 2])
    sty = torch.tensor([0, 0, 3])
    ints = torch.tensor([1.0, 0.5, 1.0])
    target_codes = torch.randint(0, 512, (B, N))

    # Training forward
    planner.train()
    out = planner(phoneme_ids, phoneme_mask, spk, emo, sty, ints, target_codes=target_codes)
    print(f"train logits: {tuple(out['logits'].shape)}  (expect (B={B}, N={N}, 512))")
    loss = F.cross_entropy(out["logits"].reshape(-1, 512), target_codes.reshape(-1))
    loss.backward()
    print(f"training loss + backward OK, loss={loss.item():.4f}")

    # Inference
    codes = planner.generate(phoneme_ids, phoneme_mask, spk, emo, sty, ints,
                             temperature=0.7, top_k=20)
    print(f"AR-generated codes: {tuple(codes.shape)}  example[0]: {codes[0].tolist()[:10]}")
    # CFG test
    codes_cfg = planner.generate(phoneme_ids, phoneme_mask, spk, emo, sty, ints,
                                 temperature=0.7, top_k=20, cfg_scale=2.0)
    print(f"CFG-guided codes:   {tuple(codes_cfg.shape)}  example[0]: {codes_cfg[0].tolist()[:10]}")
