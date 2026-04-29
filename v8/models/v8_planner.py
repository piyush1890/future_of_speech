"""
v8 planner: predicts per-phoneme continuous style embeddings.

Two modes (controlled by `causal` flag):
  causal=False (non-AR, original):
      Bidirectional attention. All phonemes predicted in parallel.
      MSE collapses each prediction to the mean over training utterances with
      same (text, speaker, knobs) → utterance-level uniform output.

  causal=True (AR, structured):
      Causal mask + previous-z input. Phoneme[t]'s prediction conditions on
      phoneme[0..t-1]'s previously-predicted z's. The model can build up
      structured prosody (e.g. "I'll stress this word, so de-emphasize what
      came before / after"). Training uses teacher-forcing on shifted GT z.
      Inference uses AR generation, optionally with Gaussian sampling
      (`sampling_std > 0`) to break determinism between runs.
"""
import math

import torch
import torch.nn as nn

from .phoneme_tts import PositionalEncoding


class V8Planner(nn.Module):
    def __init__(
        self,
        vocab_size: int = 73,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        knob_dim: int = 3,
        speaker_emb_dim: int = 64,
        style_dim: int = 256,
        knob_dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.knob_dim = knob_dim
        self.knob_dropout_p = knob_dropout
        self.causal = causal
        self.style_dim = style_dim

        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        self.knob_proj = nn.Linear(knob_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)

        # AR-only: project previous z back into d_model space + learnable BOS-z.
        if causal:
            self.z_in_proj = nn.Linear(style_dim, d_model)
            self.bos_z = nn.Parameter(torch.zeros(1, 1, style_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.style_out = nn.Linear(d_model, style_dim)
        nn.init.zeros_(self.style_out.weight)
        nn.init.zeros_(self.style_out.bias)

    def _build_causal_mask(self, N: int, device):
        # True where attention is BLOCKED (above diagonal)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, phoneme_ids, speaker_emb, knobs, phoneme_mask=None,
                prev_z=None, force_drop_knobs: bool = False):
        """
        phoneme_ids:  (B, N) long
        speaker_emb:  (B, speaker_emb_dim)
        knobs:        (B, knob_dim)
        phoneme_mask: (B, N) bool
        prev_z:       (B, N, style_dim) shifted previous z's (AR only). For training
                      we pass [bos_z, z[0], z[1], ..., z[N-2]]. None on first step
                      of inference (we build it iteratively).
        force_drop_knobs: at inference, drop knobs to use unconditional path.

        Returns: (B, N, style_dim)
        """
        B, N = phoneme_ids.shape
        device = phoneme_ids.device

        if self.training and self.knob_dropout_p > 0:
            drop = torch.rand(B, 1, device=device) < self.knob_dropout_p
            knobs = torch.where(drop, torch.zeros_like(knobs), knobs)
        if force_drop_knobs:
            knobs = torch.zeros_like(knobs)

        x = self.phoneme_embedding(phoneme_ids)                      # (B, N, D)
        x = self.pe(x)
        x = x + self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + self.knob_proj(knobs).unsqueeze(1)

        if self.causal:
            if prev_z is None:
                # No prev_z given — replicate BOS for all positions (untrained baseline).
                # In normal training/inference we always pass prev_z.
                prev_z = self.bos_z.expand(B, N, self.style_dim)
            x = x + self.z_in_proj(prev_z)
            attn_mask = self._build_causal_mask(N, device)
        else:
            attn_mask = None

        padding_mask = ~phoneme_mask if phoneme_mask is not None else None
        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return self.style_out(x)

    @torch.no_grad()
    def generate(self, phoneme_ids, speaker_emb, knobs, phoneme_mask=None,
                 sampling_std: float = 0.0, force_drop_knobs: bool = False):
        """AR generation. Builds z[0..N-1] one at a time using teacher-feedback."""
        assert self.causal, "generate() requires causal=True"
        B, N = phoneme_ids.shape
        device = phoneme_ids.device

        # prev_z starts with BOS for all positions; we fill in column t after generating.
        prev_z = self.bos_z.expand(B, N, self.style_dim).clone().to(device)

        for t in range(N):
            out = self.forward(phoneme_ids, speaker_emb, knobs, phoneme_mask,
                               prev_z=prev_z, force_drop_knobs=force_drop_knobs)
            z_t = out[:, t, :]
            if sampling_std > 0:
                z_t = z_t + sampling_std * torch.randn_like(z_t)
            if t + 1 < N:
                prev_z[:, t + 1] = z_t

        return self.forward(phoneme_ids, speaker_emb, knobs, phoneme_mask,
                            prev_z=prev_z, force_drop_knobs=force_drop_knobs)


def shift_for_teacher_forcing(z: torch.Tensor, bos_z: torch.Tensor) -> torch.Tensor:
    """Build prev_z from GT z by shifting right by one and prepending BOS.
    z: (B, N, D) — GT per-phoneme z's.
    bos_z: (1, 1, D) — learnable BOS embedding from the planner.
    Returns: (B, N, D) where pos[0]=bos, pos[t]=z[t-1] for t>=1.
    """
    B, N, D = z.shape
    bos = bos_z.expand(B, 1, D)
    return torch.cat([bos, z[:, :-1]], dim=1)


class V8CodebookPlanner(nn.Module):
    """AR planner with discrete codebook output (v5-style architecture).

    Predicts code IDs from a frozen v5 codebook (512 entries × 256-d).
    CE loss preserves multimodality; sampling at inference breaks mean-collapse.
    AR feedback uses the codebook embeddings of previously-predicted codes.

    Output: logits over codebook_size; downstream lookup gives continuous z.
    """
    def __init__(
        self,
        vocab_size: int,
        codebook_entries: torch.Tensor,        # (codebook_size, style_dim) frozen
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        knob_dim: int = 3,
        speaker_emb_dim: int = 64,
        knob_dropout: float = 0.1,
        max_context: int = 100,                 # sliding-window cap
    ):
        super().__init__()
        cb_size, style_dim = codebook_entries.shape
        self.codebook_size = cb_size
        self.style_dim = style_dim
        self.knob_dropout_p = knob_dropout
        self.max_context = max_context

        self.phoneme_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.speaker_proj = nn.Linear(speaker_emb_dim, d_model)
        self.knob_proj = nn.Linear(knob_dim, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)

        # Frozen codebook (v5 entries) + projection from style_dim into d_model
        self.register_buffer("code_embeddings", codebook_entries.float(), persistent=True)
        self.code_in_proj = nn.Linear(style_dim, d_model)
        # BOS represented as a learnable 256-d vector (in style space)
        self.bos_emb = nn.Parameter(torch.zeros(1, 1, style_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Output: logits over codebook
        self.out_head = nn.Linear(d_model, cb_size)

    def _build_causal_mask(self, N: int, device):
        # Standard causal: block above diagonal (j > i)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        # Sliding window cap: also block j < i - max_context + 1 (positions farther than K back)
        K = self.max_context
        if K > 0 and K < N:
            i = torch.arange(N, device=device).unsqueeze(1)
            j = torch.arange(N, device=device).unsqueeze(0)
            far_back = j < (i - K + 1)
            mask = mask | far_back
        return mask

    def embed_codes(self, code_ids: torch.Tensor) -> torch.Tensor:
        """code_ids: (B, N) long. Returns (B, N, style_dim) embeddings.
        Codes < 0 (e.g. PAD = -1) → zero embedding."""
        valid = (code_ids >= 0)
        safe = code_ids.clamp(min=0)
        out = self.code_embeddings[safe]
        return out * valid.unsqueeze(-1).float()

    def forward(self, phoneme_ids, speaker_emb, knobs, phoneme_mask=None,
                prev_z: torch.Tensor = None, force_drop_knobs: bool = False):
        """
        prev_z: (B, N, style_dim) — embedded previous codes (or BOS at pos 0).
                Required at training (teacher forcing) and inference.
        Returns logits over codebook: (B, N, codebook_size).
        """
        B, N = phoneme_ids.shape
        device = phoneme_ids.device

        if self.training and self.knob_dropout_p > 0:
            drop = torch.rand(B, 1, device=device) < self.knob_dropout_p
            knobs = torch.where(drop, torch.zeros_like(knobs), knobs)
        if force_drop_knobs:
            knobs = torch.zeros_like(knobs)

        if prev_z is None:
            # Default: BOS for all positions (effectively no AR feedback)
            prev_z = self.bos_emb.expand(B, N, self.style_dim)

        x = self.phoneme_embedding(phoneme_ids)
        x = self.pe(x)
        x = x + self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + self.knob_proj(knobs).unsqueeze(1)
        x = x + self.code_in_proj(prev_z)

        attn_mask = self._build_causal_mask(N, device)
        padding_mask = ~phoneme_mask if phoneme_mask is not None else None
        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return self.out_head(x)         # (B, N, codebook_size)

    @torch.no_grad()
    def generate(self, phoneme_ids, speaker_emb, knobs, phoneme_mask=None,
                 temperature: float = 1.0, top_k: int = 0,
                 force_drop_knobs: bool = False):
        """AR generation. Returns (code_ids, z) both per-phoneme.
        code_ids: (B, N) long, sampled from logits at each step.
        z:        (B, N, style_dim), embedding of those codes (= input to stage 1).
        """
        B, N = phoneme_ids.shape
        device = phoneme_ids.device

        # prev_z starts with BOS for all positions; we update column t after step t.
        prev_z = self.bos_emb.expand(B, N, self.style_dim).clone().to(device)
        code_ids = torch.zeros(B, N, dtype=torch.long, device=device)

        for t in range(N):
            logits = self.forward(phoneme_ids, speaker_emb, knobs, phoneme_mask,
                                  prev_z=prev_z, force_drop_knobs=force_drop_knobs)
            logits_t = logits[:, t, :]            # (B, codebook_size)
            if temperature != 1.0:
                logits_t = logits_t / max(1e-8, temperature)
            if top_k > 0 and top_k < self.codebook_size:
                topv, topi = logits_t.topk(top_k, dim=-1)
                mask_full = torch.full_like(logits_t, float("-inf"))
                mask_full.scatter_(-1, topi, topv)
                logits_t = mask_full
            if temperature == 0.0:
                code_t = logits_t.argmax(dim=-1)
            else:
                probs = torch.softmax(logits_t, dim=-1)
                code_t = torch.multinomial(probs, num_samples=1).squeeze(-1)
            code_ids[:, t] = code_t
            if t + 1 < N:
                prev_z[:, t + 1] = self.code_embeddings[code_t]

        z = self.code_embeddings[code_ids]
        return code_ids, z


def shift_codes_for_teacher_forcing(code_ids: torch.Tensor,
                                    bos_emb: torch.Tensor,
                                    code_embeddings: torch.Tensor) -> torch.Tensor:
    """code_ids: (B, N) long. Returns prev_z: (B, N, style_dim) where
    pos[0]=bos_emb, pos[t]=code_embeddings[code_ids[:, t-1]] for t>=1.
    Codes < 0 (PAD) at body positions are treated as zeros."""
    B, N = code_ids.shape
    style_dim = bos_emb.shape[-1]
    bos = bos_emb.expand(B, 1, style_dim)
    # Embed code_ids[:, :-1], handling PAD = -1 → zero embedding
    src = code_ids[:, :-1]
    valid = (src >= 0).unsqueeze(-1).float()
    safe = src.clamp(min=0)
    body_emb = code_embeddings[safe] * valid           # (B, N-1, style_dim)
    return torch.cat([bos, body_emb], dim=1)           # (B, N, style_dim)
