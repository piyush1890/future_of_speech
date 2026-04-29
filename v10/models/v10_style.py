"""
v10 per-phoneme style encoder + VQ codebook.

Per body phoneme i:
  attention-pool over its frames (no phoneme symbol input)
  → continuous style embedding (D)
  → VQ → discrete style code in [0, codebook_size)

BOS/EOS positions get a sentinel PAD code (= codebook_size). The renderer
encoder's style_code_emb has padding_idx for this PAD position.

Trained jointly with the renderer (gradient flows through VQ via straight-through
estimator).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class V10StyleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 256,
        codebook_size: int = 64,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.PAD_CODE = codebook_size

        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # attn_query large init: with small init (0.02), attention is near-uniform at
        # start → all utterances pool to similar embedding → kmeans clusters all codes
        # near one point → permanent codebook collapse. randn * 1.0 forces peaked
        # attention from step 0, giving the encoder real differentiation signal.
        self.attn_query = nn.Parameter(torch.randn(hidden_dim) * 1.0)
        self.attn_scale = hidden_dim ** -0.5
        self.out = nn.Linear(hidden_dim, hidden_dim)

        # kmeans_init=False (random init) + threshold_ema_dead_code=2 (revive any
        # code whose EMA usage drops to zero by replacing it with a randomly-sampled
        # active embedding). Together these prevent the start-of-training collapse.
        self.vq = VectorQuantize(
            dim=hidden_dim,
            codebook_size=codebook_size,
            decay=ema_decay,
            commitment_weight=commitment_weight,
            kmeans_init=False,
            threshold_ema_dead_code=2,
        )

    def forward(
        self,
        frames: torch.Tensor,            # (B, T, 14)
        frame_mask: torch.Tensor,        # (B, T) bool
        frame_to_enc_pos: torch.Tensor,  # (B, T) long; values in [1, N_body+1]
        n_total: int,                    # N+2 (encoder positions: BOS + body + EOS)
    ):
        """Returns:
          codes:   (B, N+2) long — per-phoneme style code (PAD at BOS/EOS)
          z_q:     (B, N+2, D)  — quantized embedding (zeros at BOS/EOS)
          commit:  scalar loss
        """
        B, T, _ = frames.shape
        D = self.proj_in.out_features
        device = frames.device
        n_body_max = n_total - 2

        # Frame embedding (no phoneme conditioning)
        x = self.norm0(self.proj_in(frames))                  # (B, T, D)
        x = self.norm1(x + F.gelu(self.layer1(x)))
        x = self.norm2(x + F.gelu(self.layer2(x)))            # (B, T, D)

        # For each body phoneme i (encoder pos i+1), attention-pool its frames.
        # Build (B, N_body_max, T) match mask.
        pos_q = torch.arange(1, n_body_max + 1, device=device).view(1, -1, 1)   # (1, Nb, 1)
        match = (frame_to_enc_pos.unsqueeze(1) == pos_q) & frame_mask.unsqueeze(1)  # (B, Nb, T)

        # Score: q · x → (B, Nb, T)
        scores = (x @ self.attn_query) * self.attn_scale       # (B, T)
        scores = scores.unsqueeze(1).expand(B, n_body_max, T).clone()
        scores = scores.masked_fill(~match, -1e9)
        # Phonemes with zero frames in batch: all -1e9 → softmax NaN. Mask after.
        any_frame = match.any(dim=-1)                          # (B, Nb) bool
        scores = scores.masked_fill(~any_frame.unsqueeze(-1), 0.0)
        weights = torch.softmax(scores, dim=-1)                # (B, Nb, T)
        weights = weights * any_frame.unsqueeze(-1).float()    # zero out missing phonemes

        pooled = torch.einsum("bnt,btd->bnd", weights, x)      # (B, Nb, D)
        emb_body = self.out(pooled)                            # (B, Nb, D)

        # Quantize body embeddings.
        z_q_body, codes_body, commit = self.vq(emb_body)       # (B, Nb, D), (B, Nb), scalar
        if commit.dim() > 0:
            commit = commit.sum()

        # Zero out missing phonemes in z_q (no gradient flow there)
        z_q_body = z_q_body * any_frame.unsqueeze(-1).float()

        # Build full (B, N+2) outputs with PAD at BOS/EOS
        codes_full = torch.full((B, n_total), self.PAD_CODE, dtype=torch.long, device=device)
        codes_full[:, 1:1 + n_body_max] = torch.where(
            any_frame, codes_body, torch.full_like(codes_body, self.PAD_CODE)
        )
        z_q_full = torch.zeros(B, n_total, D, device=device, dtype=z_q_body.dtype)
        z_q_full[:, 1:1 + n_body_max] = z_q_body

        return {"codes": codes_full, "z_q": z_q_full, "commit_loss": commit}
