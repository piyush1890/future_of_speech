"""
v9 phoneme-level RVQ tokenizer (batched).

Per phoneme:
  frame block (F, 14) → split into start half + end half
                      → encode each → 2 latent vectors
                      → 2× ResidualVQ → 2 stacks of K token IDs
                      → decoder: (start_q, end_q, phoneme_id, F) → (F, 14)

The whole thing is implemented to operate on BATCHES of phonemes — we pack
every phoneme block from a training batch into a single padded tensor of
shape (B_phonemes, F_max, 14) plus a mask, and run all phonemes through a
single forward pass. This avoids the per-phoneme Python loop that ate all
the GPU dispatch overhead.

Two tokens per phoneme so that within-phoneme transitions (rising contours,
coarticulation glides) can be expressed via different start vs end codes.
The decoder's cross-attention with smooth sinusoidal positional queries gives
within-phoneme smoothness for free, so we don't need an explicit smoothness
penalty (which would over-penalize legitimate plosive bursts).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


class HalfBlockEncoder(nn.Module):
    """Batched half-block encoder.

    Input:
      frames:  (B, F_max, 14)
      ph_emb:  (B, hidden_dim)
      mask:    (B, F_max) bool — True at valid frames for THIS half (start or end)
    Output: (B, latent_dim)

    Uses Linear+LayerNorm+GELU residual stack (no Conv1d to dodge an MPS
    shape-validation bug on small inputs), with a learnable-query attention
    pool over the masked frames.
    """
    def __init__(self, input_dim: int = 14, hidden_dim: int = 128, latent_dim: int = 256):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attn_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.attn_scale = hidden_dim ** -0.5

        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, frames: torch.Tensor, ph_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # frames: (B, F_max, 14); ph_emb: (B, H); mask: (B, F_max)
        # Phoneme conditioning: add ph_emb to every frame
        x = self.norm0(self.proj_in(frames) + ph_emb.unsqueeze(1))      # (B, F_max, H)
        x = self.norm1(x + F.gelu(self.layer1(x)))
        x = self.norm2(x + F.gelu(self.layer2(x)))

        # Attention-pool with mask: scores[b, t] = (q · x[b, t]) / sqrt(H)
        scores = (x @ self.attn_query) * self.attn_scale                # (B, F_max)
        scores = scores.masked_fill(~mask, -1e9)
        # If a row has all-False mask, softmax gives NaN — caller must avoid.
        weights = torch.softmax(scores, dim=1)                          # (B, F_max)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)                 # (B, H)
        return self.out(pooled)                                         # (B, latent_dim)


class FrameDecoder(nn.Module):
    """Batched cross-attention decoder.

    Input:
      start_z, end_z: (B, latent_dim)   — quantized half latents
      ph_emb_dec:     (B, d_model)      — phoneme embedding (decoder dim)
      lengths:        (B,) long         — actual F per phoneme
    Output: (B, F_max, output_dim)
    Caller masks the loss at positions ≥ length[b].
    """
    def __init__(self, latent_dim: int = 256, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, output_dim: int = 14):
        super().__init__()
        self.d_model = d_model

        self.mem_proj = nn.Linear(latent_dim, d_model)
        # Memory has 3 slots: [start, end, phoneme]
        self.mem_type = nn.Embedding(3, d_model)

        self.alpha_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, output_dim)

    @staticmethod
    def _sinusoidal_pos_batch(alpha: torch.Tensor, d_model: int) -> torch.Tensor:
        """alpha: (B, F_max) in [0, 1]. Returns (B, F_max, d_model)."""
        device = alpha.device
        div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                        * (-math.log(10000.0) / d_model))
        pos = alpha.unsqueeze(-1) * 100.0                     # (B, F_max, 1)
        pe = torch.zeros(alpha.shape[0], alpha.shape[1], d_model, device=device)
        pe[:, :, 0::2] = torch.sin(pos * div)
        pe[:, :, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, start_z: torch.Tensor, end_z: torch.Tensor,
                ph_emb_dec: torch.Tensor, lengths: torch.Tensor, F_max: int) -> torch.Tensor:
        device = start_z.device
        B = start_z.shape[0]

        # Build memory: (B, 3, d_model)
        type_ids = torch.tensor([0, 1, 2], device=device)
        type_emb = self.mem_type(type_ids)                              # (3, d_model)
        mem_start = self.mem_proj(start_z) + type_emb[0]
        mem_end   = self.mem_proj(end_z)   + type_emb[1]
        mem_ph    = ph_emb_dec              + type_emb[2]
        mem = torch.stack([mem_start, mem_end, mem_ph], dim=1)          # (B, 3, d_model)

        # Build per-row alpha values:
        #   alpha[b, t] = t / (length[b] - 1) for t < length[b], padded after.
        #   For length=1, original behaviour was alpha=[0.5] (single midpoint).
        idx_t = torch.arange(F_max, device=device).unsqueeze(0).float() # (1, F_max)
        denom = (lengths - 1).clamp(min=1).unsqueeze(-1).float()         # (B, 1)
        alpha = idx_t / denom                                            # (B, F_max)
        # Handle F=1 rows: alpha should be 0.5 at t=0 (and ignored for t≥1)
        is_single = (lengths == 1).unsqueeze(-1)
        alpha = torch.where(is_single, torch.full_like(alpha, 0.5), alpha)

        pe = self._sinusoidal_pos_batch(alpha, self.d_model)             # (B, F_max, d_model)
        q = self.alpha_proj(pe)                                          # (B, F_max, d_model)

        # Cross-attention. Pad-position queries still cost compute but produce
        # outputs that the loss masks out — simpler than tgt_key_padding_mask.
        out = self.decoder(q, mem)                                       # (B, F_max, d_model)
        return self.out_proj(out)                                        # (B, F_max, output_dim)


class PhonemeRVQTokenizer(nn.Module):
    """Top-level batched tokenizer: encode → 2× RVQ → decode.

    Main entry: `forward_batch(frames, ph_ids, lengths)` operates on a
    full batch of phoneme blocks at once.
    """
    def __init__(
        self,
        vocab_size: int = 73,
        input_dim: int = 14,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        codebook_size: int = 512,
        num_quantizers: int = 4,
        decoder_d_model: int = 256,
        decoder_nhead: int = 4,
        decoder_layers: int = 2,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        # Phoneme symbol embedding shared across encoder + decoder
        self.phoneme_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.phoneme_to_decoder = nn.Linear(hidden_dim, decoder_d_model)

        self.start_encoder = HalfBlockEncoder(input_dim, hidden_dim, latent_dim)
        self.end_encoder   = HalfBlockEncoder(input_dim, hidden_dim, latent_dim)

        self.rvq_start = ResidualVQ(
            dim=latent_dim, num_quantizers=num_quantizers, codebook_size=codebook_size,
            decay=ema_decay, commitment_weight=commitment_weight,
            kmeans_init=True, kmeans_iters=10,
        )
        self.rvq_end = ResidualVQ(
            dim=latent_dim, num_quantizers=num_quantizers, codebook_size=codebook_size,
            decay=ema_decay, commitment_weight=commitment_weight,
            kmeans_init=True, kmeans_iters=10,
        )

        self.decoder = FrameDecoder(
            latent_dim=latent_dim, d_model=decoder_d_model,
            nhead=decoder_nhead, num_layers=decoder_layers, output_dim=input_dim,
        )

    @staticmethod
    def _build_masks(lengths: torch.Tensor, F_max: int):
        """Build (start_mask, end_mask, valid_mask), each (B, F_max) bool.

        Splitting rule (matches original per-phoneme code):
          - F == 1: both halves see the single frame (mid := 1, end_mask := start_mask)
          - F >= 2: mid = F // 2; start = [0, mid), end = [mid, F)
        """
        device = lengths.device
        idx_t = torch.arange(F_max, device=device).unsqueeze(0)         # (1, F_max)
        valid_mask = idx_t < lengths.unsqueeze(-1)                       # (B, F_max)

        mid = (lengths // 2).clamp(min=1).unsqueeze(-1)                  # (B, 1)
        start_mask = (idx_t < mid) & valid_mask
        end_mask   = (idx_t >= mid) & valid_mask
        # F==1 special case
        end_mask = torch.where(lengths.unsqueeze(-1) == 1, start_mask, end_mask)
        return start_mask, end_mask, valid_mask

    def forward_batch(self, frames: torch.Tensor, ph_ids: torch.Tensor, lengths: torch.Tensor):
        """
        frames:  (B, F_max, 14) — padded
        ph_ids:  (B,) long
        lengths: (B,) long      — actual F per row
        Returns: (recon (B, F_max, 14), info dict)
        """
        B, F_max, _ = frames.shape
        start_mask, end_mask, valid_mask = self._build_masks(lengths, F_max)

        ph_emb_enc = self.phoneme_embedding(ph_ids)                      # (B, H)
        ph_emb_dec = self.phoneme_to_decoder(ph_emb_enc)                 # (B, d_model)

        start_z = self.start_encoder(frames, ph_emb_enc, start_mask)     # (B, latent_dim)
        end_z   = self.end_encoder(frames,   ph_emb_enc, end_mask)

        # ResidualVQ expects (B, T, D); we set T=1 by unsqueezing
        s_q, s_idx, s_cl = self.rvq_start(start_z.unsqueeze(1))          # s_q: (B,1,D), s_idx: (B,1,K)
        e_q, e_idx, e_cl = self.rvq_end(end_z.unsqueeze(1))
        s_q   = s_q.squeeze(1); e_q   = e_q.squeeze(1)
        s_idx = s_idx.squeeze(1); e_idx = e_idx.squeeze(1)
        commit = (s_cl.sum() if s_cl.dim() > 0 else s_cl) \
                 + (e_cl.sum() if e_cl.dim() > 0 else e_cl)

        recon = self.decoder(s_q, e_q, ph_emb_dec, lengths, F_max)        # (B, F_max, 14)
        return recon, {
            "start_idx": s_idx, "end_idx": e_idx,
            "commit_loss": commit, "valid_mask": valid_mask,
        }

    def decode_indices_batch(self, s_idx: torch.Tensor, e_idx: torch.Tensor,
                             ph_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Inference path: token IDs → frames.

        s_idx, e_idx: (B, K) long
        ph_ids:       (B,) long
        lengths:      (B,) long

        Returns: list of (length[b], 14) tensors (variable lengths) — concatenated
        by caller into the full utterance frame stream.
        """
        F_max = int(lengths.max().item())
        # ResidualVQ accepts (B, T, K) indices when T=1
        s_q = self.rvq_start.get_output_from_indices(s_idx.unsqueeze(1)).squeeze(1)   # (B, D)
        e_q = self.rvq_end.get_output_from_indices(e_idx.unsqueeze(1)).squeeze(1)
        ph_emb_enc = self.phoneme_embedding(ph_ids)
        ph_emb_dec = self.phoneme_to_decoder(ph_emb_enc)
        recon = self.decoder(s_q, e_q, ph_emb_dec, lengths, F_max)        # (B, F_max, 14)
        # Slice each row to its true length
        return [recon[b, :int(lengths[b].item())] for b in range(recon.shape[0])]
