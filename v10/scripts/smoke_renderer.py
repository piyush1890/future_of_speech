"""Synthetic forward+backward sanity for V10StyleEncoder + V10Renderer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from v10.models.v10_style import V10StyleEncoder
from v10.models.v10_renderer import V10Renderer


def synthetic():
    torch.manual_seed(0)
    B = 2
    N_body = 6
    N_total = N_body + 2  # BOS + body + EOS = 8
    T = 40
    K = 4
    C = 256
    style_C = 32

    # Phonemes (random in [3, 72])
    phoneme_ids = torch.randint(3, 73, (B, N_total))
    phoneme_ids[:, 0] = 1   # BOS
    phoneme_ids[:, -1] = 2  # EOS
    phoneme_mask = torch.ones(B, N_total, dtype=torch.bool)

    # Frames
    frames = torch.randn(B, T, 14)
    frame_mask = torch.ones(B, T, dtype=torch.bool)
    frame_mask[1, 30:] = False   # row 1 has T=30

    # frame_to_enc_pos: distribute frames across body phonemes
    # Body positions are 1..N_body (= 1..6). Easy: chunk T into 6 equal-ish.
    frame_to_enc_pos = torch.zeros(B, T, dtype=torch.long)
    eop = torch.zeros(B, T, dtype=torch.float)
    body_dur = torch.zeros(B, N_body, dtype=torch.long)
    for b in range(B):
        valid_T = int(frame_mask[b].sum().item())
        sizes = [valid_T // N_body] * N_body
        for i in range(valid_T - sum(sizes)):
            sizes[i] += 1
        cursor = 0
        for i, s in enumerate(sizes):
            frame_to_enc_pos[b, cursor:cursor + s] = i + 1
            if s > 0:
                eop[b, cursor + s - 1] = 1.0
            body_dur[b, i] = s
            cursor += s

    # Frame codes (random RVQ tokens)
    frame_codes = torch.randint(0, C, (B, T, K))

    spk_emb = torch.randn(B, 64)
    knobs = torch.randn(B, 6)

    style_enc = V10StyleEncoder(codebook_size=style_C)
    renderer = V10Renderer(
        codebook_size=C, num_quantizers=K, style_codebook_size=style_C,
        d_model=128, num_encoder_layers=2, num_decoder_layers=2,
        knob_dim=6, max_phonemes=20, max_frames=64,
    )

    print(f"synthetic batch: B={B} N_total={N_total} T={T} K={K}")

    style_out = style_enc(frames, frame_mask, frame_to_enc_pos, n_total=N_total)
    print(f"  style codes: {style_out['codes'].shape}  (expect {(B, N_total)})")
    print(f"  style codes BOS={style_out['codes'][:, 0].tolist()}  EOS={style_out['codes'][:, -1].tolist()}  "
          f"(expect PAD={style_C})")
    print(f"  style commit: {float(style_out['commit_loss'].item()):.4f}")

    r_out = renderer(
        phoneme_ids, style_out["codes"], spk_emb, knobs, phoneme_mask,
        frame_codes, frame_to_enc_pos, frame_mask,
    )
    print(f"  frame_logits: {r_out['frame_logits'].shape}  (expect {(B, T, K, C)})")
    print(f"  eop_logit:    {r_out['eop_logit'].shape}     (expect {(B, T)})")

    # Compute losses
    fmask = frame_mask.float()
    denom = fmask.sum().clamp(min=1.0)
    ce = 0.0
    for k in range(K):
        ce_k = F.cross_entropy(
            r_out["frame_logits"][..., k, :].reshape(-1, C),
            frame_codes[..., k].reshape(-1),
            reduction="none",
        ).reshape(B, T)
        ce = ce + (ce_k * fmask).sum() / denom

    eop_loss = F.binary_cross_entropy_with_logits(
        r_out["eop_logit"], eop, reduction="none",
    )
    eop_loss = (eop_loss * fmask).sum() / denom

    total = ce + 0.5 * eop_loss + 0.25 * style_out["commit_loss"]
    print(f"  ce={ce.item():.3f}  eop={eop_loss.item():.3f}  total={total.item():.3f}")
    total.backward()
    print("  backward ok")


if __name__ == "__main__":
    synthetic()
