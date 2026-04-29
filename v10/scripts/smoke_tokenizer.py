"""Synthetic + real-data sanity for V10Tokenizer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from v10.models.v10_tokenizer import V10Tokenizer
from v10.training.dataset_v10 import V10Dataset, collate_v10_tokenizer


def synthetic():
    print("\n=== synthetic ===")
    tok = V10Tokenizer(d_model=128, num_encoder_layers=2, num_decoder_layers=2,
                       codebook_size=256, num_quantizers=4)
    B, T, D = 3, 80, 14
    frames = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 60:] = False  # row 1 has length 60
    mask[2, 40:] = False  # row 2 has length 40

    out = tok(frames, mask)
    assert out["recon"].shape == (B, T, D), out["recon"].shape
    assert out["idx"].shape == (B, T, 4), out["idx"].shape
    assert out["commit_loss"].dim() == 0
    print(f"  recon: {out['recon'].shape}  idx: {out['idx'].shape}  commit: {out['commit_loss'].item():.4f}")

    valid = mask.unsqueeze(-1).float()
    mse = ((out["recon"] - frames) ** 2 * valid).sum() / (valid.sum() * D)
    loss = mse + 0.25 * out["commit_loss"]
    loss.backward()
    print(f"  loss: {loss.item():.4f}  mse: {mse.item():.4f}")
    print("  backward ok")

    # tokens_to_frames roundtrip
    with torch.no_grad():
        recon2 = tok.tokens_to_frames(out["idx"], mask)
        diff = (recon2 - out["recon"]).abs().mean().item()
        print(f"  tokens_to_frames diff vs forward recon: {diff:.6f}  (≈0 expected)")


def real_data():
    print("\n=== real data ===")
    try:
        ds = V10Dataset(preload=False, max_frames=600)
    except FileNotFoundError as e:
        print(f"  skip (no data): {e}")
        return
    print(f"  dataset: {len(ds)} utterances")
    if len(ds) == 0:
        print("  no utterances after filter; skip")
        return

    item = ds[0]
    print(f"  uid={item['uid']}  T={item['frames'].shape[0]}  N_body={item['body_durations'].shape[0]}")
    print(f"  eop sum={int(item['eop'].sum())}  expected={item['body_durations'].shape[0]}")
    assert int(item["eop"].sum()) == int((item["body_durations"] > 0).sum()), \
        "EOP count must match nonzero body phonemes"

    batch = collate_v10_tokenizer([ds[0], ds[1]])
    tok = V10Tokenizer(d_model=128, num_encoder_layers=2, num_decoder_layers=2)
    out = tok(batch["frames"], batch["frame_mask"])
    print(f"  batched recon: {out['recon'].shape}  idx: {out['idx'].shape}")
    print("  ok")


if __name__ == "__main__":
    torch.manual_seed(0)
    synthetic()
    real_data()
