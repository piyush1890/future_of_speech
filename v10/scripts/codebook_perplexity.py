"""Snapshot codebook perplexity for v10 tokenizer.

Loads checkpoint, runs a few batches, computes per-level perplexity + unique-code
count. Doesn't touch the running training (reads checkpoint file once).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from v10.models.v10_tokenizer import V10Tokenizer
from v10.training.dataset_v10 import V10Dataset, collate_v10_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="v10/checkpoints/tokenizer/best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-utts", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targs = ckpt["args"]
    print(f"checkpoint: epoch={ckpt['epoch']}  val_mse={ckpt['val_mse']:.4f}")
    print(f"  d_model={targs['d_model']} K={targs['num_quantizers']} cb={targs['codebook_size']}")

    K = targs["num_quantizers"]
    C = targs["codebook_size"]

    model = V10Tokenizer(
        d_model=targs["d_model"], num_encoder_layers=targs["enc_layers"],
        num_decoder_layers=targs["dec_layers"], codebook_size=C, num_quantizers=K,
        max_frames=targs["max_frames"] + 16,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = V10Dataset(max_frames=targs["max_frames"], preload=False)
    sub = Subset(ds, list(range(min(args.n_utts, len(ds)))))
    dl = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_v10_tokenizer)
    print(f"sampling {len(sub)} utterances")

    counts = [torch.zeros(C, dtype=torch.long) for _ in range(K)]
    n_frames = 0
    with torch.no_grad():
        for batch in dl:
            frames = batch["frames"].to(device)
            mask = batch["frame_mask"].to(device)
            z = model.encode(frames, mask)
            _, idx, _ = model.quantize(z)                 # (B, T, K)
            valid = mask.cpu()
            idx = idx.cpu()
            for k in range(K):
                tok_k = idx[..., k][valid]               # (n_valid,) — only real frames
                counts[k] += torch.bincount(tok_k, minlength=C)
            n_frames += int(valid.sum().item())

    print(f"\ntotal valid frames sampled: {n_frames}")
    print(f"\n{'level':<6}{'unique':<10}{'usage':<10}{'perplexity':<14}{'top-1 freq':<12}")
    print("-" * 52)
    for k in range(K):
        c = counts[k]
        used = (c > 0).sum().item()
        probs = c.float() / max(1, c.sum().item())
        ent = -(probs * (probs.clamp(min=1e-12)).log()).sum().item()
        ppl = float(np.exp(ent))
        top1 = c.max().item() / max(1, c.sum().item())
        print(f"{k:<6}{used}/{C:<6}{used/C*100:>6.1f}%   {ppl:<14.1f}{top1*100:>5.1f}%")


if __name__ == "__main__":
    main()
