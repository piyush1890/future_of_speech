"""Train v10 frame-level RVQ tokenizer (recon MSE + commit)."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from v10.models.v10_tokenizer import V10Tokenizer
from v10.training.dataset_v10 import V10Dataset, collate_v10_tokenizer


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def masked_mse(recon, target, mask):
    """recon, target: (B, T, D); mask: (B, T) bool."""
    diff = (recon - target) ** 2
    m = mask.unsqueeze(-1).float()
    denom = m.sum() * recon.shape[-1] + 1e-8
    return (diff * m).sum() / denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--commit-weight", type=float, default=0.25)
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--num-quantizers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=800)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-dir", default="v10/checkpoints/tokenizer")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ds = V10Dataset(max_frames=args.max_frames, preload=args.preload)
    n_val = max(64, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_v10_tokenizer,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_v10_tokenizer)

    model = V10Tokenizer(
        d_model=args.d_model,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        commitment_weight=args.commit_weight,
        max_frames=args.max_frames + 16,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = open(metrics_path, "a")

    def log(rec):
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_recon_sum = 0.0
        train_commit_sum = 0.0
        train_n = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for it, batch in enumerate(pbar):
            frames = batch["frames"].to(device)
            mask = batch["frame_mask"].to(device)

            lr = args.lr * cosine_with_warmup(step, args.warmup_steps, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            out = model(frames, mask)
            mse = masked_mse(out["recon"], frames, mask)
            loss = mse + args.commit_weight * out["commit_loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_recon_sum += mse.item()
            train_commit_sum += float(out["commit_loss"].item())
            train_n += 1
            step += 1

            pbar.set_postfix({"mse": f"{mse.item():.4f}",
                              "commit": f"{out['commit_loss'].item():.4f}"})
            if step % args.log_every == 0:
                log({"type": "step", "step": step, "lr": lr,
                     "mse": mse.item(),
                     "commit": float(out["commit_loss"].item()),
                     "total": loss.item()})
                tqdm.write(f"e{epoch} step {step}  mse={mse.item():.4f}  commit={out['commit_loss'].item():.4e}  lr={lr:.2e}")

        # Validation
        model.eval()
        val_recon_sum = 0.0
        val_commit_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_dl:
                frames = batch["frames"].to(device)
                mask = batch["frame_mask"].to(device)
                out = model(frames, mask)
                mse = masked_mse(out["recon"], frames, mask)
                val_recon_sum += mse.item()
                val_commit_sum += float(out["commit_loss"].item())
                val_n += 1

        train_mse = train_recon_sum / max(1, train_n)
        val_mse = val_recon_sum / max(1, val_n)
        elapsed = time.time() - t0
        print(f"=== epoch {epoch}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  time={elapsed/60:.1f}m ===")
        log({"type": "epoch", "epoch": epoch, "train_mse": train_mse,
             "val_mse": val_mse, "train_commit": train_commit_sum / max(1, train_n),
             "val_commit": val_commit_sum / max(1, val_n),
             "elapsed_sec": elapsed})

        ckpt = {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                "val_mse": val_mse}
        torch.save(ckpt, out_dir / "last.pt")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> new best (val_mse={val_mse:.4f})")

    metrics_f.close()


if __name__ == "__main__":
    main()
