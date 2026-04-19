"""
Train the VQ tokenizer on SPARC articulatory features.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer import ArticulatoryVQTokenizer
from training.dataset import VQDataset


def train_vq(args):
    device = torch.device(args.device)
    print(f"Training on {device}")

    # Dataset
    dataset = VQDataset(
        features_dir=args.features_dir,
        chunk_frames=args.chunk_frames,
        stride_frames=args.stride_frames,
    )

    # Train/val split
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = ArticulatoryVQTokenizer(
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss weights: 2x weight on pitch (idx 12) and loudness (idx 13)
    channel_weights = torch.ones(14, device=device)
    channel_weights[12] = 2.0  # pitch
    channel_weights[13] = 2.0  # loudness

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_losses = {"recon": 0, "commit": 0, "total": 0}
        train_perplexity = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            batch = batch.to(device)  # (B, T, 14)

            result = model(batch)

            # Weighted MSE reconstruction loss
            diff = (result["reconstructed"] - batch) ** 2  # (B, T, 14)
            recon_loss = (diff * channel_weights).mean()

            total_loss = recon_loss + result["commit_loss"]

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses["recon"] += recon_loss.item()
            train_losses["commit"] += result["commit_loss"].item()
            train_losses["total"] += total_loss.item()
            train_perplexity += result["perplexity"].item()

        n = len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_perplexity = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                result = model(batch)
                diff = (result["reconstructed"] - batch) ** 2
                recon_loss = (diff * channel_weights).mean()
                val_loss += recon_loss.item()
                val_perplexity += result["perplexity"].item()

        vn = len(val_loader)

        print(
            f"Epoch {epoch:3d} | "
            f"Train: recon={train_losses['recon']/n:.4f} commit={train_losses['commit']/n:.4f} | "
            f"Val: recon={val_loss/vn:.4f} | "
            f"Perplexity: train={train_perplexity/n:.1f} val={val_perplexity/vn:.1f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best model
        if val_loss / vn < best_val_loss:
            best_val_loss = val_loss / vn
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_dir / "vq_best.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss / vn,
            }, ckpt_dir / f"vq_epoch{epoch}.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--chunk-frames", type=int, default=200)
    parser.add_argument("--stride-frames", type=int, default=100)
    args = parser.parse_args()

    train_vq(args)
