"""
Train the emotion-delta predictor on the ESD-derived paired dataset.

    python training/train_emotion_delta.py --epochs 40 --device cpu

Defaults to CPU so it doesn't interfere with the main TTS training on MPS.
Pass --device mps to force MPS (will slow down whatever else is running there).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.emotion_delta_predictor import EmotionDeltaPredictor
from models.phoneme_vocab import PhonemeVocab


# Per-channel loss weighting: EMA channels are small-magnitude, pitch and loudness
# carry the emotion signal we most care about perceptually.
LOSS_WEIGHTS = np.ones(14, dtype=np.float32)
LOSS_WEIGHTS[12] = 3.0   # log_pitch
LOSS_WEIGHTS[13] = 2.0   # loudness


class DeltaDataset(Dataset):
    def __init__(self, folder: Path, max_T: int = 1200):
        self.files = sorted(folder.glob("*.npz"))
        self.max_T = max_T

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        phon = torch.from_numpy(d["phoneme_ids"].astype(np.int64))
        # New MFA dataset uses "emotional_durations"; old dataset uses "phoneme_durations"
        dur_key = "emotional_durations" if "emotional_durations" in d.files else "phoneme_durations"
        durs = torch.from_numpy(d[dur_key].astype(np.int64))
        emo = torch.tensor(int(d["emotion_idx"]), dtype=torch.long)
        spk = torch.from_numpy(d["speaker_emb"].astype(np.float32))
        delta = torch.from_numpy(d["delta"].astype(np.float32))
        # Clamp tail if too long
        if delta.shape[0] > self.max_T:
            delta = delta[:self.max_T]
            # Trim durations so sum == new T
            cum = durs.cumsum(0)
            keep = (cum <= self.max_T).sum().item()
            durs = durs[:max(1, keep)]
            phon = phon[:max(1, keep)]
            # Adjust last duration if sum < T
            used = durs.sum().item()
            if used < delta.shape[0]:
                durs[-1] = durs[-1] + (delta.shape[0] - used)
        return phon, durs, emo, spk, delta


def collate(batch):
    phons, durs_list, emos, spks, deltas = zip(*batch)
    N_max = max(p.shape[0] for p in phons)
    T_max = max(d.shape[0] for d in deltas)
    B = len(batch)
    phon_pad = torch.zeros(B, N_max, dtype=torch.long)
    dur_pad = torch.zeros(B, N_max, dtype=torch.long)
    delta_pad = torch.zeros(B, T_max, 14, dtype=torch.float32)
    for i, (p, d, dl) in enumerate(zip(phons, durs_list, deltas)):
        phon_pad[i, :p.shape[0]] = p
        dur_pad[i, :d.shape[0]] = d
        delta_pad[i, :dl.shape[0]] = dl
    emo = torch.stack(list(emos))
    spk = torch.stack(list(spks))
    return phon_pad, dur_pad, emo, spk, delta_pad


def weighted_mse(pred, target, frame_mask, channel_weights):
    # pred, target: (B, T, 14); frame_mask: (B, T); channel_weights: (14,)
    diff2 = (pred - target) ** 2
    diff2 = diff2 * channel_weights.view(1, 1, -1)
    frame_mask_f = frame_mask.float().unsqueeze(-1)
    masked = diff2 * frame_mask_f
    denom = frame_mask_f.sum() * channel_weights.sum() + 1e-8
    return masked.sum() / denom


def weighted_gaussian_nll(mu, log_var, target, frame_mask, channel_weights):
    """Gaussian negative log-likelihood: 0.5 * (((y-mu)/sigma)^2 + log(sigma^2))
    Per-channel weighted, masked to valid frames."""
    inv_var = torch.exp(-log_var)
    nll = 0.5 * ((target - mu) ** 2) * inv_var + 0.5 * log_var
    nll = nll * channel_weights.view(1, 1, -1)
    fm = frame_mask.float().unsqueeze(-1)
    masked = nll * fm
    denom = fm.sum() * channel_weights.sum() + 1e-8
    return masked.sum() / denom


def run_validation(model, loader, device, channel_weights, variational: bool):
    model.eval()
    total, n_batches = 0.0, 0
    per_ch_err = torch.zeros(14, device=device)
    per_ch_sigma = torch.zeros(14, device=device)
    with torch.no_grad():
        for phon, durs, emo, spk, delta in loader:
            phon = phon.to(device); durs = durs.to(device)
            emo = emo.to(device); spk = spk.to(device); delta = delta.to(device)
            out = model(phon, durs, emo, spk)
            if variational:
                mu, log_var, fm = out
                T = min(mu.shape[1], delta.shape[1])
                mu = mu[:, :T]; log_var = log_var[:, :T]
                tgt = delta[:, :T]; fm_ = fm[:, :T]
                loss = weighted_gaussian_nll(mu, log_var, tgt, fm_, channel_weights)
                pred = mu
                sigma = torch.exp(0.5 * log_var)
                per_ch_sigma += (sigma * fm_.unsqueeze(-1).float()).sum(dim=(0, 1)) / (fm_.sum() + 1e-8)
            else:
                pred, fm = out
                T = min(pred.shape[1], delta.shape[1])
                pred = pred[:, :T]
                tgt = delta[:, :T]; fm_ = fm[:, :T]
                loss = weighted_mse(pred, tgt, fm_, channel_weights)
            total += float(loss.item())
            n_batches += 1
            abs_err = (pred - tgt).abs() * fm_.unsqueeze(-1).float()
            per_ch_err += abs_err.sum(dim=(0, 1)) / (fm_.sum() + 1e-8)
    model.train()
    return (total / max(1, n_batches),
            (per_ch_err / n_batches).cpu().numpy(),
            (per_ch_sigma / max(1, n_batches)).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="data/emotion_delta_dataset")
    ap.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--out-dir", type=str, default="checkpoints_emotion_delta")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=400)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-phon-layers", type=int, default=4)
    ap.add_argument("--num-frame-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--variational", action="store_true",
                    help="Predict mu+log_var per frame; train with Gaussian NLL.")
    ap.add_argument("--min-sigma", type=float, default=0.05,
                    help="Floor on sigma to prevent variance collapse (only with --variational).")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    vocab = PhonemeVocab(args.vocab)
    train_ds = DeltaDataset(Path(args.dataset_dir) / "train")
    val_ds = DeltaDataset(Path(args.dataset_dir) / "val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    model = EmotionDeltaPredictor(
        vocab_size=len(vocab), n_emotions=4, spk_emb_dim=64,
        d_model=args.d_model, nhead=args.nhead,
        num_phon_layers=args.num_phon_layers,
        num_frame_layers=args.num_frame_layers,
        d_ff=args.d_ff, dropout=args.dropout, out_dim=14,
        variational=args.variational, min_sigma=args.min_sigma,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * t))

    channel_weights = torch.from_numpy(LOSS_WEIGHTS).to(device)

    best_val = float("inf")
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        n_seen = 0
        for phon, durs, emo, spk, delta in train_loader:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            phon = phon.to(device); durs = durs.to(device)
            emo = emo.to(device); spk = spk.to(device); delta = delta.to(device)

            out = model(phon, durs, emo, spk)
            if args.variational:
                mu, log_var, fm = out
                T = min(mu.shape[1], delta.shape[1])
                mu = mu[:, :T]; log_var = log_var[:, :T]
                tgt = delta[:, :T]; fm_ = fm[:, :T]
                loss = weighted_gaussian_nll(mu, log_var, tgt, fm_, channel_weights)
            else:
                pred, fm = out
                T = min(pred.shape[1], delta.shape[1])
                pred = pred[:, :T]
                tgt = delta[:, :T]; fm_ = fm[:, :T]
                loss = weighted_mse(pred, tgt, fm_, channel_weights)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            running += float(loss.item())
            n_seen += 1
            step += 1
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                lr_now = lr_at(step)
                print(f"epoch {epoch+1}/{args.epochs} step {step} lr={lr_now:.2e} "
                      f"loss={running/n_seen:.5f} elapsed={elapsed/60:.1f}min")

        val_loss, per_ch, per_sigma = run_validation(model, val_loader, device, channel_weights, args.variational)
        sigma_msg = f" sigma_pitch={per_sigma[12]:.3f} sigma_loud={per_sigma[13]:.3f}" if args.variational else ""
        print(f"[epoch {epoch+1}] train={running/max(1,n_seen):.5f} val={val_loss:.5f} "
              f"pitch_err={per_ch[12]:.4f} loud_err={per_ch[13]:.4f}{sigma_msg}")
        # Save last + best
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "vocab_size": len(vocab),
            "epoch": epoch + 1,
            "val_loss": val_loss,
        }, out_dir / "delta_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "vocab_size": len(vocab),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, out_dir / "delta_best.pt")
            print(f"  ↳ saved best (val={val_loss:.5f})")

    print(f"Done. best val={best_val:.5f}. Checkpoints: {out_dir}/delta_best.pt")


if __name__ == "__main__":
    main()
