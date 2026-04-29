"""
v8 stage 2: train planner to predict per-phoneme z from (text, speaker, V/A/D).
Targets: GT-z's extracted by frozen v5 style encoder.

Loss: MSE on z, body-only.

Usage:
  python -u v8/training/train_v8_stage2.py --device mps --epochs 15 --preload
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v8.models.v8_planner import V8Planner, shift_for_teacher_forcing
from v8.training.dataset_v8 import PhonemeAnchorsDataset, collate_v8


def train(args):
    device = torch.device(args.device)
    print(f"v8 stage 2 (planner z-prediction) — training on {device}")

    dataset = PhonemeAnchorsDataset(
        anchors_dir=args.anchors_dir,
        z_dir=args.z_dir,
        phonemes_path=args.phonemes_path,
        vad_paths=args.vad_paths,
        max_phonemes=args.max_phonemes,
        preload=args.preload,
    )
    print(f"Dataset: {len(dataset)} utterances")

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_v8, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_v8, num_workers=0)

    planner = V8Planner(
        vocab_size=args.vocab_size,
        d_model=args.planner_d_model,
        nhead=args.planner_nhead,
        num_layers=args.planner_layers,
        d_ff=args.planner_d_ff,
        dropout=args.dropout,
        knob_dim=3,
        speaker_emb_dim=64,
        style_dim=args.style_dim,
        knob_dropout=args.knob_dropout,
        causal=args.causal,
    ).to(device)
    n_params = sum(p.numel() for p in planner.parameters())
    print(f"Planner params: {n_params:,}")

    optimizer = torch.optim.AdamW(planner.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.epochs * len(train_loader) - args.warmup_steps)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(ckpt_dir / "metrics.jsonl", "a")

    best_val = float("inf")
    start_epoch = 1
    if args.resume and (ckpt_dir / "best.pt").exists():
        c = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
        planner.load_state_dict(c["model"])
        if "optim" in c and args.restore_optim:
            optimizer.load_state_dict(c["optim"])
        start_epoch = c.get("epoch", 0) + 1
        best_val = c.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}, best {best_val:.4f}")

    def compute_losses(batch):
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        spk_emb  = batch["spk_emb"].to(device)
        knobs    = batch["knobs"].to(device)
        gt_z     = batch["z"].to(device)              # (B, N, 256)

        if args.causal:
            # Teacher forcing: prev_z = [bos_z, gt_z[0], gt_z[1], ..., gt_z[N-2]]
            prev_z = shift_for_teacher_forcing(gt_z, planner.bos_z)
            pred_z = planner(phoneme_ids, spk_emb, knobs, phoneme_mask, prev_z=prev_z)
        else:
            pred_z = planner(phoneme_ids, spk_emb, knobs, phoneme_mask)

        # Body-only mask
        body_mask = phoneme_mask.clone()
        B, _ = phoneme_mask.shape
        for b in range(B):
            idx = phoneme_mask[b].nonzero(as_tuple=True)[0]
            if len(idx) >= 2:
                body_mask[b, idx[0].item()] = False
                body_mask[b, idx[-1].item()] = False
        mask_f = body_mask.unsqueeze(-1).float()
        z_dim = gt_z.shape[-1]
        l_z = (((pred_z - gt_z) ** 2) * mask_f).sum() / (mask_f.sum().clamp(min=1) * z_dim)
        return l_z

    step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        planner.train()
        agg_z = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            l_z = compute_losses(batch)
            optimizer.zero_grad()
            l_z.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            agg_z += l_z.item(); n_batches += 1
            step += 1
            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "z": l_z.item(),
                }) + "\n"); metrics_f.flush()

        planner.eval()
        v_z = 0.0; n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                v_z += compute_losses(batch).item(); n_v += 1
        train_z = agg_z / max(1, n_batches)
        val_z = v_z / max(1, n_v)
        print(f"Epoch {epoch:3d} | train z={train_z:.4f} | val z={val_z:.4f}")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train": {"z": train_z}, "val": {"z": val_z},
            "best_val": min(best_val, val_z),
        }) + "\n"); metrics_f.flush()

        if val_z < best_val:
            best_val = val_z
            torch.save({
                "epoch": epoch,
                "model": planner.state_dict(),
                "optim": optimizer.state_dict(),
                "val_loss": best_val,
                "args": vars(args),
                "vocab_size": args.vocab_size,
            }, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val z={best_val:.4f})")

    metrics_f.close()
    print(f"\nv8 stage 2 complete. Best val z: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchors-dir",     default="v8/data/phoneme_anchors")
    p.add_argument("--z-dir",           default="v8/data/phoneme_z")
    p.add_argument("--phonemes-path",   default="data/processed_merged_v3/phonemes_mfa.json")
    p.add_argument("--vad-paths",       nargs="+",
                   default=["data/librispeech_emotion_vad.json", "data/esd_emotion_vad.json"])
    p.add_argument("--checkpoint-dir",  default="v8/checkpoints/stage2_planner")
    p.add_argument("--vocab-size",      type=int, default=73)
    p.add_argument("--max-phonemes",    type=int, default=200)
    p.add_argument("--device",          default="mps")
    p.add_argument("--batch-size",      type=int, default=32)
    p.add_argument("--epochs",          type=int, default=15)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--warmup-steps",    type=int, default=500)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--planner-d-model", type=int, default=128)
    p.add_argument("--planner-layers",  type=int, default=4)
    p.add_argument("--planner-d-ff",    type=int, default=512)
    p.add_argument("--planner-nhead",   type=int, default=4)
    p.add_argument("--style-dim",       type=int, default=256)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--knob-dropout",    type=float, default=0.1)
    p.add_argument("--causal",          action="store_true", default=True,
                   help="AR transformer with causal mask + teacher forcing on z")
    p.add_argument("--non-causal",      dest="causal", action="store_false")
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--restore-optim",   action="store_true", default=True)
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
