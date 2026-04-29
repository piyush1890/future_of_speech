"""
v8 stage 1: train encoder + heads with EXACT GT-z fed as style_emb.

This is the v5-style stage-1 approach: the model learns to USE per-phoneme z's
that are extracted offline by v5's frozen style encoder running on GT audio.

No planner involved here. Loss = anchor MSE + duration MSE.

Stage 2 (separate script) trains a planner to PREDICT z from (text, knobs).

Usage:
  python -u v8/training/train_v8_stage1.py --device mps --epochs 15 --preload
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v8.models.phoneme_tts import PhonemeTTSv8
from v8.models.phoneme_classes import build_render_class_table
from v8.training.dataset_v8 import PhonemeAnchorsDataset, collate_v8
from models.phoneme_vocab import PhonemeVocab


def train(args):
    device = torch.device(args.device)
    print(f"v8 stage 1 (encoder + heads, GT-z fed in) — training on {device}")

    dataset = PhonemeAnchorsDataset(
        anchors_dir=args.anchors_dir,
        z_dir=args.z_dir if not args.use_quantized_z else None,
        codes_dir=args.codes_dir if args.use_quantized_z else None,
        phonemes_path=args.phonemes_path,
        vad_paths=None,
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

    # Build render-class lookup table from vocab (vocab_size,) — used by hybrid render
    vocab = PhonemeVocab(args.vocab_path)
    render_class_table = build_render_class_table(vocab) if args.render_mode == "hybrid" else None

    model = PhonemeTTSv8(
        vocab_size=args.vocab_size,
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        speaker_emb_dim=64,
        style_dim=args.d_model,        # z is 256-d, matches d_model
        render_mode=args.render_mode,
        render_class_table=render_class_table,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage1 params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
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

    start_epoch = 1
    best_val = float("inf")
    if args.resume and (ckpt_dir / "best.pt").exists():
        c = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(c["model"])
        if "optim" in c and args.restore_optim:
            optimizer.load_state_dict(c["optim"])
        start_epoch = c.get("epoch", 0) + 1
        best_val = c.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}, best {best_val:.4f}")

    def compute_losses(batch):
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        gt_start = batch["start"].to(device)
        gt_mid   = batch["mid"].to(device)
        gt_end   = batch["end"].to(device)
        gt_dur   = batch["durations"].to(device).float()
        spk_emb  = batch["spk_emb"].to(device)
        # Feed quantized z (z_q) if configured, else continuous z
        gt_z = batch["z_q"].to(device) if "z_q" in batch else batch["z"].to(device)

        pred_start, pred_mid, pred_end, pred_log_dur = model.predict(
            phoneme_ids, spk_emb, phoneme_mask, style_emb=gt_z,
        )

        body_mask = phoneme_mask.clone()
        B, _ = phoneme_mask.shape
        for b in range(B):
            idx = phoneme_mask[b].nonzero(as_tuple=True)[0]
            if len(idx) >= 2:
                body_mask[b, idx[0].item()] = False
                body_mask[b, idx[-1].item()] = False
        mask_f = body_mask.unsqueeze(-1).float()
        denom = mask_f.sum().clamp(min=1) * args.feature_dim
        l_start = (((pred_start - gt_start) ** 2) * mask_f).sum() / denom
        l_mid   = (((pred_mid   - gt_mid)   ** 2) * mask_f).sum() / denom
        l_end   = (((pred_end   - gt_end)   ** 2) * mask_f).sum() / denom

        log_gt_dur = torch.log(gt_dur.clamp(min=1.0))
        mask_d = phoneme_mask.float()
        l_dur = (((pred_log_dur - log_gt_dur) ** 2) * mask_d).sum() / mask_d.sum().clamp(min=1)

        total = (args.start_weight * l_start
                 + args.mid_weight   * l_mid
                 + args.end_weight   * l_end
                 + args.dur_weight   * l_dur)
        return l_start, l_mid, l_end, l_dur, total

    step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        agg = {"start": 0.0, "mid": 0.0, "end": 0.0, "dur": 0.0, "total": 0.0}
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            l_start, l_mid, l_end, l_dur, loss = compute_losses(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            agg["start"] += l_start.item()
            agg["mid"]   += l_mid.item()
            agg["end"]   += l_end.item()
            agg["dur"]   += l_dur.item()
            agg["total"] += loss.item()
            n_batches += 1
            step += 1

            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "start": l_start.item(), "mid": l_mid.item(),
                    "end": l_end.item(), "dur": l_dur.item(), "total": loss.item(),
                }) + "\n")
                metrics_f.flush()

        model.eval()
        v_agg = {"start": 0.0, "mid": 0.0, "end": 0.0, "dur": 0.0, "total": 0.0}
        n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                l_start, l_mid, l_end, l_dur, total = compute_losses(batch)
                v_agg["start"] += l_start.item(); v_agg["mid"] += l_mid.item()
                v_agg["end"]   += l_end.item();   v_agg["dur"] += l_dur.item()
                v_agg["total"] += total.item()
                n_v += 1

        train_avg = {k: v / max(1, n_batches) for k, v in agg.items()}
        val_avg = {k: v / max(1, n_v) for k, v in v_agg.items()}
        print(f"Epoch {epoch:3d} | "
              f"train s={train_avg['start']:.4f} m={train_avg['mid']:.4f} e={train_avg['end']:.4f} d={train_avg['dur']:.4f} | "
              f"val s={val_avg['start']:.4f} m={val_avg['mid']:.4f} e={val_avg['end']:.4f} d={val_avg['dur']:.4f} "
              f"total={val_avg['total']:.4f}")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train": train_avg, "val": val_avg,
            "best_val": min(best_val, val_avg["total"]),
        }) + "\n")
        metrics_f.flush()

        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "val_loss": best_val,
                "args": vars(args),
                "vocab_size": args.vocab_size,
            }, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val total={best_val:.4f})")

    metrics_f.close()
    print(f"\nv8 stage 1 complete. Best val total: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchors-dir",     default="v8/data/phoneme_anchors")
    p.add_argument("--z-dir",           default="v8/data/phoneme_z")
    p.add_argument("--codes-dir",       default="v8/data/phoneme_codes")
    p.add_argument("--use-quantized-z", action="store_true",
                   help="Feed z_q (codebook-quantized z) instead of continuous z")
    p.add_argument("--phonemes-path",   default="data/processed_merged_v3/phonemes_mfa.json")
    p.add_argument("--checkpoint-dir",  default="v8/checkpoints/stage1_z")
    p.add_argument("--vocab-size",      type=int, default=73)
    p.add_argument("--feature-dim",     type=int, default=14)
    p.add_argument("--max-phonemes",    type=int, default=200)
    p.add_argument("--device",          default="mps")
    p.add_argument("--batch-size",      type=int, default=32)
    p.add_argument("--epochs",          type=int, default=15)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--warmup-steps",    type=int, default=500)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--d-model",         type=int, default=256)
    p.add_argument("--nhead",           type=int, default=4)
    p.add_argument("--num-layers",      type=int, default=4)
    p.add_argument("--d-ff",            type=int, default=1024)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--start-weight",    type=float, default=1.0)
    p.add_argument("--mid-weight",      type=float, default=1.0)
    p.add_argument("--end-weight",      type=float, default=1.0)
    p.add_argument("--dur-weight",      type=float, default=0.1)
    p.add_argument("--render-mode",     choices=["hybrid", "hmm", "linear"], default="hybrid")
    p.add_argument("--vocab-path",      default="data/processed_all/vocab_mfa.json")
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--restore-optim",   action="store_true", default=True)
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
