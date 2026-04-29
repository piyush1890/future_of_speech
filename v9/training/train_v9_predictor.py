"""
v9 Step 3: train the AR predictor.

Inputs come from v9/data/phoneme_tokens/<uid>.npz (per-phoneme RVQ tokens
extracted by the trained tokenizer in Step 2). We never touch frames here.

Loss:
  CE on start_logits per RVQ level (level-weighted)
  CE on end_logits   per RVQ level (level-weighted)
  MSE on log-durations
  All masked to body positions (excluding BOS/EOS).

Optimizations (matching v4/v5):
  - BucketBatchSampler over utterances (sorted by phoneme count) → minimal padding.
  - Length rounding to multiples of 16.
  - AdamW + warmup → cosine, weight_decay=0.01.
  - Gradient clipping (norm=1.0).
  - Knob dropout in the model (CFG-ready).
  - Resume from last.pt (preferred) or best.pt.
  - Per-RVQ-level accuracy tracking + token-usage tracking (dead-code monitor).
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

from v9.models.v9_predictor import V9Predictor
from v9.training.dataset_v9_pred import (
    V9PredictorDataset, collate_v9_pred, BucketBatchSampler,
)


def train(args):
    device = torch.device(args.device)
    print(f"v9 predictor — training on {device}")

    dataset = V9PredictorDataset(
        tokens_dir=args.tokens_dir,
        knob_source=args.knob_source,
        vad_paths=args.vad_paths,
        metadata_path=args.metadata_path,
        max_phonemes=args.max_phonemes,
        preload=args.preload,
    )
    print(f"  knob_dim = {dataset.knob_dim} (knob_source={args.knob_source})")

    val_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    train_sampler = BucketBatchSampler(train_set, batch_size=args.batch_size, shuffle=True)
    val_sampler   = BucketBatchSampler(val_set,   batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              collate_fn=collate_v9_pred, num_workers=0)
    val_loader   = DataLoader(val_set, batch_sampler=val_sampler,
                              collate_fn=collate_v9_pred, num_workers=0)

    model = V9Predictor(
        vocab_size=args.vocab_size,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        speaker_emb_dim=64,
        knob_dim=dataset.knob_dim,
        knob_dropout=args.knob_dropout,
        max_phonemes=args.max_phonemes,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Predictor params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Level weights: equal by default; user can pass comma-sep.
    if args.level_weights:
        weights = [float(w) for w in args.level_weights.split(",")]
        assert len(weights) == args.num_quantizers, "level-weights length mismatch"
    else:
        weights = [1.0] * args.num_quantizers
    level_w = torch.tensor(weights, device=device)
    level_w = level_w / level_w.sum()
    print(f"Level weights (normalized): {level_w.tolist()}")

    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(ckpt_dir / "metrics.jsonl", "a")

    best_val = float("inf")
    start_epoch = 1
    step = 0

    if args.resume:
        path = ckpt_dir / "last.pt"
        if not path.exists():
            path = ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else None
        if path is not None:
            print(f"Resuming from {path}")
            c = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(c["model"])
            if "optim" in c and args.restore_optim:
                optimizer.load_state_dict(c["optim"])
            if "scheduler" in c:
                scheduler.load_state_dict(c["scheduler"])
            step = int(c.get("step", 0))
            start_epoch = int(c.get("epoch", 0)) + 1
            best_val = float(c.get("val_loss", float("inf")))
            print(f"  resumed at epoch {start_epoch - 1}, step {step}, best_val {best_val:.4f}")

    def compute_losses(batch):
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        body_mask = batch["body_mask"].to(device)
        gt_start = batch["start_idx"].to(device)
        gt_end   = batch["end_idx"].to(device)
        gt_log_dur = batch["log_durations"].to(device)
        spk = batch["spk_emb"].to(device)
        knobs = batch.get("knobs")
        if knobs is not None: knobs = knobs.to(device)

        out = model(phoneme_ids, spk, knobs, phoneme_mask, gt_start, gt_end)
        s_logits = out["start_logits"]                                        # (B, N, K, C)
        e_logits = out["end_logits"]
        log_dur  = out["log_dur"]                                              # (B, N)
        B, N, K, C = s_logits.shape

        body_f = body_mask.float()
        denom_body = body_f.sum().clamp(min=1)

        # CE per level (only at body positions)
        ce_total = torch.tensor(0.0, device=device)
        s_acc = torch.zeros(K, device=device)
        e_acc = torch.zeros(K, device=device)
        for k in range(K):
            for logits_k, gt_k, acc_acc in [
                (s_logits[..., k, :], gt_start[..., k], s_acc),
                (e_logits[..., k, :], gt_end[..., k],   e_acc),
            ]:
                ce_per_pos = F.cross_entropy(
                    logits_k.reshape(-1, C), gt_k.reshape(-1), reduction="none"
                ).reshape(B, N)
                ce_masked = (ce_per_pos * body_f).sum() / denom_body
                ce_total = ce_total + level_w[k] * ce_masked
                with torch.no_grad():
                    pred = logits_k.argmax(-1)
                    correct = ((pred == gt_k).float() * body_f).sum() / denom_body
                    acc_acc[k] += correct
        # Average start vs end accuracy per level
        s_acc = s_acc.detach().cpu().numpy()
        e_acc = e_acc.detach().cpu().numpy()

        # Duration MSE (log-space)
        dur_mse = (((log_dur - gt_log_dur) ** 2) * body_f).sum() / denom_body

        return ce_total, dur_mse, s_acc, e_acc

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        agg = {"ce": 0.0, "dur": 0.0, "total": 0.0}
        agg_s_acc = np.zeros(args.num_quantizers)
        agg_e_acc = np.zeros(args.num_quantizers)
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            ce, dur_mse, s_acc, e_acc = compute_losses(batch)
            loss = ce + args.dur_weight * dur_mse
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            agg["ce"]    += ce.item()
            agg["dur"]   += dur_mse.item()
            agg["total"] += loss.item()
            agg_s_acc += s_acc; agg_e_acc += e_acc
            n_batches += 1
            step += 1
            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "ce": ce.item(), "dur": dur_mse.item(), "total": loss.item(),
                    "s_acc": s_acc.tolist(), "e_acc": e_acc.tolist(),
                }) + "\n"); metrics_f.flush()

        model.eval()
        v = {"ce": 0.0, "dur": 0.0, "total": 0.0}; v_s = np.zeros(args.num_quantizers); v_e = np.zeros(args.num_quantizers); n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                ce, dur_mse, s_acc, e_acc = compute_losses(batch)
                total = ce + args.dur_weight * dur_mse
                v["ce"] += ce.item(); v["dur"] += dur_mse.item(); v["total"] += total.item()
                v_s += s_acc; v_e += e_acc; n_v += 1

        train_avg = {k: x / max(1, n_batches) for k, x in agg.items()}
        val_avg   = {k: x / max(1, n_v) for k, x in v.items()}
        train_s_acc = (agg_s_acc / max(1, n_batches)).tolist()
        train_e_acc = (agg_e_acc / max(1, n_batches)).tolist()
        val_s_acc   = (v_s / max(1, n_v)).tolist()
        val_e_acc   = (v_e / max(1, n_v)).tolist()
        print(f"Epoch {epoch:3d} | train CE={train_avg['ce']:.3f} dur={train_avg['dur']:.4f} | "
              f"val CE={val_avg['ce']:.3f} dur={val_avg['dur']:.4f}")
        print(f"  start acc per lvl: train={['%.1f%%' % (x*100) for x in train_s_acc]}  val={['%.1f%%' % (x*100) for x in val_s_acc]}")
        print(f"  end   acc per lvl: train={['%.1f%%' % (x*100) for x in train_e_acc]}  val={['%.1f%%' % (x*100) for x in val_e_acc]}")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train": train_avg, "val": val_avg,
            "train_s_acc": train_s_acc, "train_e_acc": train_e_acc,
            "val_s_acc":   val_s_acc,   "val_e_acc":   val_e_acc,
        }) + "\n"); metrics_f.flush()

        ckpt_payload = {
            "epoch": epoch, "step": step,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_avg["total"],
            "args": vars(args),
            "knob_dim": dataset.knob_dim,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            ckpt_payload["val_loss"] = best_val
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val={best_val:.4f})")

    metrics_f.close()
    print(f"\nv9 predictor training complete. Best val: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir",      default="v9/data/phoneme_tokens")
    p.add_argument("--vad-paths",       nargs="+",
                   default=["data/librispeech_emotion_vad.json", "data/esd_emotion_vad.json"])
    p.add_argument("--metadata-path",   default="data/utterance_metadata_v5.json")
    p.add_argument("--knob-source",     choices=["none", "vad", "emotion"], default="emotion")
    p.add_argument("--checkpoint-dir",  default="v9/checkpoints/predictor")
    p.add_argument("--vocab-size",      type=int, default=73)
    p.add_argument("--codebook-size",   type=int, default=512)
    p.add_argument("--num-quantizers",  type=int, default=4)
    p.add_argument("--max-phonemes",    type=int, default=200)
    p.add_argument("--device",          default="mps")
    p.add_argument("--batch-size",      type=int, default=16)
    p.add_argument("--epochs",          type=int, default=50)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--warmup-steps",    type=int, default=2000)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--d-model",         type=int, default=256)
    p.add_argument("--nhead",           type=int, default=4)
    p.add_argument("--num-layers",      type=int, default=4)
    p.add_argument("--d-ff",            type=int, default=1024)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--knob-dropout",    type=float, default=0.1)
    p.add_argument("--dur-weight",      type=float, default=0.1)
    p.add_argument("--level-weights",   type=str, default="")
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--restore-optim",   action="store_true", default=True)
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
