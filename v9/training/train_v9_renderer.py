"""
v9 Stage 1 (renderer) trainer — joint training of:
  - V9PerPhonemeStyleEncoder (extracts per-phoneme z from frames)
  - V9StyleCodebook (quantizes z → 1 of 512 per phoneme)
  - V9Renderer (predicts RVQ tokens given phoneme + spk + style code + knobs)

The style encoder receives gradient through the VQ commitment loss; the
codebook updates via EMA. Renderer is trained with CE on RVQ tokens
(level-weighted, 4 start + 4 end stacks) plus duration MSE in log-space.
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

from v9.models.style_encoder import V9PerPhonemeStyleEncoder, V9StyleCodebook
from v9.models.v9_renderer import V9Renderer
from v9.training.dataset_v9_renderer import (
    V9RendererDataset, collate_v9_renderer, BucketBatchSampler,
)


def train(args):
    device = torch.device(args.device)
    print(f"v9 stage 1 (renderer + style encoder + codebook) — training on {device}")

    dataset = V9RendererDataset(
        tokens_dir=args.tokens_dir,
        features_dir=args.features_dir,
        norm_stats_path=args.norm_stats,
        knob_source=args.knob_source,
        metadata_path=args.metadata_path,
        max_phonemes=args.max_phonemes,
        f_pad_per_phoneme=args.f_pad_phoneme,
        preload=args.preload,
    )
    print(f"  knob_dim = {dataset.knob_dim}")

    val_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    train_sampler = BucketBatchSampler(train_set, batch_size=args.batch_size, shuffle=True)
    val_sampler   = BucketBatchSampler(val_set,   batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              collate_fn=collate_v9_renderer, num_workers=0)
    val_loader   = DataLoader(val_set, batch_sampler=val_sampler,
                              collate_fn=collate_v9_renderer, num_workers=0)

    style_encoder = V9PerPhonemeStyleEncoder(
        vocab_size=args.vocab_size, input_dim=14,
        hidden_dim=args.style_hidden, latent_dim=args.style_latent,
    ).to(device)
    style_codebook = V9StyleCodebook(
        codebook_size=args.style_codebook_size, latent_dim=args.style_latent,
        decay=args.ema_decay, commitment_weight=args.commit_weight,
    ).to(device)
    renderer = V9Renderer(
        vocab_size=args.vocab_size, codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        style_codebook_size=args.style_codebook_size,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        speaker_emb_dim=64, knob_dim=dataset.knob_dim,
        knob_dropout=args.knob_dropout, max_phonemes=args.max_phonemes,
    ).to(device)

    n_params = (sum(p.numel() for p in style_encoder.parameters()) +
                sum(p.numel() for p in style_codebook.parameters()) +
                sum(p.numel() for p in renderer.parameters()))
    print(f"Total stage-1 params: {n_params:,}")

    trainable = list(style_encoder.parameters()) + \
                list(style_codebook.parameters()) + \
                list(renderer.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.level_weights:
        weights = [float(w) for w in args.level_weights.split(",")]
    else:
        weights = [1.0] * args.num_quantizers
    level_w = torch.tensor(weights, device=device)
    level_w = level_w / level_w.sum()

    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(ckpt_dir / "metrics.jsonl", "a")

    best_val = float("inf")
    start_epoch = 1
    step = 0
    if args.resume:
        path = ckpt_dir / "last.pt"
        if not path.exists() and (ckpt_dir / "best.pt").exists():
            path = ckpt_dir / "best.pt"
        if path.exists():
            print(f"Resuming from {path}")
            c = torch.load(path, map_location=device, weights_only=False)
            style_encoder.load_state_dict(c["style_encoder"])
            style_codebook.load_state_dict(c["style_codebook"])
            renderer.load_state_dict(c["renderer"])
            if "optim" in c and args.restore_optim:
                optimizer.load_state_dict(c["optim"])
            if "scheduler" in c:
                scheduler.load_state_dict(c["scheduler"])
            step = int(c.get("step", 0))
            start_epoch = int(c.get("epoch", 0)) + 1
            best_val = float(c.get("val_loss", float("inf")))
            print(f"  resumed at epoch {start_epoch - 1}, step {step}, best_val {best_val:.4f}")

    def compute_losses(batch):
        phoneme_ids   = batch["phoneme_ids"].to(device)
        phoneme_mask  = batch["phoneme_mask"].to(device)
        body_mask     = batch["body_mask"].to(device)
        gt_start      = batch["start_idx"].to(device)
        gt_end        = batch["end_idx"].to(device)
        gt_log_dur    = batch["log_durations"].to(device)
        spk           = batch["spk_emb"].to(device)
        knobs         = batch.get("knobs")
        if knobs is not None: knobs = knobs.to(device)

        blocks        = batch["phoneme_blocks"].to(device)               # (B*body_max, F_PAD, 14)
        block_lens    = batch["block_lens"].to(device)
        block_ph_ids  = batch["block_ph_ids"].to(device)
        block_valid   = batch["block_valid"].to(device)
        block_to_pos  = batch["block_to_pos"].to(device)
        Bsz, N_max    = phoneme_ids.shape

        # ── Style encoder: extract z per body phoneme ──
        # Build frame-level mask from block_lens
        F_PAD = blocks.shape[1]
        idx_t = torch.arange(F_PAD, device=device).unsqueeze(0)
        frame_mask = idx_t < block_lens.unsqueeze(-1)                    # (B*body_max, F_PAD)
        # Set padded blocks (block_valid==False) to a 1-frame "all-zero" mask so attention pool is defined
        # but they won't contribute to loss. Use a dummy length=1 for invalid blocks.
        z = style_encoder(blocks, block_ph_ids, frame_mask)              # (B*body_max, latent)
        # Quantize ONLY valid blocks; mask invalid out from commit loss
        valid_z = z[block_valid]
        valid_q, valid_codes, commit_loss = style_codebook(valid_z)
        # Reassemble: full z_q tensor (B*body_max, latent) — invalid slots get zeros,
        # then placed back into per-utterance (B, N_max) layout with args.style_codebook_size elsewhere.
        z_q_full = torch.zeros_like(z)
        z_q_full[block_valid] = valid_q
        codes_full = torch.full((z.shape[0],), args.style_codebook_size, dtype=torch.long, device=device)
        codes_full[block_valid] = valid_codes

        # Place codes into per-utterance layout (B, N_max) initialized with args.style_codebook_size
        codes_seq = torch.full((Bsz, N_max), args.style_codebook_size, dtype=torch.long, device=device)
        utt_idx = block_to_pos[:, 0]
        pos_idx = block_to_pos[:, 1]
        codes_seq[utt_idx[block_valid], pos_idx[block_valid]] = codes_full[block_valid]

        # ── Renderer forward ──
        out = renderer(phoneme_ids, spk, knobs, phoneme_mask, codes_seq,
                       gt_start, gt_end)
        s_logits = out["start_logits"]; e_logits = out["end_logits"]
        log_dur = out["log_dur"]
        B_, N_, K, C = s_logits.shape

        body_f = body_mask.float()
        denom = body_f.sum().clamp(min=1)
        start_ce = torch.tensor(0.0, device=device)
        end_ce   = torch.tensor(0.0, device=device)
        s_acc = torch.zeros(K, device=device); e_acc = torch.zeros(K, device=device)
        for k in range(K):
            s_ce_per = F.cross_entropy(
                s_logits[..., k, :].reshape(-1, C), gt_start[..., k].reshape(-1), reduction="none"
            ).reshape(B_, N_)
            e_ce_per = F.cross_entropy(
                e_logits[..., k, :].reshape(-1, C), gt_end[..., k].reshape(-1), reduction="none"
            ).reshape(B_, N_)
            start_ce = start_ce + level_w[k] * (s_ce_per * body_f).sum() / denom
            end_ce   = end_ce   + level_w[k] * (e_ce_per * body_f).sum() / denom
            with torch.no_grad():
                s_acc[k] += ((s_logits[..., k, :].argmax(-1) == gt_start[..., k]).float() * body_f).sum() / denom
                e_acc[k] += ((e_logits[..., k, :].argmax(-1) == gt_end[..., k]).float() * body_f).sum() / denom
        ce_total = start_ce + end_ce

        dur_mse = (((log_dur - gt_log_dur) ** 2) * body_f).sum() / denom
        return (ce_total, start_ce, end_ce, dur_mse, commit_loss,
                s_acc.detach().cpu().numpy(), e_acc.detach().cpu().numpy())

    for epoch in range(start_epoch, args.epochs + 1):
        style_encoder.train(); style_codebook.train(); renderer.train()
        agg = {"ce": 0.0, "start_ce": 0.0, "end_ce": 0.0,
               "dur": 0.0, "commit": 0.0, "total": 0.0}
        agg_s = np.zeros(args.num_quantizers); agg_e = np.zeros(args.num_quantizers)
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            ce, start_ce, end_ce, dur, commit, s_acc, e_acc = compute_losses(batch)
            loss = ce + args.dur_weight * dur + args.commit_loss_weight * commit
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            agg["ce"]       += ce.item()
            agg["start_ce"] += start_ce.item()
            agg["end_ce"]   += end_ce.item()
            agg["dur"]      += dur.item()
            agg["commit"]   += commit.item()
            agg["total"]    += loss.item()
            agg_s += s_acc; agg_e += e_acc; n_batches += 1
            step += 1
            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "ce": ce.item(), "start_ce": start_ce.item(), "end_ce": end_ce.item(),
                    "dur": dur.item(), "commit": commit.item(),
                    "total": loss.item(),
                }) + "\n"); metrics_f.flush()

        # Val
        style_encoder.eval(); style_codebook.eval(); renderer.eval()
        v = {"ce": 0.0, "start_ce": 0.0, "end_ce": 0.0,
             "dur": 0.0, "commit": 0.0, "total": 0.0}
        v_s = np.zeros(args.num_quantizers); v_e = np.zeros(args.num_quantizers); n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                ce, start_ce, end_ce, dur, commit, s_acc, e_acc = compute_losses(batch)
                total = ce + args.dur_weight * dur + args.commit_loss_weight * commit
                v["ce"] += ce.item(); v["start_ce"] += start_ce.item(); v["end_ce"] += end_ce.item()
                v["dur"] += dur.item(); v["commit"] += commit.item()
                v["total"] += total.item()
                v_s += s_acc; v_e += e_acc; n_v += 1

        train_avg = {k: x / max(1, n_batches) for k, x in agg.items()}
        val_avg   = {k: x / max(1, n_v) for k, x in v.items()}
        train_s = (agg_s / max(1, n_batches)).tolist()
        train_e = (agg_e / max(1, n_batches)).tolist()
        val_s   = (v_s / max(1, n_v)).tolist()
        val_e   = (v_e / max(1, n_v)).tolist()
        print(f"Epoch {epoch:3d} | train CE start={train_avg['start_ce']:.3f} end={train_avg['end_ce']:.3f} dur={train_avg['dur']:.4f} | "
              f"val CE start={val_avg['start_ce']:.3f} end={val_avg['end_ce']:.3f}")
        print(f"  start lvl-acc train={['%.1f%%' % (x*100) for x in train_s]} val={['%.1f%%' % (x*100) for x in val_s]}")
        print(f"  end   lvl-acc train={['%.1f%%' % (x*100) for x in train_e]} val={['%.1f%%' % (x*100) for x in val_e]}")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train": train_avg, "val": val_avg,
            "train_s_acc": train_s, "train_e_acc": train_e,
            "val_s_acc":   val_s,   "val_e_acc":   val_e,
        }) + "\n"); metrics_f.flush()

        ckpt = {
            "epoch": epoch, "step": step,
            "style_encoder": style_encoder.state_dict(),
            "style_codebook": style_codebook.state_dict(),
            "renderer": renderer.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_avg["total"],
            "args": vars(args),
            "knob_dim": dataset.knob_dim,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            ckpt["val_loss"] = best_val
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val={best_val:.4f})")

    metrics_f.close()
    print(f"\nv9 stage 1 complete. Best val: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens-dir",      default="v9/data/phoneme_tokens")
    p.add_argument("--features-dir",    default="data/features_merged_logpitch_v2")
    p.add_argument("--norm-stats",      default="data/features_merged_logpitch_v2/norm_stats.npz")
    p.add_argument("--metadata-path",   default="data/utterance_metadata_v5.json")
    p.add_argument("--knob-source",     choices=["none", "emotion"], default="emotion")
    p.add_argument("--checkpoint-dir",  default="v9/checkpoints/stage1_renderer")
    p.add_argument("--vocab-size",      type=int, default=73)
    p.add_argument("--codebook-size",   type=int, default=512)
    p.add_argument("--num-quantizers",  type=int, default=4)
    p.add_argument("--style-codebook-size", type=int, default=64,
                   help="Smaller than 73 phonemes — forces codes to be shared across phonemes "
                        "instead of collapsing to phoneme-identity buckets.")
    p.add_argument("--style-latent",    type=int, default=256)
    p.add_argument("--style-hidden",    type=int, default=128)
    p.add_argument("--max-phonemes",    type=int, default=200)
    p.add_argument("--f-pad-phoneme",   type=int, default=32)
    p.add_argument("--device",          default="mps")
    p.add_argument("--batch-size",      type=int, default=16)
    p.add_argument("--epochs",          type=int, default=30)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--warmup-steps",    type=int, default=2000)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--d-model",         type=int, default=256)
    p.add_argument("--nhead",           type=int, default=4)
    p.add_argument("--num-layers",      type=int, default=4)
    p.add_argument("--d-ff",            type=int, default=1024)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--knob-dropout",    type=float, default=0.3,
                   help="Higher dropout → stronger CFG signal at inference (knob more impactful).")
    p.add_argument("--dur-weight",      type=float, default=0.1)
    p.add_argument("--commit-weight",   type=float, default=0.25)
    p.add_argument("--commit-loss-weight", type=float, default=1.0)
    p.add_argument("--ema-decay",       type=float, default=0.99)
    p.add_argument("--level-weights",   type=str, default="")
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--restore-optim",   action="store_true", default=True)
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
