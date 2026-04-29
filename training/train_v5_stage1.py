"""
v5 stage 1 training: joint codebook + frame decoder.

Per-utterance flow at training:
  features (T, 14) + GT durations (N)
       ↓
  per-phoneme style encoder → z (B, N, 256)
       ↓
  style codebook (VQ) → quantized (B, N, 256), codes (B, N), commit_loss
       ↓
  transformer (phonemes + speaker + style_emb=quantized) → logits, pred_durations
       ↓
  losses: CE on RVQ tokens + dur MSE + smoothness (acceleration deadzone) + VQ commitment

From scratch — no init_from. The architecture has changed (per-phoneme style codes
replace pooled style_vec), so partial-loading v4 weights would mix incompatible
representations. Watchdog can resume from checkpoints_v5_stage1/ once one exists.

Default smooth_weight=0.10 (target ~14% gradient ratio in v5; live monitored).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.per_phoneme_style_encoder import PerPhonemeStyleEncoder
from models.style_codebook import StyleCodebook, PAD_CODE
from models.phoneme_vocab import PhonemeVocab
from training.dataset_rvq import TTSDatasetRVQ, collate_tts_rvq, BucketBatchSampler
from training.metrics_logger import (
    MetricsLogger, grad_norm_total, codebook_usage,
)


# Channel-aware smoothness weights (from σ² of inference smoothing physics).
SMOOTH_CHANNEL_WEIGHTS = [0.12, 0.12,
                          0.33, 0.33, 0.33, 0.33, 0.33,
                          0.33, 0.33, 0.33, 0.33, 0.33,
                          3.0, 1.33]

# Per-channel acceleration "noise floor" thresholds — 1× median GT acceleration
# in de-normalized feature space. With the GT-conditional deadzone (see
# compute_smooth_loss), the actual per-frame threshold is max(|gt_accel|, floor).
# These floors only apply to frames where GT itself is quieter than 1× median —
# preventing the loss from forcing the model to match unobservably-tiny accelerations.
# (Old v5 thresholds were 5× median to give static-deadzone room for prosody;
# now prosody is protected dynamically by tracking gt_accel, so the static piece
# can drop to 1× median.)
ACCEL_THRESHOLDS = [0.14362, 0.11598, 0.13951, 0.11471, 0.13586, 0.13048,
                    0.11978, 0.11836, 0.12746, 0.12548, 0.12324, 0.13593,
                    0.01060, 0.10540]


def soft_decode_features(logits, rvq_model, temperature=0.5):
    """Differentiable soft-decode: logits → soft token probs → weighted codebook
    sum → (B, T, 14). Sums residuals across RVQ levels."""
    B, T, K, C = logits.shape
    soft_features = None
    for k in range(K):
        probs_k = F.softmax(logits[:, :, k, :] / temperature, dim=-1)
        codebook_weight = rvq_model.vq.layers[k]._codebook.embed
        if codebook_weight.dim() == 3:
            codebook_weight = codebook_weight.squeeze(0)
        decoded_k = probs_k @ codebook_weight
        soft_features = decoded_k if soft_features is None else soft_features + decoded_k
    return rvq_model.decoder(soft_features)


def compute_smooth_loss(soft_features, gt_features, frame_mask,
                        channel_weights, accel_thresholds):
    """Acceleration-based smoothness loss with GT-CONDITIONAL deadzone.

    Per-frame, per-channel threshold = max(|gt_accel|, global_floor). This
    protects natural prosody dynamically (where GT accelerations are high,
    the model is allowed to be just as fast) while still suppressing model
    jitter beyond what GT shows. Replaces the static 5×-median deadzone
    which had to be lenient to avoid over-suppressing prosody, and which
    therefore left a hard ~5× GT pitch-jitter floor at inference.

    soft_features: (B, T, 14) — predicted features (de-normalized, soft-decoded)
    gt_features:   (B, T, 14) — GT features in same scale (raw, un-normalized)
    """
    pred_accel = soft_features[:, 2:] - 2 * soft_features[:, 1:-1] + soft_features[:, :-2]
    gt_accel   = gt_features[:, 2:]   - 2 * gt_features[:, 1:-1]   + gt_features[:, :-2]
    # Per-frame, per-channel: protect the larger of |gt_accel| or the global noise floor
    local_threshold = torch.maximum(gt_accel.abs(), accel_thresholds.view(1, 1, -1))
    excess = F.relu(pred_accel.abs() - local_threshold)
    mask = frame_mask[:, 2:].float().unsqueeze(-1)
    weighted = (excess ** 2) * channel_weights.view(1, 1, -1) * mask
    return weighted.sum() / mask.sum().clamp(min=1)


def train(args):
    device = torch.device(args.device)
    print(f"v5 stage 1 — training on {device}")

    vocab = PhonemeVocab(args.vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens")

    # Frozen RVQ for soft-decode smoothness loss
    rvq_ckpt = torch.load(args.rvq_checkpoint, map_location=device, weights_only=True)
    ra = rvq_ckpt["args"]
    rvq_model = ArticulatoryRVQTokenizer(
        codebook_size=ra["codebook_size"], num_quantizers=ra["num_quantizers"],
        latent_dim=ra["latent_dim"], hidden_dim=ra["hidden_dim"],
    ).to(device)
    rvq_model.load_state_dict(rvq_ckpt["model_state_dict"])
    rvq_model.eval()
    for p in rvq_model.parameters():
        p.requires_grad = False
    print(f"Loaded frozen RVQ for smoothness soft-decode")

    smooth_channel_weights = torch.tensor(SMOOTH_CHANNEL_WEIGHTS, device=device)
    accel_thresholds = torch.tensor(ACCEL_THRESHOLDS, device=device)
    norm_stats = np.load(Path(args.features_dir) / "norm_stats.npz")
    feat_mean = torch.tensor(norm_stats["mean"], device=device, dtype=torch.float32)
    feat_std  = torch.tensor(norm_stats["std"],  device=device, dtype=torch.float32)

    dataset = TTSDatasetRVQ(
        features_dir=args.features_dir, phonemes_path=args.phonemes_path,
        alignments_path=args.alignments_path, vq_tokens_dir=args.vq_tokens_dir,
        vocab_path=args.vocab_path, preload=args.preload, max_frames=args.max_frames,
        metadata_path=args.metadata_path,
    )
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    train_sampler = BucketBatchSampler(train_set, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              collate_fn=collate_tts_rvq, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_tts_rvq, num_workers=0)

    # ─── modules ──────────────────────────────────────────
    model = ArticulatoryTTSModelRVQHier(
        vocab_size=len(vocab),
        codebook_size=args.codebook_size, num_quantizers=args.num_quantizers,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        d_ff=args.d_ff, dropout=args.dropout, speaker_emb_dim=64,
        tied_output=args.tied_output,
        codebook_latent_dim=ra["latent_dim"],
    ).to(device)
    if args.tied_output:
        # Populate the model's frozen-codebook buffer from the loaded RVQ.
        # Required: tied output computes logits as similarity to these codebook entries.
        model.init_tied_codebooks(rvq_model)

    style_encoder = PerPhonemeStyleEncoder(
        input_dim=14, style_dim=args.d_model, hidden=128, n_conv_layers=3,
    ).to(device)

    style_codebook = StyleCodebook(
        latent_dim=args.d_model,
        codebook_size=args.style_codebook_size,
        commitment_weight=args.vq_commit_weight,
    ).to(device)

    n_tf = sum(p.numel() for p in model.parameters())
    n_se = sum(p.numel() for p in style_encoder.parameters())
    n_cb = args.style_codebook_size * args.d_model
    print(f"Params: transformer={n_tf:,}  style_encoder={n_se:,}  "
          f"style_codebook={n_cb:,} (buffer)")

    # ─── optimizer ────────────────────────────────────────
    trainable_params = list(model.parameters()) + list(style_encoder.parameters())
    # NOTE: style_codebook entries are EMA-updated (not in optimizer)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    if args.level_weights:
        weights = [float(w) for w in args.level_weights.split(",")]
        assert len(weights) == args.num_quantizers
    else:
        weights = [1.0] * args.num_quantizers
    level_weights = torch.tensor(weights, device=device)
    level_weights = level_weights / level_weights.sum()
    print(f"Level weights (normalized): {level_weights.tolist()}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics = MetricsLogger(
        log_path=ckpt_dir / "metrics.jsonl",
        grad_ratio_every=args.grad_ratio_every,
        scalar_every=args.scalar_every,
    )

    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0

    # Resume support — keeps the watchdog usable
    resume_path = ckpt_dir / "transformer_best.pt"
    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "style_encoder_state_dict" in ckpt:
            style_encoder.load_state_dict(ckpt["style_encoder_state_dict"])
        if "style_codebook_state_dict" in ckpt:
            style_codebook.load_state_dict(ckpt["style_codebook_state_dict"])
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        if args.restore_optim and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {ckpt.get('epoch',0)}, val={best_val_loss:.4f}")

    # ─── train loop ──────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); style_encoder.train(); style_codebook.train()
        metrics.epoch_start(epoch=epoch)

        agg = dict(ce=0.0, dur=0.0, smooth=0.0, vq=0.0, total=0.0)
        agg_acc = [0.0] * args.num_quantizers

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            phoneme_ids = batch["phoneme_ids"].to(device)
            phoneme_mask = batch["phoneme_mask"].to(device)
            durations = batch["durations"].to(device)
            vq_tokens = batch["vq_tokens"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            speaker_embs = batch["speaker_embs"].to(device)
            style_feats = batch["style_features"].to(device)

            # Per-phoneme style: encode then quantize
            z = style_encoder(style_feats, durations, phoneme_mask)
            quantized, codes, commit_loss = style_codebook(z, phoneme_mask)

            target_len = vq_tokens.size(1)
            result = model(
                phoneme_ids=phoneme_ids, speaker_emb=speaker_embs,
                durations=durations, target_len=target_len, phoneme_mask=phoneme_mask,
                target_tokens=vq_tokens, style_emb=quantized,
            )
            logits = result["logits"]                 # (B, T, K, C)
            B, T, K, C = logits.shape
            mask_t = frame_mask[:, :T].float()

            # CE per RVQ level
            ce_total = 0
            for k in range(K):
                logits_k = logits[:, :, k, :]
                target_k = vq_tokens[:, :T, k]
                ce = ce_loss_fn(logits_k.reshape(-1, C), target_k.reshape(-1)).reshape(B, T)
                ce_masked = (ce * mask_t).sum() / mask_t.sum().clamp(min=1)
                ce_total = ce_total + level_weights[k] * ce_masked
                pred = logits_k.argmax(dim=-1)
                correct = ((pred == target_k).float() * mask_t).sum()
                agg_acc[k] += (correct / mask_t.sum().clamp(min=1)).item()

            # Duration loss
            pred_dur = result["pred_durations"]
            dur_mask = (durations > 0).float()
            if dur_mask.sum() > 0:
                dur_loss = ((torch.log(pred_dur + 1) - torch.log(durations + 1)) ** 2 * dur_mask).sum() / dur_mask.sum()
            else:
                dur_loss = torch.tensor(0.0, device=device)

            # Smoothness loss with GT-conditional deadzone
            if args.smooth_weight > 0:
                soft_norm = soft_decode_features(logits, rvq_model, temperature=0.5)
                soft_feats = soft_norm * feat_std.unsqueeze(0).unsqueeze(0) + feat_mean.unsqueeze(0).unsqueeze(0)
                # GT features in the same de-normalized scale as soft_feats.
                # style_feats is already raw (un-normalized) per dataset_rvq.py;
                # truncate/pad to the prediction time dimension T.
                gt_feats = style_feats[:, :T, :]
                if gt_feats.shape[1] < T:
                    pad = T - gt_feats.shape[1]
                    gt_feats = F.pad(gt_feats, (0, 0, 0, pad))
                smooth_loss = compute_smooth_loss(
                    soft_feats, gt_feats, frame_mask[:, :T],
                    smooth_channel_weights, accel_thresholds,
                )
            else:
                smooth_loss = torch.tensor(0.0, device=device)

            # Total loss (and the four components for logging)
            ce_term     = ce_total
            dur_term    = args.dur_weight * dur_loss
            smooth_term = args.smooth_weight * smooth_loss
            vq_term     = commit_loss   # already scaled by VectorQuantize's commitment_weight
            total_loss  = ce_term + dur_term + smooth_term + vq_term

            optimizer.zero_grad()

            # Optionally measure per-loss gradient norms (expensive — only every N steps)
            grad_ratios = None; grad_norms = None
            if args.grad_ratio_every > 0 and (global_step % args.grad_ratio_every == 0) and global_step > 0:
                norms = {}
                for name, l in [("ce", ce_term), ("dur", dur_term),
                                ("smooth", smooth_term), ("vq", vq_term)]:
                    optimizer.zero_grad()
                    if l.requires_grad:
                        l.backward(retain_graph=True)
                        norms[name] = grad_norm_total(trainable_params)
                    else:
                        norms[name] = 0.0
                # Total norm
                optimizer.zero_grad()
                total_loss.backward(retain_graph=False)
                norms["total"] = grad_norm_total(trainable_params)
                # Ratios
                tot = max(1e-12, norms["total"])
                grad_ratios = {f"{k}/total": v / tot for k, v in norms.items() if k != "total"}
                grad_norms  = norms
            else:
                total_loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step(); scheduler.step(); global_step += 1

            metrics.step(
                step=global_step, lr=optimizer.param_groups[0]["lr"],
                losses={"ce": ce_term.item(), "dur": dur_loss.item(),
                        "smooth": smooth_loss.item(), "vq": commit_loss.item(),
                        "total": total_loss.item()},
                grad_norms=grad_norms, grad_ratios=grad_ratios,
            )

            agg["ce"]     += ce_term.item()
            agg["dur"]    += dur_loss.item()
            agg["smooth"] += smooth_loss.item()
            agg["vq"]     += commit_loss.item()
            agg["total"]  += total_loss.item()

        n = len(train_loader)
        train_metrics = {k: v / n for k, v in agg.items()}
        train_metrics.update({f"acc_L{k}": agg_acc[k] / n for k in range(args.num_quantizers)})

        # ─── validation ──────────────────────────────────────
        model.eval(); style_encoder.eval(); style_codebook.eval()
        val_ce = 0.0
        val_ce_per_level = [0.0] * args.num_quantizers
        val_acc_per_level = [0.0] * args.num_quantizers
        val_codes_seen = []

        with torch.no_grad():
            for batch in val_loader:
                phoneme_ids = batch["phoneme_ids"].to(device)
                phoneme_mask = batch["phoneme_mask"].to(device)
                durations = batch["durations"].to(device)
                vq_tokens = batch["vq_tokens"].to(device)
                frame_mask = batch["frame_mask"].to(device)
                speaker_embs = batch["speaker_embs"].to(device)
                style_feats = batch["style_features"].to(device)

                z = style_encoder(style_feats, durations, phoneme_mask)
                quantized, codes, _ = style_codebook(z, phoneme_mask)
                # FLATTEN per-batch — different batches have different N due to
                # bucket sampler; can't torch.cat along dim 0 with varying dim 1.
                val_codes_seen.append(codes.detach().cpu().flatten())

                target_len = vq_tokens.size(1)
                result = model(
                    phoneme_ids=phoneme_ids, speaker_emb=speaker_embs,
                    durations=durations, target_len=target_len, phoneme_mask=phoneme_mask,
                    target_tokens=vq_tokens, style_emb=quantized,
                )
                logits = result["logits"]
                B, T, K, C = logits.shape
                mask_t = frame_mask[:, :T].float()
                for k in range(K):
                    logits_k = logits[:, :, k, :]
                    target_k = vq_tokens[:, :T, k]
                    ce = ce_loss_fn(logits_k.reshape(-1, C), target_k.reshape(-1)).reshape(B, T)
                    ce_masked = (ce * mask_t).sum() / mask_t.sum().clamp(min=1)
                    val_ce_per_level[k] += ce_masked.item()
                    val_ce += (level_weights[k] * ce_masked).item()
                    pred = logits_k.argmax(dim=-1)
                    correct = ((pred == target_k).float() * mask_t).sum()
                    val_acc_per_level[k] += (correct / mask_t.sum().clamp(min=1)).item()

        vn = len(val_loader)
        val_metrics = {
            "ce_total": val_ce / vn,
            **{f"ce_L{k}": val_ce_per_level[k] / vn for k in range(args.num_quantizers)},
            **{f"acc_L{k}": val_acc_per_level[k] / vn for k in range(args.num_quantizers)},
        }

        # Style codebook usage audit on val
        all_codes = torch.cat(val_codes_seen, dim=0) if val_codes_seen else torch.empty(0, dtype=torch.long)
        cb_stats = codebook_usage(all_codes, args.style_codebook_size, pad_value=PAD_CODE)
        metrics.audit(epoch=epoch, name="style_codebook_usage", payload=cb_stats)

        ce_strs = " ".join(f"CE{k}={val_ce_per_level[k]/vn:.3f}" for k in range(args.num_quantizers))
        acc_strs = " ".join(f"L{k}={val_acc_per_level[k]/vn:.1%}" for k in range(args.num_quantizers))
        print(
            f"Epoch {epoch:3d} | "
            f"Train: CE={train_metrics['ce']:.4f} dur={train_metrics['dur']:.4f} "
            f"smooth={train_metrics['smooth']:.4f} vq={train_metrics['vq']:.4f} | "
            f"Val: CE={val_ce/vn:.4f} ({ce_strs}) | "
            f"Acc: {acc_strs} | "
            f"Codebook: active={cb_stats['active']}/{args.style_codebook_size} "
            f"perplexity={cb_stats['perplexity']:.1f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_ce / vn < best_val_loss:
            best_val_loss = val_ce / vn
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "style_encoder_state_dict": style_encoder.state_dict(),
                "style_codebook_state_dict": style_codebook.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "val_loss": best_val_loss,
                "args": vars(args),
                "vocab_size": len(vocab),
                "train_loader_len": len(train_loader),
                "model_type": "v5_per_phoneme_style",
            }, ckpt_dir / "transformer_best.pt")
            print(f"  -> Saved best (val_CE={best_val_loss:.4f})")

        metrics.epoch_end(epoch=epoch, train_metrics=train_metrics,
                          val_metrics=val_metrics, best_val_loss=best_val_loss)

    metrics.close()
    print(f"\nTraining complete. Best val CE: {best_val_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=str, default="data/features_merged_logpitch_v2")
    p.add_argument("--phonemes-path", type=str, default="data/processed_merged_v3/phonemes_mfa.json")
    p.add_argument("--alignments-path", type=str, default="data/processed_merged_v3/alignments_mfa.json")
    p.add_argument("--vq-tokens-dir", type=str, default="data/rvq_tokens_logpitch_v3")
    p.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    p.add_argument("--metadata-path", type=str, default="data/utterance_metadata_v5.json")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_v5_stage1")
    p.add_argument("--rvq-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch_v2/rvq_best.pt")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--dur-weight", type=float, default=0.1)
    p.add_argument("--smooth-weight", type=float, default=0.10)        # ← v5 default
    p.add_argument("--vq-commit-weight", type=float, default=0.25)
    p.add_argument("--codebook-size", type=int, default=512)
    p.add_argument("--num-quantizers", type=int, default=4)
    p.add_argument("--style-codebook-size", type=int, default=512)
    p.add_argument("--tied-output", action="store_true", default=True,
                   help="Use codebook-tied output projection (logit = h_proj · codebook[i]). "
                        "Probability becomes structurally feature-aware. Default True for v6.")
    p.add_argument("--no-tied-output", dest="tied_output", action="store_false",
                   help="Use independent Linear output projection (v3/v4/v5 style).")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--level-weights", type=str, default="")
    p.add_argument("--max-frames", type=int, default=800)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--restore-optim", action="store_true", default=True)
    p.add_argument("--scalar-every", type=int, default=50)
    p.add_argument("--grad-ratio-every", type=int, default=200)
    args = p.parse_args()
    train(args)
