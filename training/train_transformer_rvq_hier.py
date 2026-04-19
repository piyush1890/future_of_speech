"""
Train hierarchical-RVQ TTS transformer.

Key difference from flat training:
  - Pass ground-truth vq_tokens as `target_tokens` → teacher forcing through hier heads
  - Default uniform per-level loss weights (hier makes L1+ targets sharp, so down-weighting
    them no longer helps)
  - Partial-load from flat checkpoint with strict=False: cb_embeds stay at zero init.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.phoneme_vocab import PhonemeVocab
from training.dataset_rvq import TTSDatasetRVQ, collate_tts_rvq


def train(args):
    device = torch.device(args.device)
    print(f"Training HIERARCHICAL RVQ transformer on {device}")

    vocab = PhonemeVocab(args.vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens")

    dataset = TTSDatasetRVQ(
        features_dir=args.features_dir,
        phonemes_path=args.phonemes_path,
        alignments_path=args.alignments_path,
        vq_tokens_dir=args.vq_tokens_dir,
        vocab_path=args.vocab_path,
    )

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_tts_rvq, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_tts_rvq, num_workers=0,
    )

    model = ArticulatoryTTSModelRVQHier(
        vocab_size=len(vocab),
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        speaker_emb_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Hierarchical conditioning sharpens targets at every level, so uniform weights
    # are appropriate. Override via --level-weights if desired.
    if args.level_weights:
        weights = [float(w) for w in args.level_weights.split(",")]
        assert len(weights) == args.num_quantizers, "level_weights must match num_quantizers"
    else:
        weights = [1.0] * args.num_quantizers
    level_weights = torch.tensor(weights, device=device)
    level_weights = level_weights / level_weights.sum()
    print(f"Level weights (normalized): {level_weights.tolist()}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0

    resume_path = ckpt_dir / "transformer_best.pt"
    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"  Missing keys (expected for partial load): {len(missing)} -> {missing[:3]}...")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)} -> {unexpected[:3]}...")
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1

        if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt and args.restore_optim:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            global_step = ckpt.get("global_step", 0)
            print(f"  Restored optimizer + scheduler at global_step={global_step}")
        print(f"Resumed from epoch {ckpt.get('epoch', 0)}, val={best_val_loss:.4f}")
    elif args.init_from and Path(args.init_from).exists():
        # Partial load from flat checkpoint (weights only, fresh optimizer/scheduler)
        print(f"Init from flat checkpoint: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Missing keys (cb_embeds, expected): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        print(f"  (cb_embeds remain at zero init — model numerically equivalent to flat at step 0)")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_ce_total = 0
        train_dur = 0
        train_acc_per_level = [0] * args.num_quantizers

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            phoneme_ids = batch["phoneme_ids"].to(device)
            phoneme_mask = batch["phoneme_mask"].to(device)
            durations = batch["durations"].to(device)
            vq_tokens = batch["vq_tokens"].to(device)  # (B, T, K)
            frame_mask = batch["frame_mask"].to(device)
            speaker_embs = batch["speaker_embs"].to(device)

            target_len = vq_tokens.size(1)

            result = model(
                phoneme_ids=phoneme_ids,
                speaker_emb=speaker_embs,
                durations=durations,
                target_len=target_len,
                phoneme_mask=phoneme_mask,
                target_tokens=vq_tokens,   # teacher forcing through hierarchy
            )

            logits = result["logits"]  # (B, T, K, C)
            B, T, K, C = logits.shape
            assert K == args.num_quantizers

            total_ce = 0
            mask_t = frame_mask[:, :T].float()

            for k in range(K):
                logits_k = logits[:, :, k, :]
                target_k = vq_tokens[:, :T, k]
                ce = ce_loss_fn(logits_k.reshape(-1, C), target_k.reshape(-1)).reshape(B, T)
                ce_masked = (ce * mask_t).sum() / mask_t.sum().clamp(min=1)
                total_ce = total_ce + level_weights[k] * ce_masked

                pred = logits_k.argmax(dim=-1)
                correct = ((pred == target_k).float() * mask_t).sum()
                train_acc_per_level[k] += (correct / mask_t.sum().clamp(min=1)).item()

            pred_dur = result["pred_durations"]
            dur_mask = (durations > 0).float()
            if dur_mask.sum() > 0:
                dur_loss = ((torch.log(pred_dur + 1) - torch.log(durations + 1)) ** 2 * dur_mask).sum() / dur_mask.sum()
            else:
                dur_loss = torch.tensor(0.0, device=device)

            total_loss = total_ce + args.dur_weight * dur_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_ce_total += total_ce.item()
            train_dur += dur_loss.item()

        n = len(train_loader)

        model.eval()
        val_ce = 0
        val_acc_per_level = [0] * args.num_quantizers

        with torch.no_grad():
            for batch in val_loader:
                phoneme_ids = batch["phoneme_ids"].to(device)
                phoneme_mask = batch["phoneme_mask"].to(device)
                durations = batch["durations"].to(device)
                vq_tokens = batch["vq_tokens"].to(device)
                frame_mask = batch["frame_mask"].to(device)
                speaker_embs = batch["speaker_embs"].to(device)

                target_len = vq_tokens.size(1)

                result = model(
                    phoneme_ids=phoneme_ids,
                    speaker_emb=speaker_embs,
                    durations=durations,
                    target_len=target_len,
                    phoneme_mask=phoneme_mask,
                    target_tokens=vq_tokens,   # teacher-forced val
                )

                logits = result["logits"]
                B, T, K, C = logits.shape
                mask_t = frame_mask[:, :T].float()

                for k in range(K):
                    logits_k = logits[:, :, k, :]
                    target_k = vq_tokens[:, :T, k]
                    ce = ce_loss_fn(logits_k.reshape(-1, C), target_k.reshape(-1)).reshape(B, T)
                    ce_masked = (ce * mask_t).sum() / mask_t.sum().clamp(min=1)
                    val_ce += (level_weights[k] * ce_masked).item()

                    pred = logits_k.argmax(dim=-1)
                    correct = ((pred == target_k).float() * mask_t).sum()
                    val_acc_per_level[k] += (correct / mask_t.sum().clamp(min=1)).item()

        vn = len(val_loader)

        acc_strs = " ".join(f"L{k}={val_acc_per_level[k]/vn:.1%}" for k in range(K))
        print(
            f"Epoch {epoch:3d} | "
            f"Train: CE={train_ce_total/n:.4f} dur={train_dur/n:.4f} | "
            f"Val: CE={val_ce/vn:.4f} | "
            f"Val Acc: {acc_strs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_ce / vn < best_val_loss:
            best_val_loss = val_ce / vn
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "val_loss": best_val_loss,
                "args": vars(args),
                "vocab_size": len(vocab),
                "train_loader_len": len(train_loader),
                "model_type": "hierarchical",
            }, ckpt_dir / "transformer_best.pt")
            print(f"  -> Saved best model (val_CE={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val CE: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features_merged")
    parser.add_argument("--phonemes-path", type=str, default="data/processed_merged/phonemes_mfa.json")
    parser.add_argument("--alignments-path", type=str, default="data/processed_merged/alignments_mfa.json")
    parser.add_argument("--vq-tokens-dir", type=str, default="data/rvq_tokens_merged")
    parser.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_rvq_hier")
    parser.add_argument("--init-from", type=str, default="",
                        help="Path to flat checkpoint to init weights from (partial load, cb_embeds zero-init)")
    parser.add_argument("--restore-optim", action="store_true",
                        help="When resuming, restore optimizer+scheduler state (only for hier checkpoints)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--dur-weight", type=float, default=1.0)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--num-quantizers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--level-weights", type=str, default="",
                        help="Comma-sep weights per level, e.g. '1,1,1,1' (default uniform)")
    args = parser.parse_args()

    train(args)
