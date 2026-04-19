"""
Train TTS transformer with multi-codebook RVQ tokens.
Predicts K codebooks independently per frame.
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

from models.transformer_rvq import ArticulatoryTTSModelRVQ
from models.phoneme_vocab import PhonemeVocab
from training.dataset_rvq import TTSDatasetRVQ, collate_tts_rvq


def train(args):
    device = torch.device(args.device)
    print(f"Training RVQ transformer on {device}")

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
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_tts_rvq, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_tts_rvq, num_workers=0,
    )

    model = ArticulatoryTTSModelRVQ(
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

    # Weight earlier codebooks higher (they capture coarser, more important structure)
    level_weights = torch.tensor([1.0, 0.5, 0.25, 0.125][:args.num_quantizers], device=device)
    level_weights = level_weights / level_weights.sum()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 1
    global_step = 0

    # Resume
    resume_path = ckpt_dir / "transformer_best.pt"
    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt["val_loss"]
        start_epoch = ckpt["epoch"] + 1

        # Always rebuild scheduler from ORIGINAL training's hyperparameters and
        # fast-forward to the current epoch. This is robust to:
        #   - old checkpoints with no scheduler state (the original 31K run)
        #   - stale scheduler state from a prior constant-LR resume (step 0 despite epoch>0)
        #   - any checkpoint where the saved scheduler's total_steps no longer matches
        #     the current dataset size
        ckpt_args = ckpt.get("args", {})
        orig_epochs = ckpt_args.get("epochs", args.epochs)
        orig_warmup = ckpt_args.get("warmup_steps", args.warmup_steps)
        orig_loader_len = ckpt.get("train_loader_len", len(train_loader))
        orig_total_steps = orig_epochs * orig_loader_len
        orig_lr = ckpt_args.get("lr", args.lr)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Reset base LR so the reconstructed scheduler drives the trajectory
        for pg in optimizer.param_groups:
            pg["lr"] = orig_lr
            pg["initial_lr"] = orig_lr

        def orig_lr_lambda(step):
            if step < orig_warmup:
                return step / max(1, orig_warmup)
            progress = (step - orig_warmup) / max(1, orig_total_steps - orig_warmup)
            return 0.5 * (1 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, orig_lr_lambda)
        global_step = ckpt["epoch"] * orig_loader_len
        for _ in range(global_step):
            scheduler.step()

        use_constant_lr = False
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Reconstructed scheduler: epoch={ckpt['epoch']}, "
            f"global_step={global_step}/{orig_total_steps}, "
            f"warmup={orig_warmup}, base_lr={orig_lr:.6f}, current_lr={current_lr:.6f}"
        )
        print(f"Resumed from epoch {ckpt['epoch']}, val={best_val_loss:.4f}")
    else:
        use_constant_lr = False

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
            )

            logits = result["logits"]  # (B, T, K, codebook_size)
            B, T, K, C = logits.shape
            assert K == args.num_quantizers, f"K={K} != num_quantizers={args.num_quantizers}"

            # CE loss per level
            total_ce = 0
            mask_t = frame_mask[:, :T].float()

            for k in range(K):
                logits_k = logits[:, :, k, :]  # (B, T, C)
                target_k = vq_tokens[:, :T, k]  # (B, T)
                ce = ce_loss_fn(logits_k.reshape(-1, C), target_k.reshape(-1)).reshape(B, T)
                ce_masked = (ce * mask_t).sum() / mask_t.sum().clamp(min=1)
                total_ce = total_ce + level_weights[k] * ce_masked

                # Accuracy
                pred = logits_k.argmax(dim=-1)
                correct = ((pred == target_k).float() * mask_t).sum()
                train_acc_per_level[k] += (correct / mask_t.sum().clamp(min=1)).item()

            # Duration loss
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
            if not use_constant_lr:
                scheduler.step()
            global_step += 1

            train_ce_total += total_ce.item()
            train_dur += dur_loss.item()

        n = len(train_loader)

        # Validation
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
                )

                logits = result["logits"]
                B, T, K, C = logits.shape
                assert K == args.num_quantizers
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
            }, ckpt_dir / "transformer_best.pt")
            print(f"  -> Saved best model (val_CE={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val CE: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features_all")
    parser.add_argument("--phonemes-path", type=str, default="data/processed_all/phonemes_mfa.json")
    parser.add_argument("--alignments-path", type=str, default="data/processed_all/alignments_mfa.json")
    parser.add_argument("--vq-tokens-dir", type=str, default="data/rvq_tokens_all")
    parser.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_rvq")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
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
    parser.add_argument("--resume-lr", type=float, default=0.0,
                        help="Constant LR for fallback fine-tuning when resuming from an OLD checkpoint "
                             "that doesn't have optimizer/scheduler state saved. Ignored if checkpoint has "
                             "scheduler state (those are restored exactly). 0 = use default 0.000302.")
    args = parser.parse_args()

    train(args)
