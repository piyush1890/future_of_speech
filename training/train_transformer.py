"""
Train the TTS transformer: phonemes → articulatory token IDs.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import ArticulatoryTTSModel
from models.phoneme_vocab import PhonemeVocab
from training.dataset import TTSDataset, collate_tts


def train_transformer(args):
    device = torch.device(args.device)
    print(f"Training on {device}")

    # Vocabulary
    vocab = PhonemeVocab(args.vocab_path)
    print(f"Vocabulary: {len(vocab)} tokens")

    # Dataset
    dataset = TTSDataset(
        features_dir=args.features_dir,
        phonemes_path=args.phonemes_path,
        alignments_path=args.alignments_path,
        vq_tokens_dir=args.vq_tokens_dir,
        vocab_path=args.vocab_path,
    )

    # Train/val split
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_tts, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_tts, num_workers=0,
    )

    # Model
    model = ArticulatoryTTSModel(
        vocab_size=len(vocab),
        codebook_size=args.codebook_size,
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup + cosine schedule
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")
    dur_loss_fn = nn.MSELoss()

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    # Resume from checkpoint if available
    resume_path = ckpt_dir / "transformer_best.pt"
    if args.resume and resume_path.exists():
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt["val_loss"]
        start_epoch = ckpt["epoch"] + 1
        # Advance scheduler to correct position
        global_step = ckpt["epoch"] * len(train_loader)
        for _ in range(global_step):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}, val_CE={best_val_loss:.4f}, starting epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Training
        model.train()
        train_ce = 0
        train_dur = 0
        train_acc = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
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

            # Cross-entropy loss on articulatory tokens (masked)
            logits = result["logits"]  # (B, T, codebook_size)
            B, T, C = logits.shape
            ce = ce_loss_fn(logits.reshape(-1, C), vq_tokens[:, :T].reshape(-1))
            ce = ce.reshape(B, T)
            mask_t = frame_mask[:, :T].float()
            ce_masked = (ce * mask_t).sum() / mask_t.sum().clamp(min=1)

            # Duration loss (on phonemes, log space)
            pred_dur = result["pred_durations"]
            gt_dur = durations
            # Only compute on non-BOS/EOS phonemes (dur > 0)
            dur_mask = (gt_dur > 0).float()
            if dur_mask.sum() > 0:
                dur_loss = ((torch.log(pred_dur + 1) - torch.log(gt_dur + 1)) ** 2 * dur_mask).sum() / dur_mask.sum()
            else:
                dur_loss = torch.tensor(0.0, device=device)

            total_loss = ce_masked + args.dur_weight * dur_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # Token accuracy
            pred_tokens = logits.argmax(dim=-1)  # (B, T)
            correct = ((pred_tokens == vq_tokens[:, :T]).float() * mask_t).sum()
            acc = correct / mask_t.sum().clamp(min=1)

            train_ce += ce_masked.item()
            train_dur += dur_loss.item()
            train_acc += acc.item()

        n = len(train_loader)

        # Validation
        model.eval()
        val_ce = 0
        val_acc = 0

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
                B, T, C = logits.shape
                ce = ce_loss_fn(logits.reshape(-1, C), vq_tokens[:, :T].reshape(-1)).reshape(B, T)
                mask_t = frame_mask[:, :T].float()
                val_ce += (ce * mask_t).sum().item() / mask_t.sum().clamp(min=1).item()

                pred_tokens = logits.argmax(dim=-1)
                correct = ((pred_tokens == vq_tokens[:, :T]).float() * mask_t).sum()
                val_acc += (correct / mask_t.sum().clamp(min=1)).item()

        vn = len(val_loader)

        print(
            f"Epoch {epoch:3d} | "
            f"Train: CE={train_ce/n:.4f} dur={train_dur/n:.4f} acc={train_acc/n:.2%} | "
            f"Val: CE={val_ce/vn:.4f} acc={val_acc/vn:.2%} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save best model
        if val_ce / vn < best_val_loss:
            best_val_loss = val_ce / vn
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
                "vocab_size": len(vocab),
            }, ckpt_dir / "transformer_best.pt")
            print(f"  -> Saved best model (val_CE={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val CE: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--phonemes-path", type=str, default="data/processed/phonemes.json")
    parser.add_argument("--alignments-path", type=str, default="data/processed/alignments.json")
    parser.add_argument("--vq-tokens-dir", type=str, default="data/vq_tokens")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--dur-weight", type=float, default=1.0)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()

    train_transformer(args)
