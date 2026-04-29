"""
v9 Step 2: train the phoneme-level RVQ tokenizer.

For each utterance, loop over body phonemes (skip BOS/EOS sentinels),
encode → 2× RVQ → decode, accumulate reconstruction MSE + commit loss.

Usage:
  python -u v9/training/train_phoneme_rvq.py --device mps --epochs 5 --preload
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

from v9.models.phoneme_rvq import PhonemeRVQTokenizer
from v9.training.dataset_v9 import V9PhonemeBlocksDataset, collate_v9


def train(args):
    device = torch.device(args.device)
    print(f"v9 phoneme RVQ tokenizer — training on {device}")

    dataset = V9PhonemeBlocksDataset(
        features_dir=args.features_dir,
        phonemes_path=args.phonemes_path,
        alignments_path=args.alignments_path,
        spk_emb_dir=args.spk_emb_dir,
        norm_stats_path=args.norm_stats,
        knob_source="none",                  # tokenizer doesn't need knobs
        max_phonemes=args.max_phonemes,
        normalize=True,
        preload=args.preload,
    )
    print(f"Dataset: {len(dataset)} utterances")

    val_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_v9, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_v9, num_workers=0)

    model = PhonemeRVQTokenizer(
        vocab_size=args.vocab_size,
        input_dim=14, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size, num_quantizers=args.num_quantizers,
        decoder_d_model=args.decoder_d_model, decoder_nhead=args.decoder_nhead,
        decoder_layers=args.decoder_layers,
        commitment_weight=args.commit_weight, ema_decay=args.ema_decay,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tokenizer params: {n_params:,}")

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

    best_val = float("inf")
    start_epoch = 1
    step = 0

    # ── Resume: load last.pt (preferred) or best.pt to continue training ──
    if args.resume:
        resume_path = None
        if (ckpt_dir / "last.pt").exists():
            resume_path = ckpt_dir / "last.pt"
        elif (ckpt_dir / "best.pt").exists():
            resume_path = ckpt_dir / "best.pt"
        if resume_path is not None:
            print(f"Resuming from {resume_path}")
            c = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(c["model"])
            if "optim" in c and args.restore_optim:
                optimizer.load_state_dict(c["optim"])
            if "scheduler" in c:
                scheduler.load_state_dict(c["scheduler"])
            if "step" in c:
                step = int(c["step"])
            start_epoch = int(c.get("epoch", 0)) + 1
            best_val = float(c.get("val_loss", float("inf")))
            print(f"  resumed at epoch {start_epoch - 1}, step {step}, best_val {best_val:.4f}")
        else:
            print(f"  --resume set but no last.pt or best.pt in {ckpt_dir}; starting fresh")

    F_PAD = args.f_pad        # fixed pad length for phoneme body frames
    F_PAD_EXT = F_PAD + 2     # +1 context frame on each side, for boundary continuity
    B_PAD = args.b_pad

    def compute_losses(utt_batch):
        """Pack all body phonemes into a fixed-shape (B_PAD, F_PAD_EXT, 14) tensor.
        Each phoneme's block is laid out as
          [prev_ctx, frame_0, frame_1, ..., frame_{L-1}, next_ctx, pad, ...]
        where prev_ctx = full_feats[s_off-1] (last frame of previous phoneme; zeros
        for the first body phoneme since BOS is zero) and next_ctx similarly.

        The tokenizer encoder still sees the L+2 frames (L originals + 2 context)
        — which is fine; we want the encoder to be context-aware. The DECODER is
        trained to reconstruct all L+2 positions, which gives it explicit GT
        supervision on the boundary frames. Result: at inference, the predicted
        frame at phoneme i's right boundary (next_ctx) and phoneme i+1's left
        boundary (prev_ctx) both target the same GT frame — so they agree, and
        the natural boundary delta in real audio is preserved (sharp on
        plosives, smooth on vowels).
        """
        all_blocks, all_ph_ids, all_lens = [], [], []
        for utt in utt_batch:
            frames_np   = utt["frames"]                                   # (T+2, 14)
            offsets     = utt["frame_offsets"]
            phoneme_ids = utt["phoneme_ids"]
            n_phon = len(phoneme_ids)
            for p_idx in range(1, n_phon - 1):
                s_off = int(offsets[p_idx]); e_off = int(offsets[p_idx + 1])
                if e_off <= s_off:
                    continue
                # Truncate body to F_PAD frames; context frames always present
                L_orig = min(e_off - s_off, F_PAD)
                ext = np.zeros((F_PAD_EXT, 14), dtype=np.float32)
                # prev context (BOS sentinel = zeros if s_off==0, naturally handled)
                ext[0] = frames_np[s_off - 1] if s_off > 0 else 0.0
                ext[1:1 + L_orig] = frames_np[s_off:s_off + L_orig]
                # next context (EOS sentinel = zeros if at end)
                ext[1 + L_orig] = frames_np[e_off] if e_off < frames_np.shape[0] else 0.0
                all_blocks.append(ext)
                all_ph_ids.append(int(phoneme_ids[p_idx]))
                all_lens.append(L_orig + 2)                                # length includes both ctx + body

        if not all_blocks:
            return (torch.tensor(0.0, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device), 0)

        if len(all_blocks) > B_PAD:
            idx = np.random.choice(len(all_blocks), B_PAD, replace=False)
            all_blocks = [all_blocks[i] for i in idx]
            all_ph_ids = [all_ph_ids[i] for i in idx]
            all_lens   = [all_lens[i]   for i in idx]
        elif len(all_blocks) < B_PAD:
            n_pad = B_PAD - len(all_blocks)
            for _ in range(n_pad):
                all_blocks.append(np.zeros((F_PAD_EXT, 14), dtype=np.float32))
                all_ph_ids.append(0)
                all_lens.append(1)

        B = B_PAD
        padded = np.stack(all_blocks)                                     # (B, F_PAD_EXT, 14)
        frames_t  = torch.from_numpy(padded).to(device)
        ph_ids_t  = torch.tensor(all_ph_ids, dtype=torch.long, device=device)
        lengths_t = torch.tensor(all_lens,   dtype=torch.long, device=device)

        recon, info = model.forward_batch(frames_t, ph_ids_t, lengths_t)
        mask = info["valid_mask"].unsqueeze(-1).float()
        se = ((recon - frames_t) ** 2) * mask
        denom = mask.sum().clamp(min=1) * 14
        recon_mean  = se.sum() / denom
        commit_mean = info["commit_loss"] / B
        return recon_mean, commit_mean, B

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        agg_recon = 0.0; agg_commit = 0.0; agg_phons = 0; n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            recon, commit, n_phons = compute_losses(batch)
            loss = recon + args.commit_loss_weight * commit
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            agg_recon += recon.item(); agg_commit += commit.item()
            agg_phons += n_phons; n_batches += 1
            step += 1
            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "recon": recon.item(), "commit": commit.item(), "n_phons": n_phons,
                }) + "\n"); metrics_f.flush()

        # Val
        model.eval()
        v_recon = 0.0; v_commit = 0.0; v_phons = 0; n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                recon, commit, n_phons = compute_losses(batch)
                v_recon += recon.item(); v_commit += commit.item()
                v_phons += n_phons; n_v += 1

        train_recon = agg_recon / max(1, n_batches)
        train_commit = agg_commit / max(1, n_batches)
        val_recon = v_recon / max(1, n_v)
        val_commit = v_commit / max(1, n_v)
        print(f"Epoch {epoch:3d} | train recon={train_recon:.4f} commit={train_commit:.4f} | "
              f"val recon={val_recon:.4f} commit={val_commit:.4f}")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train": {"recon": train_recon, "commit": train_commit},
            "val":   {"recon": val_recon,   "commit": val_commit},
        }) + "\n"); metrics_f.flush()

        # Always save last.pt for resume; save best.pt only when val improves
        ckpt_payload = {
            "epoch": epoch,
            "step":  step,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_recon,
            "args": vars(args),
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        if val_recon < best_val:
            best_val = val_recon
            ckpt_payload["val_loss"] = best_val
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val recon={best_val:.4f})")

    metrics_f.close()
    print(f"\nv9 phoneme RVQ training complete. Best val recon: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir",    default="data/features_merged_logpitch_v2")
    p.add_argument("--phonemes-path",   default="data/processed_merged_v3/phonemes_mfa.json")
    p.add_argument("--alignments-path", default="data/processed_merged_v3/alignments_mfa.json")
    p.add_argument("--spk-emb-dir",     default="v8/data/phoneme_anchors")
    p.add_argument("--norm-stats",      default="data/features_merged_logpitch_v2/norm_stats.npz")
    p.add_argument("--checkpoint-dir",  default="v9/checkpoints/phoneme_rvq")
    p.add_argument("--vocab-size",      type=int, default=73)
    p.add_argument("--resume",          action="store_true",
                   help="Load last.pt (or best.pt if no last.pt) and continue from saved epoch.")
    p.add_argument("--restore-optim",   action="store_true", default=True,
                   help="When resuming, also restore optimizer state.")
    p.add_argument("--f-pad",           type=int, default=32,
                   help="Fixed F per phoneme (truncate longer, pad shorter). Stable shape for MPS.")
    p.add_argument("--b-pad",           type=int, default=256,
                   help="Fixed phoneme count per batch. Subsample if more, dummy-pad if fewer.")
    p.add_argument("--max-phonemes",    type=int, default=200)
    p.add_argument("--device",          default="mps")
    p.add_argument("--batch-size",      type=int, default=8)
    p.add_argument("--epochs",          type=int, default=5)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--warmup-steps",    type=int, default=500)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--latent-dim",      type=int, default=256)
    p.add_argument("--hidden-dim",      type=int, default=128)
    p.add_argument("--codebook-size",   type=int, default=512)
    p.add_argument("--num-quantizers",  type=int, default=4)
    p.add_argument("--decoder-d-model", type=int, default=256)
    p.add_argument("--decoder-nhead",   type=int, default=4)
    p.add_argument("--decoder-layers",  type=int, default=2)
    p.add_argument("--commit-weight",   type=float, default=0.25)
    p.add_argument("--commit-loss-weight", type=float, default=1.0)
    p.add_argument("--ema-decay",       type=float, default=0.99)
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
