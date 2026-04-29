"""
v8 stage 2 (codebook AR): predict per-phoneme code IDs from text + speaker + V/A/D.
AR with causal mask + teacher forcing. CE loss against GT code IDs (from v5 codebook).
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

from v8.models.v8_planner import V8CodebookPlanner, shift_codes_for_teacher_forcing
from v8.training.dataset_v8 import PhonemeAnchorsDataset, collate_v8
from models.style_codebook import StyleCodebook


def train(args):
    device = torch.device(args.device)
    print(f"v8 stage 2 (codebook AR planner) — training on {device}")

    # Load v5 codebook (frozen) — used to provide initial code embeddings to planner
    s1 = torch.load(args.v5_checkpoint, map_location=device, weights_only=False)
    sa = s1["args"]
    cb = StyleCodebook(latent_dim=sa["d_model"], codebook_size=sa["style_codebook_size"]).to(device)
    cb.load_state_dict(s1["style_codebook_state_dict"])
    cb.eval()
    # Codebook entries: vector_quantize_pytorch stores them at vq._codebook.embed (1, C, D)
    cb_entries = cb.vq._codebook.embed.detach().clone()
    if cb_entries.dim() == 3:
        cb_entries = cb_entries.squeeze(0)            # (C, D)
    cb_entries = cb_entries.to(device)
    print(f"v5 codebook entries: {cb_entries.shape}")

    # Dataset (with code IDs)
    dataset = PhonemeAnchorsDataset(
        anchors_dir=args.anchors_dir,
        z_dir=None,
        codes_dir=args.codes_dir,
        phonemes_path=args.phonemes_path,
        vad_paths=args.vad_paths,
        knob_source=args.knob_source,
        max_phonemes=args.max_phonemes,
        preload=args.preload,
    )
    print(f"  knob_dim = {dataset.knob_dim}  (knob_source={args.knob_source})")
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

    planner = V8CodebookPlanner(
        vocab_size=args.vocab_size,
        codebook_entries=cb_entries,
        d_model=args.planner_d_model,
        nhead=args.planner_nhead,
        num_layers=args.planner_layers,
        d_ff=args.planner_d_ff,
        dropout=args.dropout,
        knob_dim=dataset.knob_dim, speaker_emb_dim=64,
        knob_dropout=args.knob_dropout,
        max_context=args.max_context,
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

    def compute_losses(batch):
        phoneme_ids = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        spk_emb  = batch["spk_emb"].to(device)
        knobs    = batch["knobs"].to(device)
        gt_code_id = batch["code_id"].to(device)              # (B, N) long, -1 at pad

        prev_z = shift_codes_for_teacher_forcing(
            gt_code_id, planner.bos_emb, planner.code_embeddings,
        )
        logits = planner(phoneme_ids, spk_emb, knobs, phoneme_mask, prev_z=prev_z)
        # Body-only mask: exclude BOS (first valid) and EOS (last valid)
        body_mask = phoneme_mask.clone()
        B, _ = phoneme_mask.shape
        for b in range(B):
            idx = phoneme_mask[b].nonzero(as_tuple=True)[0]
            if len(idx) >= 2:
                body_mask[b, idx[0].item()] = False
                body_mask[b, idx[-1].item()] = False
        # Compute CE only at body positions where code_id != -1
        valid = body_mask & (gt_code_id >= 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        logits_flat = logits[valid]               # (M, codebook_size)
        targets_flat = gt_code_id[valid]          # (M,)
        ce = F.cross_entropy(logits_flat, targets_flat)
        # Top-1 accuracy
        with torch.no_grad():
            acc = (logits_flat.argmax(-1) == targets_flat).float().mean()
        return ce, acc

    step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        planner.train()
        agg_ce = 0.0; agg_acc = 0.0; n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            ce, acc = compute_losses(batch)
            optimizer.zero_grad()
            ce.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            agg_ce += ce.item(); agg_acc += acc.item(); n_batches += 1
            step += 1
            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "ce": ce.item(), "acc": acc.item(),
                }) + "\n"); metrics_f.flush()

        planner.eval()
        v_ce = 0.0; v_acc = 0.0; n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                ce, acc = compute_losses(batch)
                v_ce += ce.item(); v_acc += acc.item(); n_v += 1
        train_ce = agg_ce / max(1, n_batches); train_acc = agg_acc / max(1, n_batches)
        val_ce = v_ce / max(1, n_v); val_acc = v_acc / max(1, n_v)
        print(f"Epoch {epoch:3d} | train CE={train_ce:.4f} acc={train_acc:.1%} | "
              f"val CE={val_ce:.4f} acc={val_acc:.1%}")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch,
            "train": {"ce": train_ce, "acc": train_acc},
            "val": {"ce": val_ce, "acc": val_acc},
            "best_val": min(best_val, val_ce),
        }) + "\n"); metrics_f.flush()

        if val_ce < best_val:
            best_val = val_ce
            torch.save({
                "epoch": epoch,
                "model": planner.state_dict(),
                "optim": optimizer.state_dict(),
                "val_loss": best_val,
                "args": vars(args),
                "vocab_size": args.vocab_size,
                "codebook_size": planner.codebook_size,
                "style_dim": planner.style_dim,
            }, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val CE={best_val:.4f})")

    metrics_f.close()
    print(f"\nv8 stage 2 codebook AR complete. Best val CE: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--anchors-dir",     default="v8/data/phoneme_anchors")
    p.add_argument("--codes-dir",       default="v8/data/phoneme_codes")
    p.add_argument("--phonemes-path",   default="data/processed_merged_v3/phonemes_mfa.json")
    p.add_argument("--vad-paths",       nargs="+",
                   default=["data/librispeech_emotion_vad.json", "data/esd_emotion_vad.json"])
    p.add_argument("--knob-source",     choices=["vad", "emotion"], default="emotion",
                   help="vad: 3-d V/A/D; emotion: 6-d (one-hot 5-emotion + intensity)")
    p.add_argument("--v5-checkpoint",
                   default="checkpoints_v5_stage1_archived/transformer_best.pt")
    p.add_argument("--checkpoint-dir",  default="v8/checkpoints/stage2_cb_ar")
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
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--knob-dropout",    type=float, default=0.1)
    p.add_argument("--max-context",     type=int, default=100,
                   help="AR sliding-window context size (phonemes); 0 = unlimited")
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
