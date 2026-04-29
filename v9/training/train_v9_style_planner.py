"""
v9 Stage 2 trainer — train the style planner.

Inputs come from v9/data/style_codes/<uid>.npz (extracted in Stage 1.5).
Predicts 1 of 512 style codes per body phoneme from (text, spk, knobs).
Loss: CE on body positions only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.v9_style_planner import V9StylePlanner


EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}
PAD_STYLE_CODE = 64    # = style_codebook_size; padding index for BOS/EOS slots


class V9StylePlannerDataset(Dataset):
    def __init__(self, codes_dir: str = "v9/data/style_codes",
                 metadata_path: str = "data/utterance_metadata_v5.json",
                 max_phonemes: int = 200, preload: bool = False):
        self.codes_dir = Path(codes_dir)
        self.preload = preload
        self.max_phonemes = max_phonemes
        self.knob_dim = 6

        self.knobs = {}
        if Path(metadata_path).exists():
            meta = json.load(open(metadata_path))
            for k, r in meta.items():
                eid = EMOTION_TO_ID.get(r.get("emotion_label", "neutral"), 0)
                intensity = float(r.get("intensity", 0.5))
                one_hot = [0.0] * 5; one_hot[eid] = 1.0
                self.knobs[k] = tuple(one_hot + [intensity])

        self.utt_ids = []
        for p in sorted(self.codes_dir.glob("*.npz")):
            uid = p.stem
            try:
                f = np.load(p, allow_pickle=False)
                n_full = len(f["phoneme_ids"])
                if 4 < n_full <= max_phonemes + 2:
                    self.utt_ids.append(uid)
            except Exception:
                continue

        self._cache = {}
        if preload:
            print(f"Preloading {len(self.utt_ids)} planner items ...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load(uid)
                if (i+1) % 5000 == 0: print(f"  {i+1}/{len(self.utt_ids)}")
        else:
            print(f"V9StylePlannerDataset: {len(self.utt_ids)} utterances")

    def _load(self, uid):
        f = np.load(self.codes_dir / f"{uid}.npz", allow_pickle=False)
        default = tuple([0.0] * 5 + [0.5])
        return {
            "uid": uid,
            "phoneme_ids": f["phoneme_ids"].astype(np.int64),
            "style_codes": f["style_codes"].astype(np.int64),
            "spk_emb":     f["spk_emb"].astype(np.float32),
            "knobs":       np.asarray(self.knobs.get(uid, default), dtype=np.float32),
        }

    def __len__(self): return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        return self._cache.get(uid) if self.preload else self._load(uid)


def _round_up(n, b): return ((n + b - 1) // b) * b


def collate_planner(batch, pad_phoneme_id=0):
    B = len(batch)
    N_max = _round_up(max(b["phoneme_ids"].shape[0] for b in batch), 16)
    phoneme_ids = np.full((B, N_max), pad_phoneme_id, dtype=np.int64)
    phoneme_mask = np.zeros((B, N_max), dtype=bool)
    body_mask = np.zeros((B, N_max), dtype=bool)
    style_codes = np.full((B, N_max), PAD_STYLE_CODE, dtype=np.int64)
    spk_emb = np.stack([b["spk_emb"] for b in batch])
    knobs = np.stack([b["knobs"] for b in batch])

    for i, b in enumerate(batch):
        n_full = len(b["phoneme_ids"])
        n_body = n_full - 2
        phoneme_ids[i, :n_full] = b["phoneme_ids"]
        phoneme_mask[i, :n_full] = True
        body_mask[i, 1:1+n_body] = True
        style_codes[i, 1:1+n_body] = b["style_codes"]

    return {
        "phoneme_ids": torch.from_numpy(phoneme_ids),
        "phoneme_mask": torch.from_numpy(phoneme_mask),
        "body_mask": torch.from_numpy(body_mask),
        "style_codes": torch.from_numpy(style_codes),
        "spk_emb": torch.from_numpy(spk_emb),
        "knobs": torch.from_numpy(knobs),
    }


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size; self.shuffle = shuffle
        underlying = dataset.dataset if hasattr(dataset, "dataset") else dataset
        indices = list(dataset.indices) if hasattr(dataset, "indices") else list(range(len(dataset)))
        lengths = []
        for local_i, ds_idx in enumerate(indices):
            uid = underlying.utt_ids[ds_idx]
            n = len(underlying._cache[uid]["phoneme_ids"]) if underlying.preload \
                else len(underlying._load(uid)["phoneme_ids"])
            lengths.append((local_i, n))
        lengths.sort(key=lambda x: x[1])
        self.batches = [
            [idx for idx, _ in lengths[i:i+batch_size]]
            for i in range(0, len(lengths), batch_size)
        ]
    def __iter__(self):
        if self.shuffle: random.shuffle(self.batches)
        for batch in self.batches: yield batch
    def __len__(self): return len(self.batches)


def train(args):
    device = torch.device(args.device)
    print(f"v9 stage 2 (style planner) — training on {device}")

    dataset = V9StylePlannerDataset(
        codes_dir=args.codes_dir, metadata_path=args.metadata_path,
        max_phonemes=args.max_phonemes, preload=args.preload,
    )

    val_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42),
    )
    train_sampler = BucketBatchSampler(train_set, batch_size=args.batch_size, shuffle=True)
    val_sampler   = BucketBatchSampler(val_set,   batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              collate_fn=collate_planner, num_workers=0)
    val_loader   = DataLoader(val_set, batch_sampler=val_sampler,
                              collate_fn=collate_planner, num_workers=0)

    planner = V9StylePlanner(
        vocab_size=args.vocab_size, style_codebook_size=args.style_codebook_size,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        speaker_emb_dim=64, knob_dim=dataset.knob_dim,
        knob_dropout=args.knob_dropout, max_phonemes=args.max_phonemes,
    ).to(device)
    print(f"Planner params: {sum(p.numel() for p in planner.parameters()):,}")

    optimizer = torch.optim.AdamW(planner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(ckpt_dir / "metrics.jsonl", "a")

    best_val = float("inf"); start_epoch = 1; step = 0
    if args.resume:
        path = ckpt_dir / "last.pt"
        if not path.exists() and (ckpt_dir / "best.pt").exists():
            path = ckpt_dir / "best.pt"
        if path.exists():
            c = torch.load(path, map_location=device, weights_only=False)
            planner.load_state_dict(c["model"])
            if "optim" in c and args.restore_optim: optimizer.load_state_dict(c["optim"])
            if "scheduler" in c: scheduler.load_state_dict(c["scheduler"])
            step = int(c.get("step", 0))
            start_epoch = int(c.get("epoch", 0)) + 1
            best_val = float(c.get("val_loss", float("inf")))

    def compute_losses(batch):
        phoneme_ids  = batch["phoneme_ids"].to(device)
        phoneme_mask = batch["phoneme_mask"].to(device)
        body_mask    = batch["body_mask"].to(device)
        gt_codes     = batch["style_codes"].to(device)
        spk = batch["spk_emb"].to(device); knobs = batch["knobs"].to(device)

        logits = planner(phoneme_ids, spk, knobs, phoneme_mask, gt_codes)
        B, N, C = logits.shape
        body_f = body_mask.float()
        denom = body_f.sum().clamp(min=1)
        # ignore_index=PAD_STYLE_CODE prevents F.cross_entropy from indexing out-of-range
        # at non-body positions (which are filled with PAD_STYLE_CODE = codebook_size).
        ce = F.cross_entropy(
            logits.reshape(-1, C), gt_codes.reshape(-1),
            ignore_index=PAD_STYLE_CODE, reduction="none",
        ).reshape(B, N)
        ce_m = (ce * body_f).sum() / denom
        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = ((pred == gt_codes).float() * body_f).sum() / denom
        return ce_m, acc

    for epoch in range(start_epoch, args.epochs + 1):
        planner.train()
        agg = {"ce": 0.0, "acc": 0.0}; n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            ce, acc = compute_losses(batch)
            optimizer.zero_grad(); ce.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step()
            agg["ce"] += ce.item(); agg["acc"] += acc.item(); n_batches += 1
            step += 1
            if step % args.log_every == 0:
                metrics_f.write(json.dumps({
                    "type": "step", "step": step, "lr": scheduler.get_last_lr()[0],
                    "ce": ce.item(), "acc": acc.item(),
                }) + "\n"); metrics_f.flush()

        planner.eval()
        v = {"ce": 0.0, "acc": 0.0}; n_v = 0
        with torch.no_grad():
            for batch in val_loader:
                ce, acc = compute_losses(batch)
                v["ce"] += ce.item(); v["acc"] += acc.item(); n_v += 1

        train_avg = {k: x / max(1, n_batches) for k, x in agg.items()}
        val_avg   = {k: x / max(1, n_v) for k, x in v.items()}
        print(f"Epoch {epoch:3d} | train CE={train_avg['ce']:.3f} acc={train_avg['acc']*100:.1f}% | "
              f"val CE={val_avg['ce']:.3f} acc={val_avg['acc']*100:.1f}%")
        metrics_f.write(json.dumps({
            "type": "epoch", "epoch": epoch, "train": train_avg, "val": val_avg,
        }) + "\n"); metrics_f.flush()

        ckpt = {
            "epoch": epoch, "step": step, "model": planner.state_dict(),
            "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "val_loss": val_avg["ce"], "args": vars(args),
            "knob_dim": dataset.knob_dim,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_avg["ce"] < best_val:
            best_val = val_avg["ce"]
            ckpt["val_loss"] = best_val
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> Saved best (val={best_val:.4f})")

    metrics_f.close()
    print(f"\nv9 stage 2 (planner) complete. Best val: {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--codes-dir",       default="v9/data/style_codes")
    p.add_argument("--metadata-path",   default="data/utterance_metadata_v5.json")
    p.add_argument("--checkpoint-dir",  default="v9/checkpoints/stage2_planner")
    p.add_argument("--vocab-size",      type=int, default=73)
    p.add_argument("--style-codebook-size", type=int, default=64)
    p.add_argument("--max-phonemes",    type=int, default=200)
    p.add_argument("--device",          default="mps")
    p.add_argument("--batch-size",      type=int, default=16)
    p.add_argument("--epochs",          type=int, default=20)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--warmup-steps",    type=int, default=1000)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--d-model",         type=int, default=192)
    p.add_argument("--nhead",           type=int, default=4)
    p.add_argument("--num-layers",      type=int, default=4)
    p.add_argument("--d-ff",            type=int, default=768)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--knob-dropout",    type=float, default=0.3,
                   help="Higher dropout → stronger CFG at inference.")
    p.add_argument("--preload",         action="store_true")
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--restore-optim",   action="store_true", default=True)
    p.add_argument("--log-every",       type=int, default=200)
    args = p.parse_args()
    train(args)
