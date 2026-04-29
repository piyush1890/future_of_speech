"""Train v10 stage 2 (style planner). Targets are pre-extracted style codes."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from v10.models.v10_planner import V10StylePlanner


EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


class V10PlannerDataset(Dataset):
    """Lightweight: only loads what the planner needs (phoneme_ids, spk_emb, knobs, codes)."""
    def __init__(
        self,
        codes_dir: str = "v10/data/style_codes",
        phonemes_path: str = "data/processed_merged_v3/phonemes_mfa.json",
        spk_emb_dir: str = "v8/data/phoneme_anchors",
        metadata_path: str = "data/utterance_metadata_v5.json",
        max_phonemes: int = 200,
        preload: bool = False,
    ):
        self.codes_dir = Path(codes_dir)
        self.spk_emb_dir = Path(spk_emb_dir)
        self.phon_data = json.load(open(phonemes_path))
        self.max_phonemes = max_phonemes
        self.preload = preload
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
            if uid not in self.phon_data:
                continue
            n = len(self.phon_data[uid].get("indices", []))
            if not (4 < n <= max_phonemes + 2):
                continue
            if not (self.spk_emb_dir / f"{uid}.npz").exists():
                continue
            self.utt_ids.append(uid)

        self._cache = {}
        if preload:
            print(f"Preloading {len(self.utt_ids)} planner items ...")
            for i, uid in enumerate(self.utt_ids):
                self._cache[uid] = self._load(uid)
                if (i + 1) % 5000 == 0:
                    print(f"  {i+1}/{len(self.utt_ids)}")

    def _load(self, uid):
        f = np.load(self.codes_dir / f"{uid}.npz", allow_pickle=False)
        codes_full = f["codes"].astype(np.int64)              # (N+2,) — includes BOS/EOS PAD
        phoneme_ids = np.asarray(self.phon_data[uid]["indices"], dtype=np.int64)
        spk = np.load(self.spk_emb_dir / f"{uid}.npz", allow_pickle=False)
        spk_emb = spk["spk_emb"].astype(np.float32)
        default = tuple([0.0] * 5 + [0.5])
        knobs = np.asarray(self.knobs.get(uid, default), dtype=np.float32)
        return {
            "uid": uid, "phoneme_ids": phoneme_ids,
            "style_codes": codes_full, "spk_emb": spk_emb, "knobs": knobs,
        }

    def __len__(self): return len(self.utt_ids)
    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        return self._cache.get(uid) if self.preload else self._load(uid)


def collate_planner(batch, pad_phoneme_id=0, PAD_STYLE_CODE=64):
    B = len(batch)
    # Round to multiple of 16 to keep MPS kernel shape constant across batches.
    raw_N = max(b["phoneme_ids"].shape[0] for b in batch)
    N_max = ((raw_N + 15) // 16) * 16
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
        body_mask[i, 1:1 + n_body] = True
        # Style codes from extraction are already in (N+2,) layout with PAD at BOS/EOS.
        sc = b["style_codes"]
        style_codes[i, :sc.shape[0]] = sc

    return {
        "phoneme_ids": torch.from_numpy(phoneme_ids),
        "phoneme_mask": torch.from_numpy(phoneme_mask),
        "body_mask": torch.from_numpy(body_mask),
        "style_codes": torch.from_numpy(style_codes),
        "spk_emb": torch.from_numpy(spk_emb),
        "knobs": torch.from_numpy(knobs),
    }


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codes-dir", default="v10/data/style_codes")
    p.add_argument("--checkpoint-dir", default="v10/checkpoints/stage2_planner")
    p.add_argument("--style-codebook-size", type=int, default=64)
    p.add_argument("--max-phonemes", type=int, default=200)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--knob-dropout", type=float, default=0.3)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    PAD_STYLE_CODE = args.style_codebook_size

    ds = V10PlannerDataset(codes_dir=args.codes_dir, max_phonemes=args.max_phonemes,
                           preload=args.preload)
    n_val = max(64, int(len(ds) * 0.05))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_planner(b, PAD_STYLE_CODE=PAD_STYLE_CODE),
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_planner(b, PAD_STYLE_CODE=PAD_STYLE_CODE))

    planner = V10StylePlanner(
        style_codebook_size=args.style_codebook_size,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        d_ff=args.d_ff, dropout=args.dropout,
        knob_dim=ds.knob_dim, knob_dropout=args.knob_dropout,
        max_phonemes=args.max_phonemes,
    ).to(device)
    n_params = sum(p.numel() for p in planner.parameters())
    print(f"planner params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(planner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(out_dir / "metrics.jsonl", "a")
    def log(rec):
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        planner.train()
        agg_ce = 0.0; agg_acc = 0.0; agg_n = 0
        t0 = time.time()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            for k in ("phoneme_ids", "phoneme_mask", "body_mask",
                      "style_codes", "spk_emb", "knobs"):
                batch[k] = batch[k].to(device)
            lr = args.lr * cosine_with_warmup(step, args.warmup_steps, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            logits = planner(batch["phoneme_ids"], batch["spk_emb"], batch["knobs"],
                             batch["phoneme_mask"], batch["style_codes"])
            B, N, C = logits.shape
            body_f = batch["body_mask"].float()
            denom = body_f.sum().clamp(min=1.0)
            ce = F.cross_entropy(
                logits.reshape(-1, C), batch["style_codes"].reshape(-1),
                ignore_index=PAD_STYLE_CODE, reduction="none",
            ).reshape(B, N)
            ce_m = (ce * body_f).sum() / denom
            with torch.no_grad():
                pred = logits.argmax(-1)
                acc = ((pred == batch["style_codes"]).float() * body_f).sum() / denom

            opt.zero_grad()
            ce_m.backward()
            torch.nn.utils.clip_grad_norm_(planner.parameters(), args.grad_clip)
            opt.step()

            agg_ce += ce_m.item(); agg_acc += acc.item(); agg_n += 1
            step += 1
            pbar.set_postfix({"ce": f"{ce_m.item():.3f}", "acc": f"{acc.item()*100:.1f}%"})
            if step % args.log_every == 0:
                log({"type": "step", "step": step, "lr": lr,
                     "ce": ce_m.item(), "acc": acc.item()})
                tqdm.write(f"e{epoch} step {step}  ce={ce_m.item():.3f}  acc={acc.item()*100:.1f}%  lr={lr:.2e}")

        planner.eval()
        v_ce = 0.0; v_acc = 0.0; v_n = 0
        with torch.no_grad():
            for batch in val_dl:
                for k in ("phoneme_ids", "phoneme_mask", "body_mask",
                          "style_codes", "spk_emb", "knobs"):
                    batch[k] = batch[k].to(device)
                logits = planner(batch["phoneme_ids"], batch["spk_emb"], batch["knobs"],
                                 batch["phoneme_mask"], batch["style_codes"])
                B, N, C = logits.shape
                body_f = batch["body_mask"].float()
                denom = body_f.sum().clamp(min=1.0)
                ce = F.cross_entropy(
                    logits.reshape(-1, C), batch["style_codes"].reshape(-1),
                    ignore_index=PAD_STYLE_CODE, reduction="none",
                ).reshape(B, N)
                ce_m = (ce * body_f).sum() / denom
                pred = logits.argmax(-1)
                acc = ((pred == batch["style_codes"]).float() * body_f).sum() / denom
                v_ce += ce_m.item(); v_acc += acc.item(); v_n += 1

        train_ce = agg_ce / max(1, agg_n); train_acc = agg_acc / max(1, agg_n)
        val_ce_avg = v_ce / max(1, v_n); val_acc_avg = v_acc / max(1, v_n)
        elapsed = time.time() - t0
        print(f"=== epoch {epoch}  train_ce={train_ce:.3f} acc={train_acc*100:.1f}% | "
              f"val_ce={val_ce_avg:.3f} acc={val_acc_avg*100:.1f}%  time={elapsed/60:.1f}m ===")
        log({"type": "epoch", "epoch": epoch,
             "train": {"ce": train_ce, "acc": train_acc},
             "val": {"ce": val_ce_avg, "acc": val_acc_avg},
             "elapsed_sec": elapsed})

        ckpt = {"model": planner.state_dict(), "args": vars(args), "epoch": epoch,
                "val_ce": val_ce_avg, "knob_dim": ds.knob_dim}
        torch.save(ckpt, out_dir / "last.pt")
        if val_ce_avg < best_val:
            best_val = val_ce_avg
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> new best (val_ce={val_ce_avg:.3f})")

    metrics_f.close()


if __name__ == "__main__":
    main()
