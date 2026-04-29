"""Train v10 renderer + style encoder jointly.

Inputs per batch:
  phoneme_ids, phoneme_mask, spk_emb, knobs        — encoder inputs
  frames, frame_mask, frame_to_enc_pos             — for style encoder
  frame_codes (pre-extracted from disk)            — RVQ token targets
  eop, body_durations                              — EOP target + EOP class balance

Losses:
  frame CE (per RVQ level, level-weighted, masked)
  EOP BCE (pos-weighted by avg phoneme length)
  style codebook commit loss
"""
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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from v10.models.v10_renderer import V10Renderer
from v10.models.v10_style import V10StyleEncoder
from v10.training.dataset_v10 import V10Dataset, collate_v10_renderer


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def compute_losses(style_out, render_out, batch, K: int, level_w, codebooks=None,
                   ce_lev_const: float = 1.0):
    """Returns dict of scalar losses.

    codebooks: optional list of K tensors (codebook_size, D). When provided,
    adds a vector-distance loss = ||softmax(logits) @ codebook - codebook[gt]||²
    per level. CE alone treats every wrong code as equally wrong; the vector
    loss rewards predictions that are CLOSE to GT in vector space, which is
    what the decoder actually consumes.
    """
    frame_logits = render_out["frame_logits"]                # (B, T, K, C)
    eop_logit = render_out["eop_logit"]                       # (B, T)
    frame_codes = batch["frame_codes"]                        # (B, T, K) long
    frame_mask = batch["frame_mask"]                          # (B, T) bool
    eop = batch["eop"]                                        # (B, T) float
    body_dur = batch["body_durations"]                        # (B, N_max) long

    B, T, _, C = frame_logits.shape
    fmask = frame_mask.float()
    denom = fmask.sum().clamp(min=1.0)

    # Per-level CE + accuracy + (optional) vector-distance
    ce_levels = []
    vec_levels = []
    acc_levels = []
    for k in range(K):
        lg = frame_logits[..., k, :]                          # (B, T, C)
        target = frame_codes[..., k]                          # (B, T)
        ce_k = F.cross_entropy(
            lg.reshape(-1, C), target.reshape(-1), reduction="none",
        ).reshape(B, T)
        ce_k = (ce_k * fmask).sum() / denom
        ce_levels.append(ce_k)

        with torch.no_grad():
            argmax = lg.argmax(-1)
            acc_k = ((argmax == target).float() * fmask).sum() / denom
            acc_levels.append(acc_k)

        if codebooks is not None:
            cb = codebooks[k]                                 # (C, D)
            probs = F.softmax(lg, dim=-1)                     # (B, T, C)
            pred_vec = probs @ cb                             # (B, T, D)
            gt_vec = cb[target]                               # (B, T, D)
            vec_k = (pred_vec - gt_vec).pow(2).sum(-1)        # (B, T)
            vec_k = (vec_k * fmask).sum() / denom
            vec_levels.append(vec_k)

    ce_total = sum(level_w[k] * ce_levels[k] for k in range(K))
    vec_total = (sum(level_w[k] * vec_levels[k] for k in range(K))
                 if codebooks is not None else None)

    # EOP BCE with pos-weighting (negatives:positives = avg_phoneme_len - 1)
    n_pos = (body_dur > 0).float().sum().clamp(min=1.0)
    n_frames = body_dur.float().sum().clamp(min=1.0)
    pos_weight = ((n_frames - n_pos) / n_pos).clamp(min=1.0)
    pos_weight = pos_weight.detach()
    eop_loss_per = F.binary_cross_entropy_with_logits(
        eop_logit, eop, reduction="none", pos_weight=pos_weight,
    )
    eop_loss = (eop_loss_per * fmask).sum() / denom

    commit = style_out["commit_loss"]

    return {
        "ce_total": ce_total,
        "ce_per_level": ce_levels,
        "acc_per_level": acc_levels,
        "vec_total": vec_total,
        "vec_per_level": vec_levels if codebooks is not None else None,
        "eop_loss": eop_loss,
        "commit": commit,
        "pos_weight": pos_weight.item() if hasattr(pos_weight, "item") else float(pos_weight),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--style-codebook-size", type=int, default=64)
    p.add_argument("--num-quantizers", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--enc-layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=6)
    p.add_argument("--knob-source", default="emotion")
    p.add_argument("--knob-dropout", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=800)
    p.add_argument("--max-phonemes", type=int, default=200)
    p.add_argument("--frame-codes-dir", default="v10/data/frame_codes")
    p.add_argument("--tokenizer-checkpoint", default="v10/checkpoints/tokenizer/best.pt",
                   help="Path to tokenizer best.pt — used to extract codebooks for vec loss.")
    p.add_argument("--eop-weight", type=float, default=0.5)
    p.add_argument("--commit-weight", type=float, default=0.25)
    p.add_argument("--vec-weight", type=float, default=0.5,
                   help="Weight for vector-distance loss (rewards predicting tokens "
                        "close to GT in codebook vector space). 0 disables.")
    p.add_argument("--level-weights", default="1.0,0.7,0.5,0.4")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-dir", default="v10/checkpoints/stage1_renderer")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    level_w = [float(x) for x in args.level_weights.split(",")]
    assert len(level_w) == args.num_quantizers, "level-weights length must equal num-quantizers"

    ds = V10Dataset(
        max_frames=args.max_frames, max_phonemes=args.max_phonemes,
        knob_source=args.knob_source, frame_codes_dir=args.frame_codes_dir,
        preload=args.preload,
    )
    n_val = max(64, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    print(f"train={len(train_ds)}  val={len(val_ds)}  knob_dim={ds.knob_dim}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_v10_renderer,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_v10_renderer)

    style_enc = V10StyleEncoder(
        codebook_size=args.style_codebook_size,
    ).to(device)
    renderer = V10Renderer(
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        style_codebook_size=args.style_codebook_size,
        d_model=args.d_model,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        knob_dim=ds.knob_dim,
        knob_dropout=args.knob_dropout,
        max_phonemes=args.max_phonemes + 4,
        max_frames=args.max_frames + 16,
    ).to(device)

    n_params = sum(p.numel() for p in style_enc.parameters()) \
             + sum(p.numel() for p in renderer.parameters())
    print(f"style+renderer params: {n_params/1e6:.2f}M")

    # Load tokenizer codebooks (frozen) for vector-distance loss
    codebooks = None
    if args.vec_weight > 0:
        from v10.models.v10_tokenizer import V10Tokenizer
        tc = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
        ta = tc["args"]
        tok = V10Tokenizer(
            d_model=ta["d_model"], num_encoder_layers=ta["enc_layers"],
            num_decoder_layers=ta["dec_layers"], codebook_size=ta["codebook_size"],
            num_quantizers=ta["num_quantizers"], max_frames=ta["max_frames"] + 16,
        )
        tok.load_state_dict(tc["model"])
        codebooks = [layer._codebook.embed[0].detach().clone().to(device)
                     for layer in tok.rvq.layers]
        del tok
        print(f"loaded {len(codebooks)} codebooks for vec loss; "
              f"shape per level: {codebooks[0].shape}; vec_weight={args.vec_weight}")

    params = list(style_enc.parameters()) + list(renderer.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_f = open(out_dir / "metrics.jsonl", "a")
    def log(rec):
        metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()

    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        style_enc.train(); renderer.train()
        t0 = time.time()
        running = {"ce_total": 0.0, "vec_total": 0.0, "eop": 0.0, "commit": 0.0}
        running_n = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            for key in ("phoneme_ids", "phoneme_mask", "spk_emb", "knobs", "frames",
                        "frame_mask", "frame_to_enc_pos", "eop", "frame_codes",
                        "body_durations"):
                batch[key] = batch[key].to(device)

            lr = args.lr * cosine_with_warmup(step, args.warmup_steps, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            n_total = batch["phoneme_ids"].shape[1]
            style_out = style_enc(batch["frames"], batch["frame_mask"],
                                  batch["frame_to_enc_pos"], n_total=n_total)
            render_out = renderer(
                batch["phoneme_ids"], style_out["codes"], batch["spk_emb"],
                batch["knobs"], batch["phoneme_mask"],
                batch["frame_codes"], batch["frame_to_enc_pos"], batch["frame_mask"],
            )
            losses = compute_losses(style_out, render_out, batch, args.num_quantizers,
                                    level_w, codebooks=codebooks)
            total = losses["ce_total"] + args.eop_weight * losses["eop_loss"] \
                  + args.commit_weight * losses["commit"]
            if losses["vec_total"] is not None:
                total = total + args.vec_weight * losses["vec_total"]

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            running["ce_total"] += losses["ce_total"].item()
            running["eop"] += losses["eop_loss"].item()
            running["commit"] += float(losses["commit"].item())
            if losses["vec_total"] is not None:
                running["vec_total"] += losses["vec_total"].item()
            running_n += 1
            step += 1

            # Live tqdm postfix every step (cheap), full log every log_every steps
            vec_val = losses["vec_total"].item() if losses["vec_total"] is not None else None
            acc_per = [a.item() for a in losses["acc_per_level"]]
            pbar.set_postfix({
                "ce": f"{losses['ce_total'].item():.2f}",
                "acc0": f"{acc_per[0]*100:.0f}%",
                "eop": f"{losses['eop_loss'].item():.2f}",
                **({"vec": f"{vec_val:.2f}"} if vec_val is not None else {}),
            })
            if step % args.log_every == 0:
                rec = {"type": "step", "step": step, "lr": lr,
                       "ce_total": losses["ce_total"].item(),
                       "ce_per_level": [c.item() for c in losses["ce_per_level"]],
                       "acc_per_level": acc_per,
                       "vec_total": vec_val,
                       "vec_per_level": [v.item() for v in losses["vec_per_level"]] if losses["vec_per_level"] else None,
                       "eop": losses["eop_loss"].item(),
                       "commit": float(losses["commit"].item()),
                       "pos_weight": losses["pos_weight"],
                       "total": total.item()}
                log(rec)
                lvl_str = " ".join(f"{c.item():.2f}" for c in losses["ce_per_level"])
                acc_str = " ".join(f"{a*100:.0f}%" for a in acc_per)
                vec_str = f"  vec={vec_val:.3f}" if vec_val is not None else ""
                tqdm.write(f"e{epoch} step {step}  ce_tot={losses['ce_total'].item():.3f} "
                           f"[{lvl_str}]  acc=[{acc_str}]{vec_str}  "
                           f"eop={losses['eop_loss'].item():.3f}  lr={lr:.2e}")

        # Validation
        style_enc.eval(); renderer.eval()
        v = {"ce_total": 0.0, "eop": 0.0, "commit": 0.0,
             "ce_per_level": [0.0] * args.num_quantizers,
             "acc_per_level": [0.0] * args.num_quantizers}
        v_n = 0
        with torch.no_grad():
            for batch in val_dl:
                for key in ("phoneme_ids", "phoneme_mask", "spk_emb", "knobs", "frames",
                            "frame_mask", "frame_to_enc_pos", "eop", "frame_codes",
                            "body_durations"):
                    batch[key] = batch[key].to(device)
                n_total = batch["phoneme_ids"].shape[1]
                style_out = style_enc(batch["frames"], batch["frame_mask"],
                                      batch["frame_to_enc_pos"], n_total=n_total)
                render_out = renderer(
                    batch["phoneme_ids"], style_out["codes"], batch["spk_emb"],
                    batch["knobs"], batch["phoneme_mask"],
                    batch["frame_codes"], batch["frame_to_enc_pos"], batch["frame_mask"],
                )
                losses = compute_losses(style_out, render_out, batch, args.num_quantizers,
                                        level_w, codebooks=codebooks)
                v["ce_total"] += losses["ce_total"].item()
                v["eop"] += losses["eop_loss"].item()
                v["commit"] += float(losses["commit"].item())
                if losses["vec_total"] is not None:
                    v["vec_total"] = v.get("vec_total", 0.0) + losses["vec_total"].item()
                for k in range(args.num_quantizers):
                    v["ce_per_level"][k] += losses["ce_per_level"][k].item()
                    v["acc_per_level"][k] += losses["acc_per_level"][k].item()
                v_n += 1

        for k in v: v[k] = v[k] / max(1, v_n) if not isinstance(v[k], list) else \
                            [x / max(1, v_n) for x in v[k]]
        elapsed = time.time() - t0
        vec_str = f"  val_vec={v.get('vec_total', 0):.3f}" if 'vec_total' in v else ''
        print(f"=== epoch {epoch}  val_ce={v['ce_total']:.3f}  val_eop={v['eop']:.3f}{vec_str}  "
              f"time={elapsed/60:.1f}m ===")
        log({"type": "epoch", "epoch": epoch,
             "train_ce": running["ce_total"] / max(1, running_n),
             "train_vec": running["vec_total"] / max(1, running_n),
             "train_eop": running["eop"] / max(1, running_n),
             "train_commit": running["commit"] / max(1, running_n),
             "val_ce": v["ce_total"], "val_vec": v.get("vec_total", None),
             "val_eop": v["eop"], "val_commit": v["commit"],
             "val_ce_per_level": v["ce_per_level"], "elapsed_sec": elapsed})

        ckpt = {"style_enc": style_enc.state_dict(),
                "renderer": renderer.state_dict(),
                "args": vars(args), "epoch": epoch,
                "val_ce": v["ce_total"], "val_eop": v["eop"]}
        torch.save(ckpt, out_dir / "last.pt")
        if v["ce_total"] < best_val:
            best_val = v["ce_total"]
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> new best (val_ce={v['ce_total']:.3f})")

    metrics_f.close()


if __name__ == "__main__":
    main()
