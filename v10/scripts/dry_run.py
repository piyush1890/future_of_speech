"""End-to-end integration smoke for v10 on a tiny subset.

  step 1: train tokenizer ~50 steps
  step 2: extract frame codes for the same subset
  step 3: train renderer + style joint ~30 steps

If this completes without crashing the full pipeline is sound.
"""
import sys
import time
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from v10.models.v10_tokenizer import V10Tokenizer
from v10.models.v10_style import V10StyleEncoder
from v10.models.v10_renderer import V10Renderer
from v10.training.dataset_v10 import V10Dataset, collate_v10_tokenizer, collate_v10_renderer
from v10.training.train_v10_renderer import compute_losses


N_UTTS = 200          # subset size
TOK_STEPS = 60
REN_STEPS = 30
DEVICE = "mps"
TMP_DIR = Path("/tmp/v10_dry_run")
FRAME_CODES_DIR = TMP_DIR / "frame_codes"


def banner(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


def step1_tokenizer():
    banner("STEP 1 — tokenizer training (small)")
    ds = V10Dataset(max_frames=600, knob_source="emotion", preload=False)
    print(f"  full dataset: {len(ds)} utterances")
    sub = Subset(ds, list(range(N_UTTS)))
    print(f"  subset: {len(sub)}")

    dl = DataLoader(sub, batch_size=4, shuffle=True, collate_fn=collate_v10_tokenizer)
    device = torch.device(DEVICE)
    model = V10Tokenizer(
        d_model=128, num_encoder_layers=2, num_decoder_layers=2,
        codebook_size=256, num_quantizers=4,
        max_frames=620,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    t0 = time.time()
    step = 0
    while step < TOK_STEPS:
        for batch in dl:
            frames = batch["frames"].to(device)
            mask = batch["frame_mask"].to(device)
            out = model(frames, mask)
            denom = (mask.float().sum() * frames.shape[-1]).clamp(min=1.0)
            mse = (((out["recon"] - frames) ** 2 * mask.unsqueeze(-1).float()).sum()) / denom
            loss = mse + 0.25 * out["commit_loss"]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"  step {step}/{TOK_STEPS}  mse={mse.item():.4f}  commit={float(out['commit_loss'].item()):.4f}")
            if step >= TOK_STEPS:
                break
    print(f"  tokenizer done in {time.time()-t0:.1f}s")
    return model, ds


def step2_extract(tok, ds):
    banner("STEP 2 — frame code extraction")
    FRAME_CODES_DIR.mkdir(parents=True, exist_ok=True)
    sub = Subset(ds, list(range(N_UTTS)))
    dl = DataLoader(sub, batch_size=8, shuffle=False, collate_fn=collate_v10_tokenizer)
    device = torch.device(DEVICE)
    tok.eval()

    t0 = time.time()
    written = 0
    with torch.no_grad():
        for batch in dl:
            frames = batch["frames"].to(device)
            mask = batch["frame_mask"].to(device)
            z = tok.encode(frames, mask)
            _, idx, _ = tok.quantize(z)
            idx_cpu = idx.cpu().numpy().astype(np.int32)
            for i, uid in enumerate(batch["uids"]):
                T = int(mask[i].sum().item())
                np.savez_compressed(FRAME_CODES_DIR / f"{uid}.npz", idx=idx_cpu[i, :T])
                written += 1
    print(f"  extracted {written} files in {time.time()-t0:.1f}s")
    sample = next(FRAME_CODES_DIR.glob("*.npz"))
    arr = np.load(sample)["idx"]
    print(f"  sample: {sample.name}  shape={arr.shape}  dtype={arr.dtype}  range=[{arr.min()}, {arr.max()}]")


def step3_renderer():
    banner("STEP 3 — renderer + style joint training")
    ds = V10Dataset(
        max_frames=600, knob_source="emotion",
        frame_codes_dir=str(FRAME_CODES_DIR), preload=False,
    )
    print(f"  dataset (with codes): {len(ds)} utterances")
    if len(ds) == 0:
        raise RuntimeError("no utts have frame codes — extraction step must have failed")
    n = min(N_UTTS, len(ds))
    sub = Subset(ds, list(range(n)))
    dl = DataLoader(sub, batch_size=4, shuffle=True, collate_fn=collate_v10_renderer)

    device = torch.device(DEVICE)
    style_enc = V10StyleEncoder(codebook_size=64).to(device)
    renderer = V10Renderer(
        codebook_size=256, num_quantizers=4, style_codebook_size=64,
        d_model=128, num_encoder_layers=2, num_decoder_layers=3,
        knob_dim=ds.knob_dim, knob_dropout=0.3,
        max_phonemes=204, max_frames=620,
    ).to(device)
    print(f"  style+renderer params: "
          f"{(sum(p.numel() for p in style_enc.parameters()) + sum(p.numel() for p in renderer.parameters()))/1e6:.2f}M")

    params = list(style_enc.parameters()) + list(renderer.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)

    level_w = [1.0, 0.7, 0.5, 0.4]

    t0 = time.time()
    step = 0
    while step < REN_STEPS:
        for batch in dl:
            for k in ("phoneme_ids", "phoneme_mask", "spk_emb", "knobs", "frames",
                      "frame_mask", "frame_to_enc_pos", "eop", "frame_codes",
                      "body_durations"):
                batch[k] = batch[k].to(device)
            n_total = batch["phoneme_ids"].shape[1]
            style_out = style_enc(batch["frames"], batch["frame_mask"],
                                  batch["frame_to_enc_pos"], n_total=n_total)
            render_out = renderer(
                batch["phoneme_ids"], style_out["codes"], batch["spk_emb"],
                batch["knobs"], batch["phoneme_mask"],
                batch["frame_codes"], batch["frame_to_enc_pos"], batch["frame_mask"],
            )
            losses = compute_losses(style_out, render_out, batch, 4, level_w)
            total = losses["ce_total"] + 0.5 * losses["eop_loss"] + 0.25 * losses["commit"]

            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            step += 1
            if step % 5 == 0:
                lvl = " ".join(f"{c.item():.2f}" for c in losses["ce_per_level"])
                print(f"  step {step}/{REN_STEPS}  ce_tot={losses['ce_total'].item():.3f} "
                      f"[{lvl}]  eop={losses['eop_loss'].item():.3f}  "
                      f"commit={float(losses['commit'].item()):.4f}  "
                      f"pos_w={losses['pos_weight']:.2f}")
            if step >= REN_STEPS:
                break
    print(f"  renderer done in {time.time()-t0:.1f}s")


def main():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)

    torch.manual_seed(0)
    tok, ds = step1_tokenizer()
    step2_extract(tok, ds)
    step3_renderer()
    banner("ALL STEPS PASSED")
    print(f"  artifacts at {TMP_DIR} (you can delete)")


if __name__ == "__main__":
    main()
