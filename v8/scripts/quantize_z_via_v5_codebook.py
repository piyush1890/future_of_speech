"""
Quantize the per-phoneme continuous z's (already extracted) through v5's frozen
StyleCodebook. Save:
  - code_id (N,) int — index into 512-entry codebook
  - z_q (N, 256) float — codebook-embedded z (quantized)

These two outputs feed two separate downstream uses:
  - z_q: stage-1 training input (replacing continuous z; ensures train/inference match)
  - code_id: stage-2 (AR codebook planner) target via CE loss

Output dir: v8/data/phoneme_codes/<uid>.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.style_codebook import StyleCodebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-checkpoint",
                    default="checkpoints_v5_stage1_archived/transformer_best.pt")
    ap.add_argument("--z-dir",       default="v8/data/phoneme_z")
    ap.add_argument("--out-dir",     default="v8/data/phoneme_codes")
    ap.add_argument("--device",      default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    s1 = torch.load(args.v5_checkpoint, map_location=device, weights_only=False)
    sa = s1["args"]
    codebook = StyleCodebook(
        latent_dim=sa["d_model"],
        codebook_size=sa["style_codebook_size"],
    ).to(device)
    codebook.load_state_dict(s1["style_codebook_state_dict"])
    codebook.eval()
    for p in codebook.parameters():
        p.requires_grad = False
    print(f"Loaded v5 codebook: {sa['style_codebook_size']} entries × {sa['d_model']} dims")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_dir = Path(args.z_dir)
    z_files = sorted(z_dir.glob("*.npy"))
    print(f"Quantizing {len(z_files)} z files → {out_dir}")

    written = 0
    skipped = 0
    with torch.no_grad():
        for p in tqdm(z_files, desc="quantizing z"):
            try:
                z = np.load(p).astype(np.float32)        # (N, 256)
                z_t = torch.from_numpy(z).unsqueeze(0).to(device)
                mask = torch.ones((1, z.shape[0]), dtype=torch.bool, device=device)
                q, codes, _ = codebook(z_t, mask)        # q: (1, N, 256), codes: (1, N) long
                np.savez_compressed(
                    out_dir / f"{p.stem}.npz",
                    code_id=codes.squeeze(0).cpu().numpy().astype(np.int64),
                    z_q=q.squeeze(0).cpu().numpy().astype(np.float32),
                )
                written += 1
            except Exception:
                skipped += 1
    print(f"Done: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
