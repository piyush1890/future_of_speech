"""Run trained v10 style encoder over all utterances. Save (N+2,) per-phoneme codes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from v10.models.v10_style import V10StyleEncoder
from v10.training.dataset_v10 import V10Dataset, collate_v10_renderer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--checkpoint", required=True,
                   help="Stage 1 checkpoint with style_enc state_dict")
    p.add_argument("--out-dir", default="v10/data/style_codes")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=800)
    p.add_argument("--max-phonemes", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    print(f"loaded checkpoint args: style_codebook_size={targs.get('style_codebook_size')}")

    style_enc = V10StyleEncoder(
        codebook_size=targs.get("style_codebook_size", 64),
    ).to(device)
    style_enc.load_state_dict(ckpt["style_enc"])
    style_enc.eval()

    ds = V10Dataset(
        max_frames=args.max_frames, max_phonemes=args.max_phonemes,
        knob_source=targs.get("knob_source", "none"), preload=False,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_v10_renderer)
    print(f"extracting style codes from {len(ds)} utterances")

    written = 0
    skipped = 0
    with torch.no_grad():
        for batch in dl:
            uids = batch["uids"]
            todo = [(i, u) for i, u in enumerate(uids) if not (out_dir / f"{u}.npz").exists()]
            if not todo:
                skipped += len(uids)
                continue

            frames = batch["frames"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            frame_to_enc_pos = batch["frame_to_enc_pos"].to(device)
            n_total = batch["phoneme_ids"].shape[1]

            out = style_enc(frames, frame_mask, frame_to_enc_pos, n_total=n_total)
            codes = out["codes"].cpu().numpy().astype(np.int32)         # (B, N+2)

            phoneme_mask = batch["phoneme_mask"].numpy()
            for i, uid in todo:
                # Trim to actual N+2 for this row
                n_valid = int(phoneme_mask[i].sum())
                np.savez_compressed(out_dir / f"{uid}.npz", codes=codes[i, :n_valid])
                written += 1

            if (written + skipped) % 1000 == 0:
                print(f"  {written + skipped} processed (wrote={written}, skipped={skipped})")

    print(f"done — wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
