"""Run trained v10 tokenizer over all utterances. Save (T, K) frame token IDs."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from v10.models.v10_tokenizer import V10Tokenizer
from v10.training.dataset_v10 import V10Dataset, collate_v10_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="v10/data/frame_codes")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=800)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ckpt.get("args", {})
    print(f"loaded checkpoint args: d_model={targs.get('d_model')} "
          f"K={targs.get('num_quantizers')} cb={targs.get('codebook_size')}")

    model = V10Tokenizer(
        d_model=targs.get("d_model", 256),
        num_encoder_layers=targs.get("enc_layers", 4),
        num_decoder_layers=targs.get("dec_layers", 4),
        codebook_size=targs.get("codebook_size", 1024),
        num_quantizers=targs.get("num_quantizers", 4),
        max_frames=targs.get("max_frames", args.max_frames) + 16,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = V10Dataset(max_frames=args.max_frames, preload=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_v10_tokenizer)
    print(f"extracting from {len(ds)} utterances")

    written = 0
    skipped = 0
    with torch.no_grad():
        for batch in dl:
            uids = batch["uids"]
            # Skip rows already extracted
            todo = [(i, u) for i, u in enumerate(uids) if not (out_dir / f"{u}.npz").exists()]
            if not todo:
                skipped += len(uids)
                continue

            frames = batch["frames"].to(device)
            mask = batch["frame_mask"].to(device)
            z = model.encode(frames, mask)
            _, idx, _ = model.quantize(z)            # (B, T_max, K)

            idx_cpu = idx.cpu().numpy().astype(np.int32)
            for i, uid in todo:
                T = int(mask[i].sum().item())
                np.savez_compressed(out_dir / f"{uid}.npz", idx=idx_cpu[i, :T])
                written += 1

            if (written + skipped) % 1000 == 0:
                print(f"  {written + skipped} processed (wrote={written}, skipped={skipped})")

    print(f"done — wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
