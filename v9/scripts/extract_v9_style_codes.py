"""
After stage-1 (renderer + style encoder + codebook) trains, extract per-phoneme
style codes for every utterance and save to v9/data/style_codes/<uid>.npz.

These codes are the prediction target for the stage-2 planner.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.style_encoder import V9PerPhonemeStyleEncoder, V9StyleCodebook
from v9.training.dataset_v9_renderer import V9RendererDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-checkpoint", default="v9/checkpoints/stage1_renderer/best.pt")
    ap.add_argument("--out-dir",           default="v9/data/style_codes")
    ap.add_argument("--device",            default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    c = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    a = c["args"]
    print(f"stage1: epoch={c['epoch']} val={c['val_loss']:.4f}")

    enc = V9PerPhonemeStyleEncoder(
        vocab_size=a["vocab_size"], input_dim=14,
        hidden_dim=a["style_hidden"], latent_dim=a["style_latent"],
    ).to(device)
    enc.load_state_dict(c["style_encoder"]); enc.eval()
    cb = V9StyleCodebook(
        codebook_size=a["style_codebook_size"], latent_dim=a["style_latent"],
        decay=a["ema_decay"], commitment_weight=a["commit_weight"],
    ).to(device)
    cb.load_state_dict(c["style_codebook"]); cb.eval()

    # Use the renderer dataset for loading frames per phoneme
    dataset = V9RendererDataset(
        tokens_dir=a["tokens_dir"],
        features_dir=a["features_dir"],
        norm_stats_path=a["norm_stats"],
        knob_source="none",  # not needed for extraction
        max_phonemes=a["max_phonemes"],
        f_pad_per_phoneme=a["f_pad_phoneme"],
        preload=False,
    )

    saved = 0
    pbar = tqdm(dataset.utt_ids, desc="extracting style codes")
    for uid in pbar:
        out_path = out_dir / f"{uid}.npz"
        if out_path.exists():
            saved += 1; continue
        item = dataset._load(uid)
        blocks = torch.from_numpy(item["phoneme_blocks"]).to(device)        # (N_body, F_PAD, 14)
        block_lens = torch.from_numpy(item["block_lens"]).to(device)
        ph_ids = torch.from_numpy(item["phoneme_ids"][1:1+len(item["durations"])]).to(device)

        F_PAD = blocks.shape[1]
        idx = torch.arange(F_PAD, device=device).unsqueeze(0)
        frame_mask = idx < block_lens.unsqueeze(-1)

        with torch.no_grad():
            z = enc(blocks, ph_ids, frame_mask)
            _, codes, _ = cb(z)
        codes_np = codes.cpu().numpy().astype(np.int32)

        np.savez_compressed(
            out_path,
            phoneme_ids=item["phoneme_ids"],
            style_codes=codes_np,
            durations=item["durations"].astype(np.int32),
            spk_emb=item["spk_emb"],
        )
        saved += 1
        if saved % 5000 == 0:
            pbar.set_postfix(saved=saved)
    print(f"\nDone. saved={saved}")


if __name__ == "__main__":
    main()
