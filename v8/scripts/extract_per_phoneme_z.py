"""
Extract per-phoneme continuous z vectors from GT audio using v5's frozen style encoder.

For each utterance:
  GT features (T, 14) + GT durations (N,) → v5 PerPhonemeStyleEncoder → z (N, 256)

Save z to v8/data/phoneme_z/<uid>.npy (body-only, before BOS/EOS padding).

These z's serve as explicit per-phoneme targets for v8's planner — forcing it to
learn per-phoneme variation instead of collapsing to mean.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.per_phoneme_style_encoder import PerPhonemeStyleEncoder


def load_features(path):
    f = np.load(path)
    T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
    pitch = f["pitch"][:T].astype(np.float32)
    if pitch.max() > 20:
        pitch = np.log(pitch + 1.0)
    feats = np.concatenate([
        f["ema"][:T].astype(np.float32),
        pitch[:, None],
        f["loudness"][:T, None].astype(np.float32),
    ], axis=1)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-checkpoint",
                    default="checkpoints_v5_stage1_archived/transformer_best.pt")
    ap.add_argument("--features-dir", default="data/features_merged_logpitch_v2")
    ap.add_argument("--alignments",   default="data/processed_merged_v3/alignments_mfa.json")
    ap.add_argument("--phonemes",     default="data/processed_merged_v3/phonemes_mfa.json")
    ap.add_argument("--out-dir",      default="v8/data/phoneme_z")
    ap.add_argument("--device",       default="cpu")
    ap.add_argument("--max-utts", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load v5 style encoder (frozen)
    s1 = torch.load(args.v5_checkpoint, map_location=device, weights_only=True)
    sa = s1["args"]
    style_encoder = PerPhonemeStyleEncoder(
        input_dim=14, style_dim=sa["d_model"], hidden=128, n_conv_layers=3,
    ).to(device)
    style_encoder.load_state_dict(s1["style_encoder_state_dict"])
    style_encoder.eval()
    for p in style_encoder.parameters():
        p.requires_grad = False
    print(f"v5 style encoder loaded (epoch={s1['epoch']}, val={s1['val_loss']:.4f}, d_model={sa['d_model']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    align_data = json.load(open(args.alignments))
    phons_data = json.load(open(args.phonemes))
    feat_dir = Path(args.features_dir)

    utt_ids = sorted([u for u in phons_data
                      if u in align_data and (feat_dir / f"{u}.npz").exists()])
    if args.max_utts:
        utt_ids = utt_ids[:args.max_utts]
    print(f"Processing {len(utt_ids)} utterances → {out_dir}")

    skipped = 0
    written = 0
    with torch.no_grad():
        for uid in tqdm(utt_ids, desc="extracting z"):
            try:
                feats = load_features(feat_dir / f"{uid}.npz")
                durs = np.asarray(align_data[uid]["durations"], dtype=np.int64)
                phon_strs = phons_data[uid]["phonemes"]
                if len(durs) != len(phon_strs):
                    skipped += 1; continue
                if durs.sum() > len(feats):
                    skipped += 1; continue

                feat_t = torch.from_numpy(feats[:durs.sum()].astype(np.float32)).unsqueeze(0).to(device)
                durs_t = torch.from_numpy(durs).unsqueeze(0).to(device)
                mask_t = torch.ones_like(durs_t, dtype=torch.bool)
                z = style_encoder(feat_t, durs_t.float(), mask_t).squeeze(0)   # (N, D)
                np.save(out_dir / f"{uid}.npy", z.cpu().numpy().astype(np.float32))
                written += 1
            except Exception:
                skipped += 1

    print(f"\nDone: wrote {written}, skipped {skipped}")


if __name__ == "__main__":
    main()
