"""
Export stage 1's per-phoneme style codes for every training utterance.

Run after stage 1 converges. Loads the trained per-phoneme style encoder +
style codebook, runs every utterance's features through them with the GT
durations, and saves the resulting integer code sequence to disk.

These per-utterance code sequences are the TARGETS for stage 2 planner
training. The planner learns: predict these codes from (text, knobs).

Output: data/style_codes_v5/<utt_id>.npy — int64 array of shape (N,) where
N is the utterance's phoneme count (matches what the dataset uses for
phoneme_ids minus BOS/EOS).

~1-2h on MPS for 112K utterances.
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.per_phoneme_style_encoder import PerPhonemeStyleEncoder
from models.style_codebook import StyleCodebook, PAD_CODE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-checkpoint", default="checkpoints_v5_stage1/transformer_best.pt")
    p.add_argument("--features-dir",      default="data/features_merged_logpitch_v2")
    p.add_argument("--phonemes-path",     default="data/processed_merged_v3/phonemes_mfa.json")
    p.add_argument("--alignments-path",   default="data/processed_merged_v3/alignments_mfa.json")
    p.add_argument("--output-dir",        default="data/style_codes_v5")
    p.add_argument("--device",            default="mps")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load stage 1 modules ────────────────────────────
    s1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=True)
    sa = s1["args"]
    print(f"Loaded stage1: epoch={s1['epoch']}, val_loss={s1['val_loss']:.4f}")

    style_encoder = PerPhonemeStyleEncoder(
        input_dim=14, style_dim=sa["d_model"], hidden=128, n_conv_layers=3,
    ).to(device)
    style_encoder.load_state_dict(s1["style_encoder_state_dict"])
    style_encoder.eval()

    style_codebook = StyleCodebook(
        latent_dim=sa["d_model"], codebook_size=sa["style_codebook_size"],
    ).to(device)
    style_codebook.load_state_dict(s1["style_codebook_state_dict"])
    style_codebook.eval()

    # ─── Iterate all utterances ─────────────────────────
    phons_data = json.load(open(args.phonemes_path))
    align_data = json.load(open(args.alignments_path))
    feat_dir = Path(args.features_dir)

    utt_ids = [u for u in phons_data
               if u in align_data and (feat_dir / f"{u}.npz").exists()]
    print(f"{len(utt_ids)} utterances to encode")

    code_usage = np.zeros(sa["style_codebook_size"], dtype=np.int64)
    skipped = 0
    written = 0

    for utt_id in tqdm(utt_ids, desc="Exporting codes"):
        out_path = out_dir / f"{utt_id}.npy"
        if out_path.exists():
            # Skip already-done (idempotent re-runs)
            written += 1
            continue
        try:
            f = np.load(feat_dir / f"{utt_id}.npz")
            T_feat = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
            features = np.concatenate([
                f["ema"][:T_feat], f["pitch"][:T_feat, None], f["loudness"][:T_feat, None],
            ], axis=1).astype(np.float32)

            durations = align_data[utt_id]["durations"]   # per-phoneme frame counts
            # Adjust durations to sum to exactly T_feat (same logic as dataset_rvq)
            total_dur = sum(durations)
            if total_dur != T_feat:
                diff = T_feat - total_dur
                if abs(diff) > 5:
                    skipped += 1; continue
                # adjust last positive duration
                for i in range(len(durations) - 1, -1, -1):
                    if durations[i] + diff >= 1:
                        durations[i] += diff; break
            durs = np.asarray(durations, dtype=np.int64)
        except Exception as e:
            skipped += 1
            continue

        feat_t = torch.from_numpy(features).unsqueeze(0).to(device)
        durs_t = torch.from_numpy(durs).unsqueeze(0).to(device)
        mask_t = torch.ones_like(durs_t, dtype=torch.bool)

        with torch.no_grad():
            z = style_encoder(feat_t, durs_t.float(), mask_t)
            _, codes, _ = style_codebook(z, mask_t)
        codes_np = codes.squeeze(0).cpu().numpy().astype(np.int64)

        # Sanity-check: PAD_CODE shouldn't appear here (no real-utterance phoneme is padded)
        if (codes_np == PAD_CODE).any():
            print(f"WARN {utt_id}: PAD_CODE in output — unexpected")

        np.save(out_path, codes_np)

        # Track usage
        valid = codes_np[codes_np >= 0]
        np.add.at(code_usage, valid, 1)
        written += 1

    print(f"\nDone: {written}/{len(utt_ids)} written, {skipped} skipped")
    print(f"Codebook usage: {(code_usage > 0).sum()}/{sa['style_codebook_size']} active "
          f"({(code_usage > 0).mean()*100:.0f}%)")
    if (code_usage > 0).sum() > 0:
        nz = code_usage[code_usage > 0]
        nz_p = nz / nz.sum()
        print(f"Effective perplexity: {np.exp(-(nz_p * np.log(nz_p)).sum()):.1f}")


if __name__ == "__main__":
    main()
