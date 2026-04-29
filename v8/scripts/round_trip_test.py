"""
Round-trip test: GT features → phoneme anchors → linear interpolation → SPARC → audio.
Compare to: GT features → SPARC → audio (the unmodified ceiling).

If the interpolated audio sounds close to the unmodified audio, the v8 architecture
is feasible. If it sounds significantly worse, reconsider.

Outputs (per utterance):
  v8/outputs/roundtrip/<uid>_gt.wav             — direct GT → SPARC
  v8/outputs/roundtrip/<uid>_anchors_linear.wav — anchors → interp → SPARC
  v8/outputs/roundtrip/<uid>_anchors_3pt.wav    — start/mid/end → 2-segment interp → SPARC

Per-channel MSE between interpolated and GT features is also reported.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CHANS = ["TDx", "TDy", "TBx", "TBy", "TTx", "TTy",
         "ULx", "ULy", "LIx", "LIy", "LLx", "LLy",
         "logP", "Loud"]


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
    return feats, f["spk_emb"].astype(np.float32) if "spk_emb" in f.files else None


def linear_interp_2pt(start, end, durs):
    """(N, 14) start, (N, 14) end, (N,) durs → (T, 14) interpolated."""
    out = []
    for i, d in enumerate(durs):
        d = int(d)
        if d == 0:
            continue
        if d == 1:
            out.append((0.5 * (start[i] + end[i]))[None, :])     # (1, 14)
        else:
            t = np.linspace(0.0, 1.0, d)[:, None]
            out.append(start[i] * (1 - t) + end[i] * t)           # (d, 14)
    return np.concatenate(out, axis=0)


def interp_3pt(start, mid, end, durs):
    """3-anchor: split each phoneme into two halves, linearly interp each."""
    out = []
    for i, d in enumerate(durs):
        d = int(d)
        if d == 0:
            continue
        if d == 1:
            out.append(mid[i:i+1])                                # (1, 14)
        elif d == 2:
            out.append(np.stack([start[i], end[i]]))              # (2, 14)
        else:
            half = d // 2
            t1 = np.linspace(0.0, 1.0, half)[:, None]
            t2 = np.linspace(0.0, 1.0, d - half)[:, None]
            seg1 = start[i] * (1 - t1) + mid[i] * t1
            seg2 = mid[i]   * (1 - t2) + end[i] * t2
            out.append(np.concatenate([seg1, seg2], axis=0))
    return np.concatenate(out, axis=0)


def render_to_wav(feats_logp, spk_emb, sparc):
    feats = feats_logp.copy()
    feats[:, 12] = np.exp(feats[:, 12]) - 1.0   # log-pitch → Hz
    ema = feats[:, :12]
    pitch = feats[:, 12].copy()
    pitch[pitch < 30] = 0.0
    loud = feats[:, 13]
    wav = sparc.decode(ema, pitch, loud, spk_emb)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().squeeze().cpu().numpy()
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/features_merged_logpitch_v2")
    ap.add_argument("--alignments",   default="data/processed_merged_v3/alignments_mfa.json")
    ap.add_argument("--out-dir",      default="v8/outputs/roundtrip")
    ap.add_argument("--n-utts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    align_data = json.load(open(args.alignments))
    feat_dir = Path(args.features_dir)
    candidates = [u for u in align_data
                  if "-" in u
                  and 200 <= align_data[u]["total_frames"] <= 350
                  and (feat_dir / f"{u}.npz").exists()]

    import random
    random.seed(args.seed)
    samples = random.sample(candidates, min(args.n_utts, len(candidates)))

    print(f"\nRound-trip test on {len(samples)} utterances:\n")
    print(f"{'uid':<22} {'2-pt MSE':>10} {'3-pt MSE':>10} {'2-pt pitch':>12} {'3-pt pitch':>12}")
    print('-' * 70)

    for uid in samples:
        feats, spk_emb = load_features(feat_dir / f"{uid}.npz")
        durs = np.asarray(align_data[uid]["durations"], dtype=np.int64)
        T_total = int(durs.sum())
        feats = feats[:T_total]   # trim to alignment span

        # Extract anchors
        N = len(durs)
        start = np.zeros((N, 14), dtype=np.float32)
        end   = np.zeros((N, 14), dtype=np.float32)
        mid   = np.zeros((N, 14), dtype=np.float32)
        cursor = 0
        for i, d in enumerate(durs):
            d = int(d)
            if d == 0: continue
            start[i] = feats[cursor]
            end[i]   = feats[cursor + d - 1]
            mid[i]   = feats[cursor + d // 2]
            cursor += d

        # Linear 2-pt and 3-pt interpolation
        recon_2pt = linear_interp_2pt(start, end, durs)
        recon_3pt = interp_3pt(start, mid, end, durs)

        # Trim/pad to match
        T_use = min(len(feats), len(recon_2pt), len(recon_3pt))
        feats_u = feats[:T_use]
        recon_2pt = recon_2pt[:T_use]
        recon_3pt = recon_3pt[:T_use]

        # Per-channel MSE (overall + pitch only)
        mse_2pt = ((feats_u - recon_2pt) ** 2).mean()
        mse_3pt = ((feats_u - recon_3pt) ** 2).mean()
        pitch_mse_2pt = ((feats_u[:, 12] - recon_2pt[:, 12]) ** 2).mean()
        pitch_mse_3pt = ((feats_u[:, 12] - recon_3pt[:, 12]) ** 2).mean()

        print(f"{uid:<22} {mse_2pt:>10.5f} {mse_3pt:>10.5f} "
              f"{pitch_mse_2pt:>12.5f} {pitch_mse_3pt:>12.5f}")

        # Render audio
        wav_gt = render_to_wav(feats_u, spk_emb, sparc)
        wav_2pt = render_to_wav(recon_2pt, spk_emb, sparc)
        wav_3pt = render_to_wav(recon_3pt, spk_emb, sparc)
        sf.write(out_dir / f"{uid}_gt.wav", wav_gt, sparc.sr)
        sf.write(out_dir / f"{uid}_anchors_linear.wav", wav_2pt, sparc.sr)
        sf.write(out_dir / f"{uid}_anchors_3pt.wav", wav_3pt, sparc.sr)

    print(f"\nAudio saved to {out_dir}/")
    print("Listen to <uid>_gt.wav (ceiling) vs <uid>_anchors_linear.wav (2-pt) vs <uid>_anchors_3pt.wav (3-pt).")


if __name__ == "__main__":
    main()
