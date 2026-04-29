"""
Evaluate anchor quality: GT features → extract anchors → hybrid render → measure MSE.

Reports per-channel reconstruction error (the upper bound — best case if model
predicts GT anchors exactly). Compares hybrid render vs naive 3-pt linear.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v8.models.phoneme_classes import build_render_class_table, PLATEAU, LINEAR
from v8.models.interpolator import hmm_render_3pt, linear_interpolate_3pt, hybrid_render_3pt
from models.phoneme_vocab import PhonemeVocab

CHANS = ["TDx","TDy","TBx","TBy","TTx","TTy","ULx","ULy","LIx","LIy","LLx","LLy","logP","Loud"]


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


def extract(feats, durs):
    N = len(durs)
    start = np.zeros((N, 14), dtype=np.float32)
    mid   = np.zeros((N, 14), dtype=np.float32)
    end   = np.zeros((N, 14), dtype=np.float32)
    cursor = 0
    for i, d in enumerate(durs):
        d = int(d)
        if d == 0:
            idx = min(cursor, len(feats)-1)
            start[i] = feats[idx]; mid[i] = feats[idx]; end[i] = feats[idx]
        else:
            start[i] = feats[cursor]
            mid[i]   = feats[min(cursor + d//2, len(feats)-1)]
            end[i]   = feats[min(cursor + d - 1, len(feats)-1)]
            cursor += d
    return start, mid, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utts", type=int, default=200)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--vocab",  default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--phons",  default="data/processed_merged_v3/phonemes_mfa.json")
    ap.add_argument("--aligns", default="data/processed_merged_v3/alignments_mfa.json")
    ap.add_argument("--feats-dir", default="data/features_merged_logpitch_v2")
    args = ap.parse_args()

    random.seed(args.seed)
    vocab = PhonemeVocab(args.vocab)
    rct = build_render_class_table(vocab)

    phons = json.load(open(args.phons))
    aligns = json.load(open(args.aligns))
    feat_dir = Path(args.feats_dir)
    cands = [u for u in phons if u in aligns and "-" in u
             and 200 <= aligns[u]["total_frames"] <= 400
             and (feat_dir / f"{u}.npz").exists()]
    samples = random.sample(cands, min(args.n_utts, len(cands)))

    err_hybrid = []
    err_linear = []
    err_hmm    = []

    for uid in samples:
        feats = load_features(feat_dir / f"{uid}.npz")
        durs = np.asarray(aligns[uid]["durations"], dtype=np.int64)
        T_total = int(durs.sum())
        feats = feats[:T_total]
        if len(durs) != len(phons[uid]["phonemes"]):
            continue

        start, mid, end = extract(feats, durs)

        # Build phoneme_ids body-only and look up render class
        ids_body = np.asarray([vocab.token2idx.get(p, vocab.pad_idx) for p in phons[uid]["phonemes"]],
                              dtype=np.int64)
        render_cls_body = rct[torch.from_numpy(ids_body)].numpy()

        s_t = torch.from_numpy(start).unsqueeze(0)
        m_t = torch.from_numpy(mid).unsqueeze(0)
        e_t = torch.from_numpy(end).unsqueeze(0)
        d_t = torch.from_numpy(durs).unsqueeze(0)
        rc_t = torch.from_numpy(render_cls_body).unsqueeze(0)

        # Render in three modes
        recon_h, _ = hybrid_render_3pt(s_t, m_t, e_t, d_t, rc_t)
        recon_h = recon_h.squeeze(0).numpy()
        recon_l, _ = linear_interpolate_3pt(s_t, m_t, e_t, d_t)
        recon_l = recon_l.squeeze(0).numpy()
        recon_hmm, _ = hmm_render_3pt(s_t, m_t, e_t, d_t)
        recon_hmm = recon_hmm.squeeze(0).numpy()

        T = min(len(feats), len(recon_h), len(recon_l), len(recon_hmm))
        feats_u = feats[:T]
        recon_h = recon_h[:T]
        recon_l = recon_l[:T]
        recon_hmm = recon_hmm[:T]

        err_hybrid.append((feats_u - recon_h) ** 2)
        err_linear.append((feats_u - recon_l) ** 2)
        err_hmm.append((feats_u - recon_hmm) ** 2)

    err_h = np.concatenate(err_hybrid, axis=0)   # (Σ T, 14)
    err_l = np.concatenate(err_linear, axis=0)
    err_hmm = np.concatenate(err_hmm, axis=0)

    print(f"Per-channel MSE (lower = better reconstruction)")
    print(f"  Sampled {len(samples)} utterances, {err_h.shape[0]:,} frames\n")
    print(f"{'chan':>5}  {'hybrid':>10}  {'linear':>10}  {'hmm-only':>10}")
    print('-' * 45)
    for c in range(14):
        print(f"{CHANS[c]:>5}  {err_h[:, c].mean():>10.5f}  {err_l[:, c].mean():>10.5f}  {err_hmm[:, c].mean():>10.5f}")

    print(f"\n{'overall MSE':>15}  hybrid={err_h.mean():.5f}  linear={err_l.mean():.5f}  hmm-only={err_hmm.mean():.5f}")

    # Stats on render-class distribution in this sample
    plateau_count = 0
    linear_count = 0
    for uid in samples:
        ids_body = [vocab.token2idx.get(p, vocab.pad_idx) for p in phons[uid]["phonemes"]]
        for i in ids_body:
            cls = rct[i].item()
            if cls == PLATEAU: plateau_count += 1
            elif cls == LINEAR: linear_count += 1
    total = plateau_count + linear_count
    print(f"\nPhoneme distribution: PLATEAU={plateau_count} ({plateau_count/total:.1%})  "
          f"LINEAR={linear_count} ({linear_count/total:.1%})")


if __name__ == "__main__":
    main()
