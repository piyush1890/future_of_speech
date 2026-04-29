"""
v9 token extraction (one-time, post-tokenizer-training).

For each utterance in the dataset, run the trained tokenizer encoder over
every body phoneme block to get (start_idx, end_idx) RVQ tokens. Save to
v9/data/phoneme_tokens/<uid>.npz with start_idx (N, 4), end_idx (N, 4),
durations (N,), phoneme_ids (N+2,), spk_emb (64,).

Predictor training reads these GT tokens — it never re-encodes from frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.phoneme_rvq import PhonemeRVQTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-checkpoint", default="v9/checkpoints/phoneme_rvq/best.pt")
    ap.add_argument("--features-dir",         default="data/features_merged_logpitch_v2")
    ap.add_argument("--phonemes-path",        default="data/processed_merged_v3/phonemes_mfa.json")
    ap.add_argument("--alignments-path",      default="data/processed_merged_v3/alignments_mfa.json")
    ap.add_argument("--spk-emb-dir",          default="v8/data/phoneme_anchors")
    ap.add_argument("--norm-stats",           default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--out-dir",              default="v9/data/phoneme_tokens")
    ap.add_argument("--device",               default="mps")
    ap.add_argument("--max-phonemes",         type=int, default=200)
    ap.add_argument("--f-pad",                type=int, default=32)
    ap.add_argument("--b-pad",                type=int, default=256,
                    help="Phonemes per forward batch when extracting (just for speed)")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load tokenizer
    c = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    a = c["args"]
    print(f"Tokenizer: epoch={c['epoch']} val_loss={c['val_loss']:.4f}")
    model = PhonemeRVQTokenizer(
        vocab_size=a["vocab_size"], input_dim=14,
        latent_dim=a["latent_dim"], hidden_dim=a["hidden_dim"],
        codebook_size=a["codebook_size"], num_quantizers=a["num_quantizers"],
        decoder_d_model=a["decoder_d_model"], decoder_nhead=a["decoder_nhead"],
        decoder_layers=a["decoder_layers"],
        commitment_weight=a["commit_weight"], ema_decay=a["ema_decay"],
    ).to(device)
    model.load_state_dict(c["model"]); model.eval()

    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32)
    feat_std  = stats["std"].astype(np.float32)

    phon_data  = json.load(open(args.phonemes_path))
    align_data = json.load(open(args.alignments_path))

    feats_dir   = Path(args.features_dir)
    spk_dir     = Path(args.spk_emb_dir)
    out_dir     = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    F_PAD       = args.f_pad

    uids = sorted(phon_data.keys())
    skipped = saved = 0
    K = a["num_quantizers"]

    pbar = tqdm(uids, desc="extracting")
    for uid in pbar:
        out_path = out_dir / f"{uid}.npz"
        if out_path.exists():
            saved += 1; continue
        if uid not in align_data:
            skipped += 1; continue
        phoneme_ids = np.asarray(phon_data[uid]["indices"], dtype=np.int64)   # BOS+body+EOS
        n_full = len(phoneme_ids); n_body = n_full - 2
        if not (4 < n_full <= args.max_phonemes + 2):
            skipped += 1; continue
        feats_p = feats_dir  / f"{uid}.npz"
        spk_p   = spk_dir    / f"{uid}.npz"
        if not (feats_p.exists() and spk_p.exists()):
            skipped += 1; continue

        body_durs = np.asarray(align_data[uid]["durations"], dtype=np.int64)
        if len(body_durs) != n_body:
            skipped += 1; continue
        body_T = int(body_durs.sum())

        f = np.load(feats_p, allow_pickle=False)
        T_full = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
        if T_full < body_T:
            running = 0
            new_durs = body_durs.copy()
            for i, d in enumerate(body_durs):
                if running + d > T_full:
                    new_durs[i] = max(0, T_full - running); new_durs[i+1:] = 0; break
                running += d
            body_durs = new_durs; body_T = int(body_durs.sum())
        feats = np.concatenate([
            f["ema"][:body_T].astype(np.float32),
            f["pitch"][:body_T, None].astype(np.float32),
            f["loudness"][:body_T, None].astype(np.float32),
        ], axis=1)
        feats_norm = (feats - feat_mean) / (feat_std + 1e-8)
        # Sentinel: BOS at -1, EOS after body — both zeros (matches training dataset)
        bos = np.zeros((1, 14), dtype=np.float32)
        eos = np.zeros((1, 14), dtype=np.float32)
        full_feats = np.concatenate([bos, feats_norm, eos], axis=0)
        # Offsets are into full_feats; body phoneme i covers [1+cum[i], 1+cum[i+1])
        cum = np.concatenate([[0], np.cumsum(body_durs)])

        # Each phoneme block extended by 1 context frame on each side
        F_PAD_EXT = F_PAD + 2
        blocks, ph_ids, lens = [], [], []
        for p_idx in range(n_body):
            s = 1 + int(cum[p_idx]); e = 1 + int(cum[p_idx + 1])
            if e <= s:
                blocks.append(np.zeros((F_PAD_EXT, 14), dtype=np.float32))
                ph_ids.append(int(phoneme_ids[1 + p_idx]))
                lens.append(2)                                        # 0 body + 2 ctx
                continue
            L_orig = min(e - s, F_PAD)
            ext = np.zeros((F_PAD_EXT, 14), dtype=np.float32)
            ext[0] = full_feats[s - 1]
            ext[1:1 + L_orig] = full_feats[s:s + L_orig]
            ext[1 + L_orig] = full_feats[s + L_orig]                  # next context (or 0 if EOS)
            blocks.append(ext)
            ph_ids.append(int(phoneme_ids[1 + p_idx]))
            lens.append(L_orig + 2)

        B = len(blocks)
        padded = np.stack(blocks)                                     # (B, F_PAD_EXT, 14)
        frames_t  = torch.from_numpy(padded).to(device)
        ph_ids_t  = torch.tensor(ph_ids, dtype=torch.long, device=device)
        lens_t    = torch.tensor(lens, dtype=torch.long, device=device)

        # Encode (no gradient; no decode needed)
        with torch.no_grad():
            B_total, F_max, _ = frames_t.shape
            start_mask, end_mask, _ = model._build_masks(lens_t, F_max)
            ph_emb_enc = model.phoneme_embedding(ph_ids_t)
            start_z = model.start_encoder(frames_t, ph_emb_enc, start_mask)
            end_z   = model.end_encoder(frames_t, ph_emb_enc, end_mask)
            _, s_idx, _ = model.rvq_start(start_z.unsqueeze(1))
            _, e_idx, _ = model.rvq_end(end_z.unsqueeze(1))
            s_idx = s_idx.squeeze(1)        # (B, K)
            e_idx = e_idx.squeeze(1)

        spk_emb = np.load(spk_p, allow_pickle=False)["spk_emb"].astype(np.float32)
        np.savez_compressed(
            out_path,
            phoneme_ids=phoneme_ids,                                        # (n_full,)
            start_idx=s_idx.cpu().numpy().astype(np.int32),                 # (n_body, K)
            end_idx=e_idx.cpu().numpy().astype(np.int32),                   # (n_body, K)
            durations=body_durs.astype(np.int32),                           # (n_body,)
            spk_emb=spk_emb,                                                # (64,)
        )
        saved += 1
        if saved % 5000 == 0:
            pbar.set_postfix(saved=saved, skipped=skipped)

    print(f"\nDone. saved={saved}  skipped={skipped}")


if __name__ == "__main__":
    main()
