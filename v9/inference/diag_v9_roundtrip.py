"""
v9 tokenizer round-trip on real emotion utterances:
  GT frames → tokenizer.encode → tokens → tokenizer.decode → recon frames → SPARC → audio

Compares against a direct GT-frames → SPARC render (the upper bound).
Listening test answers: does v9's quantization preserve emotion?
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.phoneme_rvq import PhonemeRVQTokenizer


CKPT         = "v9/checkpoints/phoneme_rvq/best.pt"
PHONEMES     = "data/processed_merged_v3/phonemes_mfa.json"
ALIGNMENTS   = "data/processed_merged_v3/alignments_mfa.json"
FEATS_DIR    = Path("data/features_merged_logpitch_v2")
ANCHORS_DIR  = Path("v8/data/phoneme_anchors")
NORM_STATS   = "data/features_merged_logpitch_v2/norm_stats.npz"
OUT_DIR      = Path("v9/outputs/diag_v9_roundtrip")

TARGETS = {
    "angry":   "0011_Angry_0011_000564",
    "happy":   "0011_Happy_0011_000936",
    "sad":     "0011_Sad_0011_001286",
    "neutral": "0011_Neutral_0011_000313",
}


def main():
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ──
    c = torch.load(CKPT, map_location=device, weights_only=False)
    a = c["args"]
    print(f"v9 tokenizer: epoch={c['epoch']} val_loss={c['val_loss']:.4f}")
    model = PhonemeRVQTokenizer(
        vocab_size=a["vocab_size"], input_dim=14,
        latent_dim=a["latent_dim"], hidden_dim=a["hidden_dim"],
        codebook_size=a["codebook_size"], num_quantizers=a["num_quantizers"],
        decoder_d_model=a["decoder_d_model"], decoder_nhead=a["decoder_nhead"],
        decoder_layers=a["decoder_layers"],
        commitment_weight=a["commit_weight"], ema_decay=a["ema_decay"],
    ).to(device)
    model.load_state_dict(c["model"]); model.eval()

    stats = np.load(NORM_STATS)
    feat_mean = stats["mean"].astype(np.float32)
    feat_std  = stats["std"].astype(np.float32)

    phon_data = json.load(open(PHONEMES))
    align_data = json.load(open(ALIGNMENTS))

    print("Loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    for label, uid in TARGETS.items():
        feats_path = FEATS_DIR / f"{uid}.npz"
        anch_path  = ANCHORS_DIR / f"{uid}.npz"
        if uid not in phon_data or uid not in align_data: print(f"skip {label}"); continue
        if not (feats_path.exists() and anch_path.exists()): print(f"skip {label}"); continue

        # Load + normalize GT features
        f = np.load(feats_path, allow_pickle=False)
        T_full = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
        gt_feats = np.concatenate([
            f["ema"][:T_full].astype(np.float32),
            f["pitch"][:T_full, None].astype(np.float32),
            f["loudness"][:T_full, None].astype(np.float32),
        ], axis=1)                                                  # (T, 14)
        gt_norm = (gt_feats - feat_mean) / (feat_std + 1e-8)

        # Phonemes + body durations
        phoneme_ids = np.asarray(phon_data[uid]["indices"], dtype=np.int64)  # BOS + body + EOS
        body_durs = np.asarray(align_data[uid]["durations"], dtype=np.int64)
        n_body = len(body_durs)
        body_T = int(body_durs.sum())
        gt_norm = gt_norm[:body_T]                                    # match alignment

        # Slice body frames into per-phoneme blocks
        offsets = np.concatenate([[0], np.cumsum(body_durs)])
        blocks, ph_ids, lens = [], [], []
        for p_idx in range(n_body):
            s = int(offsets[p_idx]); e = int(offsets[p_idx + 1])
            if e <= s: continue
            blocks.append(gt_norm[s:e])
            ph_ids.append(int(phoneme_ids[1 + p_idx]))   # +1 to skip BOS in phoneme_ids
            lens.append(e - s)

        # Pack into batch (truncate at F_PAD=32 to match training)
        F_PAD = a.get("f_pad", 32)
        B = len(blocks)
        padded = np.zeros((B, F_PAD, 14), dtype=np.float32)
        clipped_lens = []
        for i, b in enumerate(blocks):
            L = min(len(b), F_PAD)
            padded[i, :L] = b[:L]
            clipped_lens.append(L)
        frames_t  = torch.from_numpy(padded).to(device)
        ph_ids_t  = torch.tensor(ph_ids, dtype=torch.long, device=device)
        lengths_t = torch.tensor(clipped_lens, dtype=torch.long, device=device)

        with torch.no_grad():
            recon, info = model.forward_batch(frames_t, ph_ids_t, lengths_t)
        # Slice each phoneme to its real length and concatenate
        recon_chunks = [recon[i, :clipped_lens[i]].cpu().numpy() for i in range(B)]
        # If any phoneme was truncated, pad recon with zero-residual to match GT length
        for i, b in enumerate(blocks):
            if clipped_lens[i] < len(b):
                pad_extra = np.zeros((len(b) - clipped_lens[i], 14), dtype=np.float32)
                recon_chunks[i] = np.concatenate([recon_chunks[i], pad_extra], axis=0)
        recon_norm = np.concatenate(recon_chunks, axis=0)             # (body_T, 14)

        # Denormalize
        recon_feats = recon_norm * feat_std + feat_mean
        gt_for_sparc = gt_feats[:body_T].copy()

        # log(p+1) → Hz for SPARC
        for arr in (recon_feats, gt_for_sparc):
            arr[:, 12] = np.exp(arr[:, 12]) - 1.0
            arr[arr[:, 12] < 30, 12] = 0.0

        # SPARC render
        spk = np.load(anch_path, allow_pickle=False)["spk_emb"].astype(np.float32)
        wav_v9 = sparc.decode(recon_feats[:, :12], recon_feats[:, 12], recon_feats[:, 13], spk)
        wav_gt = sparc.decode(gt_for_sparc[:, :12], gt_for_sparc[:, 12], gt_for_sparc[:, 13], spk)
        if isinstance(wav_v9, torch.Tensor): wav_v9 = wav_v9.detach().squeeze().cpu().numpy()
        if isinstance(wav_gt, torch.Tensor): wav_gt = wav_gt.detach().squeeze().cpu().numpy()

        out_v9 = OUT_DIR / f"{label}_v9_roundtrip.wav"
        out_gt = OUT_DIR / f"{label}_gt_render.wav"
        sf.write(out_v9, wav_v9, sparc.sr)
        sf.write(out_gt, wav_gt, sparc.sr)

        # Per-utterance recon MSE on normalized features
        mse = float(((recon_norm - gt_norm) ** 2).mean())
        print(f"\n[{label}] uid={uid}")
        print(f"  body frames: {body_T}  phonemes: {B}  recon MSE: {mse:.4f}")
        print(f"  v9 -> {out_v9}")
        print(f"  GT -> {out_gt}")


if __name__ == "__main__":
    main()
