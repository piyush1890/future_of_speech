"""
v8 stage 1 round-trip on real utterances: same phonemes, same per-phoneme z,
same speaker — does the 3-anchor + linear interp preserve emotion?
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v8.models.phoneme_tts import PhonemeTTSv8
from v8.models.phoneme_classes import build_render_class_table
from models.phoneme_vocab import PhonemeVocab


STAGE1_CKPT  = "v8/checkpoints/stage1_zq/best.pt"
VOCAB        = "data/processed_all/vocab_mfa.json"
PHONEMES     = "data/processed_merged_v3/phonemes_mfa.json"
ANCHORS_DIR  = Path("v8/data/phoneme_anchors")
Z_DIR        = Path("v8/data/phoneme_z")
FEATS_DIR    = Path("data/features_merged_logpitch_v2")
OUT_DIR      = Path("v8/outputs/diag_v8_roundtrip")

TARGETS = {
    "angry":   "0011_Angry_0011_000564",
    "happy":   "0011_Happy_0011_000936",
    "sad":     "0011_Sad_0011_001286",
    "neutral": "0011_Neutral_0011_000313",
}


def main():
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s1 = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    sa = s1["args"]
    print(f"stage1: epoch={s1['epoch']} val={s1['val_loss']:.4f}")
    vocab = PhonemeVocab(VOCAB)
    phon_data = json.load(open(PHONEMES))
    rmode = sa.get("render_mode", "hybrid")
    rct = build_render_class_table(vocab) if rmode == "hybrid" else None

    stage1 = PhonemeTTSv8(
        vocab_size=s1["vocab_size"], feature_dim=sa.get("feature_dim", 14),
        d_model=sa["d_model"], nhead=sa["nhead"],
        num_layers=sa["num_layers"], d_ff=sa["d_ff"],
        dropout=sa.get("dropout", 0.0), speaker_emb_dim=64,
        style_dim=sa["d_model"], render_mode=rmode, render_class_table=rct,
    ).to(device)
    stage1.load_state_dict(s1["model"]); stage1.eval()

    print("loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    for label, uid in TARGETS.items():
        anch_p = ANCHORS_DIR / f"{uid}.npz"
        z_p    = Z_DIR / f"{uid}.npy"
        feats_p = FEATS_DIR / f"{uid}.npz"
        if not (anch_p.exists() and z_p.exists() and feats_p.exists()):
            print(f"skip {label}: missing files"); continue

        # Phoneme IDs (BOS + body + EOS)
        ph_ids = np.asarray(phon_data[uid]["indices"], dtype=np.int64)   # (N+2,)
        N = len(ph_ids)
        n_body = N - 2

        # Speaker emb
        anch = np.load(anch_p, allow_pickle=False)
        spk = anch["spk_emb"].astype(np.float32)

        # Per-phoneme z (body-only, pad with zeros at BOS/EOS)
        z_body = np.load(z_p).astype(np.float32)              # (n_body, 256)
        z_full = np.zeros((N, 256), dtype=np.float32)
        z_full[1:1+n_body] = z_body

        # ─── (a) v8 stage1 round-trip render ───
        ph_t = torch.from_numpy(ph_ids).unsqueeze(0).to(device)
        spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)
        z_t = torch.from_numpy(z_full).unsqueeze(0).to(device)
        with torch.no_grad():
            frames, durations, frame_mask = stage1.generate(
                ph_t, spk_t, style_emb=z_t, duration_scale=1.0,
            )
        v8_logp = frames.squeeze(0).cpu().numpy()
        v8_logp = v8_logp[frame_mask.squeeze(0).cpu().numpy()]
        v8_feats = v8_logp.copy()
        v8_feats[:, 12] = np.exp(v8_feats[:, 12]) - 1.0
        v8_pitch = v8_feats[:, 12].copy(); v8_pitch[v8_pitch < 30] = 0.0
        wav = sparc.decode(v8_feats[:, :12], v8_pitch, v8_feats[:, 13], spk)
        if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
        out = OUT_DIR / f"{label}_v8_render.wav"
        sf.write(out, wav, sparc.sr)
        print(f"\n[{label}] uid={uid}")
        print(f"  v8 render   -> {out}  frames={v8_feats.shape[0]}  dur={v8_feats.shape[0]/50:.2f}s  "
              f"pitch_mean={v8_pitch[v8_pitch>0].mean() if (v8_pitch>0).any() else 0:.0f}Hz")

        # ─── (b) GT frames → SPARC (the upper bound) ───
        feats_npz = np.load(feats_p, allow_pickle=False)
        gt_ema = feats_npz["ema"].astype(np.float32)            # (T, 12)
        gt_pitch_log = feats_npz["pitch"].astype(np.float32)
        gt_pitch = np.exp(gt_pitch_log) - 1.0                   # log(p+1) -> p
        gt_loud = feats_npz["loudness"].astype(np.float32)
        gt_pitch[gt_pitch < 30] = 0.0
        wav_gt = sparc.decode(gt_ema, gt_pitch, gt_loud, spk)
        if isinstance(wav_gt, torch.Tensor): wav_gt = wav_gt.detach().squeeze().cpu().numpy()
        out_gt = OUT_DIR / f"{label}_gt_render.wav"
        sf.write(out_gt, wav_gt, sparc.sr)
        print(f"  GT render   -> {out_gt}  frames={gt_ema.shape[0]}  dur={gt_ema.shape[0]/50:.2f}s  "
              f"pitch_mean={gt_pitch[gt_pitch>0].mean() if (gt_pitch>0).any() else 0:.0f}Hz")


if __name__ == "__main__":
    main()
