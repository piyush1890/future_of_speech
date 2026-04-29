"""
Diagnostic: bypass planner, feed GT-extracted z (mean over a target utterance)
into stage 1 directly. Tests whether stage 1 actually uses z.
"""
import sys
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from g2p_en import G2p

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v8.models.phoneme_tts import PhonemeTTSv8
from v8.models.phoneme_classes import build_render_class_table
from models.phoneme_vocab import PhonemeVocab


TEXT = "You know, I tried that the other day, and it actually made me feel a lot better."
DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"
STAGE1_CKPT = "v8/checkpoints/stage1_zq/best.pt"
VOCAB = "data/processed_all/vocab_mfa.json"
Z_DIR = Path("v8/data/phoneme_z")
OUT_DIR = Path("v8/outputs/diag_gt_z")

# 4 utterances spanning emotions for speaker 0011 (high intensity)
TARGETS = {
    "angry":   "0011_Angry_0011_000564",
    "happy":   "0011_Happy_0011_000936",
    "sad":     "0011_Sad_0011_001286",
    "neutral": "0011_Neutral_0011_000313",
}


def text_to_phonemes(text, g2p):
    parts = re.split(r"(?<=[.!?,;:—–])\s+", text.strip())
    phs = ["<sil>"]
    for si, part in enumerate(parts):
        if not part.strip(): continue
        if si > 0: phs.append("<sil>")
        for p in g2p(part):
            if p and p[0].isalpha() and p[0].isupper():
                phs.append(p)
    phs.append("<sil>")
    return phs


def main():
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s1 = torch.load(STAGE1_CKPT, map_location=device, weights_only=False)
    sa = s1["args"]
    print(f"stage1: epoch={s1['epoch']} val={s1['val_loss']:.4f}")
    vocab = PhonemeVocab(VOCAB)
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

    g2p = G2p()
    phs = text_to_phonemes(TEXT, g2p)
    phoneme_ids = torch.tensor([vocab.encode(phs, add_bos_eos=True)], dtype=torch.long, device=device)
    N = phoneme_ids.shape[1]
    print(f"phonemes: {len(phs)}, encoded len: {N}")

    spk = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)

    print("loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    for label, uid in TARGETS.items():
        z_path = Z_DIR / f"{uid}.npy"
        if not z_path.exists():
            print(f"  skip {label}: {z_path} missing"); continue
        z = np.load(z_path)                              # (n_body, 256)
        z_mean = z.mean(axis=0)                          # (256,)
        print(f"\n[{label}] uid={uid}  z shape={z.shape}  mean-z norm={np.linalg.norm(z_mean):.3f}")

        # Broadcast mean z across all N phoneme positions (incl. BOS/EOS)
        z_full = torch.from_numpy(np.broadcast_to(z_mean, (1, N, z.shape[1])).copy()).float().to(device)

        with torch.no_grad():
            frames, durations, frame_mask = stage1.generate(
                phoneme_ids, spk_t, style_emb=z_full, duration_scale=1.0,
            )
        feats_logp = frames.squeeze(0).cpu().numpy()
        feats_logp = feats_logp[frame_mask.squeeze(0).cpu().numpy()]
        feats = feats_logp.copy()
        feats[:, 12] = np.exp(feats[:, 12]) - 1.0
        ema = feats[:, :12]
        pitch = feats[:, 12].copy(); pitch[pitch < 30] = 0.0
        loud = feats[:, 13]
        wav = sparc.decode(ema, pitch, loud, spk)
        if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
        out = OUT_DIR / f"{label}.wav"
        sf.write(out, wav, sparc.sr)
        print(f"  -> {out}  frames={feats.shape[0]}  dur={feats.shape[0]/50:.2f}s  "
              f"pitch_mean={pitch[pitch>0].mean() if (pitch>0).any() else 0:.0f}Hz  "
              f"loud_mean={loud.mean():.2f}")


if __name__ == "__main__":
    main()
