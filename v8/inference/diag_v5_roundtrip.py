"""
v5 round-trip on the same 4 real utterances we ran through v8 stage 1.
Bypass v5 planner — quantize the GT per-phoneme z directly and feed to v5
transformer. Output sits next to v8 round-trip for A/B comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.style_codebook import StyleCodebook, PAD_CODE
from models.phoneme_vocab import PhonemeVocab


V5_CKPT      = "checkpoints_v5_stage1_archived/transformer_best.pt"
RVQ_CKPT     = "checkpoints_rvq_logpitch_v2/rvq_best.pt"
NORM_STATS   = "data/features_merged_logpitch_v2/norm_stats.npz"
VOCAB        = "data/processed_all/vocab_mfa.json"
PHONEMES     = "data/processed_merged_v3/phonemes_mfa.json"
ANCHORS_DIR  = Path("v8/data/phoneme_anchors")
Z_DIR        = Path("v8/data/phoneme_z")
OUT_DIR      = Path("v8/outputs/diag_v8_roundtrip")  # share dir w/ v8 outputs

TARGETS = {
    "angry":   "0011_Angry_0011_000564",
    "happy":   "0011_Happy_0011_000936",
    "sad":     "0011_Sad_0011_001286",
    "neutral": "0011_Neutral_0011_000313",
}


def main():
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = np.load(NORM_STATS)
    feat_mean, feat_std = stats["mean"], stats["std"]

    # RVQ
    rvq_ckpt = torch.load(RVQ_CKPT, map_location=device, weights_only=True)
    ra = rvq_ckpt["args"]
    rvq = ArticulatoryRVQTokenizer(
        codebook_size=ra["codebook_size"], num_quantizers=ra["num_quantizers"],
        latent_dim=ra["latent_dim"], hidden_dim=ra["hidden_dim"],
    ).to(device)
    rvq.load_state_dict(rvq_ckpt["model_state_dict"]); rvq.eval()

    # Transformer + style codebook
    s1 = torch.load(V5_CKPT, map_location=device, weights_only=False)
    sa = s1["args"]
    print(f"v5 stage1: epoch={s1['epoch']} val={s1['val_loss']:.4f}")
    transformer = ArticulatoryTTSModelRVQHier(
        vocab_size=s1["vocab_size"],
        d_model=sa["d_model"], nhead=sa["nhead"], num_encoder_layers=sa["num_layers"],
        num_decoder_layers=sa["num_layers"], d_ff=sa["d_ff"], dropout=sa.get("dropout", 0.0),
        speaker_emb_dim=64, style_dim=sa["d_model"],
        codebook_size=sa["codebook_size"], num_quantizers=sa["num_quantizers"],
    ).to(device)
    transformer.load_state_dict(s1["model_state_dict"]); transformer.eval()

    style_codebook = StyleCodebook(
        latent_dim=sa["d_model"], codebook_size=sa["style_codebook_size"],
    ).to(device)
    style_codebook.load_state_dict(s1["style_codebook_state_dict"]); style_codebook.eval()

    vocab = PhonemeVocab(VOCAB)
    phon_data = json.load(open(PHONEMES))

    print("loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    for label, uid in TARGETS.items():
        anch_p = ANCHORS_DIR / f"{uid}.npz"
        z_p    = Z_DIR / f"{uid}.npy"
        if not (anch_p.exists() and z_p.exists()):
            print(f"skip {label}"); continue

        ph_ids = np.asarray(phon_data[uid]["indices"], dtype=np.int64)   # (N+2,)
        N = len(ph_ids)
        n_body = N - 2

        anch = np.load(anch_p, allow_pickle=False)
        spk = anch["spk_emb"].astype(np.float32)

        z_body = np.load(z_p).astype(np.float32)              # (n_body, 256)
        # Pad BOS/EOS with zeros (will get PAD_CODE via codebook)
        z_full = np.zeros((N, sa["d_model"]), dtype=np.float32)
        z_full[1:1+n_body] = z_body

        ph_t  = torch.from_numpy(ph_ids).unsqueeze(0).to(device)
        spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)
        z_t   = torch.from_numpy(z_full).unsqueeze(0).to(device)
        # Mask: True for body, False for BOS/EOS — same convention v5 uses for codebook
        mask_body = torch.zeros((1, N), dtype=torch.bool, device=device)
        mask_body[0, 1:1+n_body] = True

        with torch.no_grad():
            # Quantize body z's to codes
            _, codes_body, _ = style_codebook(z_t, mask_body)            # (1, N) — body slots filled
            # Build full-length codes: BOS/EOS = PAD_CODE
            codes = torch.full((1, N), PAD_CODE, dtype=codes_body.dtype, device=device)
            codes[0, 1:1+n_body] = codes_body[0, 1:1+n_body]
            style_emb = style_codebook.embed_codes(codes)                # (1, N, D)
            phoneme_mask = ph_t != 0

            enc = transformer.encode_phonemes(ph_t, spk_t, phoneme_mask, style_emb=style_emb)
            pred_dur = transformer.duration_predictor(enc, phoneme_mask)
            durations = pred_dur.round().clamp(min=1)
            T = int(durations.sum().item())
            decoded, _ = transformer._decode_frames(enc, durations, T, phoneme_mask)
            logits = transformer._run_hierarchical_heads(decoded, target_tokens=None)
            token_ids = logits.argmax(dim=-1)
            feats_norm = rvq.decode_indices(token_ids).squeeze(0).cpu().numpy()

        feats = feats_norm * feat_std + feat_mean
        feats = feats[:T]
        feats[:, 12] = np.exp(feats[:, 12]) - 1.0
        ema = feats[:, :12]
        pitch = feats[:, 12].copy(); pitch[pitch < 30] = 0.0
        loud = feats[:, 13]

        wav = sparc.decode(ema, pitch, loud, spk)
        if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
        out = OUT_DIR / f"{label}_v5_render.wav"
        sf.write(out, wav, sparc.sr)
        print(f"\n[{label}] uid={uid}")
        print(f"  v5 render -> {out}  frames={feats.shape[0]}  dur={feats.shape[0]/50:.2f}s  "
              f"pitch_mean={pitch[pitch>0].mean() if (pitch>0).any() else 0:.0f}Hz")


if __name__ == "__main__":
    main()
