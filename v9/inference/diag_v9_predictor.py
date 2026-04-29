"""Deep predictor debug: feed MFA-aligned phonemes (not g2p) for a known utterance
and compare predicted tokens vs GT tokens. Render audio for both.

If audio from predictor-tokens-on-MFA-phonemes sounds OK but g2p path is bad,
the bug is in g2p ↔ MFA phoneme set mismatch.
If both are bad, the predictor itself is the bottleneck.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.phoneme_rvq import PhonemeRVQTokenizer
from v9.models.v9_predictor import V9Predictor
from models.phoneme_vocab import PhonemeVocab


TOK_CKPT  = "v9/checkpoints/phoneme_rvq/best.pt"
PRED_CKPT = "v9/checkpoints/predictor/best.pt"
TOKENS_DIR = Path("v9/data/phoneme_tokens")
PHONEMES = "data/processed_merged_v3/phonemes_mfa.json"
ALIGNMENTS = "data/processed_merged_v3/alignments_mfa.json"
NORM_STATS = "data/features_merged_logpitch_v2/norm_stats.npz"
VOCAB = "data/processed_merged_v3/vocab_mfa.json"
METADATA = "data/utterance_metadata_v5.json"
OUT = Path("v9/outputs/diag_predictor")

UID = "0011_Neutral_0011_000313"
EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


def sample_argmax(logits): return logits.argmax(-1)


@torch.no_grad()
def predictor_argmax(predictor, phoneme_ids, spk_emb, knobs, phoneme_mask, K):
    """Greedy argmax generation (deterministic). Returns (start, end, logdur)."""
    B, N = phoneme_ids.shape
    device = phoneme_ids.device
    enc = predictor.encode(phoneme_ids, spk_emb, knobs, phoneme_mask, force_drop_knobs=False)

    gen_start = torch.zeros(B, N, K, dtype=torch.long, device=device)
    gen_end = torch.zeros(B, N, K, dtype=torch.long, device=device)
    gen_logdur = torch.zeros(B, N, device=device)

    for i in range(N):
        dec_inp = predictor._make_decoder_input(gen_start, gen_end)
        dec_inp = predictor.decoder_pe(dec_inp)
        causal = torch.triu(torch.full((N, N), float("-inf"), device=device), diagonal=1)
        h = predictor.decoder(dec_inp, enc, tgt_mask=causal,
                              tgt_key_padding_mask=~phoneme_mask,
                              memory_key_padding_mask=~phoneme_mask)
        h_pos = h[:, i, :]
        for k in range(K):
            cl_s = predictor.start_heads.step_logits_one(h_pos, k, gen_start[:, i, :k])
            gen_start[:, i, k] = sample_argmax(cl_s)
        for k in range(K):
            cl_e = predictor.end_heads.step_logits_one(h_pos, k, gen_end[:, i, :k])
            gen_end[:, i, k] = sample_argmax(cl_e)
        gen_logdur[:, i] = predictor.duration_head(h_pos).squeeze(-1)
    return gen_start, gen_end, gen_logdur


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Load models
    tc = torch.load(TOK_CKPT, map_location=device, weights_only=False); ta = tc["args"]
    tokenizer = PhonemeRVQTokenizer(
        vocab_size=ta["vocab_size"], input_dim=14,
        latent_dim=ta["latent_dim"], hidden_dim=ta["hidden_dim"],
        codebook_size=ta["codebook_size"], num_quantizers=ta["num_quantizers"],
        decoder_d_model=ta["decoder_d_model"], decoder_nhead=ta["decoder_nhead"],
        decoder_layers=ta["decoder_layers"],
        commitment_weight=ta["commit_weight"], ema_decay=ta["ema_decay"],
    ).to(device)
    tokenizer.load_state_dict(tc["model"]); tokenizer.eval()

    pc = torch.load(PRED_CKPT, map_location=device, weights_only=False); pa = pc["args"]
    knob_dim = pc.get("knob_dim", 6)
    predictor = V9Predictor(
        vocab_size=pa["vocab_size"], codebook_size=pa["codebook_size"],
        num_quantizers=pa["num_quantizers"],
        d_model=pa["d_model"], nhead=pa["nhead"],
        num_encoder_layers=pa["num_layers"], num_decoder_layers=pa["num_layers"],
        d_ff=pa["d_ff"], dropout=0.0, speaker_emb_dim=64,
        knob_dim=knob_dim, knob_dropout=0.0,
        max_phonemes=pa["max_phonemes"],
    ).to(device)
    predictor.load_state_dict(pc["model"]); predictor.eval()
    K = pa["num_quantizers"]

    # Load utterance data
    phon_data = json.load(open(PHONEMES))
    align_data = json.load(open(ALIGNMENTS))
    meta = json.load(open(METADATA))
    vocab = PhonemeVocab(VOCAB)
    text = phon_data[UID].get("text", "(no text)")
    mfa_phs = phon_data[UID]["phonemes"]
    mfa_indices = phon_data[UID]["indices"]
    print(f"\nUID: {UID}\ntext: {text}")
    print(f"MFA phonemes ({len(mfa_phs)}): {mfa_phs[:25]}{'...' if len(mfa_phs)>25 else ''}")

    phoneme_ids = torch.tensor([mfa_indices], dtype=torch.long, device=device)
    N = phoneme_ids.shape[1]
    phoneme_mask = phoneme_ids != 0
    spk = np.load(f"v8/data/phoneme_anchors/{UID}.npz")["spk_emb"].astype(np.float32)
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)
    emo_id = EMOTION_TO_ID.get(meta.get(UID, {}).get("emotion_label", "neutral"), 0)
    intensity = float(meta.get(UID, {}).get("intensity", 0.5))
    one_hot = [0.0]*5; one_hot[emo_id] = 1.0
    knobs = torch.tensor([one_hot + [intensity]], dtype=torch.float32, device=device)

    # GT tokens
    tok = np.load(TOKENS_DIR / f"{UID}.npz", allow_pickle=False)
    gt_start = tok["start_idx"].astype(np.int64)        # (N_body, K)
    gt_end = tok["end_idx"].astype(np.int64)
    gt_durations = tok["durations"].astype(np.int64)
    n_body = len(gt_durations)

    # Predictor argmax (deterministic, no sampling)
    pred_start, pred_end, pred_logdur = predictor_argmax(
        predictor, phoneme_ids, spk_t, knobs, phoneme_mask, K
    )
    body_pred_start = pred_start[0, 1:N-1, :].cpu().numpy()
    body_pred_end = pred_end[0, 1:N-1, :].cpu().numpy()
    pred_durations = (torch.exp(pred_logdur[0, 1:N-1])).round().clamp(min=1).long().cpu().numpy()

    print(f"\n=== TOKEN COMPARISON (argmax vs GT) ===")
    print(f"start match by level: {[float((body_pred_start[:, k]==gt_start[:, k]).mean()) for k in range(K)]}")
    print(f"end   match by level: {[float((body_pred_end[:, k]==gt_end[:, k]).mean()) for k in range(K)]}")
    print(f"\ndurations  GT  : {gt_durations.tolist()}")
    print(f"durations PRED: {pred_durations.tolist()}")
    print(f"duration match (exact): {(pred_durations == gt_durations).mean()*100:.1f}%")
    print(f"duration MSE (frames): {float(((pred_durations - gt_durations)**2).mean()):.2f}")

    # Render 4 variants:
    #   GT_tokens + GT_dur     ← upper bound (you said this sounds fine)
    #   PRED_tokens + GT_dur   ← isolates token quality
    #   GT_tokens + PRED_dur   ← isolates duration quality
    #   PRED_tokens + PRED_dur ← full pipeline (what synthesize_v9 does)
    body_ph_ids = torch.from_numpy(np.asarray(mfa_indices[1:1+n_body], dtype=np.int64))
    stats = np.load(NORM_STATS)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)

    print("\nLoading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    def render(s_idx, e_idx, dur, label):
        s_t = torch.from_numpy(s_idx).long()
        e_t = torch.from_numpy(e_idx).long()
        d_t = torch.from_numpy(dur).long()
        with torch.no_grad():
            blocks = tokenizer.decode_indices_batch(s_t, e_t, body_ph_ids, d_t)
        body_norm = torch.cat(blocks, dim=0).cpu().numpy()
        feats = body_norm * feat_std + feat_mean
        feats[:, 12] = np.exp(feats[:, 12]) - 1.0
        feats[feats[:, 12] < 30, 12] = 0.0
        wav = sparc.decode(feats[:, :12], feats[:, 12], feats[:, 13], spk)
        if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
        out_path = OUT / f"{UID}_{label}.wav"
        sf.write(out_path, wav, sparc.sr)
        print(f"  {label}: {out_path} ({len(wav)/sparc.sr:.2f}s)  T_frames={body_norm.shape[0]}")

    print("\n=== RENDERING 4 VARIANTS ===")
    render(gt_start, gt_end, gt_durations, "1_GTtok_GTdur")
    render(body_pred_start, body_pred_end, gt_durations, "2_PREDtok_GTdur")
    render(gt_start, gt_end, pred_durations, "3_GTtok_PREDdur")
    render(body_pred_start, body_pred_end, pred_durations, "4_PREDtok_PREDdur")


if __name__ == "__main__":
    main()
