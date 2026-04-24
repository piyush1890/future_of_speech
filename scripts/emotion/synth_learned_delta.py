"""
Synth using the trained emotion-delta predictor.

    python scripts/emotion/synth_learned_delta.py \\
        --text-file reference_audio/conv/transcript.txt \\
        --emotion Happy --intensity 1.0 \\
        --out outputs/learned/jre_happy.wav
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.emotion_delta_predictor import EmotionDeltaPredictor
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


EMOTIONS = ["Happy", "Sad", "Angry", "Surprise"]
EMO_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


@torch.no_grad()
def synthesize(text, tf, rvq, delta_model, vocab, g2p, sparc, spk_emb,
               fm, fs, emotion, intensity, device=torch.device("cpu"),
               ema_scale=0.0, sample_temperature=1.0):
    # Phonemes
    raw = g2p(text)
    phonemes = ["<sil>"] + [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
    indices_bos = vocab.encode(phonemes, add_bos_eos=True)
    indices_nobos = vocab.encode(phonemes, add_bos_eos=False)

    # Main TTS forward
    phoneme_ids = torch.tensor([indices_bos], dtype=torch.long, device=device)
    phoneme_mask = phoneme_ids != 0
    spk_t = torch.from_numpy(spk_emb).unsqueeze(0).to(device)

    enc = tf.encode_phonemes(phoneme_ids, spk_t, phoneme_mask)
    pred_dur = tf.duration_predictor(enc, phoneme_mask)
    durations_full = pred_dur.round().clamp(min=1)  # (1, N+2) incl BOS/EOS
    T = int(durations_full.sum().item())

    decoded, _ = tf._decode_frames(enc, durations_full, T, phoneme_mask)
    logits = tf._run_hierarchical_heads(decoded, target_tokens=None)
    tok = logits.argmax(dim=-1)
    feats_norm = rvq.decode_indices(tok).squeeze(0).cpu().numpy()[:T]
    features = feats_norm * fs + fm

    # Delta predictor — uses the SAME phoneme sequence WITHOUT BOS/EOS (that's how
    # the training dataset was built). Durations: strip BOS (index 0) and EOS (last).
    phon_nobos = torch.tensor([indices_nobos], dtype=torch.long, device=device)
    durs_nobos = durations_full[:, 1:-1]  # drop BOS and EOS positions
    # Guard: the main TTS pads; trim durations to actual phoneme count
    actual_N = phon_nobos.shape[1]
    durs_nobos = durs_nobos[:, :actual_N].long()

    emo_idx = torch.tensor([EMO_TO_IDX[emotion]], dtype=torch.long, device=device)
    spk_for_delta = spk_t  # same 64-dim spk_emb

    # Use sample() — stochastic if model is variational, deterministic otherwise.
    pred_delta, fm_mask = delta_model.sample(
        phon_nobos, durs_nobos, emo_idx, spk_for_delta,
        temperature=sample_temperature,
    )
    pred_delta = pred_delta[0].cpu().numpy()   # (T_delta, 14)

    # The main TTS includes BOS+EOS frames (~2 extra frames). Align delta to the
    # phoneme body of our output. BOS contributes durations_full[0,0] frames, EOS
    # durations_full[0,-1] frames. Shift delta into the middle.
    bos_f = int(durations_full[0, 0].item())
    eos_f = int(durations_full[0, -1].item())
    body_start = bos_f
    body_end = T - eos_f
    body_len = body_end - body_start
    if pred_delta.shape[0] < body_len:
        # Pad delta to body length (should rarely happen if durations are consistent)
        pad = np.zeros((body_len - pred_delta.shape[0], 14), dtype=np.float32)
        pred_delta = np.concatenate([pred_delta, pad], axis=0)
    pred_delta = pred_delta[:body_len]

    delta_full = np.zeros((T, 14), dtype=np.float32)
    delta_full[body_start:body_end] = pred_delta

    # Apply with intensity — EMA (chans 0..11) gets its own scale (default 0)
    # because amplified EMA deltas shift articulators off-manifold across speakers.
    out = features.copy()
    if ema_scale != 0.0:
        out[:, :12] = out[:, :12] + ema_scale * delta_full[:, :12]
    out[:, 12] = out[:, 12] + intensity * delta_full[:, 12]
    out[:, 13] = out[:, 13] + intensity * delta_full[:, 13]
    features = out

    # Log-pitch → Hz + smoothing (ORIGINAL order — smooth AFTER emotion to tame
    # any artifacts without killing what the model learned)
    pitch_log = features[:, 12]
    pitch_hz = np.exp(pitch_log) - 1.0
    pitch_hz[pitch_hz < 30] = 0.0
    pitch_hz = median_filter(pitch_hz, size=5, mode="nearest")
    pitch_hz = gaussian_filter1d(pitch_hz, sigma=1.0, mode="nearest")
    features[:, 12] = pitch_hz
    for k in range(12):
        features[:, k] = gaussian_filter1d(features[:, k], sigma=1.0, mode="nearest")
    features[:, 13] = gaussian_filter1d(features[:, 13], sigma=1.0, mode="nearest")

    wav = sparc.decode(features[:, :12], features[:, 12], features[:, 13], spk_emb)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().squeeze().cpu().numpy()
    return wav, sparc.sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", type=str, nargs="?", default="")
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--emotion", type=str, default="Happy",
                   choices=["Neutral"] + EMOTIONS)
    p.add_argument("--intensity", type=float, default=1.0,
                   help="Multiply the pitch+loudness delta by this factor")
    p.add_argument("--ema-scale", type=float, default=0.0,
                   help="Scale on EMA (articulator) delta. Default 0 — amplified EMA "
                        "shifts speech off-manifold. Try 0.2-0.5 for some articulatory color.")
    p.add_argument("--sample-temperature", type=float, default=1.0,
                   help="(Variational delta only) Temperature on sigma at sampling. "
                        "0=deterministic mean. 1=natural variance. >1=more variation.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for sampling (variational delta). Different seeds = "
                        "different takes of same text.")
    p.add_argument("--sentence-gap", type=float, default=0.25)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--speaker-utt", type=str, default="1322-137588-0000")
    p.add_argument("--rvq-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch/rvq_best.pt")
    p.add_argument("--transformer-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch_hier/transformer_best.pt")
    p.add_argument("--delta-checkpoint", type=str,
                   default="checkpoints_emotion_delta/delta_best.pt")
    p.add_argument("--features-dir", type=str, default="data/features_merged_logpitch")
    p.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    a = p.parse_args()

    device = torch.device("cpu")

    # Main TTS
    rvq_ckpt = torch.load(a.rvq_checkpoint, map_location=device, weights_only=True)
    ra = rvq_ckpt["args"]
    rvq = ArticulatoryRVQTokenizer(
        codebook_size=ra["codebook_size"], num_quantizers=ra["num_quantizers"],
        latent_dim=ra["latent_dim"], hidden_dim=ra["hidden_dim"],
    ).to(device)
    rvq.load_state_dict(rvq_ckpt["model_state_dict"]); rvq.eval()

    tf_ckpt = torch.load(a.transformer_checkpoint, map_location=device, weights_only=True)
    ta = tf_ckpt["args"]
    tf = ArticulatoryTTSModelRVQHier(
        vocab_size=tf_ckpt["vocab_size"], codebook_size=ta["codebook_size"],
        num_quantizers=ta["num_quantizers"], d_model=ta["d_model"], nhead=ta["nhead"],
        num_encoder_layers=ta["num_layers"], num_decoder_layers=ta["num_layers"],
        d_ff=ta["d_ff"], dropout=ta.get("dropout", 0.1), speaker_emb_dim=64,
    ).to(device)
    tf.load_state_dict(tf_ckpt["model_state_dict"], strict=False); tf.eval()

    # Delta predictor
    dm_ckpt = torch.load(a.delta_checkpoint, map_location=device, weights_only=True)
    da = dm_ckpt["args"]
    delta_model = EmotionDeltaPredictor(
        vocab_size=dm_ckpt["vocab_size"], n_emotions=4, spk_emb_dim=64,
        d_model=da["d_model"], nhead=da["nhead"],
        num_phon_layers=da["num_phon_layers"],
        num_frame_layers=da["num_frame_layers"],
        d_ff=da["d_ff"], dropout=0.0, out_dim=14,
        variational=da.get("variational", False),
        min_sigma=da.get("min_sigma", 0.05),
    ).to(device)
    delta_model.load_state_dict(dm_ckpt["model_state_dict"]); delta_model.eval()
    print(f"Loaded delta predictor: epoch {dm_ckpt['epoch']}, val={dm_ckpt['val_loss']:.4f}, "
          f"variational={da.get('variational', False)}")

    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

    stats = np.load(Path(a.features_dir) / "norm_stats.npz")
    fm, fs = stats["mean"], stats["std"]
    spk_emb = np.load(Path(a.features_dir) / f"{a.speaker_utt}.npz")["spk_emb"].astype(np.float32)

    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    g2p = G2p()
    vocab = PhonemeVocab(a.vocab)

    # Text → segments (split on .!? for multi-sentence)
    if a.text_file:
        text = Path(a.text_file).read_text().strip()
        rough = re.split(r"(?<=[.!?])\s+", text)
        segments = []
        for s in rough:
            s = s.strip()
            if not s:
                continue
            words = s.split()
            if len(words) > 18:
                parts = re.split(r",\s*", s)
                cur = ""
                for pt in parts:
                    pt = pt.strip()
                    if not pt:
                        continue
                    merged = (cur + ", " + pt) if cur else pt
                    if len(merged.split()) > 18:
                        if cur: segments.append(cur)
                        cur = pt
                    else:
                        cur = merged
                if cur: segments.append(cur)
            else:
                segments.append(s)
    else:
        if not a.text:
            raise SystemExit("Provide TEXT or --text-file")
        segments = [a.text]

    print(f"Rendering {len(segments)} segments, emotion={a.emotion}, intensity={a.intensity}")
    chunks = []
    sr = None
    for i, s in enumerate(segments):
        if len(s) < 2:
            continue
        wav, sr = synthesize(s, tf, rvq, delta_model, vocab, g2p, sparc, spk_emb,
                             fm, fs, a.emotion, a.intensity, device=device,
                             ema_scale=a.ema_scale,
                             sample_temperature=a.sample_temperature)
        print(f"  [{i+1}/{len(segments)}] ({len(wav)/sr:.1f}s) {s[:60]}...")
        chunks.append(wav)
        chunks.append(np.zeros(int(sr * a.sentence_gap), dtype=wav.dtype))
    full = np.concatenate(chunks[:-1]) if chunks else np.zeros(int((sr or 16000) * 0.1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, full, sr)
    print(f"Saved {a.out}  ({len(full)/sr:.1f}s)")


if __name__ == "__main__":
    main()
