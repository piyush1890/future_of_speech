"""
Synth driver using exemplar-based emotion rendering (no mean-delta averaging).

    python scripts/emotion/synth_exemplar.py \\
        "So tell me, what exactly are you doing to me?" \\
        --emotion Happy --intensity 1.0 --idx 7 \\
        --out outputs/exemplar/happy_7.wav
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p

from emotion_exemplar import (
    apply_to_features, apply_envelopes_to_features, list_exemplars, EMOTIONS,
)


@torch.no_grad()
def synthesize(text, tf, rvq, vocab, g2p, sparc, spk_emb, fm, fs,
               emotion, intensity, idx, speaker, device=torch.device("cpu"),
               ema_scale=0.0, mode="additive",
               pitch_intensity=1.0, loud_intensity=1.0,
               delta_smooth_sigma=5.0, post_smooth_sigma=0.5):
    raw = g2p(text)
    phonemes = ["<sil>"] + [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
    indices = vocab.encode(phonemes, add_bos_eos=True)
    phoneme_ids = torch.tensor([indices], dtype=torch.long, device=device)
    phoneme_mask = phoneme_ids != 0
    spk_t = torch.from_numpy(spk_emb).unsqueeze(0).to(device)

    enc = tf.encode_phonemes(phoneme_ids, spk_t, phoneme_mask)
    pred_dur = tf.duration_predictor(enc, phoneme_mask)
    durations = pred_dur.round().clamp(min=1)
    T = int(durations.sum().item())
    decoded, _ = tf._decode_frames(enc, durations, T, phoneme_mask)
    logits = tf._run_hierarchical_heads(decoded, target_tokens=None)
    tok = logits.argmax(dim=-1)

    feats_norm = rvq.decode_indices(tok).squeeze(0).cpu().numpy()[:T]
    features = feats_norm * fs + fm

    # --- Apply emotion FIRST (on raw RVQ output) ---
    meta = None
    if emotion and emotion.lower() != "neutral":
        if mode == "envelope":
            is_q = text.strip().endswith("?")
            features, meta = apply_envelopes_to_features(
                features, emotion=emotion,
                idx=(idx if idx is not None and idx >= 0 else None),
                target_is_question=is_q,
                pitch_intensity=pitch_intensity,
                loud_intensity=loud_intensity,
                speaker=speaker,
            )
        else:
            features, meta = apply_to_features(
                features, emotion=emotion, intensity=intensity,
                idx=idx, speaker=speaker, ema_scale=ema_scale,
                smooth_delta_sigma=delta_smooth_sigma,
            )

    # --- Pitch log→Hz + voicing floor ---
    pitch_log = features[:, 12]
    pitch_hz = np.exp(pitch_log) - 1.0
    pitch_hz[pitch_hz < 30] = 0.0
    pitch_hz = median_filter(pitch_hz, size=5, mode="nearest")
    # --- Original post-hoc smoothing (sigma=1.0) — cleans RVQ jitter AND
    # frame-rate artifacts from emotion delta addition. ---
    pitch_hz = gaussian_filter1d(pitch_hz, sigma=1.0, mode="nearest")
    features[:, 12] = pitch_hz
    for k in range(12):
        features[:, k] = gaussian_filter1d(features[:, k], sigma=1.0, mode="nearest")
    features[:, 13] = gaussian_filter1d(features[:, 13], sigma=1.0, mode="nearest")

    wav = sparc.decode(features[:, :12], features[:, 12], features[:, 13], spk_emb)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().squeeze().cpu().numpy()
    return wav, sparc.sr, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", type=str, nargs="?", default="",
                   help="Text to synthesize. Ignored if --text-file is given.")
    p.add_argument("--text-file", type=str, default=None,
                   help="Long text; split on .!? into sentences, render each with an exemplar, "
                        "concatenate with pauses.")
    p.add_argument("--sentence-gap", type=float, default=0.25)
    p.add_argument("--rotate-exemplars", action="store_true",
                   help="When using --text-file, cycle through several exemplars "
                        "(vs using the same idx for every sentence).")
    p.add_argument("--rotate-indices", type=str, default="0,2,4,6,8,10",
                   help="Comma-separated exemplar indices to rotate through.")
    p.add_argument("--emotion", type=str, default="Happy",
                   choices=["neutral", "Neutral"] + list(EMOTIONS))
    p.add_argument("--intensity", type=float, default=1.0,
                   help="Scale on pitch+loudness delta")
    p.add_argument("--ema-scale", type=float, default=0.0,
                   help="Scale on EMA (articulator) delta. Default 0 — EMA is phoneme-specific "
                        "and cross-applying another speaker's EMA garbles speech. Try 0.2-0.4 for "
                        "mild articulatory color; >0.5 risks garbling.")
    p.add_argument("--mode", type=str, default="additive", choices=["additive", "envelope"],
                   help="additive: add exemplar-neutral delta (raw trajectory). "
                        "envelope: multiplicative — expand OUR pitch/loudness variance using "
                        "exemplar's excitement envelope. Safer for pronunciation.")
    p.add_argument("--pitch-intensity", type=float, default=1.0,
                   help="(envelope mode) how much the pitch envelope multiplier takes effect")
    p.add_argument("--loud-intensity", type=float, default=1.0,
                   help="(envelope mode) how much the loudness envelope multiplier takes effect")
    p.add_argument("--delta-smooth", type=float, default=5.0,
                   help="(additive mode) Gaussian sigma applied to the exemplar delta before "
                        "adding it to our features. Larger = keeps slow emotional arc, removes "
                        "fast phoneme-boundary jitter that causes chirps. 1.5 = old default.")
    p.add_argument("--post-smooth", type=float, default=0.5,
                   help="Gaussian sigma applied to final pitch/loudness. 0 = off. 0.5 = mild "
                        "chirp removal without killing envelope dynamics.")
    p.add_argument("--auto-match", action="store_true",
                   help="(envelope mode) auto-pick best-matching exemplar per sentence")
    p.add_argument("--idx", type=int, default=0, help="Which ESD exemplar to use (0..349)")
    p.add_argument("--esd-speaker", type=str, default="0011")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--speaker-utt", type=str, default="1322-137588-0000")
    p.add_argument("--rvq-checkpoint", type=str, default="checkpoints_rvq_logpitch/rvq_best.pt")
    p.add_argument("--transformer-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch_hier/transformer_best.pt")
    p.add_argument("--features-dir", type=str, default="data/features_merged_logpitch")
    p.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    p.add_argument("--list-exemplars", action="store_true",
                   help="List first 20 exemplar texts for this emotion/speaker and exit.")
    a = p.parse_args()

    if a.list_exemplars:
        pairs = list_exemplars(a.emotion, speaker=a.esd_speaker)
        for i, (nf, ef) in enumerate(pairs[:20]):
            e = np.load(ef)
            print(f"  {i:3d}  {e['text']}")
        return

    device = torch.device("cpu")
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
    tf.load_state_dict(tf_ckpt["model_state_dict"]); tf.eval()

    stats = np.load(Path(a.features_dir) / "norm_stats.npz")
    fm, fs = stats["mean"], stats["std"]
    spk_emb = np.load(Path(a.features_dir) / f"{a.speaker_utt}.npz")["spk_emb"].astype(np.float32)

    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    g2p = G2p()
    vocab = PhonemeVocab(a.vocab)

    if a.text_file:
        import re
        text = Path(a.text_file).read_text().strip()
        # Split on .!? and keep sentences roughly <= 15 words by also splitting on
        # long comma-joined fragments.
        rough = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        for s in rough:
            s = s.strip()
            if not s:
                continue
            # If sentence is very long, split further on commas
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
                        if cur:
                            sentences.append(cur)
                        cur = pt
                    else:
                        cur = merged
                if cur:
                    sentences.append(cur)
            else:
                sentences.append(s)

        rot_idxs = [int(x) for x in a.rotate_indices.split(",")]
        print(f"Rendering {len(sentences)} sentences with {a.emotion} exemplar"
              f"(s) {'rotating' if a.rotate_exemplars else f'idx={a.idx}'}")

        chunks = []
        sr = None
        for si, s in enumerate(sentences):
            ex_idx = rot_idxs[si % len(rot_idxs)] if a.rotate_exemplars else a.idx
            idx_for_call = -1 if (a.mode == "envelope" and a.auto_match) else ex_idx
            wav, sr, meta = synthesize(
                s, tf, rvq, vocab, g2p, sparc, spk_emb, fm, fs,
                emotion=a.emotion, intensity=a.intensity, idx=idx_for_call,
                speaker=a.esd_speaker, device=device, ema_scale=a.ema_scale,
                mode=a.mode, pitch_intensity=a.pitch_intensity,
                loud_intensity=a.loud_intensity,
                delta_smooth_sigma=a.delta_smooth, post_smooth_sigma=a.post_smooth,
            )
            msg = f"[{si+1}/{len(sentences)}] exemplar#{ex_idx}  ({len(wav)/sr:.1f}s)  \"{s[:60]}...\""
            print(msg, flush=True)
            chunks.append(wav)
            chunks.append(np.zeros(int(sr * a.sentence_gap), dtype=wav.dtype))
        full = np.concatenate(chunks[:-1]) if chunks else np.zeros(int((sr or 16000) * 0.1))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        sf.write(a.out, full, sr)
        print(f"Saved {a.out}  ({len(full)/sr:.1f}s total)")
        return

    if not a.text:
        raise SystemExit("Provide either TEXT argument or --text-file")

    idx_for_call = -1 if (a.mode == "envelope" and a.auto_match) else a.idx
    wav, sr, meta = synthesize(
        a.text, tf, rvq, vocab, g2p, sparc, spk_emb, fm, fs,
        emotion=a.emotion, intensity=a.intensity, idx=idx_for_call,
        speaker=a.esd_speaker, device=device, ema_scale=a.ema_scale,
        mode=a.mode, pitch_intensity=a.pitch_intensity, loud_intensity=a.loud_intensity,
        delta_smooth_sigma=a.delta_smooth, post_smooth_sigma=a.post_smooth,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, wav, sr)
    if meta:
        neutral = meta.get('neutral_file', 'N/A')
        print(f"Exemplar: {meta['emotion_file']} (pair neutral={neutral})")
        print(f"         text='{meta['text']}' T_e={meta['T_e']} frames")
    print(f"Saved {a.out}  ({len(wav)/sr:.2f}s)")


if __name__ == "__main__":
    main()
