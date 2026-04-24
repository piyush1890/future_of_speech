"""
Test-harness for emotion-delta synthesis.

Runs our log-pitch transformer → RVQ decode → features → applies per-frame
emotion delta from data/emotion_deltas.npz → SPARC decode → audio.

This isolates the emotion-delta application logic for testing before integrating
into synthesize_rvq_hier.py.

Usage:
    python scripts/emotion/synth_with_delta.py "Hello, how are you?" \\
        --emotion happy --intensity 1.0 --out outputs/emo_delta/happy.wav
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


def load_delta_table(path: str):
    data = np.load(path, allow_pickle=True)
    deltas = data["deltas"]              # (V, E, 14)
    phonemes = list(data["phonemes"])    # list of V symbols
    emotions = list(data["emotions"])    # list of E names
    phon_to_idx = {p: i for i, p in enumerate(phonemes)}
    emo_to_idx  = {e: i for i, e in enumerate(emotions)}
    return deltas, phon_to_idx, emo_to_idx


def build_frame_phoneme_ids(phoneme_ids: torch.Tensor, durations: torch.Tensor) -> np.ndarray:
    """Given (1, N) phoneme ids and (1, N) integer durations, return a flat (T,) array
    where frame t's value is the phoneme id active at that frame."""
    ids = phoneme_ids[0].cpu().numpy()
    durs = durations[0].cpu().numpy().astype(int)
    return np.repeat(ids, durs)


@torch.no_grad()
def synthesize(
    text: str,
    emotion: str,
    intensity: float,
    tf: ArticulatoryTTSModelRVQHier,
    rvq: ArticulatoryRVQTokenizer,
    vocab: PhonemeVocab,
    g2p: G2p,
    sparc,
    spk_emb: np.ndarray,
    fm: np.ndarray,
    fs: np.ndarray,
    deltas: np.ndarray,
    phon_to_idx: dict,
    emo_to_idx: dict,
    duration_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
    smooth_delta_sigma: float = 2.0,
):
    # Phonemes: MFA-style, single leading <sil>, drop interior spaces/punct
    raw = g2p(text)
    phonemes = ["<sil>"] + [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
    indices = vocab.encode(phonemes, add_bos_eos=True)
    phoneme_ids = torch.tensor([indices], dtype=torch.long, device=device)
    phoneme_mask = phoneme_ids != 0

    spk_t = torch.from_numpy(spk_emb).unsqueeze(0).to(device)

    # Step-by-step: need durations for per-frame emotion mapping
    enc = tf.encode_phonemes(phoneme_ids, spk_t, phoneme_mask)
    pred_dur = tf.duration_predictor(enc, phoneme_mask)
    durations = (pred_dur * duration_scale).round().clamp(min=1)
    T = int(durations.sum().item())

    decoded, _ = tf._decode_frames(enc, durations, T, phoneme_mask)
    logits = tf._run_hierarchical_heads(decoded, target_tokens=None)
    token_ids = logits.argmax(dim=-1)

    feats_norm = rvq.decode_indices(token_ids).squeeze(0).cpu().numpy()[:T]
    features = feats_norm * fs + fm  # de-normalize (global norm_stats)

    # --- EMOTION DELTA APPLICATION ---
    if emotion != "neutral" and emotion in emo_to_idx:
        e_idx = emo_to_idx[emotion]
        frame_phon_ids = build_frame_phoneme_ids(phoneme_ids, durations)  # (T,)
        # Translate vocab ids → delta-table phoneme indices (same vocab, same indexing,
        # since we built the table from vocab_mfa.json).
        # Any frame whose phoneme_id is out of range gets zero delta (safety)
        V = deltas.shape[0]
        valid = frame_phon_ids < V
        per_frame_delta = np.zeros((T, 14), dtype=np.float32)
        per_frame_delta[valid] = deltas[frame_phon_ids[valid], e_idx]

        # Temporal smoothing to avoid step changes at phoneme boundaries
        if smooth_delta_sigma > 0:
            for ch in range(14):
                per_frame_delta[:, ch] = gaussian_filter1d(
                    per_frame_delta[:, ch], sigma=smooth_delta_sigma, mode="nearest"
                )

        features = features + intensity * per_frame_delta

    # --- LOG-PITCH INVERSE + POST-PROCESSING ---
    pitch_log = features[:, 12]
    pitch_hz = np.exp(pitch_log) - 1.0
    pitch_hz[pitch_hz < 30] = 0.0  # unvoiced floor
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
    p.add_argument("text", type=str)
    p.add_argument("--emotion", type=str, default="neutral",
                   choices=["neutral", "happy", "sad", "angry", "calm"])
    p.add_argument("--intensity", type=float, default=1.0,
                   help="Multiplier for the delta vector. 0=no emotion, 1=full, 2=exaggerated")
    p.add_argument("--out", type=str, default="outputs/emo_delta/out.wav")
    p.add_argument("--duration-scale", type=float, default=1.0)
    p.add_argument("--speaker-utt", type=str, default="1885-136863-0000",
                   help="Training utterance whose spk_emb to use")
    p.add_argument("--rvq-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch/rvq_best.pt")
    p.add_argument("--transformer-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch_hier/transformer_best.pt")
    p.add_argument("--deltas", type=str, default="data/emotion_deltas.npz")
    p.add_argument("--features-dir", type=str, default="data/features_merged_logpitch")
    p.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load RVQ
    rvq_ckpt = torch.load(args.rvq_checkpoint, map_location=device, weights_only=True)
    ra = rvq_ckpt["args"]
    rvq = ArticulatoryRVQTokenizer(
        codebook_size=ra["codebook_size"], num_quantizers=ra["num_quantizers"],
        latent_dim=ra["latent_dim"], hidden_dim=ra["hidden_dim"],
    ).to(device)
    rvq.load_state_dict(rvq_ckpt["model_state_dict"])
    rvq.eval()

    # Load transformer
    tf_ckpt = torch.load(args.transformer_checkpoint, map_location=device, weights_only=True)
    ta = tf_ckpt["args"]
    tf = ArticulatoryTTSModelRVQHier(
        vocab_size=tf_ckpt["vocab_size"], codebook_size=ta["codebook_size"],
        num_quantizers=ta["num_quantizers"], d_model=ta["d_model"], nhead=ta["nhead"],
        num_encoder_layers=ta["num_layers"], num_decoder_layers=ta["num_layers"],
        d_ff=ta["d_ff"], dropout=ta.get("dropout", 0.1), speaker_emb_dim=64,
    ).to(device)
    tf.load_state_dict(tf_ckpt["model_state_dict"])
    tf.eval()

    # Norm stats
    stats = np.load(Path(args.features_dir) / "norm_stats.npz")
    fm, fs = stats["mean"], stats["std"]

    # Speaker embedding from a single training utterance (on-manifold, per Concept 5)
    ref_npz = np.load(Path(args.features_dir) / f"{args.speaker_utt}.npz")
    spk_emb = ref_npz["spk_emb"].astype(np.float32)

    # Delta table
    deltas, phon_to_idx, emo_to_idx = load_delta_table(args.deltas)

    # SPARC (memory-heaviest, load last)
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    g2p = G2p()
    vocab = PhonemeVocab(args.vocab)

    wav, sr = synthesize(
        args.text, args.emotion, args.intensity,
        tf, rvq, vocab, g2p, sparc, spk_emb, fm, fs,
        deltas, phon_to_idx, emo_to_idx,
        duration_scale=args.duration_scale, device=device,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav, sr)
    print(f"Saved {args.out}  ({len(wav)/sr:.2f}s, emotion={args.emotion}, intensity={args.intensity})")


if __name__ == "__main__":
    main()
