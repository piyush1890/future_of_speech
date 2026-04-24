"""
Inference-side prosody layer addressing three perceptual gaps vs natural speech:
  1) Per-word speaking-rate variation (content vs function words)
  2) Pitch range expansion (compensate model's regression-to-mean)
  3) Phrase-level pitch contours (declarative fall, question rise)

Entirely inference-side. No retraining, no model changes.
"""
import argparse
import json
import sys
import re
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

import spacy


# POS-to-prosody-class mapping
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
FUNCTION_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}
# Everything else (INTJ, SYM, PUNCT, X): treated as neutral (scale 1.0)


PROSODY_PRESETS = {
    "standard": {
        "content_dur": 1.30, "content_pitch": 0.05,
        "function_dur": 0.70, "function_pitch": -0.03,
        "pitch_range_factor": 1.25,
        "sentence_final_fall": -0.15,
        "question_rise": 0.20,
    },
    "charming": {
        # Highly expressive: bigger duration swing, stronger pitch lift on content,
        # aggressive range expansion, lower silence density (use short gaps in caller)
        "content_dur": 1.40, "content_pitch": 0.10,
        "function_dur": 0.60, "function_pitch": -0.05,
        "pitch_range_factor": 1.80,
        "sentence_final_fall": -0.10,   # gentler fall (charming speech often doesn't drop)
        "question_rise": 0.25,
    },
}


def classify_word_prosody(token, preset):
    """Return (duration_scale, pitch_boost_log) for a spaCy token under a preset."""
    pos = token.pos_
    if pos in CONTENT_POS:
        return preset["content_dur"], preset["content_pitch"]
    if pos in FUNCTION_POS:
        return preset["function_dur"], preset["function_pitch"]
    return 1.00, 0.0


def split_text_into_words(text):
    """Tokenize preserving punctuation and sentence boundaries via spaCy."""
    return text


def build_frame_word_map(phoneme_ids, durations_per_phoneme, phoneme_word_ids):
    """Expand phoneme-level info into a frame-level word-id array."""
    phoneme_ids_np = phoneme_ids[0].cpu().numpy()
    durs = durations_per_phoneme[0].cpu().numpy().astype(int)
    pw = np.asarray(phoneme_word_ids, dtype=np.int64)
    # For each phoneme, repeat its word_id `d` times
    frame_word_ids = np.repeat(pw, durs)
    return frame_word_ids


def apply_sentence_contour(pitch_log, sentence_ends_with_question: bool, fade_frames: int = 12):
    """Apply phrase-level pitch contour at the end of an utterance.
    pitch_log: (T,) log-space pitch. Fade effect into the last `fade_frames`.
    """
    T = len(pitch_log)
    if T < fade_frames * 2:
        return pitch_log
    out = pitch_log.copy()
    # Build a ramp from 0 at T-fade to the target shift at T-1
    ramp = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    if sentence_ends_with_question:
        shift = +0.20   # rising intonation (~22% pitch rise at end)
    else:
        shift = -0.15   # declarative fall (~14% drop at end)
    out[-fade_frames:] = out[-fade_frames:] + ramp * shift
    return out


def expand_pitch_range(pitch_hz, factor: float = 1.25):
    """Expand pitch range around the voiced mean. Only affects voiced frames.
    Addresses regression-to-mean in the model's predictions."""
    voiced = pitch_hz > 40.0
    if voiced.sum() < 5:
        return pitch_hz
    mean_p = pitch_hz[voiced].mean()
    out = pitch_hz.copy()
    out[voiced] = mean_p + (pitch_hz[voiced] - mean_p) * factor
    out = np.clip(out, 40, 500)
    return out


@torch.no_grad()
def synthesize_sentence(
    sent_text: str,
    nlp,
    tf, rvq, vocab, g2p, sparc, spk_emb,
    fm, fs,
    device=torch.device("cpu"),
    prosody_enabled: bool = True,
    preset_name: str = "standard",
):
    preset = PROSODY_PRESETS[preset_name]
    pitch_range_factor = preset["pitch_range_factor"]
    # Tokenize with spaCy to get word-level POS and duration/pitch scales
    doc = nlp(sent_text)
    word_info = []   # list of (text, dur_scale, pitch_boost_log) for non-punctuation tokens
    for tok in doc:
        if tok.is_punct or tok.is_space or not tok.text.strip():
            continue
        dur_scale, pitch_boost = classify_word_prosody(tok, preset)
        word_info.append((tok.text, dur_scale, pitch_boost))

    # Build phoneme sequence with per-phoneme (word_id, dur_scale, pitch_boost)
    phonemes = ["<sil>"]
    phoneme_word_ids = [-1]      # -1 means "leading silence, no word"
    phoneme_dur_scales = [1.0]   # no stretch on leading silence
    phoneme_pitch_boost = [0.0]
    for word_idx, (word, dur_s, pb) in enumerate(word_info):
        raw = g2p(word)
        arp = [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
        for ph in arp:
            phonemes.append(ph)
            phoneme_word_ids.append(word_idx)
            phoneme_dur_scales.append(dur_s)
            phoneme_pitch_boost.append(pb)

    if len(phonemes) == 1:  # only <sil>, no actual phonemes — skip
        return np.zeros(int(0.1 * sparc.sr), dtype=np.float32), sparc.sr

    # Encode + predict durations step by step (need to apply per-phoneme dur scale)
    indices = vocab.encode(phonemes, add_bos_eos=True)
    phoneme_ids = torch.tensor([indices], dtype=torch.long, device=device)
    phoneme_mask = phoneme_ids != 0
    spk_t = torch.from_numpy(spk_emb).unsqueeze(0).to(device)

    enc = tf.encode_phonemes(phoneme_ids, spk_t, phoneme_mask)
    pred_dur = tf.duration_predictor(enc, phoneme_mask)  # (1, N+2) [BOS, phonemes..., EOS]

    if prosody_enabled:
        # Build per-phoneme duration multiplier aligned with pred_dur's positions.
        # pred_dur indices: 0=BOS, 1..=phonemes, last=EOS.
        # phoneme_dur_scales is ordered as [<sil>, phoneme1, phoneme2, ...].
        dur_mults = [1.0]   # BOS
        for d_s in phoneme_dur_scales:
            dur_mults.append(d_s)
        dur_mults.append(1.0)  # EOS
        assert len(dur_mults) == pred_dur.shape[1], f'{len(dur_mults)} vs {pred_dur.shape[1]}'
        mult_tensor = torch.tensor(dur_mults, device=device, dtype=pred_dur.dtype).unsqueeze(0)
        pred_dur = pred_dur * mult_tensor

    durations = pred_dur.round().clamp(min=1)
    T = int(durations.sum().item())

    decoded, _ = tf._decode_frames(enc, durations, T, phoneme_mask)
    logits = tf._run_hierarchical_heads(decoded, target_tokens=None)
    tok = logits.argmax(dim=-1)
    feats_norm = rvq.decode_indices(tok).squeeze(0).cpu().numpy()[:T]
    features = feats_norm * fs + fm  # de-normalize

    if prosody_enabled:
        # Apply per-frame pitch boost for content words (in log space)
        # Build frame-level word-id + pitch-boost arrays
        # pred_dur has BOS + sil + phonemes + EOS; we need pitch_boost per phoneme (including BOS/EOS=0, sil=0)
        per_phoneme_pitch_boost = [0.0] + phoneme_pitch_boost + [0.0]
        pb_per_phoneme = np.asarray(per_phoneme_pitch_boost, dtype=np.float32)
        durs_np = durations[0].cpu().numpy().astype(int)
        frame_pb = np.repeat(pb_per_phoneme, durs_np)
        # Smooth with a short gaussian to avoid step changes at word boundaries
        frame_pb = gaussian_filter1d(frame_pb, sigma=2.0, mode="nearest")
        features[:, 12] = features[:, 12] + frame_pb

        # Apply sentence-final contour based on punctuation (preset-specific magnitudes)
        q = sent_text.strip().endswith('?')
        shift = preset["question_rise"] if q else preset["sentence_final_fall"]
        fade = 15
        if len(features) > fade * 2:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            features[-fade:, 12] = features[-fade:, 12] + ramp * shift

    # Log-pitch → Hz, smooth, optionally expand range
    pitch_log = features[:, 12]
    pitch_hz = np.exp(pitch_log) - 1.0
    pitch_hz[pitch_hz < 30] = 0.0
    pitch_hz = median_filter(pitch_hz, size=5, mode="nearest")
    pitch_hz = gaussian_filter1d(pitch_hz, sigma=1.0, mode="nearest")

    if prosody_enabled and pitch_range_factor != 1.0:
        pitch_hz = expand_pitch_range(pitch_hz, factor=pitch_range_factor)

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
    p.add_argument("--text-file", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--no-prosody", action="store_true", help="Disable prosody layer (baseline)")
    p.add_argument("--preset", type=str, default="standard", choices=list(PROSODY_PRESETS.keys()))
    p.add_argument("--speaker-utt", type=str, default="1322-137588-0000")
    p.add_argument("--rvq-checkpoint", type=str, default="checkpoints_rvq_logpitch/rvq_best.pt")
    p.add_argument("--transformer-checkpoint", type=str,
                   default="checkpoints_rvq_logpitch_hier/transformer_best.pt")
    p.add_argument("--features-dir", type=str, default="data/features_merged_logpitch")
    p.add_argument("--vocab", type=str, default="data/processed_all/vocab_mfa.json")
    p.add_argument("--sentence-gap", type=float, default=0.30)
    a = p.parse_args()

    device = torch.device("cpu")
    rvq_ckpt = torch.load(a.rvq_checkpoint, map_location=device, weights_only=True)
    ra = rvq_ckpt["args"]
    rvq = ArticulatoryRVQTokenizer(
        codebook_size=ra["codebook_size"], num_quantizers=ra["num_quantizers"],
        latent_dim=ra["latent_dim"], hidden_dim=ra["hidden_dim"],
    ).to(device)
    rvq.load_state_dict(rvq_ckpt["model_state_dict"])
    rvq.eval()

    tf_ckpt = torch.load(a.transformer_checkpoint, map_location=device, weights_only=True)
    ta = tf_ckpt["args"]
    tf = ArticulatoryTTSModelRVQHier(
        vocab_size=tf_ckpt["vocab_size"], codebook_size=ta["codebook_size"],
        num_quantizers=ta["num_quantizers"], d_model=ta["d_model"], nhead=ta["nhead"],
        num_encoder_layers=ta["num_layers"], num_decoder_layers=ta["num_layers"],
        d_ff=ta["d_ff"], dropout=ta.get("dropout", 0.1), speaker_emb_dim=64,
    ).to(device)
    tf.load_state_dict(tf_ckpt["model_state_dict"])
    tf.eval()

    stats = np.load(Path(a.features_dir) / "norm_stats.npz")
    fm, fs = stats["mean"], stats["std"]
    spk_emb = np.load(Path(a.features_dir) / f"{a.speaker_utt}.npz")["spk_emb"].astype(np.float32)

    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    g2p = G2p()
    vocab = PhonemeVocab(a.vocab)
    nlp = spacy.load("en_core_web_sm")

    with open(a.text_file) as f:
        text = f.read().strip()
    # Split on BOTH sentence-ending punctuation and commas.
    # Each segment keeps its trailing punctuation so the sentence-final contour logic sees it.
    # We annotate each chunk with its follow-on gap length.
    COMMA_GAP = 0.12       # pause at commas / semicolons / colons / dashes
    SENTENCE_GAP = a.sentence_gap
    # Regex splits on any of .!?,;: while keeping the delimiter at end of each piece.
    tokens = re.split(r"([.!?,;:—-])", text)
    segments = []   # list of (text, gap_after)
    buf = ""
    for tok in tokens:
        if not tok:
            continue
        if tok in ".!?,;:—-":
            buf += tok
            if tok in ".!?":
                segments.append((buf.strip(), SENTENCE_GAP))
                buf = ""
            elif tok in ",;:—-":
                if buf.strip():
                    segments.append((buf.strip(), COMMA_GAP))
                buf = ""
        else:
            buf += tok
    if buf.strip():
        segments.append((buf.strip(), SENTENCE_GAP))

    preset_label = 'OFF' if a.no_prosody else a.preset
    print(f"Synthesizing {len(segments)} segments, prosody preset={preset_label}")

    chunks = []
    sr = sparc.sr
    for s, gap in segments:
        if len(s) < 2:
            continue
        wav, sr = synthesize_sentence(
            s, nlp, tf, rvq, vocab, g2p, sparc, spk_emb, fm, fs,
            device=device,
            prosody_enabled=not a.no_prosody,
            preset_name=a.preset,
        )
        chunks.append(wav)
        chunks.append(np.zeros(int(sr * gap), dtype=wav.dtype))
    full = np.concatenate(chunks[:-1]) if chunks else np.zeros(int(sr*0.1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, full, sr)
    print(f"Saved {a.out}  ({len(full)/sr:.1f}s)")


if __name__ == "__main__":
    main()
