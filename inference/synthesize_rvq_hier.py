"""
Hierarchical RVQ inference pipeline: text → phonemes → hier transformer → RVQ decode → SPARC.

Three lessons baked in from diagnostics:
  1. Training data has no <sil> between words (MFA-style). g2p_en inserting one per space
     creates a listy, start-stop delivery. We keep only a leading <sil>.
  2. Predicted pitch has ~2.6× GT frame-to-frame jitter, causing vocal tremor. A 5-frame
     median filter on pitch removes it with no audible blur. Enabled by default; disable
     via --no-smooth-pitch.
  3. The 14-dim SPARC features (EMA + pitch + loudness) give us usable knobs for emotion
     without any retraining: pitch mean/variance, loudness offset, duration. F0=0 across
     the whole utterance gives passable whisper since SPARC's decoder learned unvoiced
     synthesis from training-set unvoiced consonants.

Emotion presets are linear feature transforms applied after the transformer predicts
features. No emotional training data is used or needed for these.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


# Emotion presets. Keys:
#   dur          — global duration scale (multiplies all phoneme durations)
#   pm           — pitch mean scale (applied to voiced frames)
#   pv           — pitch variance scale (around mean, on voiced frames)
#   p_override   — if set, force pitch to this value globally (0 = whisper)
#   loud_delta   — additive offset to loudness feature BEFORE SPARC decode.
#                  NEGATIVE ONLY. Positive values push SPARC outside training
#                  distribution and produce buzzy artifacts — use wav_scale instead.
#   wav_scale    — multiplicative scaling of the output waveform AFTER SPARC.
#                  Safe lever for "louder" effects.
#   stress_pm    — extra pitch-mean boost applied ONLY on frames of words marked
#                  as stressed via *word* in the input text. 1.0 = no stress emphasis.
#   stress_dur   — extra duration stretch on stressed phonemes (pre-length-regulator).
#   stress_gain  — extra waveform amplitude on stressed frame ranges.
# Stress requires the user to mark words with asterisks, e.g.
#   "I can *not* believe *you* did *that*."
# Global prosody (pm/pv/loud_delta/wav_scale) still applies everywhere; stress
# adds local emphasis on top.
EMOTION_PRESETS = {
    "neutral": dict(dur=1.00, pm=1.00, pv=1.00, p_override=None, loud_delta= 0.0,
                    wav_scale=1.00, stress_pm=1.00, stress_dur=1.00, stress_gain=1.00),
    "whisper": dict(dur=1.10, pm=1.00, pv=1.00, p_override=0.0,  loud_delta=-2.0,
                    wav_scale=1.00, stress_pm=1.00, stress_dur=1.00, stress_gain=1.00),
    "happy":   dict(dur=0.92, pm=1.15, pv=1.40, p_override=None, loud_delta= 0.0,
                    wav_scale=1.10, stress_pm=1.15, stress_dur=1.05, stress_gain=1.10),
    "sad":     dict(dur=1.25, pm=0.92, pv=0.50, p_override=None, loud_delta=-0.5,
                    wav_scale=1.00, stress_pm=1.00, stress_dur=1.00, stress_gain=1.00),
    "angry":   dict(dur=1.00, pm=1.00, pv=0.85, p_override=None, loud_delta= 0.0,
                    wav_scale=1.15, stress_pm=1.25, stress_dur=1.20, stress_gain=1.30),
    "calm":    dict(dur=1.10, pm=0.95, pv=0.80, p_override=None, loud_delta=-0.2,
                    wav_scale=1.00, stress_pm=1.00, stress_dur=1.00, stress_gain=1.00),
    # Seductive: MEASURED against 2 ElevenLabs Natasha refs + 1 neutral training
    # speaker via SPARC encoding. Corrections vs earlier guessed preset:
    #   pv is NOT reduced (ref vs neutral ratio = 0.92, essentially same)
    #   loud is NOT reduced (delta = +0.01)
    #   wav_scale is NOT reduced (same presence as neutral)
    # The real differentiator is pitch MEAN (~17% lower, ratio 0.83) and slower,
    # continuous delivery (reference clips show 100% voiced frames — drawn-out,
    # no word-final cutoffs). Stress lingers longer at same volume, pitch drops
    # slightly on stressed word for intimate emphasis.
    "seductive": dict(dur=1.20, pm=0.83, pv=0.95, p_override=None, loud_delta=0.0,
                     wav_scale=1.00, stress_pm=0.95, stress_dur=1.25, stress_gain=1.00),
}


def apply_emotion(features: np.ndarray, preset: dict, ramp_frames: int = 15) -> np.ndarray:
    """features: (T, 14), returns modified (T, 14). Non-destructive.

    Applies pitch and loudness-feature transforms to voiced frames only,
    with linear ramp-in/out at voicing boundaries to prevent click artifacts.
    Does NOT apply wav_scale — that's handled post-decode.
    """
    pm_scale = preset["pm"]
    pv_scale = preset["pv"]
    p_override = preset["p_override"]
    loud_delta = preset["loud_delta"]

    out = features.copy()
    pitch = out[:, 12]
    T = features.shape[0]

    voiced = pitch > 40.0
    voiced_idx = np.where(voiced)[0]

    if p_override is not None:
        # Whisper: F0=0 everywhere. Loudness delta applied uniformly (not ramped) —
        # whisper is quiet *throughout*, not just in the middle. Ramping leaves
        # edges at full (non-whisper) loudness with F0=0, which sounds glitchy.
        out[:, 12] = p_override
        if loud_delta != 0.0:
            out[:, 13] = out[:, 13] + loud_delta
        return out

    if len(voiced_idx) == 0:
        return out

    strength = _build_ramp(T, voiced_idx, ramp_frames)

    mean_p = pitch[voiced].mean()
    target = (mean_p + (pitch - mean_p) * pv_scale) * pm_scale
    target = np.clip(target, 0, None)
    out[voiced, 12] = (1 - strength[voiced]) * pitch[voiced] + strength[voiced] * target[voiced]

    if loud_delta != 0.0:
        out[:, 13] = out[:, 13] + loud_delta * strength

    return out


def _build_ramp(T: int, voiced_idx: np.ndarray, ramp_frames: int) -> np.ndarray:
    """1 in the middle of voiced region, 0 at edges/silence, linear ramp on flanks."""
    ramp = np.zeros(T, dtype=np.float32)
    if len(voiced_idx) == 0:
        return ramp
    first, last = voiced_idx[0], voiced_idx[-1]
    for i, v in enumerate(voiced_idx):
        from_start = i
        from_end = len(voiced_idx) - 1 - i
        ramp[v] = min(1.0, from_start / ramp_frames, from_end / ramp_frames)
    return ramp


class ArticulatoryTTSRVQHier:
    def __init__(
        self,
        rvq_checkpoint: str,
        transformer_checkpoint: str,
        vocab_path: str,
        norm_stats_path: str,
        sparc_model: str = "en",
        device: str = "cpu",
        smooth_pitch: bool = True,
        smooth_window: int = 5,
    ):
        self.device = torch.device(device)
        self.smooth_pitch = smooth_pitch
        self.smooth_window = smooth_window

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"]
        self.feat_std = stats["std"]

        self.vocab = PhonemeVocab(vocab_path)
        self.g2p = G2p()

        rvq_ckpt = torch.load(rvq_checkpoint, map_location=self.device, weights_only=True)
        ra = rvq_ckpt["args"]
        self.rvq = ArticulatoryRVQTokenizer(
            codebook_size=ra["codebook_size"],
            num_quantizers=ra["num_quantizers"],
            latent_dim=ra["latent_dim"],
            hidden_dim=ra["hidden_dim"],
        ).to(self.device)
        self.rvq.load_state_dict(rvq_ckpt["model_state_dict"])
        self.rvq.eval()

        tf_ckpt = torch.load(transformer_checkpoint, map_location=self.device, weights_only=True)
        ta = tf_ckpt["args"]
        self.transformer = ArticulatoryTTSModelRVQHier(
            vocab_size=tf_ckpt["vocab_size"],
            codebook_size=ta["codebook_size"],
            num_quantizers=ta["num_quantizers"],
            d_model=ta["d_model"],
            nhead=ta["nhead"],
            num_encoder_layers=ta["num_layers"],
            num_decoder_layers=ta["num_layers"],
            d_ff=ta["d_ff"],
            dropout=ta.get("dropout", 0.1),
            speaker_emb_dim=64,
        ).to(self.device)
        self.transformer.load_state_dict(tf_ckpt["model_state_dict"])
        self.transformer.eval()

        from sparc import load_model as load_sparc
        self.sparc = load_sparc(sparc_model, device="cpu")

        print(f"Hierarchical RVQ TTS loaded (epoch {tf_ckpt.get('epoch','?')}, "
              f"val={tf_ckpt.get('val_loss', float('nan')):.4f}). "
              f"Smooth pitch: {self.smooth_pitch} (window={self.smooth_window}).")

    def text_to_phonemes(self, text):
        """MFA-style: drop spaces and punctuation, single leading <sil>.
        Also parses *word* markers to return per-phoneme stress flags.
        Returns (phonemes, stress_flags) — both lists of same length."""
        import re
        parts = re.split(r"(\*[^*]+\*)", text)
        phonemes = ["<sil>"]
        stress = [False]
        for part in parts:
            if not part:
                continue
            is_stress = part.startswith("*") and part.endswith("*")
            plain = part.strip("*")
            raw = self.g2p(plain)
            for p in raw:
                if p and p[0].isalpha() and p[0].isupper():
                    phonemes.append(p)
                    stress.append(is_stress)
        return phonemes, stress

    @torch.no_grad()
    def synthesize(self, text, speaker_emb=None, duration_scale=1.0, emotion="neutral"):
        if emotion not in EMOTION_PRESETS:
            raise ValueError(f"Unknown emotion '{emotion}'. Choices: {list(EMOTION_PRESETS)}")
        preset = EMOTION_PRESETS[emotion]
        effective_dur_scale = duration_scale * preset["dur"]

        phonemes, phon_stress = self.text_to_phonemes(text)
        indices = self.vocab.encode(phonemes, add_bos_eos=True)
        # indices = [BOS] + phonemes + [EOS], length N+2
        phoneme_ids = torch.tensor([indices], dtype=torch.long, device=self.device)
        phoneme_mask = phoneme_ids != 0

        # Stress mask aligned with indices: BOS=False, each phoneme's flag, EOS=False
        stress_per_index = torch.tensor(
            [False] + phon_stress + [False], dtype=torch.bool, device=self.device
        ).unsqueeze(0)  # (1, N+2)

        if speaker_emb is None:
            speaker_emb = np.zeros(64, dtype=np.float32)
        spk = torch.from_numpy(speaker_emb).unsqueeze(0).to(self.device)

        # Run model step-by-step so we can inject per-phoneme duration control.
        enc = self.transformer.encode_phonemes(phoneme_ids, spk, phoneme_mask)
        pred_dur = self.transformer.duration_predictor(enc, phoneme_mask)  # (1, N+2) float

        # Per-phoneme stretch: stressed phonemes slightly longer.
        dur_mult = torch.where(
            stress_per_index,
            torch.tensor(preset["stress_dur"], device=self.device, dtype=pred_dur.dtype),
            torch.tensor(1.0, device=self.device, dtype=pred_dur.dtype),
        )
        durations = (pred_dur * dur_mult * effective_dur_scale).round().clamp(min=1)

        # Build per-frame stress mask from integer durations.
        dur_np = durations[0].cpu().numpy().astype(int)
        stress_np = stress_per_index[0].cpu().numpy()
        T = int(dur_np.sum())
        frame_stress = np.zeros(T, dtype=bool)
        cursor = 0
        for i, d in enumerate(dur_np):
            if stress_np[i]:
                end = min(cursor + d, T)
                frame_stress[cursor:end] = True
            cursor += d

        # Decode frames with the stress-aware durations.
        decoded, frame_mask = self.transformer._decode_frames(enc, durations, T, phoneme_mask)
        logits = self.transformer._run_hierarchical_heads(decoded, target_tokens=None)
        token_ids = logits.argmax(dim=-1)

        features_norm = self.rvq.decode_indices(token_ids).squeeze(0).cpu().numpy()
        features = features_norm * self.feat_std + self.feat_mean
        features = features[:T]  # trim to exact length

        if self.smooth_pitch and self.smooth_window > 1:
            from scipy.ndimage import median_filter, gaussian_filter1d
            # Pitch carries the tremor — smooth hard here (median+gaussian).
            features[:, 12] = median_filter(
                features[:, 12], size=max(self.smooth_window, 7), mode="nearest"
            )
            features[:, 12] = gaussian_filter1d(features[:, 12], sigma=1.5, mode="nearest")
            # EMA is the articulation — keep light so phonemes stay crisp. σ=1 ≈ 20 ms.
            for k in range(12):
                features[:, k] = gaussian_filter1d(features[:, k], sigma=1.0, mode="nearest")
            # Loudness only needs a touch.
            features[:, 13] = gaussian_filter1d(features[:, 13], sigma=1.0, mode="nearest")

        # Apply global emotion transforms (pitch mean/var, loudness, F0 override).
        features = apply_emotion(features, preset)

        # Per-frame stress pitch boost. Uses a soft (gaussian-ramped) stress mask so
        # the boost fades in/out at phoneme boundaries, and re-smooths pitch after
        # application so the boost itself doesn't amplify any residual jitter.
        if preset["stress_pm"] != 1.0 and frame_stress.any():
            from scipy.ndimage import gaussian_filter1d
            soft_mask = gaussian_filter1d(frame_stress.astype(np.float32), sigma=3.0)
            # soft_mask is 0..~1; multiplier ramps 1.0 → stress_pm as soft_mask rises.
            mult = 1.0 + (preset["stress_pm"] - 1.0) * soft_mask
            voiced = features[:, 12] > 40.0
            features[voiced, 12] = features[voiced, 12] * mult[voiced]
            # Re-smooth pitch so the boost doesn't amplify jitter inside stress regions.
            features[:, 12] = gaussian_filter1d(features[:, 12], sigma=1.5, mode="nearest")

        ema = features[:, :12]
        pitch = features[:, 12]
        loudness = features[:, 13]

        waveform = self.sparc.decode(ema, pitch, loudness, speaker_emb)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().squeeze().cpu().numpy()

        # Per-frame stress amplitude gain on the output waveform.
        if preset["stress_gain"] != 1.0 and frame_stress.any():
            samples_per_frame = self.sparc.sr / 50  # SPARC runs at 50 Hz
            gain = np.ones(len(waveform), dtype=np.float32)
            for t in range(T):
                if frame_stress[t]:
                    s = int(t * samples_per_frame)
                    e = int((t + 1) * samples_per_frame)
                    gain[s:e] = preset["stress_gain"]
            # Smooth gain boundaries (~10 ms) so gain steps don't click.
            from scipy.ndimage import gaussian_filter1d
            gain = gaussian_filter1d(gain, sigma=samples_per_frame * 0.5)
            waveform = waveform * gain

        # Global waveform scaling (soft-clipped to prevent distortion above 1.0).
        if preset["wav_scale"] != 1.0:
            waveform = np.tanh(waveform * preset["wav_scale"])

        return waveform, self.sparc.sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--rvq-checkpoint", type=str, default="checkpoints_rvq/rvq_best.pt")
    parser.add_argument("--transformer-checkpoint", type=str,
                        default="checkpoints_rvq_hier/transformer_best.pt")
    parser.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    parser.add_argument("--norm-stats", type=str, default="data/features_merged/norm_stats.npz")
    parser.add_argument("--speaker-emb", type=str, default=None,
                        help="Path to .npy (64-dim). Defaults to first speaker in features_merged.")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=str, default="outputs/output_rvq_hier.wav")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-smooth-pitch", dest="smooth_pitch", action="store_false",
                        help="Disable median-5 pitch smoothing (on by default)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Median filter window on pitch (frames, 50 Hz)")
    parser.add_argument("--mode", "--emotion", dest="emotion", type=str, default="neutral",
                        choices=list(EMOTION_PRESETS),
                        help="Emotion preset: neutral/whisper/happy/sad/angry/calm")
    parser.set_defaults(smooth_pitch=True)
    args = parser.parse_args()

    tts = ArticulatoryTTSRVQHier(
        rvq_checkpoint=args.rvq_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        vocab_path=args.vocab_path,
        norm_stats_path=args.norm_stats,
        device=args.device,
        smooth_pitch=args.smooth_pitch,
        smooth_window=args.smooth_window,
    )

    spk_emb = None
    if args.speaker_emb:
        spk_emb = np.load(args.speaker_emb)
    else:
        import json
        with open("data/features_merged/speaker_embeddings.json") as f:
            spk_embs = json.load(f)
        spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

    waveform, sr = tts.synthesize(args.text, spk_emb, args.duration_scale, emotion=args.emotion)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({len(waveform)/sr:.2f}s)")


if __name__ == "__main__":
    main()
