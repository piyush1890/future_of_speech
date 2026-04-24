"""
Exemplar-based emotion rendering.

Instead of using a single averaged delta vector per (phoneme, emotion), pull the
actual per-frame trajectory of ONE ESD utterance in the target emotion, subtract
its neutral pair (same speaker + same text), and add the resulting per-frame
delta to our synthesized features.

Rationale: averaging across thousands of emotional utterances destroys the
high-frequency zigzag (pitch spikes, energy bursts, formant sweeps) that makes
emotional speech feel alive. Using a single real exemplar preserves that zigzag.

ESD pairing: for speaker S, Neutral utterances are indices 1..350, Happy 701..1050,
Angry 351..700, Sad 1051..1400, Surprise 1401..1750. Index i in sorted Neutral
list ↔ index i in sorted Happy list (same text).
"""
from pathlib import Path
import numpy as np


EMOTIONS = ("Happy", "Sad", "Angry", "Surprise")
DEFAULT_FEATURES_DIR = "data/esd_features"
DEFAULT_SPEAKER = "0011"   # first English speaker in ESD


def _feat14(npz) -> np.ndarray:
    """Convert ESD npz → (T, 14) = [ema(12), log_pitch, loudness]."""
    ema = npz["ema"]
    pitch = npz["pitch"]
    loud = npz["loudness"]
    T = min(ema.shape[0], pitch.shape[0], loud.shape[0])
    pitch_log = np.log(pitch[:T] + 1.0)
    return np.concatenate([ema[:T], pitch_log[:, None], loud[:T, None]], axis=1).astype(np.float32)


def _time_warp(arr: np.ndarray, new_len: int) -> np.ndarray:
    """Linear time-warp a (T, D) array to (new_len, D)."""
    T, D = arr.shape
    if T == new_len:
        return arr
    idx = np.linspace(0, T - 1, new_len)
    out = np.empty((new_len, D), dtype=arr.dtype)
    src = np.arange(T)
    for c in range(D):
        out[:, c] = np.interp(idx, src, arr[:, c])
    return out


def list_exemplars(emotion: str, speaker: str = DEFAULT_SPEAKER,
                   features_dir: str = DEFAULT_FEATURES_DIR):
    """List the paired (neutral, emotional) utterance file pairs for a speaker."""
    if emotion not in EMOTIONS:
        raise ValueError(f"emotion must be one of {EMOTIONS}, got {emotion!r}")
    d = Path(features_dir)
    neu = sorted(d.glob(f"{speaker}_Neutral_{speaker}_*.npz"))
    emo = sorted(d.glob(f"{speaker}_{emotion}_{speaker}_*.npz"))
    n = min(len(neu), len(emo))
    return list(zip(neu[:n], emo[:n]))


def load_delta(emotion: str, idx: int = 0, speaker: str = DEFAULT_SPEAKER,
               features_dir: str = DEFAULT_FEATURES_DIR, smooth_neutral: bool = True):
    """Return (delta: (T_e, 14), meta: dict) for one paired exemplar."""
    pairs = list_exemplars(emotion, speaker, features_dir)
    if idx >= len(pairs):
        raise IndexError(f"only {len(pairs)} exemplars for {speaker}/{emotion}")
    neu_f, emo_f = pairs[idx]
    neu = np.load(neu_f)
    emo = np.load(emo_f)

    nf = _feat14(neu)
    ef = _feat14(emo)

    # Align neutral → emotional length (linear time-warp). Preserves the
    # emotional trajectory's zigzag; only the neutral baseline gets stretched.
    n_aligned = _time_warp(nf, ef.shape[0])

    if smooth_neutral:
        # Smooth the neutral side so small frame-level noise in the baseline
        # doesn't show up as "fake zigzag" in the delta. We want zigzag from
        # the emotional trajectory, not from the subtraction.
        from scipy.ndimage import gaussian_filter1d
        for c in range(14):
            n_aligned[:, c] = gaussian_filter1d(n_aligned[:, c], sigma=3.0, mode="nearest")

    delta = ef - n_aligned   # (T_e, 14)
    return delta, {
        "neutral_file": neu_f.name,
        "emotion_file": emo_f.name,
        "text": str(emo["text"]),
        "speaker": str(emo["speaker"]),
        "T_e": ef.shape[0],
    }


def _excitement_envelopes(emo_npz, envelope_win_ms: float = 250.0,
                          sr_frames: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """From an emotional utterance, derive two time-varying multiplier envelopes.

    pitch_excite(t): how *variable* the pitch is around its local mean, normalized
                     so the whole-utterance mean is 1. Values > 1 mean "pitch is
                     swinging more right here" → expand our pitch range here.
    loud_excite(t):  loudness magnitude relative to its mean. Values > 1 mean
                     "louder than average here" → boost our loudness here.

    Both envelopes are smoothed so application doesn't introduce frame-level jitter.
    """
    from scipy.ndimage import gaussian_filter1d
    pitch = emo_npz["pitch"].astype(np.float32)
    loud  = emo_npz["loudness"].astype(np.float32)
    T = min(pitch.shape[0], loud.shape[0])
    pitch = pitch[:T]; loud = loud[:T]

    # Rolling std of pitch over ~250ms window (voiced frames only)
    win = max(3, int(round(envelope_win_ms / 1000.0 * sr_frames)))
    voiced_mask = pitch > 40.0
    # Compute rolling std using a uniform window
    pad = win // 2
    padded = np.pad(pitch, (pad, pad), mode="edge")
    mask_padded = np.pad(voiced_mask.astype(np.float32), (pad, pad), mode="edge")
    pitch_excite = np.zeros(T, dtype=np.float32)
    for i in range(T):
        w = padded[i:i + win]
        m = mask_padded[i:i + win] > 0.5
        if m.sum() >= 3:
            pitch_excite[i] = w[m].std()
        else:
            pitch_excite[i] = 0.0
    pitch_excite = gaussian_filter1d(pitch_excite, sigma=6.0, mode="nearest")
    # Normalize: mean = 1. Clip extreme multipliers.
    pe_mean = pitch_excite.mean() + 1e-6
    pitch_excite = pitch_excite / pe_mean
    pitch_excite = np.clip(pitch_excite, 0.5, 3.0)

    loud_excite = gaussian_filter1d(loud, sigma=6.0, mode="nearest")
    le_mean = loud_excite.mean() + 1e-6
    loud_excite = loud_excite / le_mean
    loud_excite = np.clip(loud_excite, 0.6, 2.5)

    return pitch_excite, loud_excite


def find_best_exemplar(target_n_frames: int, target_is_question: bool,
                       emotion: str, speaker: str = DEFAULT_SPEAKER,
                       features_dir: str = DEFAULT_FEATURES_DIR) -> int:
    """Find the exemplar index (within the emotion pool) best matching our target.

    Scoring: favor exemplars whose frame count is close to ours, whose sentence
    type matches (question vs non-question via exemplar text's final punct).
    """
    pairs = list_exemplars(emotion, speaker, features_dir)
    best_idx, best_score = 0, -float("inf")
    for i, (neu_f, emo_f) in enumerate(pairs):
        emo = np.load(emo_f)
        T_e = int(emo["pitch"].shape[0])
        text = str(emo["text"]).strip()
        is_q = text.endswith("?")
        # Normalized duration distance: 1.0 at perfect match, → 0 as they diverge
        dur_ratio = min(T_e, target_n_frames) / max(T_e, target_n_frames, 1)
        # Sentence-type bonus
        type_bonus = 0.25 if is_q == target_is_question else 0.0
        score = dur_ratio + type_bonus
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx


def apply_envelopes_to_features(features: np.ndarray, emotion: str,
                                idx: int = None, target_is_question: bool = False,
                                pitch_intensity: float = 1.0,
                                loud_intensity: float = 1.0,
                                speaker: str = DEFAULT_SPEAKER,
                                features_dir: str = DEFAULT_FEATURES_DIR) -> tuple:
    """Multiplicative emotion application.

    If idx is None, pick the best-matching exemplar for this utterance via
    find_best_exemplar.

    Features are expected as (T, 14) = [ema(12), log_pitch, loudness].
    We expand our pitch deviations and loudness deviations around their per-frame
    smoothed mean, scaled by the exemplar's excitement envelopes time-warped to
    our length. EMA is untouched.
    """
    from scipy.ndimage import gaussian_filter1d
    T_out = features.shape[0]

    if idx is None:
        idx = find_best_exemplar(T_out, target_is_question, emotion, speaker, features_dir)

    pairs = list_exemplars(emotion, speaker, features_dir)
    neu_f, emo_f = pairs[idx]
    emo = np.load(emo_f)

    pitch_excite, loud_excite = _excitement_envelopes(emo)

    # Warp envelopes to our length
    def warp(env, T_out):
        T = len(env)
        if T == T_out:
            return env
        return np.interp(np.linspace(0, T - 1, T_out), np.arange(T), env)
    pitch_excite = warp(pitch_excite, T_out).astype(np.float32)
    loud_excite  = warp(loud_excite,  T_out).astype(np.float32)

    # Apply to our log_pitch (ch 12) around a smoothed local mean, so we expand
    # vibrato around the sentence contour instead of amplifying the contour itself.
    out = features.copy()
    lp = out[:, 12].astype(np.float32)
    lp_mean = gaussian_filter1d(lp, sigma=20.0, mode="nearest")
    lp_dev = lp - lp_mean
    # Blend 1.0 with the envelope by pitch_intensity: intensity=0 → no effect.
    pitch_mult = 1.0 + pitch_intensity * (pitch_excite - 1.0)
    lp_new = lp_mean + lp_dev * pitch_mult
    out[:, 12] = lp_new

    # Apply to loudness (ch 13) similarly
    ld = out[:, 13].astype(np.float32)
    ld_mean = gaussian_filter1d(ld, sigma=20.0, mode="nearest")
    ld_dev = ld - ld_mean
    loud_mult = 1.0 + loud_intensity * (loud_excite - 1.0)
    out[:, 13] = ld_mean + ld_dev * loud_mult

    meta = {
        "emotion_file": emo_f.name,
        "text": str(emo["text"]),
        "idx": idx,
        "T_e": int(emo["pitch"].shape[0]),
    }
    return out, meta


def apply_to_features(features: np.ndarray, emotion: str, intensity: float = 1.0,
                      idx: int = 0, speaker: str = DEFAULT_SPEAKER,
                      features_dir: str = DEFAULT_FEATURES_DIR,
                      smooth_delta_sigma: float = 1.5,
                      ema_scale: float = 0.0,
                      voicing_mask: bool = True) -> tuple:
    """Add exemplar-based emotion delta to (T, 14) features.

    EMA channels (0..11) carry phoneme identity, so applying another speaker's
    EMA trajectory onto ours pushes articulators off-manifold and garbles SPARC.
    By default `ema_scale=0.0` — only pitch (ch 12) and loudness (ch 13) get the
    delta. Set ema_scale > 0 to reintroduce EMA contribution (use cautiously).

    `intensity` scales pitch + loudness channels together.

    `voicing_mask`: gate pitch delta by both our-voiced AND exemplar-voiced flags.
    Prevents chirps at consonants from frame-scale phoneme-boundary mismatch.
    """
    pairs = list_exemplars(emotion, speaker, features_dir)
    if idx >= len(pairs):
        raise IndexError(f"only {len(pairs)} exemplars for {speaker}/{emotion}")
    neu_f, emo_f = pairs[idx]
    neu = np.load(neu_f)
    emo = np.load(emo_f)

    delta, meta = load_delta(emotion, idx, speaker, features_dir)
    T_out = features.shape[0]
    delta_warped = _time_warp(delta, T_out)

    # Build voicing masks (both sides) if enabled, BEFORE smoothing delta,
    # so the mask is crisp at frame boundaries.
    if voicing_mask:
        # Exemplar-voiced: warp the raw exemplar pitch voicing flag
        emo_pitch = emo["pitch"][:delta.shape[0]]
        emo_voiced = (emo_pitch > 40.0).astype(np.float32)
        # Smooth the voicing mask over a few frames to avoid hard on/off switching
        emo_voiced_warped = _time_warp(emo_voiced[:, None], T_out)[:, 0]
        emo_voiced_mask = (emo_voiced_warped > 0.5).astype(np.float32)

        # Our-voiced: from our log-pitch feature directly
        # log(30+1) = ~3.43, our voicing floor
        our_voiced_mask = (features[:, 12] > np.log(30 + 1.0)).astype(np.float32)

        from scipy.ndimage import gaussian_filter1d
        # Soft mask so the gate doesn't click on/off
        voice_gate = our_voiced_mask * emo_voiced_mask
        voice_gate = gaussian_filter1d(voice_gate, sigma=2.0, mode="nearest")
        # Zero pitch delta on frames that aren't voiced-on-both-sides
        delta_warped[:, 12] = delta_warped[:, 12] * voice_gate
        # Loudness: gate only by our voicing (doesn't need to match exemplar's
        # consonant pattern — loudness shift on consonants is fine)
        delta_warped[:, 13] = delta_warped[:, 13] * gaussian_filter1d(
            our_voiced_mask, sigma=2.0, mode="nearest"
        )

    if smooth_delta_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        for c in range(14):
            delta_warped[:, c] = gaussian_filter1d(
                delta_warped[:, c], sigma=smooth_delta_sigma, mode="nearest"
            )

    out = features.copy()
    if ema_scale != 0.0:
        out[:, :12] = out[:, :12] + ema_scale * delta_warped[:, :12]
    out[:, 12] = out[:, 12] + intensity * delta_warped[:, 12]
    out[:, 13] = out[:, 13] + intensity * delta_warped[:, 13]
    return out, meta
