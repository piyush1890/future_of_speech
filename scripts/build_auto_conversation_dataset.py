#!/usr/bin/env python3
"""
Build an automatic conversational speech dataset from owned media.

Pipeline:
  media/video -> ffmpeg dialogue-friendly cleanup -> Whisper ASR -> sentence chunks
  -> punctuation normalization -> emotion/intensity tags -> quality filtering
  -> clipped WAVs + JSONL manifest.

The script is intentionally dependency-light for the pilot. If Demucs is
installed, pass --separator demucs to use vocal separation before denoising.
If the Audeering V/A/D model can be loaded, --emotion-model auto will use it;
otherwise the script falls back to a deterministic prosody/text heuristic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import librosa
import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HINDI_QUESTION_RE = re.compile(
    r"(?:^|\s)(क्या|क्यों|कहाँ|कहा|कब|कैसे|कौन|किस|कितना|कितनी|कितने|"
    r"किधर|कौनसा|कौनसी|है\s*न|हैं\s*न|ना\??)$"
)
PUNCT_RE = re.compile(r"[।.!?…]+$")
SPACE_RE = re.compile(r"\s+")
DEFAULT_HINDI_PROMPT = (
    "यह हिंदी बातचीत है। देवनागरी लिपि में सही शब्द, नाम, प्रश्नवाचक चिन्ह "
    "और पूर्ण विराम लिखें।"
)

POSITIVE_WORDS = {
    "अच्छा", "अच्छी", "खुश", "खुशी", "प्यार", "वाह", "शुक्रिया",
    "धन्यवाद", "बहुत बढ़िया", "मस्त", "कमाल", "हँसी", "हंसी",
}
NEGATIVE_WORDS = {
    "नहीं", "मत", "गलत", "दर्द", "डर", "झूठ", "गुस्सा", "छोड़",
    "परेशान", "नफरत", "धोखा", "रो", "मरा", "मर", "खत्म", "चुप",
}
ANGER_WORDS = {
    "चुप", "निकल", "बकवास", "तंग", "गुस्सा", "मार", "छोड़", "मत",
}
SAD_WORDS = {
    "अकेला", "अकेली", "दर्द", "रो", "आँसू", "आंसू", "दुख", "दुखी",
}


@dataclass
class Chunk:
    source_media: str
    source_id: str
    clip_id: str
    audio: str
    text: str
    normalized_text: str
    start: float
    end: float
    duration: float
    speaker: str
    language: str
    emotion: str
    intensity: float
    valence: float
    arousal: float
    dominance: float
    avg_word_probability: float | None
    devanagari_ratio: float
    zero_duration_word_ratio: float
    word_count: int
    words_per_second: float
    unique_word_ratio: float
    repeated_ngram_fraction: float
    dominant_word_fraction: float
    quality_score: float
    snr_db: float
    clipping_ratio: float
    silence_ratio: float
    avg_logprob: float | None
    no_speech_prob: float | None
    words: list[dict[str, Any]]
    license_source: str


@dataclass
class RejectedChunk:
    source_media: str
    source_id: str
    text: str
    start: float
    end: float
    duration: float
    reason: str
    quality_score: float | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    details: dict[str, Any] | None = None


class AudeeringEmotionTagger:
    def __init__(self, device: str):
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor

        from scripts.tag_emotions_vad import MODEL_ID, EmotionModel

        self.device = torch.device(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)

        cfg_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_dict["vocab_size"] = cfg_dict.get("vocab_size") or 32
        config = Wav2Vec2Config(**cfg_dict)
        self.model = EmotionModel.from_pretrained(MODEL_ID, config=config)
        self.model = self.model.to(self.device).eval()

    def __call__(self, wav: np.ndarray, sr: int, text: str) -> tuple[float, float, float]:
        if sr != 16000:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
            sr = 16000
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        with torch.no_grad():
            inputs = self.feature_extractor(
                wav, sampling_rate=sr, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(self.device)
            vad = self.model(input_values).squeeze(0).detach().cpu().numpy()
        return tuple(float(np.clip(v, 0.0, 1.0)) for v in vad[:3])


def run(cmd: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def ffprobe_duration(path: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def stable_source_id(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def extract_audio(
    media: Path,
    out_wav: Path,
    sample_rate: int,
    max_media_seconds: float | None,
) -> None:
    filters = ",".join(
        [
            "highpass=f=80",
            f"lowpass=f={min(7600, int(sample_rate * 0.46))}",
            "afftdn=nf=-25",
            "dynaudnorm=f=250:g=15",
        ]
    )
    cmd = [
        "ffmpeg", "-y",
    ]
    if max_media_seconds:
        cmd += ["-t", f"{max_media_seconds:.3f}"]
    cmd += [
        "-i", str(media),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-af", filters,
        str(out_wav),
    ]
    run(cmd, quiet=True)


def maybe_demucs(media_wav: Path, work_dir: Path, separator: str) -> Path:
    demucs_bin = shutil.which("demucs")
    if demucs_bin is None:
        sibling = Path(sys.executable).with_name("demucs")
        if sibling.exists():
            demucs_bin = str(sibling)
    if separator == "none":
        return media_wav
    if separator == "auto" and demucs_bin is None:
        return media_wav
    if separator == "demucs" and demucs_bin is None:
        raise RuntimeError("Demucs requested, but `demucs` is not installed.")

    sep_dir = work_dir / "demucs"
    run(
        [
            demucs_bin, "--two-stems=vocals", "-n", "htdemucs", "-o",
            str(sep_dir), str(media_wav),
        ],
        quiet=False,
    )
    candidates = list(sep_dir.glob(f"**/{media_wav.stem}/vocals.wav"))
    if not candidates:
        raise RuntimeError("Demucs finished, but vocals.wav was not found.")
    return candidates[0]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_whisper_model(model_name: str, device: str):
    import whisper

    print(f"Loading Whisper {model_name} on {device}...")
    return whisper.load_model(model_name, device=device)


def transcribe(
    model,
    wav_path: Path,
    language: str,
    device: str,
    initial_prompt: str | None,
    condition_on_previous_text: bool,
    beam_size: int,
) -> dict[str, Any]:
    fp16 = device == "cuda"
    return model.transcribe(
        str(wav_path),
        language=language,
        task="transcribe",
        word_timestamps=True,
        verbose=False,
        fp16=fp16,
        temperature=0.0,
        beam_size=beam_size,
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=condition_on_previous_text,
    )


def normalize_text(text: str) -> str:
    text = SPACE_RE.sub(" ", text.replace(" ,", ",").replace(" ?", "?").strip())
    text = text.strip(" -–—")
    if not text:
        return text
    if PUNCT_RE.search(text):
        return text
    if HINDI_QUESTION_RE.search(text) or text.startswith("क्या "):
        return text + "?"
    return text + "।"


def should_close_on_word(word_text: str) -> bool:
    return bool(re.search(r"[।.!?]$", word_text.strip()))


def segment_words(segment: dict[str, Any]) -> list[dict[str, Any]]:
    words = segment.get("words") or []
    out = []
    for w in words:
        word = str(w.get("word", "")).strip()
        if not word:
            continue
        if "start" not in w or "end" not in w:
            continue
        out.append(
            {
                "word": word,
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": (
                    float(w["probability"]) if w.get("probability") is not None else None
                ),
            }
        )
    return out


def split_segments(
    transcript: dict[str, Any],
    min_duration: float,
    max_duration: float,
    min_chars: int,
) -> tuple[list[dict[str, Any]], list[RejectedChunk]]:
    chunks: list[dict[str, Any]] = []
    rejected: list[RejectedChunk] = []

    for seg in transcript.get("segments", []):
        words = segment_words(seg)
        avg_logprob = (
            float(seg["avg_logprob"]) if seg.get("avg_logprob") is not None else None
        )
        no_speech_prob = (
            float(seg["no_speech_prob"]) if seg.get("no_speech_prob") is not None else None
        )

        if not words:
            text = normalize_text(str(seg.get("text", "")).strip())
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            chunks.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "words": [],
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech_prob,
                }
            )
            continue

        current: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal current
            if not current:
                return
            text = normalize_text(" ".join(w["word"] for w in current))
            start = current[0]["start"]
            end = current[-1]["end"]
            duration = end - start
            if duration < min_duration or len(text) < min_chars:
                rejected.append(
                    RejectedChunk(
                        source_media="",
                        source_id="",
                        text=text,
                        start=start,
                        end=end,
                        duration=duration,
                        reason="too_short",
                        avg_logprob=avg_logprob,
                        no_speech_prob=no_speech_prob,
                    )
                )
            else:
                chunks.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text,
                        "words": current,
                        "avg_logprob": avg_logprob,
                        "no_speech_prob": no_speech_prob,
                    }
                )
            current = []

        for word in words:
            if not current:
                current = [word]
                continue
            proposed_duration = word["end"] - current[0]["start"]
            if proposed_duration > max_duration:
                flush()
            current.append(word)
            if should_close_on_word(word["word"]) and proposed_duration >= min_duration:
                flush()
        flush()

    return chunks, rejected


def load_audio_slice(wav_path: Path, start: float, end: float) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(wav_path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    start_i = max(0, int(round(start * sr)))
    end_i = min(len(wav), int(round(end * sr)))
    return wav[start_i:end_i].astype(np.float32), sr


def frame_rms(wav: np.ndarray, sr: int, frame_ms: float = 40.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000.0))
    if len(wav) < frame:
        return np.array([float(np.sqrt(np.mean(np.square(wav))))], dtype=np.float32)
    hop = frame // 2
    values = []
    for i in range(0, len(wav) - frame + 1, hop):
        chunk = wav[i:i + frame]
        values.append(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
    return np.asarray(values, dtype=np.float32)


def quality_metrics(
    wav: np.ndarray,
    sr: int,
    avg_logprob: float | None,
    no_speech_prob: float | None,
) -> tuple[float, float, float, float]:
    if len(wav) == 0:
        return 0.0, 0.0, 1.0, 0.0

    rms = frame_rms(wav, sr)
    speech = float(np.percentile(rms, 75))
    floor = float(np.percentile(rms, 10)) + 1e-6
    snr_db = float(np.clip(20.0 * math.log10((speech + 1e-6) / floor), 0.0, 45.0))
    silence_threshold = max(floor * 2.0, 0.003)
    silence_ratio = float(np.mean(rms < silence_threshold))
    clipping_ratio = float(np.mean(np.abs(wav) >= 0.98))

    snr_score = np.clip((snr_db - 8.0) / 22.0, 0.0, 1.0)
    silence_score = 1.0 - np.clip((silence_ratio - 0.10) / 0.70, 0.0, 1.0)
    clip_score = 1.0 - np.clip(clipping_ratio / 0.01, 0.0, 1.0)

    if avg_logprob is None:
        asr_score = 0.75
    else:
        asr_score = float(np.clip((avg_logprob + 1.4) / 1.2, 0.0, 1.0))
    if no_speech_prob is not None:
        asr_score *= float(np.clip(1.0 - no_speech_prob, 0.0, 1.0))

    quality = (
        0.35 * snr_score
        + 0.25 * silence_score
        + 0.25 * asr_score
        + 0.15 * clip_score
    )
    return round(float(quality), 4), round(snr_db, 2), round(clipping_ratio, 5), round(silence_ratio, 4)


def repeated_ngram_fraction_for_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n * 2:
        return 0.0
    grams: dict[tuple[str, ...], int] = {}
    for i in range(0, len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        grams[gram] = grams.get(gram, 0) + 1
    if not grams:
        return 0.0
    max_count = max(grams.values())
    if max_count <= 1:
        return 0.0
    return round(float((max_count * n) / max(len(tokens), 1)), 4)


def repeated_ngram_fraction(tokens: list[str]) -> float:
    return max(repeated_ngram_fraction_for_n(tokens, n) for n in (2, 3, 4))


def text_asr_stats(
    text: str,
    words: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    probs = [
        float(w["probability"])
        for w in words
        if w.get("probability") is not None and np.isfinite(float(w["probability"]))
    ]
    avg_word_probability = round(float(np.mean(probs)), 4) if probs else None

    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    if timed:
        zeroish = [
            1.0
            for w in timed
            if float(w.get("end", 0.0)) - float(w.get("start", 0.0)) <= 0.025
        ]
        zero_duration_word_ratio = round(float(len(zeroish) / len(timed)), 4)
    else:
        zero_duration_word_ratio = 0.0

    tokens = [w["word"].strip("।.!?,;:'\"()[]{}-–—…").lower() for w in words]
    tokens = [t for t in tokens if t]
    if not tokens:
        tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    word_count = len(tokens)
    words_per_second = round(float(word_count / max(duration, 0.001)), 4)
    unique_word_ratio = (
        round(float(len(set(tokens)) / word_count), 4) if word_count else 0.0
    )
    repeat_fraction = repeated_ngram_fraction(tokens)
    if word_count:
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        dominant_word_fraction = round(float(max(counts.values()) / word_count), 4)
    else:
        dominant_word_fraction = 0.0

    chars = [
        ch
        for ch in text
        if not ch.isspace() and ch not in "।.!?,;:'\"()[]{}-–—…"
    ]
    if not chars:
        devanagari_ratio = 0.0
    else:
        deva = sum(1 for ch in chars if "\u0900" <= ch <= "\u097f")
        devanagari_ratio = round(float(deva / len(chars)), 4)
    return {
        "avg_word_probability": avg_word_probability,
        "devanagari_ratio": devanagari_ratio,
        "zero_duration_word_ratio": zero_duration_word_ratio,
        "word_count": word_count,
        "words_per_second": words_per_second,
        "unique_word_ratio": unique_word_ratio,
        "repeated_ngram_fraction": repeat_fraction,
        "dominant_word_fraction": dominant_word_fraction,
        "has_replacement_char": "\ufffd" in text,
    }


def reject(
    rejected: list[RejectedChunk],
    media: Path,
    source_id: str,
    cand: dict[str, Any],
    text: str,
    start: float,
    end: float,
    reason: str,
    quality_score: float | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    rejected.append(
        RejectedChunk(
            source_media=str(media),
            source_id=source_id,
            text=text,
            start=start,
            end=end,
            duration=end - start,
            reason=reason,
            quality_score=quality_score,
            avg_logprob=cand.get("avg_logprob"),
            no_speech_prob=cand.get("no_speech_prob"),
            details=details,
        )
    )


def heuristic_vad(wav: np.ndarray, sr: int, text: str) -> tuple[float, float, float]:
    if len(wav) == 0:
        return 0.5, 0.5, 0.5

    rms = frame_rms(wav, sr)
    energy = float(np.clip((np.percentile(rms, 80) - 0.015) / 0.12, 0.0, 1.0))
    duration = max(len(wav) / sr, 0.1)
    word_count = max(len(text.split()), 1)
    speech_rate = float(np.clip((word_count / duration - 1.6) / 3.2, 0.0, 1.0))

    try:
        f0 = librosa.yin(
            wav.astype(np.float32),
            fmin=65,
            fmax=450,
            sr=sr,
            frame_length=min(2048, max(256, 2 ** int(math.log2(max(len(wav), 256))))),
        )
        f0 = f0[np.isfinite(f0)]
        pitch_var = float(np.clip(np.std(f0) / 85.0, 0.0, 1.0)) if len(f0) else 0.35
    except Exception:
        pitch_var = 0.35

    lowered = text.lower()
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in lowered)
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lowered)

    arousal = float(np.clip(0.25 + 0.35 * energy + 0.25 * speech_rate + 0.15 * pitch_var, 0.0, 1.0))
    valence = float(np.clip(0.52 + 0.12 * pos_hits - 0.11 * neg_hits, 0.0, 1.0))
    dominance = float(np.clip(0.35 + 0.45 * energy + 0.20 * speech_rate, 0.0, 1.0))
    return valence, arousal, dominance


def emotion_from_vad_text(
    valence: float,
    arousal: float,
    dominance: float,
    text: str,
) -> tuple[str, float]:
    lowered = text.lower()
    anger_hits = sum(1 for w in ANGER_WORDS if w in lowered)
    sad_hits = sum(1 for w in SAD_WORDS if w in lowered)
    question = "?" in text or HINDI_QUESTION_RE.search(text) is not None
    exclaim = "!" in text

    if anger_hits and arousal >= 0.48:
        emotion = "anger"
    elif sad_hits and valence <= 0.55:
        emotion = "sadness"
    elif arousal < 0.32 and valence >= 0.45:
        emotion = "calm"
    elif valence > 0.62 and arousal >= 0.42:
        emotion = "joy"
    elif valence < 0.42 and arousal < 0.50:
        emotion = "sadness"
    elif valence < 0.45 and arousal >= 0.58 and dominance >= 0.50:
        emotion = "anger"
    elif valence < 0.45 and arousal >= 0.58:
        emotion = "fear"
    elif arousal > 0.72 and (question or exclaim or 0.42 <= valence <= 0.62):
        emotion = "surprise"
    elif question and arousal >= 0.50:
        emotion = "confusion"
    elif valence < 0.48 and 0.45 <= arousal <= 0.65:
        emotion = "concern"
    else:
        emotion = "neutral"

    intensity = float(np.clip(abs(arousal - 0.45) * 1.35 + abs(valence - 0.5) * 0.55, 0.0, 1.0))
    if emotion == "neutral":
        intensity = min(intensity, 0.45)
    return emotion, round(intensity, 4)


def build_emotion_tagger(mode: str, device: str):
    if mode == "heuristic":
        return None
    try:
        print("Loading Audeering V/A/D emotion model...")
        return AudeeringEmotionTagger(device)
    except Exception as exc:
        if mode == "audeering":
            raise
        print(f"Audeering model unavailable; falling back to heuristic emotion tags ({exc}).")
        return None


def export_clip(src_wav: Path, out_wav: Path, start: float, end: float, sample_rate: int) -> None:
    run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-i", str(src_wav),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-af", "loudnorm=I=-23:TP=-2:LRA=11",
            str(out_wav),
        ],
        quiet=True,
    )


def speaker_feature(wav_path: Path) -> np.ndarray:
    wav, sr = sf.read(wav_path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if len(wav) < sr // 2:
        return np.zeros(48, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    rms = frame_rms(wav, sr)
    zcr = librosa.feature.zero_crossing_rate(wav).reshape(-1)

    try:
        f0 = librosa.yin(wav, fmin=65, fmax=450, sr=sr)
        f0 = f0[np.isfinite(f0)]
        pitch_stats = np.array(
            [
                np.mean(f0) if len(f0) else 0.0,
                np.std(f0) if len(f0) else 0.0,
                np.percentile(f0, 25) if len(f0) else 0.0,
                np.percentile(f0, 75) if len(f0) else 0.0,
            ],
            dtype=np.float32,
        )
    except Exception:
        pitch_stats = np.zeros(4, dtype=np.float32)

    feat = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            mfcc_delta.mean(axis=1)[:4],
            np.array(
                [
                    rms.mean(),
                    rms.std(),
                    np.percentile(rms, 75),
                    zcr.mean() if len(zcr) else 0.0,
                ],
                dtype=np.float32,
            ),
            pitch_stats,
        ]
    )
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.astype(np.float32)


def assign_speakers(
    chunks: list[Chunk],
    mode: str,
    min_speakers: int,
    max_speakers: int,
    prefix: str,
) -> None:
    if not chunks:
        return
    if mode == "single" or len(chunks) < 2:
        for c in chunks:
            c.speaker = f"{prefix}_000"
        return
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if len(chunks) <= 3:
        # Very small pilots do not have enough samples for reliable silhouette
        # scoring; use two clusters when possible so a dialogue can still split.
        labels = AgglomerativeClustering(n_clusters=min(2, len(chunks)), linkage="ward").fit_predict(
            np.stack([speaker_feature(Path(c.audio)) for c in chunks], axis=0)
        )
        first_seen: dict[int, int] = {}
        next_id = 0
        for c, label in zip(chunks, labels):
            label = int(label)
            if label not in first_seen:
                first_seen[label] = next_id
                next_id += 1
            c.speaker = f"{prefix}_{first_seen[label]:03d}"
        return

    feats = np.stack([speaker_feature(Path(c.audio)) for c in chunks], axis=0)
    x = StandardScaler().fit_transform(feats)

    n = len(chunks)
    min_k = max(1, min(min_speakers, n))
    max_k = max(min_k, min(max_speakers, n - 1))
    if max_k == 1:
        labels = np.zeros(n, dtype=np.int64)
    else:
        best_labels = np.zeros(n, dtype=np.int64)
        best_score = -1.0
        for k in range(max(2, min_k), max_k + 1):
            labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
            if len(set(labels_k)) < 2:
                continue
            score = float(silhouette_score(x, labels_k))
            if score > best_score:
                best_score = score
                best_labels = labels_k
        if min_k <= 1 and best_score < 0.08:
            labels = np.zeros(n, dtype=np.int64)
        else:
            labels = best_labels

    first_seen: dict[int, int] = {}
    next_id = 0
    for c, label in zip(chunks, labels):
        label = int(label)
        if label not in first_seen:
            first_seen[label] = next_id
            next_id += 1
        c.speaker = f"{prefix}_{first_seen[label]:03d}"


def process_media(
    media: Path,
    args: argparse.Namespace,
    whisper_model,
    emotion_tagger,
    device: str,
) -> tuple[list[Chunk], list[RejectedChunk]]:
    source_id = args.source_id or stable_source_id(media)
    media_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", media.stem)
    work_dir = Path(args.out_dir) / "work" / f"{media_stem}_{source_id}"
    clip_dir = Path(args.out_dir) / "clips"
    transcript_dir = Path(args.out_dir) / "transcripts"
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    if not args.append:
        for old_clip in clip_dir.glob(f"{media_stem}_{source_id}_*.wav"):
            old_clip.unlink()

    raw_wav = work_dir / "raw_clean.wav"
    extract_audio(media, raw_wav, args.sample_rate, args.max_media_seconds)
    separated = maybe_demucs(raw_wav, work_dir, args.separator)
    if separated != raw_wav:
        denoised_wav = work_dir / "dialogue.wav"
        extract_audio(separated, denoised_wav, args.sample_rate, None)
    else:
        denoised_wav = raw_wav

    transcript_path = transcript_dir / f"{media_stem}_{source_id}.whisper.json"
    if args.reuse_transcript and transcript_path.exists():
        print(f"Reusing transcript {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
    else:
        print(f"Transcribing {media.name}...")
        transcript = transcribe(
            whisper_model,
            denoised_wav,
            args.language,
            device,
            args.initial_prompt,
            args.condition_on_previous_text,
            args.beam_size,
        )
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

    candidates, rejected = split_segments(
        transcript,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_chars=args.min_chars,
    )
    for r in rejected:
        r.source_media = str(media)
        r.source_id = source_id

    accepted: list[Chunk] = []
    limit = args.limit_chunks if args.limit_chunks and args.limit_chunks > 0 else None

    for idx, cand in enumerate(candidates):
        start = max(0.0, float(cand["start"]) - args.pad_seconds)
        end = max(start, float(cand["end"]) + args.pad_seconds)
        duration = end - start
        text = str(cand["text"]).strip()
        normalized_text = normalize_text(text)

        if duration < args.min_duration or duration > args.max_duration + 2 * args.pad_seconds:
            reject(
                rejected,
                media,
                source_id,
                cand,
                normalized_text,
                start,
                end,
                "duration_out_of_range",
            )
            continue

        words = cand.get("words", [])
        asr_details = text_asr_stats(normalized_text, words, duration)
        avg_word_probability = asr_details["avg_word_probability"]
        devanagari_ratio = asr_details["devanagari_ratio"]
        zero_duration_word_ratio = asr_details["zero_duration_word_ratio"]
        avg_logprob = cand.get("avg_logprob")
        no_speech_prob = cand.get("no_speech_prob")

        if not normalized_text:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "empty_text", details=asr_details)
            continue
        if asr_details["has_replacement_char"]:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "bad_unicode", details=asr_details)
            continue
        if asr_details["word_count"] < args.min_words:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "too_few_words", details=asr_details)
            continue
        if asr_details["words_per_second"] < args.min_words_per_second:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "too_sparse_text", details=asr_details)
            continue
        if (
            asr_details["word_count"] >= 8
            and asr_details["unique_word_ratio"] < args.min_unique_word_ratio
        ):
            reject(rejected, media, source_id, cand, normalized_text, start, end, "low_unique_word_ratio", details=asr_details)
            continue
        if (
            asr_details["word_count"] >= 4
            and asr_details["dominant_word_fraction"] > args.max_dominant_word_fraction
        ):
            reject(rejected, media, source_id, cand, normalized_text, start, end, "dominant_word_loop", details=asr_details)
            continue
        if asr_details["repeated_ngram_fraction"] > args.max_repeated_ngram_fraction:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "repetitive_text", details=asr_details)
            continue
        if avg_logprob is not None and avg_logprob < args.min_segment_logprob:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "low_segment_logprob", details=asr_details)
            continue
        if no_speech_prob is not None and no_speech_prob > args.max_no_speech_prob:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "high_no_speech_prob", details=asr_details)
            continue
        if avg_word_probability is not None and avg_word_probability < args.min_avg_word_prob:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "low_word_probability", details=asr_details)
            continue
        if devanagari_ratio < args.min_devanagari_ratio:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "low_devanagari_ratio", details=asr_details)
            continue
        if zero_duration_word_ratio > args.max_zero_duration_word_ratio:
            reject(rejected, media, source_id, cand, normalized_text, start, end, "bad_word_timestamps", details=asr_details)
            continue

        wav, sr = load_audio_slice(denoised_wav, start, end)
        quality, snr_db, clipping_ratio, silence_ratio = quality_metrics(
            wav, sr, avg_logprob, no_speech_prob
        )
        if quality < args.min_quality:
            reject(
                rejected,
                media,
                source_id,
                cand,
                normalized_text,
                start,
                end,
                "low_quality",
                quality_score=quality,
                details=asr_details,
            )
            continue

        if emotion_tagger is not None:
            valence, arousal, dominance = emotion_tagger(wav, sr, normalized_text)
        else:
            valence, arousal, dominance = heuristic_vad(wav, sr, normalized_text)
        emotion, intensity = emotion_from_vad_text(valence, arousal, dominance, normalized_text)

        clip_id = f"{media_stem}_{source_id}_{len(accepted):06d}"
        out_wav = clip_dir / f"{clip_id}.wav"
        export_clip(denoised_wav, out_wav, start, end, args.sample_rate)

        accepted.append(
            Chunk(
                source_media=str(media),
                source_id=source_id,
                clip_id=clip_id,
                audio=str(out_wav),
                text=text,
                normalized_text=normalized_text,
                start=round(start, 3),
                end=round(end, 3),
                duration=round(duration, 3),
                speaker=args.default_speaker,
                language=args.language,
                emotion=emotion,
                intensity=intensity,
                valence=round(float(valence), 4),
                arousal=round(float(arousal), 4),
                dominance=round(float(dominance), 4),
                avg_word_probability=avg_word_probability,
                devanagari_ratio=devanagari_ratio,
                zero_duration_word_ratio=zero_duration_word_ratio,
                word_count=asr_details["word_count"],
                words_per_second=asr_details["words_per_second"],
                unique_word_ratio=asr_details["unique_word_ratio"],
                repeated_ngram_fraction=asr_details["repeated_ngram_fraction"],
                dominant_word_fraction=asr_details["dominant_word_fraction"],
                quality_score=quality,
                snr_db=snr_db,
                clipping_ratio=clipping_ratio,
                silence_ratio=silence_ratio,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                words=words,
                license_source=args.license_source,
            )
        )

        if limit is not None and len(accepted) >= limit:
            break

    assign_speakers(
        accepted,
        mode=args.speaker_mode,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        prefix=args.speaker_prefix,
    )

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    return accepted, rejected


def write_jsonl(path: Path, records: list[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            if hasattr(rec, "__dataclass_fields__"):
                rec = asdict(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize(chunks: list[Chunk], rejected: list[RejectedChunk], out_dir: Path, elapsed: float) -> None:
    durations = np.array([c.duration for c in chunks], dtype=np.float32)
    emotion_counts: dict[str, int] = {}
    for c in chunks:
        emotion_counts[c.emotion] = emotion_counts.get(c.emotion, 0) + 1
    summary = {
        "accepted_chunks": len(chunks),
        "rejected_chunks": len(rejected),
        "accepted_hours": round(float(durations.sum() / 3600.0), 4) if len(durations) else 0.0,
        "duration_seconds": {
            "min": round(float(durations.min()), 3) if len(durations) else 0.0,
            "mean": round(float(durations.mean()), 3) if len(durations) else 0.0,
            "max": round(float(durations.max()), 3) if len(durations) else 0.0,
        },
        "emotion_counts": dict(sorted(emotion_counts.items())),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("media", nargs="+", help="Owned source media files: mp4/mov/webm/wav/mp3/etc.")
    ap.add_argument("--out-dir", default="data/owned_hindi_pilot")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--initial-prompt", default=DEFAULT_HINDI_PROMPT)
    ap.add_argument("--condition-on-previous-text", action="store_true")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--reuse-transcript", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--separator", choices=["auto", "none", "demucs"], default="auto")
    ap.add_argument("--emotion-model", choices=["auto", "heuristic", "audeering"], default="auto")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--min-duration", type=float, default=1.8)
    ap.add_argument("--max-duration", type=float, default=12.0)
    ap.add_argument("--pad-seconds", type=float, default=0.04)
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--min-quality", type=float, default=0.42)
    ap.add_argument("--min-segment-logprob", type=float, default=-1.6)
    ap.add_argument("--max-no-speech-prob", type=float, default=0.85)
    ap.add_argument("--min-avg-word-prob", type=float, default=0.45)
    ap.add_argument("--min-devanagari-ratio", type=float, default=0.65)
    ap.add_argument("--max-zero-duration-word-ratio", type=float, default=0.30)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--min-words-per-second", type=float, default=0.70)
    ap.add_argument("--min-unique-word-ratio", type=float, default=0.40)
    ap.add_argument("--max-repeated-ngram-fraction", type=float, default=0.32)
    ap.add_argument("--max-dominant-word-fraction", type=float, default=0.55)
    ap.add_argument("--max-media-seconds", type=float, default=None)
    ap.add_argument("--limit-chunks", type=int, default=None)
    ap.add_argument("--default-speaker", default="spk_unknown")
    ap.add_argument("--speaker-mode", choices=["cluster", "single"], default="cluster")
    ap.add_argument("--speaker-prefix", default="spk")
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=8)
    ap.add_argument("--source-id", default=None)
    ap.add_argument("--license-source", default="owned_production")
    ap.add_argument("--keep-work", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    media_paths = [Path(p).expanduser().resolve() for p in args.media]
    missing = [str(p) for p in media_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing media files: {missing}")

    device = resolve_device(args.device)
    whisper_model = load_whisper_model(args.whisper_model, device)
    emotion_tagger = build_emotion_tagger(args.emotion_model, device)

    started = time.time()
    all_chunks: list[Chunk] = []
    all_rejected: list[RejectedChunk] = []
    for media in media_paths:
        print(f"\n=== {media} ===")
        chunks, rejected = process_media(media, args, whisper_model, emotion_tagger, device)
        all_chunks.extend(chunks)
        all_rejected.extend(rejected)

    write_jsonl(out_dir / "manifest.jsonl", all_chunks)
    write_jsonl(out_dir / "rejected.jsonl", all_rejected)
    summarize(all_chunks, all_rejected, out_dir, time.time() - started)
    print(f"\nWrote dataset to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
