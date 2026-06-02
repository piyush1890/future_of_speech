#!/usr/bin/env python3
"""
Drop entire dialogue sentences/phrases that are not clearly audible.

The script uses Whisper word timestamps as a proxy for audibility, but removes
whole sentence-like chunks, not individual words. It exports:
  - accepted whole-sentence clips
  - a compact WAV made only from accepted sentences
  - accepted/rejected manifests with reasons and confidence stats
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import soundfile as sf
import torch


SENTENCE_END_RE = re.compile(r"[।.!?…]+$")
SPACE_RE = re.compile(r"\s+")
CONTINUATION_START_TOKENS = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "while",
    "which",
    "who",
    "whom",
    "whose",
    "with",
    "without",
    "that",
    "then",
    "से",
    "के",
    "की",
    "का",
    "को",
    "में",
    "पर",
    "लिए",
    "वाला",
    "वाली",
    "वाले",
    "कि",
    "जो",
    "जिस",
    "जिसे",
    "जिसका",
    "जिसकी",
    "जिसमें",
    "जिन",
    "और",
    "या",
    "पर",
    "तो",
    "जैसे",
    "बट",
    "बढ़",
    "बढ",
    "रिलेशन्शिप",
    "एमोशन",
    "खुशी",
    "गम",
    "है",
    "हूं",
    "हूँ",
    "ना",
    "बात",
    "कुछ",
    "समझू",
}
NON_TERMINAL_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr."}
DEFAULT_PROMPT = (
    "यह हिंदी फिल्म या सीरीज का संवाद है। केवल साफ़ सुनाई देने वाले बोले गए "
    "संवाद को देवनागरी में सही शब्द और विराम चिन्ह के साथ लिखें।"
)


@dataclass
class Candidate:
    clip_id: str
    audio: str | None
    text: str
    start: float
    end: float
    duration: float
    word_count: int
    avg_word_probability: float | None
    min_word_probability: float | None
    low_word_fraction: float
    very_low_word_fraction: float
    low_word_count: int
    very_low_word_count: int
    zero_duration_word_ratio: float
    devanagari_ratio: float
    repeated_ngram_fraction: float
    unique_word_ratio: float
    avg_logprob: float | None
    no_speech_prob: float | None
    compression_ratio: float | None
    quality_score: float
    accepted: bool
    reject_reason: str | None
    words: list[dict[str, Any]]
    speech_rate_wps: float | None = None
    max_word_duration: float | None = None
    p95_word_duration: float | None = None
    max_voiced_run_seconds: float | None = None
    harmonic_ratio: float | None = None
    song_like_score: float | None = None
    turn_id: str | None = None
    turn_start: float | None = None
    turn_end: float | None = None


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_text(text: str) -> str:
    text = SPACE_RE.sub(" ", text.strip())
    text = text.replace(" ,", ",").replace(" ?", "?").replace(" ।", "।")
    return text


def clean_token(text: str) -> str:
    return text.strip().strip("।.!?,;:'\"()[]{}-–—…").lower()


def devanagari_ratio(text: str) -> float:
    chars = [
        ch
        for ch in text
        if not ch.isspace() and ch not in "।.!?,;:'\"()[]{}-–—…"
    ]
    if not chars:
        return 0.0
    deva = sum(1 for ch in chars if "\u0900" <= ch <= "\u097f")
    return float(deva / len(chars))


def repeated_ngram_fraction(tokens: list[str]) -> float:
    best = 0.0
    for n in (2, 3, 4):
        if len(tokens) < n * 2:
            continue
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
        if counts:
            best = max(best, max(counts.values()) * n / max(1, len(tokens)))
    return float(best)


def segment_words(
    transcript: dict[str, Any],
    *,
    offset_seconds: float = 0.0,
    turn_id: str | None = None,
    turn_start: float | None = None,
    turn_end: float | None = None,
) -> list[dict[str, Any]]:
    all_words: list[dict[str, Any]] = []
    for seg in transcript.get("segments", []):
        seg_meta = {
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
            "compression_ratio": seg.get("compression_ratio"),
        }
        for word in seg.get("words") or []:
            text = str(word.get("word", "")).strip()
            if not text or word.get("start") is None or word.get("end") is None:
                continue
            start = float(word["start"]) + offset_seconds
            end = float(word["end"]) + offset_seconds
            if turn_start is not None:
                start = max(float(turn_start), start)
            if turn_end is not None:
                end = min(float(turn_end), end)
            if end <= start:
                continue
            item = {
                "word": text,
                "start": start,
                "end": end,
                "probability": (
                    float(word["probability"])
                    if word.get("probability") is not None
                    else None
                ),
                **seg_meta,
            }
            if turn_id is not None:
                item["turn_id"] = turn_id
                item["turn_start"] = turn_start
                item["turn_end"] = turn_end
            all_words.append(item)
    all_words.sort(key=lambda w: (w["start"], w["end"]))
    return all_words


def split_candidates(
    words: list[dict[str, Any]],
    min_duration: float,
    _max_duration: float,
    pause_split: float,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = []

    for word in words:
        if not current:
            current = [word]
            continue

        gap = word["start"] - current[-1]["end"]
        current_duration = current[-1]["end"] - current[0]["start"]
        sentence_boundary = (
            SENTENCE_END_RE.search(str(current[-1]["word"])) is not None
            and current_duration >= min_duration
        )
        long_pause = gap >= pause_split and current_duration >= min_duration

        if sentence_boundary or long_pause:
            flush()
        current.append(word)
    flush()
    return chunks


def chunk_duration(words: list[dict[str, Any]]) -> float:
    if not words:
        return 0.0
    return max(0.0, float(words[-1]["end"]) - float(words[0]["start"]))


def chunk_gap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if not left or not right:
        return float("inf")
    return max(0.0, float(right[0]["start"]) - float(left[-1]["end"]))


def starts_with_continuation(words: list[dict[str, Any]]) -> bool:
    if not words:
        return False
    token = clean_token(str(words[0].get("word", "")))
    return token in CONTINUATION_START_TOKENS


def ends_with_nonterminal_punctuation(words: list[dict[str, Any]]) -> bool:
    if not words:
        return False
    token = str(words[-1].get("word", "")).strip()
    return token.endswith((",", ";", ":", "،"))


def merge_continuation_splits(
    chunks: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    if not args.merge_continuation_splits:
        return chunks

    merged: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        if (
            merged
            and chunk
            and not is_sentence_end_word(merged[-1][-1])
            and (starts_with_continuation(chunk) or ends_with_nonterminal_punctuation(merged[-1]))
            and chunk_gap(merged[-1], chunk) <= args.continuation_merge_max_gap
        ):
            merged[-1] = merged[-1] + chunk
            continue
        merged.append(chunk)
    return merged


def is_sentence_end_word(word: dict[str, Any]) -> bool:
    return SENTENCE_END_RE.search(str(word.get("word", ""))) is not None


def speaker_window_feature(
    wav: np.ndarray,
    sr: int,
    start: float,
    end: float,
    cache: dict[str, Any],
) -> dict[str, Any] | None:
    if cache.get("backend") in {"resemblyzer", "speechbrain"}:
        key = (round(start, 2), round(end, 2))
        if key in cache["windows"]:
            return cache["windows"][key]

        times = cache["times"]
        embeddings = cache["embeddings"]
        voiced = cache.get("voiced")
        rms_values = cache.get("rms")
        mask = (times >= start) & (times <= end)
        if voiced is not None:
            mask = mask & np.asarray(voiced, dtype=bool)
        if not np.any(mask):
            center = 0.5 * (start + end)
            eligible = (
                np.flatnonzero(np.asarray(voiced, dtype=bool))
                if voiced is not None
                else np.arange(len(times))
            )
            if len(eligible) == 0:
                cache["windows"][key] = None
                return None
            nearest = int(eligible[int(np.argmin(np.abs(times[eligible] - center)))])
            if abs(float(times[nearest]) - center) > max(0.9, (end - start) * 0.75):
                cache["windows"][key] = None
                return None
            mask = np.zeros(len(times), dtype=bool)
            mask[nearest] = True

        embedding = np.mean(embeddings[mask], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-8:
            cache["windows"][key] = None
            return None
        feature = {"embedding": embedding / norm, "voice_windows": int(np.sum(mask))}
        if rms_values is not None:
            feature["mean_voice_rms"] = float(np.mean(np.asarray(rms_values)[mask]))
        cache["windows"][key] = feature
        return feature

    acoustic_cache = cache["windows"]
    key = (round(start, 2), round(end, 2))
    if key in acoustic_cache:
        return acoustic_cache[key]

    start_i = max(0, int(round(start * sr)))
    end_i = min(len(wav), int(round(end * sr)))
    clip = wav[start_i:end_i].astype(np.float32, copy=False)
    if len(clip) < int(0.22 * sr):
        acoustic_cache[key] = None
        return None

    clip = clip - float(np.mean(clip))
    rms = float(np.sqrt(np.mean(np.square(clip)) + 1e-9))
    if rms < 1e-5:
        acoustic_cache[key] = None
        return None
    clip = clip / rms

    import librosa

    hop_length = max(80, int(sr * 0.020))
    n_fft = min(1024, max(256, 2 ** int(np.floor(np.log2(max(256, len(clip)))))))
    n_fft = min(n_fft, len(clip))
    mfcc = librosa.feature.mfcc(
        y=clip,
        sr=sr,
        n_mfcc=13,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    # Drop MFCC 0 so loudness changes do not look like speaker changes.
    timbre = np.concatenate([np.mean(mfcc[1:], axis=1), np.std(mfcc[1:], axis=1)])
    centroid = librosa.feature.spectral_centroid(
        y=clip,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=clip,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    f0_value = 0.0
    try:
        f0 = librosa.yin(
            clip,
            fmin=65,
            fmax=500,
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        valid_f0 = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 500)]
        if len(valid_f0):
            f0_value = float(np.median(valid_f0))
    except Exception:
        f0_value = 0.0

    feature = {
        "timbre": np.nan_to_num(timbre.astype(np.float32)),
        "centroid": float(np.nanmedian(centroid) / max(sr, 1)),
        "zcr": float(np.nanmean(zcr)),
        "f0": f0_value,
    }
    acoustic_cache[key] = feature
    return feature


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return clipped01(1.0 - float(np.dot(a, b) / denom))


def relative_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = 0.5 * (np.abs(a) + np.abs(b)) + 1.0
    return clipped01(float(np.mean(np.abs(a - b) / denom) / 0.55))


def speaker_feature_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    if "embedding" in left and "embedding" in right:
        return cosine_distance(
            np.asarray(left["embedding"], dtype=np.float32),
            np.asarray(right["embedding"], dtype=np.float32),
        )

    left_timbre = np.asarray(left["timbre"], dtype=np.float32)
    right_timbre = np.asarray(right["timbre"], dtype=np.float32)
    timbre_distance = (
        0.55 * cosine_distance(left_timbre, right_timbre)
        + 0.45 * relative_distance(left_timbre, right_timbre)
    )

    pitch_distance = 0.0
    if left["f0"] > 0.0 and right["f0"] > 0.0:
        pitch_distance = clipped01(
            abs(math.log2((left["f0"] + 1e-6) / (right["f0"] + 1e-6))) / 0.75
        )

    centroid_distance = clipped01(
        abs(math.log((left["centroid"] + 1e-4) / (right["centroid"] + 1e-4))) / 0.70
    )
    zcr_distance = clipped01(abs(left["zcr"] - right["zcr"]) / 0.06)
    return float(
        0.58 * timbre_distance
        + 0.24 * pitch_distance
        + 0.10 * centroid_distance
        + 0.08 * zcr_distance
    )


def should_consider_speaker_boundary(
    left_words: list[dict[str, Any]],
    right_words: list[dict[str, Any]],
    gap: float,
    args: argparse.Namespace,
) -> bool:
    if len(left_words) < args.speaker_split_min_side_words:
        return False
    if len(right_words) < args.speaker_split_min_side_words:
        return False
    if chunk_duration(left_words) < args.speaker_split_min_side_duration:
        return False
    if chunk_duration(right_words) < args.speaker_split_min_side_duration:
        return False

    if args.chunking == "speaker":
        return True

    previous_word = left_words[-1]
    sentence_boundary = is_sentence_end_word(previous_word)
    soft_pause = gap >= args.speaker_split_min_gap
    return sentence_boundary or soft_pause


def word_boundary_time(chunk: list[dict[str, Any]], index: int) -> float:
    return 0.5 * (float(chunk[index - 1]["end"]) + float(chunk[index]["start"]))


def is_utterance_boundary(
    chunk: list[dict[str, Any]],
    index: int,
    args: argparse.Namespace,
) -> bool:
    if index <= 0 or index >= len(chunk):
        return False
    gap = float(chunk[index]["start"]) - float(chunk[index - 1]["end"])
    if is_sentence_end_word(chunk[index - 1]):
        return True
    if gap < args.speaker_split_utterance_gap:
        return False
    if gap >= args.speaker_split_force_gap:
        return True

    previous_text = str(chunk[index - 1].get("word", "")).strip()
    next_token = clean_token(str(chunk[index].get("word", "")))
    if previous_text.endswith((",", ";", ":")):
        return False
    if next_token in CONTINUATION_START_TOKENS:
        return False
    return True


def snap_to_utterance_boundary(
    chunk: list[dict[str, Any]],
    peak_index: int,
    piece_start: int,
    args: argparse.Namespace,
) -> int | None:
    peak_time = word_boundary_time(chunk, peak_index)
    start = max(piece_start + 1, peak_index - args.speaker_split_snap_window_words)
    end = min(len(chunk), peak_index + args.speaker_split_snap_window_words + 1)
    choices: list[tuple[float, int, int]] = []
    for index in range(start, end):
        if not is_utterance_boundary(chunk, index, args):
            continue
        boundary_time = word_boundary_time(chunk, index)
        time_distance = abs(boundary_time - peak_time)
        if time_distance > args.speaker_split_snap_window_seconds:
            continue
        left_words = chunk[piece_start:index]
        right_words = chunk[index:]
        if len(left_words) < args.speaker_split_min_side_words:
            continue
        if len(right_words) < args.speaker_split_min_side_words:
            continue
        if chunk_duration(left_words) < args.speaker_split_min_side_duration:
            continue
        if chunk_duration(right_words) < args.speaker_split_min_side_duration:
            continue
        choices.append((time_distance, abs(index - peak_index), index))
    if not choices:
        return None
    return min(choices)[2]


def split_chunk_on_speaker_changes(
    wav: np.ndarray,
    sr: int,
    chunk: list[dict[str, Any]],
    args: argparse.Namespace,
    feature_cache: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if args.chunking == "speaker":
        return split_chunk_on_speaker_change_peaks(wav, sr, chunk, args, feature_cache)

    if len(chunk) < args.speaker_split_min_side_words * 2:
        return [chunk]

    pieces: list[list[dict[str, Any]]] = []
    piece_start = 0
    for i in range(1, len(chunk)):
        left_words = chunk[piece_start:i]
        right_words = chunk[i:]
        gap = float(chunk[i]["start"]) - float(chunk[i - 1]["end"])
        if not should_consider_speaker_boundary(left_words, right_words, gap, args):
            continue

        boundary = 0.5 * (float(chunk[i - 1]["end"]) + float(chunk[i]["start"]))
        left_start = max(float(left_words[0]["start"]), boundary - args.speaker_split_window_seconds)
        right_end = min(float(chunk[-1]["end"]), boundary + args.speaker_split_window_seconds)
        left_feature = speaker_window_feature(
            wav,
            sr,
            left_start,
            boundary,
            feature_cache,
        )
        right_feature = speaker_window_feature(
            wav,
            sr,
            boundary,
            right_end,
            feature_cache,
        )
        if left_feature is None or right_feature is None:
            continue

        distance = speaker_feature_distance(left_feature, right_feature)
        threshold = args.speaker_split_threshold
        if args.chunking != "speaker":
            if "?" in str(chunk[i - 1].get("word", "")):
                threshold = min(threshold, args.speaker_split_question_threshold)
            elif is_sentence_end_word(chunk[i - 1]):
                threshold = min(threshold, args.speaker_split_sentence_threshold)
            if gap >= args.pause_split * 0.5:
                threshold = max(0.0, threshold - args.speaker_split_pause_bonus)

        if distance >= threshold:
            pieces.append(left_words)
            piece_start = i

    pieces.append(chunk[piece_start:])
    return [piece for piece in pieces if piece]


def split_chunk_on_speaker_change_peaks(
    wav: np.ndarray,
    sr: int,
    chunk: list[dict[str, Any]],
    args: argparse.Namespace,
    feature_cache: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if len(chunk) < args.speaker_split_min_side_words * 2:
        return [chunk]

    scores = np.full(len(chunk), -np.inf, dtype=np.float32)
    for i in range(1, len(chunk)):
        left_words = chunk[:i]
        right_words = chunk[i:]
        gap = float(chunk[i]["start"]) - float(chunk[i - 1]["end"])
        if not should_consider_speaker_boundary(left_words, right_words, gap, args):
            continue

        boundary = 0.5 * (float(chunk[i - 1]["end"]) + float(chunk[i]["start"]))
        left_feature = speaker_window_feature(
            wav,
            sr,
            max(float(chunk[0]["start"]), boundary - args.speaker_split_window_seconds),
            boundary,
            feature_cache,
        )
        right_feature = speaker_window_feature(
            wav,
            sr,
            boundary,
            min(float(chunk[-1]["end"]), boundary + args.speaker_split_window_seconds),
            feature_cache,
        )
        if left_feature is None or right_feature is None:
            continue
        scores[i] = speaker_feature_distance(left_feature, right_feature)

    if args.speaker_split_adaptive_threshold and np.any(np.isfinite(scores)):
        finite_scores = scores[np.isfinite(scores)]
        adaptive = float(np.percentile(finite_scores, args.speaker_split_adaptive_percentile))
        threshold_floor = args.speaker_split_threshold
        threshold = max(threshold_floor, adaptive)
    else:
        threshold = args.speaker_split_threshold

    radius = max(1, args.speaker_split_peak_neighborhood_words)
    peak_indices: list[int] = []
    for i in range(1, len(chunk)):
        score = float(scores[i])
        if not np.isfinite(score) or score < threshold:
            continue
        left = max(1, i - radius)
        right = min(len(chunk), i + radius + 1)
        if score >= float(np.max(scores[left:right])) - 1e-6:
            peak_indices.append(i)

    selected: list[int] = []
    piece_start = 0
    for i in peak_indices:
        split_index = i
        if args.speaker_split_snap_to_utterance:
            snapped = snap_to_utterance_boundary(chunk, i, piece_start, args)
            if snapped is None:
                continue
            split_index = snapped

        if selected and split_index <= selected[-1]:
            continue

        left_words = chunk[piece_start:split_index]
        right_words = chunk[split_index:]
        if len(left_words) < args.speaker_split_min_side_words:
            continue
        if len(right_words) < args.speaker_split_min_side_words:
            continue
        if chunk_duration(left_words) < args.speaker_split_min_side_duration:
            continue
        if chunk_duration(right_words) < args.speaker_split_min_side_duration:
            continue
        selected.append(split_index)
        piece_start = split_index

    selected = sorted(set(selected))
    while selected:
        final_piece = chunk[selected[-1]:]
        if (
            len(final_piece) >= args.speaker_split_min_side_words
            and chunk_duration(final_piece) >= args.speaker_split_min_side_duration
        ):
            break
        selected.pop()

    if not selected:
        return [chunk]

    pieces: list[list[dict[str, Any]]] = []
    piece_start = 0
    for i in selected:
        pieces.append(chunk[piece_start:i])
        piece_start = i
    pieces.append(chunk[piece_start:])
    return [piece for piece in pieces if piece]


def build_speaker_feature_cache(
    wav: np.ndarray,
    sr: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.speaker_feature_backend == "resemblyzer":
        from resemblyzer import VoiceEncoder

        import librosa

        wav16 = wav.astype(np.float32, copy=False)
        if sr != 16000:
            wav16 = librosa.resample(wav16, orig_sr=sr, target_sr=16000)
        encoder = VoiceEncoder(device=args.speaker_embedding_device, verbose=False)
        _, embeddings, slices = encoder.embed_utterance(
            wav16,
            return_partials=True,
            rate=args.speaker_embedding_rate,
        )
        centers = np.asarray(
            [(float(s.start) + float(s.stop)) / (2.0 * 16000.0) for s in slices],
            dtype=np.float32,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
        return {
            "backend": "resemblyzer",
            "times": centers,
            "embeddings": embeddings / norms,
            "windows": {},
        }

    if args.speaker_feature_backend == "speechbrain":
        from speechbrain.inference.speaker import EncoderClassifier

        import librosa

        wav16 = wav.astype(np.float32, copy=False)
        if sr != 16000:
            wav16 = librosa.resample(wav16, orig_sr=sr, target_sr=16000)
        # SpeechBrain 1.1.0 currently fails when EncoderClassifier is
        # constructed directly with run_opts={"device": "mps"}. Instantiate the
        # module, then explicitly move every module and all inference tensors to
        # MPS below. PYTORCH_ENABLE_MPS_FALLBACK=0 keeps inference honest.
        load_device = "cpu" if args.speaker_embedding_device == "mps" else args.speaker_embedding_device
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=args.speechbrain_model_dir,
            run_opts={"device": load_device},
        )
        if args.speaker_embedding_device == "mps":
            mps_device = torch.device("mps")
            classifier.mods.to(mps_device)
            for value in vars(classifier.hparams).values():
                if isinstance(value, torch.nn.Module):
                    value.to(mps_device)
            classifier.device = mps_device
            classifier.device_type = "mps"

        window = max(1600, int(round(args.speaker_embedding_window_seconds * 16000)))
        hop = max(160, int(round(16000 / args.speaker_embedding_rate)))
        if len(wav16) <= window:
            starts = np.asarray([0], dtype=np.int64)
        else:
            starts = np.arange(0, len(wav16) - window + 1, hop, dtype=np.int64)
            final_start = len(wav16) - window
            if int(starts[-1]) != final_start:
                starts = np.concatenate(
                    [starts, np.asarray([final_start], dtype=np.int64)]
                )

        embeddings_list: list[np.ndarray] = []
        rms_list: list[np.ndarray] = []
        device = torch.device(args.speaker_embedding_device)
        batch_size = max(1, int(args.speaker_embedding_batch_size))
        with torch.inference_mode():
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start:batch_start + batch_size]
                batch = np.stack(
                    [wav16[int(s):int(s) + window] for s in batch_starts],
                    axis=0,
                ).astype(np.float32)
                rms_list.append(
                    np.sqrt(np.mean(np.square(batch), axis=1) + 1e-12).astype(np.float32)
                )
                batch_tensor = torch.from_numpy(batch).to(device)
                emb = classifier.encode_batch(batch_tensor, normalize=True)
                emb = emb.squeeze(1).detach().cpu().numpy().astype(np.float32)
                embeddings_list.append(emb)

        embeddings = np.concatenate(embeddings_list, axis=0)
        rms_values = np.concatenate(rms_list, axis=0)
        positive_rms = rms_values[rms_values > 1e-8]
        if len(positive_rms):
            dynamic_voice_floor = (
                float(np.percentile(positive_rms, args.speaker_embedding_voice_rms_percentile))
                * float(args.speaker_embedding_voice_rms_scale)
            )
            voice_rms_threshold = max(float(args.speaker_embedding_min_voice_rms), dynamic_voice_floor)
            voiced = rms_values >= voice_rms_threshold
        else:
            voice_rms_threshold = float(args.speaker_embedding_min_voice_rms)
            voiced = np.zeros_like(rms_values, dtype=bool)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
        centers = (starts.astype(np.float32) + window / 2.0) / 16000.0
        return {
            "backend": "speechbrain",
            "times": centers.astype(np.float32),
            "embeddings": embeddings / norms,
            "rms": rms_values,
            "voiced": voiced,
            "voice_rms_threshold": voice_rms_threshold,
            "windows": {},
        }

    return {"backend": "acoustic", "windows": {}}


def load_input_wav(input_wav: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(input_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False), sr


def detect_audio_first_speaker_turns(
    input_wav: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wav, sr = load_input_wav(input_wav)
    duration = len(wav) / max(sr, 1)
    work_dir = out_dir / "audio_first_speaker_turns"
    work_dir.mkdir(parents=True, exist_ok=True)

    if duration <= max(args.min_duration, args.audio_speaker_min_turn_duration):
        turns = [
            {
                "turn_id": "turn_000000",
                "start": 0.0,
                "end": round(duration, 3),
                "duration": round(duration, 3),
            }
        ]
        with (work_dir / "turns.json").open("w", encoding="utf-8") as handle:
            json.dump(turns, handle, ensure_ascii=False, indent=2)
        return turns, []

    feature_cache = build_speaker_feature_cache(wav, sr, args)
    raw_times = feature_cache.get("times")
    times = np.asarray(raw_times if raw_times is not None else [], dtype=np.float32)
    min_side = max(float(args.speaker_split_min_side_duration), float(args.audio_speaker_min_turn_duration))
    window = max(float(args.speaker_split_window_seconds), float(args.speaker_embedding_window_seconds) * 0.65)
    if len(times) == 0:
        times = np.arange(min_side, max(min_side, duration - min_side), 0.4, dtype=np.float32)
    candidate_times = times[(times >= min_side) & (times <= duration - min_side)]

    scores: list[dict[str, Any]] = []
    for boundary in candidate_times:
        t = float(boundary)
        left_feature = speaker_window_feature(
            wav,
            sr,
            max(0.0, t - window),
            t,
            feature_cache,
        )
        right_feature = speaker_window_feature(
            wav,
            sr,
            t,
            min(duration, t + window),
            feature_cache,
        )
        if left_feature is None or right_feature is None:
            continue
        score = speaker_feature_distance(left_feature, right_feature)
        scores.append(
            {
                "time": round(t, 3),
                "score": round(score, 4),
                "left_voice_windows": left_feature.get("voice_windows"),
                "right_voice_windows": right_feature.get("voice_windows"),
                "left_mean_voice_rms": (
                    round(float(left_feature["mean_voice_rms"]), 6)
                    if left_feature.get("mean_voice_rms") is not None
                    else None
                ),
                "right_mean_voice_rms": (
                    round(float(right_feature["mean_voice_rms"]), 6)
                    if right_feature.get("mean_voice_rms") is not None
                    else None
                ),
            }
        )

    if not scores:
        turns = [
            {
                "turn_id": "turn_000000",
                "start": 0.0,
                "end": round(duration, 3),
                "duration": round(duration, 3),
            }
        ]
        with (work_dir / "turns.json").open("w", encoding="utf-8") as handle:
            json.dump(turns, handle, ensure_ascii=False, indent=2)
        with (work_dir / "boundary_scores.json").open("w", encoding="utf-8") as handle:
            json.dump(scores, handle, ensure_ascii=False, indent=2)
        return turns, scores

    score_values = np.asarray([float(item["score"]) for item in scores], dtype=np.float32)
    threshold = float(args.speaker_split_threshold)
    if args.speaker_split_adaptive_threshold:
        threshold = max(
            threshold,
            float(np.percentile(score_values, args.speaker_split_adaptive_percentile)),
        )

    radius_seconds = float(args.audio_speaker_peak_neighborhood_seconds)
    peak_candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(scores):
        score = float(item["score"])
        if score < threshold:
            continue
        t = float(item["time"])
        neighbors = [
            float(other["score"])
            for other in scores
            if abs(float(other["time"]) - t) <= radius_seconds
        ]
        if score >= max(neighbors) - 1e-6:
            peak_candidates.append(item)

    min_boundary_gap = max(min_side, float(args.audio_speaker_min_boundary_gap_seconds))
    selected: list[dict[str, Any]] = []
    for item in sorted(peak_candidates, key=lambda row: float(row["score"]), reverse=True):
        t = float(item["time"])
        if t < min_side or duration - t < min_side:
            continue
        if any(abs(t - float(other["time"])) < min_boundary_gap for other in selected):
            continue
        selected.append(item)
    selected.sort(key=lambda row: float(row["time"]))

    boundaries = [float(item["time"]) for item in selected]
    turns: list[dict[str, Any]] = []
    start = 0.0
    for idx, boundary in enumerate([*boundaries, duration]):
        end = float(boundary)
        if end - start < min_side and turns:
            turns[-1]["end"] = round(end, 3)
            turns[-1]["duration"] = round(float(turns[-1]["end"]) - float(turns[-1]["start"]), 3)
        elif end > start:
            turns.append(
                {
                    "turn_id": f"turn_{len(turns):06d}",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "left_boundary_score": (
                        None if idx == 0 else round(float(selected[idx - 1]["score"]), 4)
                    ),
                }
            )
        start = end

    if not turns:
        turns = [
            {
                "turn_id": "turn_000000",
                "start": 0.0,
                "end": round(duration, 3),
                "duration": round(duration, 3),
            }
        ]

    with (work_dir / "turns.json").open("w", encoding="utf-8") as handle:
        json.dump(turns, handle, ensure_ascii=False, indent=2)
    with (work_dir / "boundary_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(scores, handle, ensure_ascii=False, indent=2)
    with (work_dir / "selected_boundaries.json").open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)
    return turns, scores


def transcribe_audio_first_speaker_turns(
    input_wav: Path,
    turns: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[list[dict[str, Any]]]:
    wav, sr = load_input_wav(input_wav)
    work_dir = out_dir / "audio_first_speaker_turns"
    turn_audio_dir = work_dir / "turn_audio"
    transcript_dir = work_dir / "turn_transcripts"
    turn_audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[list[dict[str, Any]]] = []
    transcript_manifest: list[dict[str, Any]] = []
    for turn in turns:
        turn_id = str(turn["turn_id"])
        start = float(turn["start"])
        end = float(turn["end"])
        start_i = max(0, int(round(start * sr)))
        end_i = min(len(wav), int(round(end * sr)))
        if end_i <= start_i:
            continue
        turn_path = turn_audio_dir / f"{turn_id}.wav"
        sf.write(turn_path, wav[start_i:end_i], sr)
        transcript = transcribe_uncached(turn_path, args)
        transcript_path = transcript_dir / f"{turn_id}.json"
        with transcript_path.open("w", encoding="utf-8") as handle:
            json.dump(transcript, handle, ensure_ascii=False, indent=2)
        turn_words = segment_words(
            transcript,
            offset_seconds=start,
            turn_id=turn_id,
            turn_start=start,
            turn_end=end,
        )
        transcript_manifest.append(
            {
                **turn,
                "audio": str(turn_path),
                "transcript": str(transcript_path),
                "words": len(turn_words),
            }
        )
        if not turn_words:
            continue
        if args.clip_unit == "sentence":
            chunks.extend(split_turns_into_complete_sentences([turn_words], args))
        else:
            chunks.append(turn_words)

    with (work_dir / "turn_transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for row in transcript_manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return chunks


def split_chunks_on_speaker_changes(
    input_wav: Path,
    chunks: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    if not args.speaker_split:
        return chunks

    wav, sr = sf.read(input_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    feature_cache = build_speaker_feature_cache(wav, sr, args)
    split_chunks: list[list[dict[str, Any]]] = []
    for chunk in chunks:
        split_chunks.extend(
            split_chunk_on_speaker_changes(wav, sr, chunk, args, feature_cache)
        )
    return split_chunks


def split_turns_into_complete_sentences(
    turns: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    sentence_chunks: list[list[dict[str, Any]]] = []
    for turn in turns:
        current: list[dict[str, Any]] = []
        for word in turn:
            current.append(word)
            if is_sentence_end_word(word) and chunk_duration(current) >= args.min_duration:
                sentence_chunks.append(current)
                current = []

        if not current or not args.keep_turn_final_utterance:
            continue
        first_token = clean_token(str(current[0].get("word", "")))
        if first_token in CONTINUATION_START_TOKENS:
            continue
        sentence_chunks.append(current)
    return sentence_chunks


def apply_turn_sentence_fallback(
    chunks: list[list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    chunks = merge_continuation_splits(chunks, args)
    if args.clip_unit != "turn_with_sentence_fallback":
        return chunks, {
            "turn_first_input_turns": len(chunks),
            "turn_first_kept_turns": len(chunks),
            "turn_first_fallback_turns": 0,
            "turn_first_fallback_sentence_candidates": 0,
        }

    final_chunks: list[list[dict[str, Any]]] = []
    kept_turns = 0
    fallback_turns = 0
    fallback_sentence_candidates = 0
    for chunk in chunks:
        if not chunk:
            continue
        turn_candidate = score_candidate(chunk, args, "turn_probe")
        if turn_candidate.accepted:
            final_chunks.append(chunk)
            kept_turns += 1
            continue

        sentence_chunks = split_turns_into_complete_sentences([chunk], args)
        if sentence_chunks:
            final_chunks.extend(sentence_chunks)
            fallback_turns += 1
            fallback_sentence_candidates += len(sentence_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks, {
        "turn_first_input_turns": len(chunks),
        "turn_first_kept_turns": kept_turns,
        "turn_first_fallback_turns": fallback_turns,
        "turn_first_fallback_sentence_candidates": fallback_sentence_candidates,
    }


def weighted_mean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def clipped01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def append_reject_reason(candidate: Candidate, reason: str) -> None:
    if candidate.reject_reason:
        reasons = candidate.reject_reason.split(";")
        if reason not in reasons:
            reasons.append(reason)
            candidate.reject_reason = ";".join(reasons)
    else:
        candidate.reject_reason = reason
    candidate.accepted = False


def score_candidate(
    words: list[dict[str, Any]],
    args: argparse.Namespace,
    clip_id: str,
) -> Candidate:
    text = normalize_text(" ".join(w["word"] for w in words))
    start = float(words[0]["start"])
    end = float(words[-1]["end"])
    duration = max(0.0, end - start)
    probs = [
        float(w["probability"])
        for w in words
        if w.get("probability") is not None and np.isfinite(float(w["probability"]))
    ]
    tokens = [clean_token(w["word"]) for w in words]
    tokens = [t for t in tokens if t]
    word_count = len(tokens)

    avg_prob = float(np.mean(probs)) if probs else None
    min_prob = float(np.min(probs)) if probs else None
    low_word_count = int(np.sum(np.asarray(probs) < args.low_word_prob)) if probs else 0
    very_low_word_count = int(np.sum(np.asarray(probs) < args.very_low_word_prob)) if probs else 0
    low_frac = float(low_word_count / len(probs)) if probs else 1.0
    very_low_frac = float(very_low_word_count / len(probs)) if probs else 1.0
    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    word_durations = [
        max(0.0, float(w["end"]) - float(w["start"]))
        for w in timed
    ]
    max_word_duration = float(max(word_durations) if word_durations else 0.0)
    p95_word_duration = float(np.percentile(word_durations, 95) if word_durations else 0.0)
    zero_ratio = (
        float(np.mean([(float(w["end"]) - float(w["start"])) <= 0.025 for w in timed]))
        if timed
        else 1.0
    )
    deva_ratio = devanagari_ratio(text)
    repeat_frac = repeated_ngram_fraction(tokens)
    unique_ratio = float(len(set(tokens)) / max(1, word_count)) if tokens else 0.0
    avg_logprob = weighted_mean([w.get("avg_logprob") for w in words])
    no_speech_prob = weighted_mean([w.get("no_speech_prob") for w in words])
    compression_ratio = weighted_mean([w.get("compression_ratio") for w in words])

    reasons: list[str] = []
    hard_too_short = duration < args.min_hard_duration
    soft_too_short = duration < args.min_duration and word_count < args.min_words
    if hard_too_short or soft_too_short:
        reasons.append("too_short")
    if word_count < args.min_words:
        reasons.append("too_few_words")
    if "\ufffd" in text:
        reasons.append("bad_unicode")
    if deva_ratio < args.min_devanagari_ratio:
        reasons.append("low_devanagari_ratio")
    if avg_prob is None or avg_prob < args.min_avg_word_prob:
        reasons.append("low_avg_word_prob")
    if min_prob is not None and min_prob < args.min_word_prob:
        reasons.append("very_uncertain_word")
    if (
        low_word_count >= args.min_low_words_for_many_uncertain
        and low_frac > args.max_low_word_fraction
    ):
        reasons.append("many_uncertain_words")
    if very_low_frac > args.max_very_low_word_fraction:
        reasons.append("very_low_word_fraction")
    if zero_ratio > args.max_zero_duration_word_ratio:
        reasons.append("bad_word_timestamps")
    if max_word_duration > args.max_asr_word_duration:
        reasons.append("bad_word_timestamps")
    if p95_word_duration > args.max_asr_p95_word_duration:
        reasons.append("bad_word_timestamps")
    # Natural conversational Hindi can repeat words for emphasis. Keep the
    # repetition/unique-word metrics for diagnostics, but don't reject otherwise
    # audible dialogue only because words repeat.
    if args.reject_continuation_start and tokens and tokens[0] in CONTINUATION_START_TOKENS:
        reasons.append("continuation_start")
    if args.require_terminal_punctuation:
        raw_last = str(words[-1].get("word", "")).strip().lower()
        if not SENTENCE_END_RE.search(text) or raw_last in NON_TERMINAL_ABBREVIATIONS:
            reasons.append("missing_terminal_punctuation")
    if avg_logprob is not None and avg_logprob < args.min_avg_logprob:
        reasons.append("low_segment_logprob")
    if no_speech_prob is not None and no_speech_prob > args.max_no_speech_prob:
        reasons.append("high_no_speech_prob")
    if compression_ratio is not None and compression_ratio > args.max_compression_ratio:
        companion_quality_issue = any(
            reason in reasons
            for reason in {
                "bad_unicode",
                "low_devanagari_ratio",
                "low_avg_word_prob",
                "very_uncertain_word",
                "many_uncertain_words",
                "very_low_word_fraction",
                "bad_word_timestamps",
                "low_segment_logprob",
                "high_no_speech_prob",
            }
        )
        repetitive_loop = word_count >= args.compression_loop_min_words and (
            (
                repeat_frac >= args.compression_loop_repeated_ngram_fraction
                and unique_ratio <= args.compression_loop_unique_word_ratio
            )
            or (
                compression_ratio >= args.compression_loop_min_ratio
                and unique_ratio <= args.compression_loop_unique_word_ratio
            )
        )
        if companion_quality_issue or repetitive_loop:
            reasons.append("high_compression_ratio")
            if repetitive_loop:
                reasons.append("asr_repetitive_loop")
    reasons = list(dict.fromkeys(reasons))

    # Weighted quality score for ranking; thresholds above determine rejection.
    prob_score = 0.0 if avg_prob is None else np.clip((avg_prob - 0.45) / 0.45, 0.0, 1.0)
    min_score = 0.0 if min_prob is None else np.clip((min_prob - 0.10) / 0.55, 0.0, 1.0)
    logp_score = 0.7 if avg_logprob is None else np.clip((avg_logprob + 1.2) / 1.0, 0.0, 1.0)
    quality = (
        0.38 * prob_score
        + 0.20 * min_score
        + 0.17 * logp_score
        + 0.12 * (1.0 - np.clip(low_frac / max(args.max_low_word_fraction, 1e-6), 0.0, 1.0))
        + 0.08 * deva_ratio
        + 0.05 * (1.0 - np.clip(repeat_frac / max(args.max_repeated_ngram_fraction, 1e-6), 0.0, 1.0))
    )

    return Candidate(
        clip_id=clip_id,
        audio=None,
        text=text,
        start=round(start, 3),
        end=round(end, 3),
        duration=round(duration, 3),
        word_count=word_count,
        avg_word_probability=round(avg_prob, 4) if avg_prob is not None else None,
        min_word_probability=round(min_prob, 4) if min_prob is not None else None,
        low_word_fraction=round(low_frac, 4),
        very_low_word_fraction=round(very_low_frac, 4),
        low_word_count=low_word_count,
        very_low_word_count=very_low_word_count,
        zero_duration_word_ratio=round(zero_ratio, 4),
        devanagari_ratio=round(deva_ratio, 4),
        repeated_ngram_fraction=round(repeat_frac, 4),
        unique_word_ratio=round(unique_ratio, 4),
        avg_logprob=round(avg_logprob, 4) if avg_logprob is not None else None,
        no_speech_prob=round(no_speech_prob, 4) if no_speech_prob is not None else None,
        compression_ratio=round(compression_ratio, 4) if compression_ratio is not None else None,
        quality_score=round(float(quality), 4),
        accepted=len(reasons) == 0,
        reject_reason=";".join(reasons) if reasons else None,
        words=[
            {
                "word": w["word"],
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "probability": (
                    round(float(w["probability"]), 4)
                    if w.get("probability") is not None
                    else None
                ),
            }
            for w in words
        ],
        speech_rate_wps=round(float(word_count / max(duration, 0.1)), 4),
        max_word_duration=round(max_word_duration, 4),
        p95_word_duration=round(p95_word_duration, 4),
        turn_id=str(words[0].get("turn_id")) if words[0].get("turn_id") is not None else None,
        turn_start=(
            round(float(words[0]["turn_start"]), 3)
            if words[0].get("turn_start") is not None
            else None
        ),
        turn_end=(
            round(float(words[0]["turn_end"]), 3)
            if words[0].get("turn_end") is not None
            else None
        ),
    )


def transcribe(input_wav: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.reuse_transcript and Path(args.transcript_json).exists():
        with open(args.transcript_json, encoding="utf-8") as f:
            return json.load(f)

    if args.asr_backend == "mlx":
        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(input_wav),
            path_or_hf_repo=args.mlx_model,
            language=args.language,
            task="transcribe",
            word_timestamps=True,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=args.initial_prompt,
        )
    else:
        import whisper

        device = resolve_device(args.device)
        model = whisper.load_model(args.model, device=device)
        result = model.transcribe(
            str(input_wav),
            language=args.language,
            task="transcribe",
            word_timestamps=True,
            verbose=False,
            fp16=device == "cuda",
            temperature=0.0,
            beam_size=args.beam_size,
            condition_on_previous_text=False,
            initial_prompt=args.initial_prompt,
        )

    with open(args.transcript_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def transcribe_uncached(input_wav: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.asr_backend == "mlx":
        import mlx.core as mx
        import mlx_whisper

        if "gpu" not in str(mx.default_device()).lower():
            raise RuntimeError(
                f"MLX ASR must run on GPU; refusing CPU fallback, got {mx.default_device()}."
            )
        return mlx_whisper.transcribe(
            str(input_wav),
            path_or_hf_repo=args.mlx_model,
            language=args.language,
            task="transcribe",
            word_timestamps=True,
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=args.initial_prompt,
        )

    import whisper

    device = resolve_device(args.device)
    if device == "cpu":
        raise RuntimeError("ASR must run on GPU; refusing CPU fallback.")
    model = whisper.load_model(args.model, device=device)
    return model.transcribe(
        str(input_wav),
        language=args.language,
        task="transcribe",
        word_timestamps=True,
        verbose=False,
        fp16=device == "cuda",
        temperature=0.0,
        beam_size=args.beam_size,
        condition_on_previous_text=False,
        initial_prompt=args.initial_prompt,
    )


def export_audio(
    input_wav: Path,
    candidates: list[Candidate],
    out_dir: Path,
    pad_seconds: float,
    compact_gap_seconds: float,
) -> None:
    wav, sr = sf.read(input_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.wav"):
        old.unlink()

    compact_parts: list[np.ndarray] = []
    gap = np.zeros(int(sr * compact_gap_seconds), dtype=np.float32)
    accepted_idx = 0
    for cand in candidates:
        if not cand.accepted:
            continue
        start_seconds = cand.start - pad_seconds
        end_seconds = cand.end + pad_seconds
        if cand.turn_start is not None:
            start_seconds = max(start_seconds, float(cand.turn_start))
        if cand.turn_end is not None:
            end_seconds = min(end_seconds, float(cand.turn_end))
        start = max(0, int(round(start_seconds * sr)))
        end = min(len(wav), int(round(end_seconds * sr)))
        if end <= start:
            cand.accepted = False
            cand.reject_reason = "empty_audio_slice"
            continue
        clip = wav[start:end]
        clip_id = f"sentence_{accepted_idx:06d}"
        clip_path = clips_dir / f"{clip_id}.wav"
        sf.write(clip_path, clip, sr)
        cand.clip_id = clip_id
        cand.audio = str(clip_path)
        if compact_parts:
            compact_parts.append(gap)
        compact_parts.append(clip)
        accepted_idx += 1

    compact = np.concatenate(compact_parts) if compact_parts else np.zeros(0, dtype=np.float32)
    sf.write(out_dir / "accepted_sentences_compact.wav", compact, sr)


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def song_word_timing_metrics(candidate: Candidate) -> dict[str, float]:
    word_durations = [
        max(0.0, float(w["end"]) - float(w["start"]))
        for w in candidate.words
        if w.get("start") is not None and w.get("end") is not None
    ]
    return {
        "speech_rate_wps": float(candidate.word_count / max(candidate.duration, 0.1)),
        "max_word_duration": float(max(word_durations) if word_durations else 0.0),
        "p95_word_duration": float(
            np.percentile(word_durations, 95) if word_durations else 0.0
        ),
    }


def librosa_song_like_metrics(
    wav: np.ndarray,
    sr: int,
    candidate: Candidate,
    pad_seconds: float,
) -> dict[str, float]:
    start = max(0, int(round((candidate.start - pad_seconds) * sr)))
    end = min(len(wav), int(round((candidate.end + pad_seconds) * sr)))
    clip = wav[start:end].astype(np.float32)

    timing = song_word_timing_metrics(candidate)
    speech_rate = timing["speech_rate_wps"]
    max_word_duration = timing["max_word_duration"]
    p95_word_duration = timing["p95_word_duration"]

    max_voiced_run = 0.0
    harmonic_ratio = 0.0
    if len(clip) >= sr // 2:
        try:
            import librosa

            hop_length = max(80, int(sr * 0.010))
            f0, voiced_flag, _ = librosa.pyin(
                clip,
                fmin=65,
                fmax=500,
                sr=sr,
                frame_length=1024,
                hop_length=hop_length,
            )
            voiced = np.asarray(voiced_flag) & np.isfinite(f0)
            max_voiced_run = longest_true_run(voiced) * hop_length / sr
        except Exception:
            max_voiced_run = 0.0

        try:
            import librosa

            harmonic, _ = librosa.effects.hpss(clip)
            harmonic_ratio = float(
                np.sum(np.square(harmonic)) / (np.sum(np.square(clip)) + 1e-9)
            )
        except Exception:
            harmonic_ratio = 0.0

    # Singing often has long held syllables, slower word cadence, lyric repetition,
    # and a continuous harmonic bed. This is intentionally conservative and only
    # rejects clear song-like candidates after the audibility filter has passed.
    song_score = (
        0.30 * clipped01((max_word_duration - 0.90) / 1.40)
        + 0.20 * clipped01((1.90 - speech_rate) / 0.80)
        + 0.18 * clipped01((max_voiced_run - 1.00) / 1.80)
        + 0.12 * clipped01((candidate.repeated_ngram_fraction - 0.16) / 0.20)
        + 0.10
        * clipped01(
            ((candidate.compression_ratio or 0.0) - 2.10) / 0.90
        )
        + 0.10 * clipped01((harmonic_ratio - 0.45) / 0.25)
    )

    return {
        "speech_rate_wps": float(speech_rate),
        "max_word_duration": float(max_word_duration),
        "p95_word_duration": float(p95_word_duration),
        "max_voiced_run_seconds": float(max_voiced_run),
        "harmonic_ratio": float(harmonic_ratio),
        "song_like_score": float(song_score),
    }


def song_feature_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_torch_song_features(
    wav: np.ndarray,
    sr: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build acoustic song features once for the whole file.

    The expensive part is the STFT, which runs on MPS when available. Candidate
    scoring then becomes cheap window lookups instead of repeated librosa.pyin
    and HPSS calls per sentence.
    """

    device = song_feature_device(args.song_device)
    n_fft = args.song_n_fft
    hop = max(1, int(sr * args.song_hop_ms / 1000.0))
    window = torch.hann_window(n_fft, device=device)
    freq = torch.linspace(0.0, sr / 2.0, n_fft // 2 + 1, device=device)
    band = (freq >= args.song_band_low_hz) & (freq <= args.song_band_high_hz)
    low_voice = (freq >= args.song_voice_low_hz) & (freq <= args.song_voice_high_hz)
    k = min(args.song_tonal_top_k, int(band.sum().item()))

    total_frames = max(0, 1 + (len(wav) - n_fft) // hop)
    chunk_samples = max(
        n_fft,
        int(max(args.song_feature_chunk_seconds, args.song_hop_ms / 1000.0) * sr),
    )
    chunk_frames = max(1, chunk_samples // hop)

    band_power_chunks: list[np.ndarray] = []
    tonal_chunks: list[np.ndarray] = []
    low_voice_chunks: list[np.ndarray] = []

    # Process long files in frame-aligned chunks. A full-movie STFT can exceed
    # the MPS shared-memory watermark even though each local window is tiny.
    with torch.inference_mode():
        for frame_start in range(0, total_frames, chunk_frames):
            frame_end = min(total_frames, frame_start + chunk_frames)
            sample_start = frame_start * hop
            sample_end = min(len(wav), (frame_end - 1) * hop + n_fft)
            chunk = wav[sample_start:sample_end]
            if len(chunk) < n_fft:
                continue
            x = torch.from_numpy(chunk.astype(np.float32, copy=False)).to(device)
            spec = torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop,
                window=window,
                center=False,
                return_complex=True,
            )
            power = spec.abs().square()
            band_power = power[band].sum(dim=0).clamp_min(1e-9)
            low_voice_ratio = power[low_voice].sum(dim=0) / band_power
            tonal_ratio = (
                torch.topk(power[band], k=k, dim=0).values.sum(dim=0) / band_power
            )

            if device == "mps":
                torch.mps.synchronize()

            band_power_chunks.append(band_power.detach().cpu().numpy())
            tonal_chunks.append(tonal_ratio.detach().cpu().numpy())
            low_voice_chunks.append(low_voice_ratio.detach().cpu().numpy())

            del x, spec, power, band_power, low_voice_ratio, tonal_ratio
            if device == "mps":
                torch.mps.empty_cache()

    if band_power_chunks:
        band_power_np = np.concatenate(band_power_chunks)[:total_frames]
        tonal_np = np.concatenate(tonal_chunks)[:total_frames]
        low_voice_np = np.concatenate(low_voice_chunks)[:total_frames]
    else:
        band_power_np = np.asarray([], dtype=np.float32)
        tonal_np = np.asarray([], dtype=np.float32)
        low_voice_np = np.asarray([], dtype=np.float32)
    active_floor = max(float(np.percentile(band_power_np, args.song_active_percentile)), 1e-9)
    active = band_power_np >= active_floor
    voiced = active & (
        (tonal_np >= args.song_tonal_voiced_threshold)
        | (low_voice_np >= args.song_low_voice_threshold)
    )
    return {
        "backend": "torch",
        "device": device,
        "hop": hop,
        "sr": sr,
        "tonal_ratio": tonal_np.astype(np.float32),
        "voiced": voiced.astype(bool),
    }


def torch_song_like_metrics(
    feature_index: dict[str, Any],
    candidate: Candidate,
    pad_seconds: float,
) -> dict[str, float]:
    timing = song_word_timing_metrics(candidate)
    sr = int(feature_index["sr"])
    hop = int(feature_index["hop"])
    start = max(0, int((candidate.start - pad_seconds) * sr / hop))
    end = int(np.ceil((candidate.end + pad_seconds) * sr / hop))
    tonal = feature_index["tonal_ratio"][start:end]
    voiced = feature_index["voiced"][start:end]
    max_voiced_run = longest_true_run(voiced) * hop / sr if len(voiced) else 0.0
    harmonic_ratio = float(np.mean(tonal)) if len(tonal) else 0.0

    speech_rate = timing["speech_rate_wps"]
    max_word_duration = timing["max_word_duration"]
    song_score = (
        0.30 * clipped01((max_word_duration - 0.90) / 1.40)
        + 0.20 * clipped01((1.90 - speech_rate) / 0.80)
        + 0.18 * clipped01((max_voiced_run - 1.00) / 1.80)
        + 0.12 * clipped01((candidate.repeated_ngram_fraction - 0.16) / 0.20)
        + 0.10
        * clipped01(
            ((candidate.compression_ratio or 0.0) - 2.10) / 0.90
        )
        + 0.10 * clipped01((harmonic_ratio - 0.18) / 0.22)
    )
    return {
        **timing,
        "max_voiced_run_seconds": float(max_voiced_run),
        "harmonic_ratio": float(harmonic_ratio),
        "song_like_score": float(song_score),
    }


def reject_song_like_candidates(
    input_wav: Path,
    candidates: list[Candidate],
    args: argparse.Namespace,
) -> None:
    if args.allow_songlike:
        return

    wav, sr = sf.read(input_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    feature_index: dict[str, Any] | None = None
    backend = args.song_acoustic_backend
    if backend == "auto":
        backend = "torch"
    if backend == "torch":
        feature_index = build_torch_song_features(wav, sr, args)

    for candidate in candidates:
        if not candidate.accepted:
            continue

        if backend == "librosa":
            metrics = librosa_song_like_metrics(wav, sr, candidate, args.clip_pad_seconds)
        elif feature_index is not None:
            metrics = torch_song_like_metrics(feature_index, candidate, args.clip_pad_seconds)
        else:
            metrics = {
                **song_word_timing_metrics(candidate),
                "max_voiced_run_seconds": 0.0,
                "harmonic_ratio": 0.0,
                "song_like_score": 0.0,
            }

        candidate.speech_rate_wps = round(metrics["speech_rate_wps"], 4)
        candidate.max_word_duration = round(metrics["max_word_duration"], 4)
        candidate.p95_word_duration = round(metrics["p95_word_duration"], 4)
        candidate.max_voiced_run_seconds = round(
            metrics["max_voiced_run_seconds"], 4
        )
        candidate.harmonic_ratio = round(metrics["harmonic_ratio"], 4)
        candidate.song_like_score = round(metrics["song_like_score"], 4)

        timestamp_unreliable = (
            candidate.zero_duration_word_ratio > args.song_max_zero_duration_word_ratio
        )
        acoustic_song_evidence = (
            candidate.max_voiced_run_seconds >= args.song_min_voiced_run_for_score
            or (
                candidate.harmonic_ratio >= args.song_min_harmonic_ratio_for_score
                and candidate.speech_rate_wps <= args.song_max_speech_rate
            )
        )
        slow_sustained_words = (
            candidate.max_word_duration >= args.song_max_word_duration
            and candidate.speech_rate_wps <= args.song_max_speech_rate
            and acoustic_song_evidence
            and not timestamp_unreliable
        )
        scored_as_song = (
            candidate.song_like_score >= args.song_score_threshold
            and acoustic_song_evidence
            and not timestamp_unreliable
        )
        long_tonal_phrase = (
            candidate.max_voiced_run_seconds >= args.song_min_voiced_run_for_song
        )
        lyric_repetition = (
            candidate.repeated_ngram_fraction >= args.song_min_repeated_ngram_for_song
        )
        lyric_like_sustain = (
            slow_sustained_words
            and lyric_repetition
            and candidate.speech_rate_wps <= args.song_max_speech_rate
        )
        if scored_as_song and (long_tonal_phrase or lyric_like_sustain):
            append_reject_reason(candidate, "song_like")
        elif slow_sustained_words or scored_as_song:
            append_reject_reason(candidate, "background_music_or_noise")


def write_jsonl(path: Path, records: list[Candidate]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def summarize(
    candidates: list[Candidate],
    out_dir: Path,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    accepted = [c for c in candidates if c.accepted]
    rejected = [c for c in candidates if not c.accepted]
    reason_counts: dict[str, int] = {}
    for c in rejected:
        for reason in (c.reject_reason or "unknown").split(";"):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "total_candidates": len(candidates),
        "accepted_sentences": len(accepted),
        "rejected_sentences": len(rejected),
        "accepted_seconds": round(sum(c.duration for c in accepted), 3),
        "rejected_seconds": round(sum(c.duration for c in rejected), 3),
        "reason_counts": dict(sorted(reason_counts.items())),
        "accepted_compact": str(out_dir / "accepted_sentences_compact.wav"),
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--language", default="hi")
    ap.add_argument("--asr-backend", choices=["openai", "mlx"], default="openai")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--mlx-model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--initial-prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--reuse-transcript", action="store_true")
    ap.add_argument("--transcript-json", default=None)
    ap.add_argument(
        "--chunking",
        choices=["sentence", "speaker"],
        default="sentence",
        help="sentence uses punctuation/pause chunks; speaker cuts only at detected speaker changes.",
    )
    ap.add_argument(
        "--clip-unit",
        choices=["turn", "sentence", "turn_with_sentence_fallback"],
        default="turn",
        help=(
            "turn exports whole speaker turns; sentence packages complete sentence "
            "clips inside speaker turns; turn_with_sentence_fallback keeps clean "
            "turns whole and falls back to sentence candidates only when the full "
            "turn fails quality checks."
        ),
    )
    ap.add_argument(
        "--speaker-boundary-mode",
        choices=["post_asr", "audio_first"],
        default="post_asr",
        help=(
            "post_asr keeps the legacy flow: ASR first, then speaker splitting "
            "at ASR word boundaries. audio_first detects speaker turns directly "
            "on audio, cuts those turns, and only then runs ASR inside each turn."
        ),
    )
    ap.add_argument(
        "--keep-turn-final-utterance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When clip-unit=sentence, keep the final unpunctuated utterance at a speaker turn end unless it starts like a continuation.",
    )
    ap.add_argument("--min-duration", type=float, default=1.4)
    ap.add_argument(
        "--min-hard-duration",
        type=float,
        default=0.55,
        help=(
            "Reject clips shorter than this absolute floor. Clips between this "
            "and --min-duration are kept when they have enough words."
        ),
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=13.0,
        help="Kept for CLI compatibility; clips are no longer cut or rejected for being long.",
    )
    ap.add_argument("--pause-split", type=float, default=0.65)
    ap.add_argument(
        "--merge-continuation-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge adjacent non-speaker chunks when the next chunk starts like a sentence continuation.",
    )
    ap.add_argument("--continuation-merge-max-gap", type=float, default=0.75)
    ap.add_argument(
        "--speaker-split",
        action="store_true",
        help="Split sentence chunks at likely speaker-turn boundaries before scoring/export.",
    )
    ap.add_argument("--speaker-split-threshold", type=float, default=0.42)
    ap.add_argument(
        "--speaker-feature-backend",
        choices=["acoustic", "resemblyzer", "speechbrain"],
        default="acoustic",
        help="Feature backend for speaker-change scoring. resemblyzer/speechbrain use pretrained speaker encoders.",
    )
    ap.add_argument(
        "--speechbrain-model-dir",
        default="models/speechbrain-spkrec-ecapa-voxceleb",
        help="Directory where the SpeechBrain ECAPA speaker model is cached.",
    )
    ap.add_argument(
        "--speaker-embedding-device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Device for the resemblyzer speaker encoder.",
    )
    ap.add_argument("--speaker-embedding-rate", type=float, default=2.5)
    ap.add_argument("--speaker-embedding-window-seconds", type=float, default=1.6)
    ap.add_argument("--speaker-embedding-batch-size", type=int, default=64)
    ap.add_argument("--speaker-embedding-min-voice-rms", type=float, default=0.0008)
    ap.add_argument("--speaker-embedding-voice-rms-percentile", type=float, default=35.0)
    ap.add_argument("--speaker-embedding-voice-rms-scale", type=float, default=0.55)
    ap.add_argument(
        "--speaker-split-adaptive-threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raise the speaker threshold to a percentile of scores for this file.",
    )
    ap.add_argument("--speaker-split-adaptive-percentile", type=float, default=92.0)
    ap.add_argument("--speaker-split-sentence-threshold", type=float, default=0.36)
    ap.add_argument("--speaker-split-question-threshold", type=float, default=0.30)
    ap.add_argument("--speaker-split-pause-bonus", type=float, default=0.05)
    ap.add_argument("--speaker-split-window-seconds", type=float, default=1.0)
    ap.add_argument("--speaker-split-peak-neighborhood-words", type=int, default=4)
    ap.add_argument(
        "--speaker-split-snap-to-utterance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Snap detected speaker-change peaks to nearby complete utterance boundaries.",
    )
    ap.add_argument("--speaker-split-snap-window-seconds", type=float, default=1.2)
    ap.add_argument("--speaker-split-snap-window-words", type=int, default=7)
    ap.add_argument("--speaker-split-utterance-gap", type=float, default=0.42)
    ap.add_argument("--speaker-split-force-gap", type=float, default=1.45)
    ap.add_argument("--speaker-split-min-gap", type=float, default=0.12)
    ap.add_argument("--speaker-split-min-side-words", type=int, default=2)
    ap.add_argument("--speaker-split-min-side-duration", type=float, default=0.55)
    ap.add_argument(
        "--audio-speaker-min-turn-duration",
        type=float,
        default=0.9,
        help="Minimum audio-first speaker-turn duration before/after a boundary.",
    )
    ap.add_argument(
        "--audio-speaker-min-boundary-gap-seconds",
        type=float,
        default=0.9,
        help="Minimum spacing between selected audio-first speaker boundaries.",
    )
    ap.add_argument(
        "--audio-speaker-peak-neighborhood-seconds",
        type=float,
        default=1.0,
        help="Local-maximum window for audio-first speaker-boundary scoring.",
    )
    ap.add_argument("--clip-pad-seconds", type=float, default=0.08)
    ap.add_argument("--compact-gap-seconds", type=float, default=0.18)
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--min-devanagari-ratio", type=float, default=0.72)
    ap.add_argument("--min-avg-word-prob", type=float, default=0.68)
    ap.add_argument("--min-word-prob", type=float, default=0.18)
    ap.add_argument("--low-word-prob", type=float, default=0.48)
    ap.add_argument("--very-low-word-prob", type=float, default=0.28)
    ap.add_argument("--max-low-word-fraction", type=float, default=0.15)
    ap.add_argument(
        "--min-low-words-for-many-uncertain",
        type=int,
        default=2,
        help="Require at least this many low-confidence words before using the low-word fraction reject.",
    )
    ap.add_argument("--max-very-low-word-fraction", type=float, default=0.04)
    ap.add_argument("--max-zero-duration-word-ratio", type=float, default=0.16)
    ap.add_argument(
        "--max-asr-word-duration",
        type=float,
        default=3.0,
        help="Reject broken ASR timestamps where one word spans too much audio. This is not a clip-length limit.",
    )
    ap.add_argument(
        "--max-asr-p95-word-duration",
        type=float,
        default=2.2,
        help="Reject clips whose word-timing distribution is globally stretched/corrupt.",
    )
    ap.add_argument("--min-unique-word-ratio", type=float, default=0.42)
    ap.add_argument("--max-repeated-ngram-fraction", type=float, default=0.28)
    ap.add_argument(
        "--compression-loop-min-words",
        type=int,
        default=5,
        help="Minimum words before high compression can reject an otherwise clean ASR loop.",
    )
    ap.add_argument(
        "--compression-loop-repeated-ngram-fraction",
        type=float,
        default=0.50,
        help="Repeated n-gram fraction used to distinguish ASR loops from valid repeated names/phrases.",
    )
    ap.add_argument(
        "--compression-loop-unique-word-ratio",
        type=float,
        default=0.55,
        help="Unique-word ratio below which high-compression text looks like an ASR loop.",
    )
    ap.add_argument(
        "--compression-loop-min-ratio",
        type=float,
        default=4.0,
        help="Very high compression ratio that can flag low-diversity ASR loops.",
    )
    ap.add_argument(
        "--reject-continuation-start",
        action="store_true",
        help="Reject clips whose first token looks like the middle of a sentence.",
    )
    ap.add_argument(
        "--require-terminal-punctuation",
        action="store_true",
        help="Reject clips that do not end with terminal sentence punctuation.",
    )
    ap.add_argument("--min-avg-logprob", type=float, default=-0.75)
    ap.add_argument("--max-no-speech-prob", type=float, default=0.45)
    ap.add_argument("--max-compression-ratio", type=float, default=2.6)
    ap.add_argument(
        "--allow-songlike",
        action="store_true",
        help="Keep sung/music-like phrases instead of rejecting them.",
    )
    ap.add_argument("--song-score-threshold", type=float, default=0.50)
    ap.add_argument("--song-max-word-duration", type=float, default=2.0)
    ap.add_argument("--song-max-speech-rate", type=float, default=1.85)
    ap.add_argument(
        "--song-min-voiced-run-for-score",
        type=float,
        default=2.10,
        help="Require a continuous voiced/tonal run this long before a high song score can reject a clip.",
    )
    ap.add_argument(
        "--song-min-harmonic-ratio-for-score",
        type=float,
        default=0.58,
        help="Require this tonal ratio before a high song score can reject a clip without a long voiced run.",
    )
    ap.add_argument(
        "--song-max-zero-duration-word-ratio",
        type=float,
        default=0.18,
        help="Do not song-reject clips whose word timestamps are too broken to trust sustained-word cues.",
    )
    ap.add_argument(
        "--song-min-voiced-run-for-song",
        type=float,
        default=4.0,
        help="Require a tonal/voiced run this long before labeling a rejection as actual song.",
    )
    ap.add_argument(
        "--song-min-repeated-ngram-for-song",
        type=float,
        default=0.22,
        help="Require this much lyric-like repetition before sustained-word cues label a rejection as song.",
    )
    ap.add_argument(
        "--song-acoustic-backend",
        choices=["auto", "torch", "librosa"],
        default="auto",
        help="Acoustic song scoring backend. auto/torch computes one GPU-friendly STFT for the whole file; librosa is slower but closer to the original per-clip analysis.",
    )
    ap.add_argument("--song-device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--song-n-fft", type=int, default=1024)
    ap.add_argument("--song-hop-ms", type=float, default=10.0)
    ap.add_argument("--song-band-low-hz", type=float, default=80.0)
    ap.add_argument("--song-band-high-hz", type=float, default=4000.0)
    ap.add_argument("--song-voice-low-hz", type=float, default=80.0)
    ap.add_argument("--song-voice-high-hz", type=float, default=500.0)
    ap.add_argument("--song-tonal-top-k", type=int, default=5)
    ap.add_argument("--song-active-percentile", type=float, default=30.0)
    ap.add_argument("--song-tonal-voiced-threshold", type=float, default=0.12)
    ap.add_argument("--song-low-voice-threshold", type=float, default=0.08)
    ap.add_argument(
        "--song-feature-chunk-seconds",
        type=float,
        default=180.0,
        help="Chunk the torch/MPS song-feature STFT to avoid full-movie memory spikes.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    input_wav = Path(args.input_wav).expanduser().resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.transcript_json = args.transcript_json or str(out_dir / "whisper_sentence_filter.json")

    extra_summary: dict[str, Any] = {
        "speaker_boundary_mode": args.speaker_boundary_mode,
    }
    if args.speaker_boundary_mode == "audio_first":
        args.speaker_split = True
        turns, boundary_scores = detect_audio_first_speaker_turns(input_wav, args, out_dir)
        chunks = transcribe_audio_first_speaker_turns(input_wav, turns, args, out_dir)
        chunks, turn_first_summary = apply_turn_sentence_fallback(chunks, args)
        original_chunk_count = len(turns)
        speaker_turn_count = len(turns)
        speaker_split_count = max(0, len(turns) - 1)
        extra_summary.update(
            {
                **turn_first_summary,
                "audio_first_turns": len(turns),
                "audio_first_boundary_candidates": len(boundary_scores),
                "audio_first_selected_boundaries": max(0, len(turns) - 1),
                "audio_first_turns_manifest": str(
                    out_dir / "audio_first_speaker_turns" / "turns.json"
                ),
                "audio_first_boundary_scores": str(
                    out_dir / "audio_first_speaker_turns" / "boundary_scores.json"
                ),
            }
        )
    else:
        transcript = transcribe(input_wav, args)
        words = segment_words(transcript)
        if args.chunking == "speaker":
            args.speaker_split = True
            chunks = [words] if words else []
        else:
            chunks = split_candidates(
                words,
                args.min_duration,
                args.max_duration,
                args.pause_split,
            )
            chunks = merge_continuation_splits(chunks, args)
        original_chunk_count = len(chunks)
        chunks = split_chunks_on_speaker_changes(input_wav, chunks, args)
        speaker_turn_count = len(chunks)
        speaker_split_count = len(chunks) - original_chunk_count
        if args.clip_unit == "sentence":
            chunks = split_turns_into_complete_sentences(chunks, args)
    candidates = [
        score_candidate(chunk, args, f"sentence_{i:06d}")
        for i, chunk in enumerate(chunks)
        if chunk
    ]
    reject_song_like_candidates(input_wav, candidates, args)
    export_audio(input_wav, candidates, out_dir, args.clip_pad_seconds, args.compact_gap_seconds)
    accepted = [c for c in candidates if c.accepted]
    rejected = [c for c in candidates if not c.accepted]
    write_jsonl(out_dir / "accepted_sentences.jsonl", accepted)
    write_jsonl(out_dir / "rejected_sentences.jsonl", rejected)
    summarize(
        candidates,
        out_dir,
        {
            **extra_summary,
            "chunking": args.chunking,
            "clip_unit": args.clip_unit,
            "speaker_split_enabled": bool(args.speaker_split),
            "speaker_split_chunks_added": speaker_split_count,
            "initial_chunks": original_chunk_count,
            "speaker_turns": speaker_turn_count,
        },
    )


if __name__ == "__main__":
    main()
