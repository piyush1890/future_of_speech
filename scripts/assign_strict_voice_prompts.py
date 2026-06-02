#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


@dataclass(frozen=True)
class ClipAudio:
    samples: np.ndarray
    length: int


@dataclass(frozen=True)
class StyleFeatures:
    duration: float
    word_count: float
    char_count: float
    speech_rate_wps: float
    chars_per_second: float
    p95_word_duration: float
    rms_db: float
    peak_db: float
    energy_iqr_db: float
    voiced_ratio: float
    zero_crossing_rate: float
    harmonic_ratio: float
    pitch_median_hz: float
    pitch_range_octaves: float


STYLE_COMPONENT_WEIGHTS = {
    "duration": 0.14,
    "word_count": 0.12,
    "speech_rate": 0.18,
    "chars_per_second": 0.08,
    "p95_word_duration": 0.08,
    "rms_db": 0.12,
    "energy_iqr_db": 0.06,
    "voiced_ratio": 0.06,
    "zero_crossing_rate": 0.05,
    "harmonic_ratio": 0.04,
    "pitch_median_hz": 0.05,
    "pitch_range_octaves": 0.02,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_audio_path(audio_path: str, project_root: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_audio_16k(path: Path) -> ClipAudio:
    import librosa

    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
    if wav.size == 0:
        wav = np.zeros(1600, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return ClipAudio(samples=wav, length=int(wav.shape[0]))


def load_ecapa_classifier(model_dir: str, device_name: str):
    from speechbrain.inference.speaker import EncoderClassifier

    load_device = "cpu" if device_name == "mps" else device_name
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": load_device},
    )
    if device_name == "mps":
        mps_device = torch.device("mps")
        classifier.mods.to(mps_device)
        for value in vars(classifier.hparams).values():
            if isinstance(value, torch.nn.Module):
                value.to(mps_device)
        classifier.device = mps_device
        classifier.device_type = "mps"
    return classifier


def embed_clips(
    audio: list[ClipAudio],
    *,
    model_dir: str,
    device_name: str,
    batch_size: int,
) -> np.ndarray:
    classifier = load_ecapa_classifier(model_dir, device_name)
    device = torch.device(device_name)
    embeddings: list[np.ndarray] = []
    batch_size = max(1, int(batch_size))

    with torch.inference_mode():
        for start in range(0, len(audio), batch_size):
            batch = audio[start : start + batch_size]
            max_len = max(item.length for item in batch)
            padded = np.zeros((len(batch), max_len), dtype=np.float32)
            lens = np.zeros((len(batch),), dtype=np.float32)
            for idx, item in enumerate(batch):
                padded[idx, : item.length] = item.samples
                lens[idx] = item.length / max_len

            wav_tensor = torch.from_numpy(padded).to(device)
            wav_lens = torch.from_numpy(lens).to(device)
            emb = classifier.encode_batch(wav_tensor, wav_lens=wav_lens, normalize=True)
            emb_np = emb.squeeze(1).detach().cpu().numpy().astype(np.float32)
            embeddings.append(emb_np)

    matrix = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    return matrix / norms


def cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def count_text_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def frame_rms(samples: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    if samples.size < frame_length:
        samples = np.pad(samples, (0, frame_length - samples.size))
    frames = []
    for start in range(0, max(1, samples.size - frame_length + 1), hop_length):
        frame = samples[start : start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        frames.append(float(np.sqrt(np.mean(np.square(frame), dtype=np.float64))))
    return np.asarray(frames or [0.0], dtype=np.float32)


def estimate_pitch_stats(samples: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    if samples.size < int(sr * 0.35):
        return 0.0, 0.0
    try:
        import librosa

        f0 = librosa.yin(
            samples.astype(np.float32, copy=False),
            fmin=60,
            fmax=450,
            sr=sr,
            frame_length=1024,
            hop_length=320,
        )
    except Exception:
        return 0.0, 0.0

    f0 = np.asarray(f0, dtype=np.float32)
    f0 = f0[np.isfinite(f0)]
    f0 = f0[(f0 >= 60.0) & (f0 <= 450.0)]
    if f0.size < 3:
        return 0.0, 0.0
    p10, p90 = np.percentile(f0, [10, 90])
    pitch_range = math.log2(max(p90, 1e-6) / max(p10, 1e-6)) if p90 > p10 > 0 else 0.0
    return float(np.median(f0)), float(pitch_range)


def style_features_for_record(record: dict[str, Any], audio: ClipAudio) -> StyleFeatures:
    samples = audio.samples.astype(np.float32, copy=False)
    audio_duration = max(audio.length / 16000.0, 1e-3)
    duration = max(safe_float(record.get("duration"), audio_duration), 1e-3)
    word_count = max(safe_float(record.get("word_count")), 0.0)
    if word_count <= 0.0:
        word_count = float(len(str(record.get("text") or "").split()))
    char_count = float(count_text_chars(str(record.get("text") or "")))
    speech_rate = safe_float(record.get("speech_rate_wps"), word_count / duration if duration else 0.0)
    chars_per_second = char_count / duration if duration else 0.0
    p95_word_duration = safe_float(record.get("p95_word_duration"))

    rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-6))
    peak_db = 20.0 * math.log10(max(peak, 1e-6))

    rms_frames = frame_rms(samples)
    rms_frame_db = 20.0 * np.log10(np.maximum(rms_frames, 1e-6))
    energy_iqr_db = float(np.percentile(rms_frame_db, 75) - np.percentile(rms_frame_db, 25))
    voiced_threshold = max(float(np.percentile(rms_frames, 75)) * 0.25, 1e-4)
    voiced_ratio = float(np.mean(rms_frames > voiced_threshold)) if rms_frames.size else 0.0

    if samples.size > 1:
        zero_crossing_rate = float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1])))
    else:
        zero_crossing_rate = 0.0

    pitch_median, pitch_range = estimate_pitch_stats(samples)

    return StyleFeatures(
        duration=duration,
        word_count=word_count,
        char_count=char_count,
        speech_rate_wps=max(speech_rate, 0.0),
        chars_per_second=max(chars_per_second, 0.0),
        p95_word_duration=max(p95_word_duration, 0.0),
        rms_db=rms_db,
        peak_db=peak_db,
        energy_iqr_db=max(energy_iqr_db, 0.0),
        voiced_ratio=max(0.0, min(1.0, voiced_ratio)),
        zero_crossing_rate=max(0.0, zero_crossing_rate),
        harmonic_ratio=max(0.0, safe_float(record.get("harmonic_ratio"))),
        pitch_median_hz=max(pitch_median, 0.0),
        pitch_range_octaves=max(pitch_range, 0.0),
    )


def build_style_features(records: list[dict[str, Any]], audio: list[ClipAudio]) -> list[StyleFeatures]:
    return [style_features_for_record(record, clip_audio) for record, clip_audio in zip(records, audio)]


def log_ratio_delta(left: float, right: float, ratio_at_one: float) -> float:
    if left <= 0.0 and right <= 0.0:
        return 0.0
    if left <= 0.0 or right <= 0.0:
        return 1.0
    scale = max(math.log(ratio_at_one), 1e-6)
    return min(abs(math.log(left / right)) / scale, 1.0)


def abs_delta(left: float, right: float, scale: float) -> float:
    return min(abs(left - right) / max(scale, 1e-6), 1.0)


def style_distance_components(left: StyleFeatures, right: StyleFeatures) -> dict[str, float]:
    components = {
        "duration": log_ratio_delta(left.duration, right.duration, 2.0),
        "word_count": log_ratio_delta(left.word_count, right.word_count, 2.0),
        "speech_rate": log_ratio_delta(left.speech_rate_wps, right.speech_rate_wps, 1.8),
        "chars_per_second": log_ratio_delta(left.chars_per_second, right.chars_per_second, 1.8),
        "p95_word_duration": log_ratio_delta(left.p95_word_duration, right.p95_word_duration, 1.8),
        "rms_db": abs_delta(left.rms_db, right.rms_db, 12.0),
        "energy_iqr_db": abs_delta(left.energy_iqr_db, right.energy_iqr_db, 14.0),
        "voiced_ratio": abs_delta(left.voiced_ratio, right.voiced_ratio, 0.5),
        "zero_crossing_rate": abs_delta(left.zero_crossing_rate, right.zero_crossing_rate, 0.08),
        "harmonic_ratio": abs_delta(left.harmonic_ratio, right.harmonic_ratio, 0.35),
        "pitch_median_hz": log_ratio_delta(left.pitch_median_hz, right.pitch_median_hz, 1.7),
        "pitch_range_octaves": abs_delta(left.pitch_range_octaves, right.pitch_range_octaves, 0.8),
    }
    return {key: round(value, 4) for key, value in components.items()}


def style_distance(left: StyleFeatures, right: StyleFeatures) -> float:
    components = style_distance_components(left, right)
    total_weight = sum(STYLE_COMPONENT_WEIGHTS.values())
    return sum(components[key] * STYLE_COMPONENT_WEIGHTS[key] for key in components) / max(total_weight, 1e-6)


def style_distance_matrix(features: list[StyleFeatures]) -> np.ndarray:
    count = len(features)
    distances = np.zeros((count, count), dtype=np.float32)
    for left in range(count):
        for right in range(left + 1, count):
            distance = style_distance(features[left], features[right])
            distances[left, right] = distance
            distances[right, left] = distance
    return distances


def rounded_style_features(features: StyleFeatures) -> dict[str, float]:
    return {
        "duration": round(features.duration, 3),
        "word_count": round(features.word_count, 3),
        "char_count": round(features.char_count, 3),
        "speech_rate_wps": round(features.speech_rate_wps, 4),
        "chars_per_second": round(features.chars_per_second, 4),
        "p95_word_duration": round(features.p95_word_duration, 4),
        "rms_db": round(features.rms_db, 3),
        "peak_db": round(features.peak_db, 3),
        "energy_iqr_db": round(features.energy_iqr_db, 3),
        "voiced_ratio": round(features.voiced_ratio, 4),
        "zero_crossing_rate": round(features.zero_crossing_rate, 5),
        "harmonic_ratio": round(features.harmonic_ratio, 4),
        "pitch_median_hz": round(features.pitch_median_hz, 3),
        "pitch_range_octaves": round(features.pitch_range_octaves, 4),
    }


def record_quality(record: dict[str, Any]) -> float:
    quality = float(record.get("quality_score") or 0.0)
    avg_prob = float(record.get("avg_word_probability") or 0.0)
    min_prob = float(record.get("min_word_probability") or 0.0)
    duration = float(record.get("duration") or 0.0)
    song_like = float(record.get("song_like_score") or 0.0)
    duration_bonus = min(duration / 12.0, 1.0) * 0.08
    return (quality * 0.45) + (avg_prob * 0.35) + (min_prob * 0.12) + duration_bonus - (song_like * 0.15)


def greedy_complete_link_clusters(
    records: list[dict[str, Any]],
    distances: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    order = sorted(range(len(records)), key=lambda idx: record_quality(records[idx]), reverse=True)
    clusters: list[list[int]] = []

    for idx in order:
        best_cluster: int | None = None
        best_max_distance = float("inf")
        for cluster_idx, members in enumerate(clusters):
            max_distance = max(float(distances[idx, member]) for member in members)
            if max_distance <= threshold and max_distance < best_max_distance:
                best_cluster = cluster_idx
                best_max_distance = max_distance

        if best_cluster is None:
            clusters.append([idx])
        else:
            clusters[best_cluster].append(idx)

    return [sorted(cluster, key=lambda idx: float(records[idx].get("start") or 0.0)) for cluster in clusters]


def prompt_candidate_reject_reasons(record: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if float(record.get("duration") or 0.0) < args.min_prompt_duration:
        reasons.append("prompt_too_short")
    if float(record.get("duration") or 0.0) > args.max_prompt_duration:
        reasons.append("prompt_too_long")
    if int(record.get("word_count") or 0) < args.min_prompt_words:
        reasons.append("prompt_too_few_words")
    if float(record.get("quality_score") or 0.0) < args.min_prompt_quality:
        reasons.append("prompt_low_quality")
    if float(record.get("avg_word_probability") or 0.0) < args.min_prompt_avg_word_prob:
        reasons.append("prompt_low_avg_word_prob")
    if float(record.get("song_like_score") or 0.0) > args.max_prompt_song_score:
        reasons.append("prompt_song_like")
    return reasons


def is_prompt_candidate(record: dict[str, Any], args: argparse.Namespace) -> bool:
    return not prompt_candidate_reject_reasons(record, args)


def same_turn_clusters(records: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        turn_id = record.get("turn_id")
        if turn_id is None:
            continue
        groups.setdefault(str(turn_id), []).append(idx)
    clusters = [members for members in groups.values() if len(members) >= 2]
    return [
        sorted(cluster, key=lambda idx: float(records[idx].get("start") or 0.0))
        for cluster in clusters
    ]


def cluster_label_from_turn(record: dict[str, Any], fallback: int) -> str:
    turn_id = str(record.get("turn_id") or f"{fallback:06d}")
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in turn_id)
    return f"spk_{safe}"


def pair_selection_score(
    idx: int,
    candidate_idx: int,
    records: list[dict[str, Any]],
    speaker_distances: np.ndarray,
    style_distances: np.ndarray,
    args: argparse.Namespace,
) -> float:
    speaker_score = float(speaker_distances[idx, candidate_idx]) * args.speaker_distance_weight
    style_score = float(style_distances[idx, candidate_idx]) * args.style_match_weight
    quality_bonus = record_quality(records[candidate_idx]) * args.prompt_quality_weight
    return speaker_score + style_score - quality_bonus


def pair_sort_key(
    idx: int,
    candidate_idx: int,
    records: list[dict[str, Any]],
    speaker_distances: np.ndarray,
    style_distances: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    return (
        pair_selection_score(idx, candidate_idx, records, speaker_distances, style_distances, args),
        float(style_distances[idx, candidate_idx]),
        float(speaker_distances[idx, candidate_idx]),
        -record_quality(records[candidate_idx]),
    )


def choose_prompts(
    records: list[dict[str, Any]],
    clusters: list[list[int]],
    distances: np.ndarray,
    style_features: list[StyleFeatures],
    style_distances: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[int, str], set[int]]:
    assigned_cluster_id: dict[int, str] = {}
    cluster_by_record: dict[int, list[int]] = {}
    cluster_source_by_record: dict[int, str] = {}
    prompt_for_record: dict[int, int] = {}
    prompt_dist_for_record: dict[int, float] = {}
    prompt_source_for_record: dict[int, str] = {}
    selected_prompt_indexes: set[int] = set()

    for next_turn_cluster_id, cluster in enumerate(same_turn_clusters(records)):
        cluster_id = cluster_label_from_turn(records[cluster[0]], next_turn_cluster_id)
        for idx in cluster:
            assigned_cluster_id[idx] = cluster_id
            cluster_by_record[idx] = cluster
            cluster_source_by_record[idx] = "audio_turn"

        candidates = [idx for idx in cluster if is_prompt_candidate(records[idx], args)]
        candidates.sort(key=lambda idx: record_quality(records[idx]), reverse=True)
        if not candidates:
            continue

        for idx in cluster:
            prompt_choices = [candidate_idx for candidate_idx in candidates if candidate_idx != idx]
            if not prompt_choices:
                continue
            prompt_idx = min(
                prompt_choices,
                key=lambda candidate_idx: pair_sort_key(
                    idx, candidate_idx, records, distances, style_distances, args
                ),
            )
            prompt_for_record[idx] = prompt_idx
            prompt_dist_for_record[idx] = float(distances[idx, prompt_idx])
            prompt_source_for_record[idx] = "audio_turn"
            selected_prompt_indexes.add(prompt_idx)

    usable_clusters = [cluster for cluster in clusters if len(cluster) >= args.min_cluster_size]
    usable_clusters.sort(key=lambda cluster: (len(cluster), max(record_quality(records[idx]) for idx in cluster)), reverse=True)

    for next_cluster_id, cluster in enumerate(usable_clusters):
        cluster_id = f"spk_{next_cluster_id:04d}"
        for idx in cluster:
            if idx not in assigned_cluster_id:
                assigned_cluster_id[idx] = cluster_id
                cluster_by_record[idx] = cluster
                cluster_source_by_record[idx] = "ecapa_complete_link"

        candidates = [idx for idx in cluster if is_prompt_candidate(records[idx], args)]
        candidates.sort(key=lambda idx: record_quality(records[idx]), reverse=True)
        if not candidates:
            continue

        for idx in cluster:
            if idx in prompt_for_record:
                continue
            strict_matches = [
                prompt_idx
                for prompt_idx in candidates
                if prompt_idx != idx and float(distances[idx, prompt_idx]) <= args.prompt_pair_threshold
            ]
            if not strict_matches:
                continue
            prompt_idx = min(
                strict_matches,
                key=lambda candidate_idx: pair_sort_key(
                    idx, candidate_idx, records, distances, style_distances, args
                ),
            )
            prompt_for_record[idx] = prompt_idx
            prompt_dist_for_record[idx] = float(distances[idx, prompt_idx])
            prompt_source_for_record[idx] = "ecapa_complete_link"
            selected_prompt_indexes.add(prompt_idx)

    all_prompt_candidates = [idx for idx in range(len(records)) if is_prompt_candidate(records[idx], args)]
    nearest_fallback_counter = 0
    for idx in range(len(records)):
        if idx in prompt_for_record:
            continue

        cluster = cluster_by_record.get(idx, [])
        same_cluster_candidates = [
            candidate_idx
            for candidate_idx in cluster
            if candidate_idx != idx and is_prompt_candidate(records[candidate_idx], args)
        ]
        candidate_pool = same_cluster_candidates or [
            candidate_idx for candidate_idx in all_prompt_candidates if candidate_idx != idx
        ]
        strict_matches = [
            prompt_idx
            for prompt_idx in candidate_pool
            if float(distances[idx, prompt_idx]) <= args.prompt_pair_threshold
        ]
        if not strict_matches:
            continue

        prompt_idx = min(
            strict_matches,
            key=lambda candidate_idx: pair_sort_key(
                idx, candidate_idx, records, distances, style_distances, args
            ),
        )
        prompt_for_record[idx] = prompt_idx
        prompt_dist_for_record[idx] = float(distances[idx, prompt_idx])
        prompt_source_for_record[idx] = (
            "ecapa_relaxed_same_cluster" if same_cluster_candidates else "ecapa_nearest_neighbor"
        )
        selected_prompt_indexes.add(prompt_idx)

        if prompt_idx not in assigned_cluster_id:
            nearest_fallback_counter += 1
            fallback_cluster_id = f"spk_nn_{nearest_fallback_counter:04d}"
            assigned_cluster_id[prompt_idx] = fallback_cluster_id
            cluster_by_record[prompt_idx] = [idx, prompt_idx]
            cluster_source_by_record[prompt_idx] = "ecapa_nearest_neighbor"
        if idx not in assigned_cluster_id:
            assigned_cluster_id[idx] = assigned_cluster_id[prompt_idx]
        cluster_by_record[idx] = sorted(
            set(cluster_by_record.get(idx, [])) | {idx, prompt_idx},
            key=lambda record_idx: float(records[record_idx].get("start") or 0.0),
        )
        cluster_source_by_record[idx] = prompt_source_for_record[idx]

    short_clip_prompt_candidates = [
        idx for idx in all_prompt_candidates if idx in assigned_cluster_id
    ]
    for idx in range(len(records)):
        if idx in prompt_for_record:
            continue

        duration = float(records[idx].get("duration") or 0.0)
        if duration <= 0.0 or duration > args.short_clip_max_duration:
            continue

        start = float(records[idx].get("start") or 0.0)
        end = float(records[idx].get("end") or start)
        midpoint = (start + end) / 2.0
        strict_matches: list[int] = []
        for prompt_idx in short_clip_prompt_candidates:
            if prompt_idx == idx:
                continue
            prompt_start = float(records[prompt_idx].get("start") or 0.0)
            prompt_end = float(records[prompt_idx].get("end") or prompt_start)
            prompt_midpoint = (prompt_start + prompt_end) / 2.0
            if abs(prompt_midpoint - midpoint) > args.short_clip_max_prompt_gap:
                continue
            if float(distances[idx, prompt_idx]) <= args.short_clip_prompt_pair_threshold:
                strict_matches.append(prompt_idx)
        if not strict_matches:
            continue

        prompt_idx = min(
            strict_matches,
            key=lambda candidate_idx: (
                pair_selection_score(idx, candidate_idx, records, distances, style_distances, args),
                float(style_distances[idx, candidate_idx]),
                abs(
                    (
                        float(records[candidate_idx].get("start") or 0.0)
                        + float(records[candidate_idx].get("end") or 0.0)
                    )
                    / 2.0
                        - midpoint
                ),
                float(distances[idx, candidate_idx]),
            ),
        )
        prompt_for_record[idx] = prompt_idx
        prompt_dist_for_record[idx] = float(distances[idx, prompt_idx])
        prompt_source_for_record[idx] = "ecapa_short_clip_nearby"
        selected_prompt_indexes.add(prompt_idx)

        assigned_cluster_id[idx] = assigned_cluster_id[prompt_idx]
        cluster_by_record[idx] = sorted(
            set(cluster_by_record.get(idx, [])) | set(cluster_by_record.get(prompt_idx, [])) | {idx, prompt_idx},
            key=lambda record_idx: float(records[record_idx].get("start") or 0.0),
        )
        cluster_source_by_record[idx] = prompt_source_for_record[idx]

    enriched: list[dict[str, Any]] = []
    prompt_name_by_index: dict[int, str] = {}
    prompt_counts_by_cluster: dict[str, int] = {}
    for prompt_idx in sorted(selected_prompt_indexes, key=lambda idx: (assigned_cluster_id[idx], idx)):
        clip_id = str(records[prompt_idx].get("clip_id") or f"clip_{prompt_idx:06d}")
        cluster_id = assigned_cluster_id[prompt_idx]
        prompt_number = prompt_counts_by_cluster.get(cluster_id, 0)
        prompt_counts_by_cluster[cluster_id] = prompt_number + 1
        prompt_name_by_index[prompt_idx] = f"{cluster_id}_prompt_{prompt_number:02d}_{clip_id}.wav"

    for idx, record in enumerate(records):
        output = dict(record)
        cluster_id = assigned_cluster_id.get(idx)
        cluster = cluster_by_record.get(idx, [])
        candidate_reject_reasons = prompt_candidate_reject_reasons(record, args)
        same_cluster_candidates = [
            candidate_idx
            for candidate_idx in cluster
            if candidate_idx != idx and is_prompt_candidate(records[candidate_idx], args)
        ]
        closest_prompt_idx: int | None = None
        closest_prompt_distance: float | None = None
        if same_cluster_candidates:
            closest_prompt_idx = min(same_cluster_candidates, key=lambda candidate_idx: float(distances[idx, candidate_idx]))
            closest_prompt_distance = float(distances[idx, closest_prompt_idx])

        output["speaker_cluster_id"] = cluster_id
        output["speaker_cluster_source"] = cluster_source_by_record.get(idx)
        output["style_features"] = rounded_style_features(style_features[idx])
        output["is_prompt_candidate"] = not candidate_reject_reasons
        output["prompt_candidate_reject_reasons"] = candidate_reject_reasons
        output["closest_prompt_clip_id"] = records[closest_prompt_idx].get("clip_id") if closest_prompt_idx is not None else None
        output["closest_prompt_distance"] = round(closest_prompt_distance, 4) if closest_prompt_distance is not None else None
        output["is_voice_prompt_clip"] = idx in selected_prompt_indexes
        if idx in prompt_for_record:
            prompt_idx = prompt_for_record[idx]
            output["voice_prompt_clip_id"] = records[prompt_idx].get("clip_id")
            output["voice_prompt"] = str(args.prompt_dir / prompt_name_by_index[prompt_idx])
            output["voice_prompt_distance"] = round(prompt_dist_for_record[idx], 4)
            output["voice_prompt_style_distance"] = round(float(style_distances[idx, prompt_idx]), 4)
            output["voice_prompt_style_components"] = style_distance_components(
                style_features[idx], style_features[prompt_idx]
            )
            output["voice_prompt_style_features"] = rounded_style_features(style_features[prompt_idx])
            output["voice_prompt_selection_score"] = round(
                pair_selection_score(idx, prompt_idx, records, distances, style_distances, args),
                4,
            )
            output["voice_prompt_match_source"] = prompt_source_for_record.get(idx)
            output["prompt_status"] = "strict_match"
            output["train_allowed"] = not (idx in selected_prompt_indexes and args.holdout_prompts)
        else:
            output["voice_prompt_clip_id"] = None
            output["voice_prompt"] = None
            output["voice_prompt_distance"] = None
            output["voice_prompt_style_distance"] = None
            output["voice_prompt_style_components"] = None
            output["voice_prompt_style_features"] = None
            output["voice_prompt_selection_score"] = None
            output["voice_prompt_match_source"] = None
            if cluster_id is None:
                output["prompt_status"] = "unpaired_small_or_singleton_cluster"
            elif not same_cluster_candidates:
                output["prompt_status"] = "unpaired_no_clean_prompt_candidate"
            else:
                output["prompt_status"] = "unpaired_prompt_distance_too_high"
            output["train_allowed"] = not (idx in selected_prompt_indexes and args.holdout_prompts)
        enriched.append(output)

    return enriched, prompt_name_by_index, selected_prompt_indexes


def copy_prompt_files(
    records: list[dict[str, Any]],
    prompt_name_by_index: dict[int, str],
    *,
    project_root: Path,
    prompt_dir: Path,
) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for old_prompt in prompt_dir.glob("*.wav"):
        old_prompt.unlink()
    for idx, name in prompt_name_by_index.items():
        src = resolve_audio_path(str(records[idx]["audio"]), project_root)
        dst = prompt_dir / name
        shutil.copy2(src, dst)


def cluster_summary(
    records: list[dict[str, Any]],
    clusters: list[list[int]],
    distances: np.ndarray,
    assigned_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    clusters_by_id: dict[str, list[int]] = {}
    for idx, record in enumerate(assigned_records):
        cluster_id = record.get("speaker_cluster_id")
        if cluster_id:
            clusters_by_id.setdefault(str(cluster_id), []).append(idx)

    for cluster_id, cluster in clusters_by_id.items():
        pairwise = [
            float(distances[left, right])
            for pos, left in enumerate(cluster)
            for right in cluster[pos + 1 :]
        ]
        prompt_count = sum(1 for idx in cluster if assigned_records[idx].get("is_voice_prompt_clip"))
        paired_count = sum(1 for idx in cluster if assigned_records[idx].get("prompt_status") == "strict_match")
        sources = sorted(
            {
                str(assigned_records[idx].get("speaker_cluster_source"))
                for idx in cluster
                if assigned_records[idx].get("speaker_cluster_source")
            }
        )
        summary.append(
            {
                "speaker_cluster_id": cluster_id,
                "sources": sources,
                "size": len(cluster),
                "paired_clips": paired_count,
                "prompt_clips": prompt_count,
                "duration_seconds": round(sum(float(records[idx].get("duration") or 0.0) for idx in cluster), 2),
                "max_pairwise_distance": round(max(pairwise), 4) if pairwise else 0.0,
                "mean_pairwise_distance": round(float(np.mean(pairwise)), 4) if pairwise else 0.0,
                "clip_ids": [records[idx].get("clip_id") for idx in cluster],
            }
        )
    return sorted(summary, key=lambda item: item["speaker_cluster_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign strict same-speaker voice prompts to accepted dialogue clips."
    )
    parser.add_argument("manifest", type=Path, help="accepted_sentences.jsonl manifest")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", default="models/speechbrain-spkrec-ecapa-voxceleb")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--same-speaker-threshold",
        type=float,
        default=0.34,
        help="Complete-link cluster threshold. Lower is stricter.",
    )
    parser.add_argument(
        "--prompt-pair-threshold",
        type=float,
        default=0.34,
        help="Maximum ECAPA cosine distance from a target clip to its prompt. Lower is stricter.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--min-prompt-duration", type=float, default=1.4)
    parser.add_argument("--max-prompt-duration", type=float, default=30.0)
    parser.add_argument("--min-prompt-words", type=int, default=8)
    parser.add_argument("--min-prompt-quality", type=float, default=0.70)
    parser.add_argument("--min-prompt-avg-word-prob", type=float, default=0.80)
    parser.add_argument("--max-prompt-song-score", type=float, default=0.32)
    parser.add_argument("--short-clip-max-duration", type=float, default=3.0)
    parser.add_argument("--short-clip-max-prompt-gap", type=float, default=45.0)
    parser.add_argument("--short-clip-prompt-pair-threshold", type=float, default=0.48)
    parser.add_argument("--style-matching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--style-match-weight",
        type=float,
        default=0.35,
        help="How strongly prosody/style distance affects prompt choice after the same-speaker gate.",
    )
    parser.add_argument("--speaker-distance-weight", type=float, default=1.0)
    parser.add_argument(
        "--prompt-quality-weight",
        type=float,
        default=0.04,
        help="Small bonus for cleaner prompt candidates when speaker/style scores are close.",
    )
    parser.add_argument("--holdout-prompts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.manifest = args.manifest.resolve()
    if args.out_dir is None:
        args.out_dir = args.manifest.parent / "strict_voice_prompts"
    args.out_dir = args.out_dir.resolve()
    args.prompt_dir = args.out_dir / "prompts"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.manifest)
    if not records:
        raise SystemExit(f"No records found in {args.manifest}")

    audio_paths = [resolve_audio_path(str(record["audio"]), args.project_root) for record in records]
    missing = [str(path) for path in audio_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing audio files: {missing[:5]}")

    audio = [load_audio_16k(path) for path in audio_paths]
    embeddings = embed_clips(
        audio,
        model_dir=args.model_dir,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    distances = cosine_distances(embeddings)
    style_features = build_style_features(records, audio)
    if args.style_matching and args.style_match_weight > 0.0:
        style_distances = style_distance_matrix(style_features)
    else:
        style_distances = np.zeros_like(distances, dtype=np.float32)
    clusters = greedy_complete_link_clusters(records, distances, args.same_speaker_threshold)
    enriched, prompt_name_by_index, selected_prompt_indexes = choose_prompts(
        records,
        clusters,
        distances,
        style_features,
        style_distances,
        args,
    )
    copy_prompt_files(
        records,
        prompt_name_by_index,
        project_root=args.project_root,
        prompt_dir=args.prompt_dir,
    )

    pair_manifest = args.out_dir / "voice_prompt_pairs.jsonl"
    paired_manifest = args.out_dir / "paired_with_prompts.jsonl"
    train_manifest = args.out_dir / "train_with_prompts.jsonl"
    write_jsonl(pair_manifest, enriched)
    paired_records = [record for record in enriched if record.get("prompt_status") == "strict_match"]
    train_records = [record for record in paired_records if record.get("train_allowed")]
    write_jsonl(paired_manifest, paired_records)
    write_jsonl(train_manifest, train_records)

    np.save(args.out_dir / "ecapa_embeddings.npy", embeddings)
    np.save(args.out_dir / "ecapa_distances.npy", distances)
    np.save(args.out_dir / "style_distances.npy", style_distances)

    upper = distances[np.triu_indices_from(distances, k=1)]
    prompt_match_source_counts: dict[str, int] = {}
    for record in paired_records:
        source = str(record.get("voice_prompt_match_source") or "unknown")
        prompt_match_source_counts[source] = prompt_match_source_counts.get(source, 0) + 1
    paired_style_distances = [
        float(record["voice_prompt_style_distance"])
        for record in paired_records
        if record.get("voice_prompt_style_distance") is not None
    ]
    style_component_means: dict[str, float] = {}
    if paired_records:
        for key in STYLE_COMPONENT_WEIGHTS:
            values = [
                float((record.get("voice_prompt_style_components") or {}).get(key, 0.0))
                for record in paired_records
                if record.get("voice_prompt_style_components") is not None
            ]
            if values:
                style_component_means[key] = round(float(np.mean(values)), 4)
    summary = {
        "manifest": str(args.manifest),
        "total_clips": len(records),
        "paired_clips": sum(1 for record in enriched if record.get("prompt_status") == "strict_match"),
        "unpaired_clips": sum(1 for record in enriched if record.get("prompt_status") != "strict_match"),
        "speaker_clusters": len({record.get("speaker_cluster_id") for record in enriched if record.get("speaker_cluster_id")}),
        "selected_prompt_clips": len(selected_prompt_indexes),
        "paired_seconds": round(sum(float(record.get("duration") or 0.0) for record in paired_records), 2),
        "train_clips": len(train_records),
        "train_seconds": round(sum(float(record.get("duration") or 0.0) for record in train_records), 2),
        "heldout_prompt_clips": sum(1 for record in enriched if not record.get("train_allowed")),
        "same_speaker_threshold": args.same_speaker_threshold,
        "prompt_pair_threshold": args.prompt_pair_threshold,
        "short_clip_max_duration": args.short_clip_max_duration,
        "short_clip_max_prompt_gap": args.short_clip_max_prompt_gap,
        "short_clip_prompt_pair_threshold": args.short_clip_prompt_pair_threshold,
        "style_matching": args.style_matching,
        "style_match_weight": args.style_match_weight,
        "speaker_distance_weight": args.speaker_distance_weight,
        "prompt_quality_weight": args.prompt_quality_weight,
        "style_component_weights": STYLE_COMPONENT_WEIGHTS,
        "style_distance_percentiles": {
            str(percentile): round(float(np.percentile(paired_style_distances, percentile)), 4)
            for percentile in (5, 10, 25, 50, 75, 90, 95)
        }
        if paired_style_distances
        else {},
        "style_component_means": style_component_means,
        "prompt_match_source_counts": dict(sorted(prompt_match_source_counts.items())),
        "distance_percentiles": {
            str(percentile): round(float(np.percentile(upper, percentile)), 4)
            for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "outputs": {
            "voice_prompt_pairs": str(pair_manifest),
            "paired_with_prompts": str(paired_manifest),
            "train_with_prompts": str(train_manifest),
            "prompts": str(args.prompt_dir),
        },
        "clusters": cluster_summary(records, clusters, distances, enriched),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
