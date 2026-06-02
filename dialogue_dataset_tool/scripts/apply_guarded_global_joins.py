#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import librosa
import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier


TERMINAL_SUFFIXES = ("।", ".", "?", "!", "…")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def resolve_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def slugify(value: str) -> str:
    value = value.strip().lower() or "unknown"
    value = re.sub(r"[^a-z0-9_+.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:120] or "unknown"


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False), int(sr)


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    return vector.astype(np.float32) / max(float(np.linalg.norm(vector)), 1e-8)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - np.dot(left, right))


class EcapaEmbedder:
    def __init__(self, model_dir: str, device: str) -> None:
        if device == "cpu":
            raise RuntimeError("ECAPA global join must run on GPU/MPS; refusing CPU fallback.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available; refusing CPU fallback.")
        load_device = "cpu" if device == "mps" else device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
            run_opts={"device": load_device},
        )
        self.device = torch.device(device)
        if device == "mps":
            self.classifier.mods.to(self.device)
            for value in vars(self.classifier.hparams).values():
                if isinstance(value, torch.nn.Module):
                    value.to(self.device)
            self.classifier.device = self.device
            self.classifier.device_type = "mps"
        self.cache: dict[Path, np.ndarray] = {}

    def waveform_embedding(self, wav: np.ndarray, sr: int, min_seconds: float) -> np.ndarray:
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        wav = wav.astype(np.float32, copy=False)
        min_samples = int(round(min_seconds * sr))
        if len(wav) < min_samples:
            wav = np.pad(wav, (0, max(0, min_samples - len(wav))))
        if not np.any(np.abs(wav) > 1e-7):
            wav = np.zeros(max(min_samples, sr // 2), dtype=np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(wav[None, :]).to(self.device)
            emb = (
                self.classifier.encode_batch(tensor, normalize=True)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        return normalize_embedding(emb)

    def file_embedding(self, path: Path, min_seconds: float) -> np.ndarray:
        path = path.resolve()
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        wav, sr = load_audio(path)
        embedding = self.waveform_embedding(wav, sr, min_seconds)
        self.cache[path] = embedding
        return embedding


def infer_clip_abs_start(row: dict[str, Any], audio_duration: float, clip_pad_seconds: float) -> float:
    start = float(row.get("start") or 0.0)
    end = float(row.get("end") or start)
    turn_start = row.get("turn_start")
    turn_end = row.get("turn_end")
    padded_start = start - clip_pad_seconds
    if turn_start is not None:
        padded_start = max(padded_start, float(turn_start))
    padded_start = max(0.0, padded_start)
    padded_end = end + clip_pad_seconds
    if turn_end is not None:
        padded_end = min(padded_end, float(turn_end))
    expected = max(0.0, padded_end - padded_start)
    if abs(expected - audio_duration) <= 0.35:
        return padded_start
    return start


def words_text(words: list[dict[str, Any]]) -> str:
    return re.sub(r"\s+", " ", " ".join(str(w.get("word", "")).strip() for w in words)).strip()


def word_stats(words: list[dict[str, Any]]) -> dict[str, Any]:
    probs = [
        float(w["probability"])
        for w in words
        if w.get("probability") is not None and np.isfinite(float(w["probability"]))
    ]
    return {
        "word_count": len([w for w in words if str(w.get("word", "")).strip()]),
        "avg_word_probability": round(float(np.mean(probs)), 4) if probs else None,
        "min_word_probability": round(float(np.min(probs)), 4) if probs else None,
    }


def selected_word_boundary_cuts(
    row: dict[str, Any],
    wav: np.ndarray,
    sr: int,
    clip_abs_start: float,
    embedder: EcapaEmbedder,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    words = row.get("words") or []
    if len(words) < 2:
        return []
    audio_duration = len(wav) / max(sr, 1)
    scored_cuts: list[dict[str, Any]] = []
    for index in range(len(words) - 1):
        current_word = str(words[index].get("word", "")).strip()
        if args.word_boundary_require_terminal_punctuation and not current_word.endswith(TERMINAL_SUFFIXES):
            continue
        boundary = float(words[index]["end"]) - clip_abs_start
        if boundary < args.word_boundary_min_side_seconds:
            continue
        if audio_duration - boundary < args.word_boundary_min_side_seconds:
            continue
        left_start = max(0.0, boundary - args.word_boundary_context_seconds)
        right_end = min(audio_duration, boundary + args.word_boundary_context_seconds)
        left = wav[int(round(left_start * sr)):int(round(boundary * sr))]
        right = wav[int(round(boundary * sr)):int(round(right_end * sr))]
        if len(left) == 0 or len(right) == 0:
            continue
        left_embedding = embedder.waveform_embedding(left, sr, args.min_embedding_seconds)
        right_embedding = embedder.waveform_embedding(right, sr, args.min_embedding_seconds)
        distance = cosine_distance(left_embedding, right_embedding)
        scored_cuts.append(
            {
                "word_index": index,
                "boundary_seconds": round(boundary, 3),
                "absolute_seconds": round(float(words[index]["end"]), 3),
                "distance": round(distance, 4),
                "after_word": current_word,
                "before_word": str(words[index + 1].get("word", "")).strip(),
                "selected": False,
            }
        )
    selected: list[dict[str, Any]] = []
    last_boundary: float | None = None
    for cut in scored_cuts:
        boundary = float(cut["boundary_seconds"])
        if float(cut["distance"]) < args.word_boundary_speaker_threshold:
            continue
        if (
            last_boundary is not None
            and boundary - last_boundary < args.word_boundary_min_gap_seconds
        ):
            continue
        cut["selected"] = True
        selected.append(cut)
        last_boundary = boundary
    return selected


def build_parts(
    rows: list[dict[str, Any]],
    project_root: Path,
    embedder: EcapaEmbedder,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parts: list[dict[str, Any]] = []
    cut_manifest: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        audio_path = resolve_path(row.get("audio"), project_root)
        if audio_path is None or not audio_path.exists():
            continue
        wav, sr = load_audio(audio_path)
        audio_duration = len(wav) / max(sr, 1)
        clip_abs_start = infer_clip_abs_start(row, audio_duration, args.clip_pad_seconds)
        cuts = selected_word_boundary_cuts(row, wav, sr, clip_abs_start, embedder, args)
        cut_manifest.append(
            {
                "clip_id": row.get("clip_id"),
                "audio": str(audio_path),
                "selected_cuts": cuts,
            }
        )
        selected_indices = [int(cut["word_index"]) for cut in cuts]
        boundaries = [0.0] + [float(cut["boundary_seconds"]) for cut in cuts] + [audio_duration]
        word_ranges: list[tuple[int, int]] = []
        start_word = 0
        for cut_index in selected_indices:
            word_ranges.append((start_word, cut_index + 1))
            start_word = cut_index + 1
        word_ranges.append((start_word, len(row.get("words") or [])))

        for part_index, ((audio_start, audio_end), (word_start, word_end)) in enumerate(
            zip(zip(boundaries[:-1], boundaries[1:]), word_ranges)
        ):
            part_words = list((row.get("words") or [])[word_start:word_end])
            if not part_words:
                continue
            start_i = max(0, int(round(audio_start * sr)))
            end_i = min(len(wav), int(round(audio_end * sr)))
            if end_i <= start_i:
                continue
            part_wav = wav[start_i:end_i].astype(np.float32, copy=False)
            embedding = embedder.waveform_embedding(part_wav, sr, args.min_embedding_seconds)
            part_id = f"{row.get('clip_id')}_part_{part_index:02d}"
            parts.append(
                {
                    "part_id": part_id,
                    "source_row_index": row_index,
                    "source_clip_id": row.get("clip_id"),
                    "source_audio": str(audio_path),
                    "source_part_index": part_index,
                    "wav": part_wav,
                    "sr": sr,
                    "embedding": embedding,
                    "words": part_words,
                    "text": words_text(part_words),
                    "start": float(part_words[0]["start"]),
                    "end": float(part_words[-1]["end"]),
                    "duration": max(0.0, float(part_words[-1]["end"]) - float(part_words[0]["start"])),
                    "source_row": row,
                    "split_from_source": bool(cuts),
                    "audio_start_seconds": round(audio_start, 3),
                    "audio_end_seconds": round(audio_end, 3),
                }
            )
    parts.sort(key=lambda item: (float(item["start"]), float(item["end"]), str(item["part_id"])))
    return parts, cut_manifest


def prompt_anchors(
    rows: list[dict[str, Any]],
    project_root: Path,
    embedder: EcapaEmbedder,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for row in rows:
        prompt_path = resolve_path(row.get("voice_prompt"), project_root)
        if prompt_path is None or not prompt_path.exists() or prompt_path in seen:
            continue
        seen.add(prompt_path)
        anchors.append(
            {
                "path": prompt_path,
                "embedding": embedder.file_embedding(prompt_path, args.min_embedding_seconds),
                "speaker_cluster_id": row.get("speaker_cluster_id"),
                "voice_prompt_clip_id": row.get("voice_prompt_clip_id"),
                "source_row": row,
            }
        )
    return anchors


def nearest_anchor(embedding: np.ndarray, anchors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not anchors:
        return None
    ranked = sorted(
        (
            (cosine_distance(embedding, np.asarray(anchor["embedding"], dtype=np.float32)), anchor)
            for anchor in anchors
        ),
        key=lambda item: item[0],
    )
    distance, anchor = ranked[0]
    return {
        "distance": float(distance),
        "speaker_cluster_id": anchor.get("speaker_cluster_id"),
        "voice_prompt_clip_id": anchor.get("voice_prompt_clip_id"),
        "path": anchor["path"],
        "top5": [
            {
                "distance": round(float(item_distance), 4),
                "speaker_cluster_id": item_anchor.get("speaker_cluster_id"),
                "voice_prompt": str(item_anchor["path"]),
            }
            for item_distance, item_anchor in ranked[:5]
        ],
    }


def add_nearest_anchors(parts: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> None:
    for part in parts:
        part["nearest_anchor"] = nearest_anchor(part["embedding"], anchors)


def should_join(left: dict[str, Any], right: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    gap = max(0.0, float(right["start"]) - float(left["end"]))
    distance = cosine_distance(left["embedding"], right["embedding"])
    left_anchor = left.get("nearest_anchor") or {}
    right_anchor = right.get("nearest_anchor") or {}
    left_anchor_distance = float(left_anchor.get("distance", 999.0))
    right_anchor_distance = float(right_anchor.get("distance", 999.0))
    left_cluster = left_anchor.get("speaker_cluster_id")
    right_cluster = right_anchor.get("speaker_cluster_id")

    if gap > args.global_join_max_gap_seconds:
        reason = "gap_too_large"
        joined = False
    else:
        speaker_to_unknown_guard = (
            left_anchor_distance <= args.global_join_strong_known_distance
            and right_anchor_distance >= args.global_join_unknownish_distance
        ) or (
            right_anchor_distance <= args.global_join_strong_known_distance
            and left_anchor_distance >= args.global_join_unknownish_distance
        )
        if distance <= args.global_join_strict_distance:
            joined = True
            reason = "strict_adjacent_same_voice"
        elif distance <= args.global_join_medium_distance and not speaker_to_unknown_guard:
            joined = True
            reason = "medium_adjacent_same_voice_no_anchor_drift"
        else:
            joined = False
            reason = "anchor_drift_guard" if speaker_to_unknown_guard else "adjacent_distance_too_high"

    return joined, {
        "left_part": left["part_id"],
        "right_part": right["part_id"],
        "gap_seconds": round(gap, 4),
        "adjacent_distance": round(distance, 4),
        "left_nearest_anchor": {
            **left_anchor,
            "distance": round(left_anchor_distance, 4),
            "path": str(left_anchor.get("path")) if left_anchor.get("path") else None,
        },
        "right_nearest_anchor": {
            **right_anchor,
            "distance": round(right_anchor_distance, 4),
            "path": str(right_anchor.get("path")) if right_anchor.get("path") else None,
        },
        "joined": joined,
        "reason": reason,
    }


def group_parts(parts: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    if not parts:
        return [], []
    groups: list[list[dict[str, Any]]] = []
    decisions: list[dict[str, Any]] = []
    current = [parts[0]]
    for left, right in zip(parts, parts[1:]):
        joined, decision = should_join(left, right, args)
        decisions.append(decision)
        if joined:
            current.append(right)
        else:
            groups.append(current)
            current = [right]
    groups.append(current)
    return groups, decisions


def concatenate_group_audio(group: list[dict[str, Any]], args: argparse.Namespace) -> tuple[np.ndarray, int]:
    sr = int(group[0]["sr"])
    chunks: list[np.ndarray] = []
    previous: dict[str, Any] | None = None
    for part in group:
        if previous is not None and part["source_clip_id"] != previous["source_clip_id"]:
            gap = max(0.0, float(part["start"]) - float(previous["end"]))
            if gap > 0.0:
                chunks.append(np.zeros(int(round(min(gap, args.global_join_max_gap_seconds) * sr)), dtype=np.float32))
        wav = np.asarray(part["wav"], dtype=np.float32)
        if int(part["sr"]) != sr:
            wav = librosa.resample(wav, orig_sr=int(part["sr"]), target_sr=sr).astype(np.float32)
        chunks.append(wav)
        previous = part
    if not chunks:
        return np.asarray([], dtype=np.float32), sr
    return np.concatenate(chunks), sr


def copy_prompt(prompt_path: Path, prompt_dir: Path, copied: dict[Path, Path]) -> Path:
    prompt_path = prompt_path.resolve()
    if prompt_path in copied:
        return copied[prompt_path]
    prompt_dir.mkdir(parents=True, exist_ok=True)
    dest = prompt_dir / prompt_path.name
    if dest.exists() and dest.resolve() != prompt_path:
        dest = prompt_dir / f"{len(copied):06d}_{prompt_path.name}"
    shutil.copy2(prompt_path, dest)
    copied[prompt_path] = dest
    return dest


def output_row(
    group: list[dict[str, Any]],
    clip_id: str,
    audio_path: Path,
    prompt_path: Path,
    nearest: dict[str, Any],
    audio_duration: float,
) -> dict[str, Any]:
    words = [word for part in group for word in part["words"]]
    words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    source_rows = [part["source_row"] for part in group]
    base = dict(source_rows[0])
    stats = word_stats(words)
    start = float(words[0]["start"])
    end = float(words[-1]["end"])
    base.update(
        {
            "clip_id": clip_id,
            "audio": str(audio_path),
            "text": words_text(words),
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "audio_duration_seconds": round(audio_duration, 3),
            "words": [
                {
                    "word": word.get("word"),
                    "start": round(float(word["start"]), 3),
                    "end": round(float(word["end"]), 3),
                    "probability": (
                        round(float(word["probability"]), 4)
                        if word.get("probability") is not None
                        else None
                    ),
                }
                for word in words
            ],
            **stats,
            "source_clip_ids": [part["source_clip_id"] for part in group],
            "source_part_ids": [part["part_id"] for part in group],
            "guarded_global_join_applied": True,
            "speaker_cluster_id": nearest.get("speaker_cluster_id"),
            "voice_prompt_clip_id": nearest.get("voice_prompt_clip_id"),
            "voice_prompt": str(prompt_path),
            "voice_prompt_distance": round(float(nearest["distance"]), 4),
            "voice_prompt_match_source": "guarded_global_join_nearest_prompt",
            "prompt_status": "strict_match",
            "train_allowed": True,
        }
    )
    return base


def apply_guarded_global_joins(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    prompt_dir = args.prompt_dir.resolve()
    sentence_dir = prompt_dir.parent
    out_dir = args.out_dir.resolve() if args.out_dir else sentence_dir / args.out_dir_name
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    prompts_dir = out_dir / "prompts"
    rejected_dir = out_dir / "rejected_unmatched_fragments"
    for directory in (clips_dir, prompts_dir, rejected_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(prompt_dir / "train_with_prompts.jsonl")
    if not rows:
        raise FileNotFoundError(f"missing or empty train manifest: {prompt_dir / 'train_with_prompts.jsonl'}")

    embedder = EcapaEmbedder(args.speechbrain_model_dir, args.device)
    anchors = prompt_anchors(rows, project_root, embedder, args)
    if not anchors:
        raise RuntimeError("No voice prompt anchors found; guarded global join cannot run.")

    parts, cut_manifest = build_parts(rows, project_root, embedder, args)
    add_nearest_anchors(parts, anchors)
    groups, join_decisions = group_parts(parts, args)

    prompt_copies: dict[Path, Path] = {}
    output_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    group_manifest: list[dict[str, Any]] = []
    for group in groups:
        audio, sr = concatenate_group_audio(group, args)
        if len(audio) == 0:
            continue
        group_embedding = embedder.waveform_embedding(audio, sr, args.min_embedding_seconds)
        nearest = nearest_anchor(group_embedding, anchors)
        if nearest is None:
            continue
        audio_duration = len(audio) / max(sr, 1)
        is_tiny_unmatched = (
            len(group) == 1
            and audio_duration < args.tiny_unmatched_seconds
            and float(nearest["distance"]) >= args.tiny_unmatched_known_distance
        )
        too_far_from_prompt = float(nearest["distance"]) > args.max_output_prompt_distance
        target_list = rejected_rows if is_tiny_unmatched or too_far_from_prompt else output_rows
        index = len(target_list)
        if target_list is rejected_rows:
            clip_id = f"rejected_global_join_{index:06d}"
            audio_path = rejected_dir / f"{clip_id}.wav"
        else:
            clip_id = f"global_join_{index:06d}"
            audio_path = clips_dir / f"{clip_id}.wav"
        sf.write(audio_path, audio, sr)
        copied_prompt = copy_prompt(Path(nearest["path"]), prompts_dir, prompt_copies)
        row = output_row(group, clip_id, audio_path, copied_prompt, nearest, audio_duration)
        if target_list is rejected_rows:
            row["accepted"] = False
            row["train_allowed"] = False
            row["reject_reason"] = (
                "unmatched_tiny_fragment"
                if is_tiny_unmatched
                else "global_join_prompt_distance_too_high"
            )
        target_list.append(row)
        group_manifest.append(
            {
                "clip_id": clip_id,
                "kept": target_list is output_rows,
                "source_part_ids": [part["part_id"] for part in group],
                "source_clip_ids": [part["source_clip_id"] for part in group],
                "audio": str(audio_path),
                "duration": round(audio_duration, 3),
                "voice_prompt_distance": row.get("voice_prompt_distance"),
                "reject_reason": row.get("reject_reason"),
            }
        )

    write_jsonl(out_dir / "train_with_prompts.jsonl", output_rows)
    write_jsonl(out_dir / "paired_with_prompts.jsonl", output_rows)
    write_jsonl(out_dir / "voice_prompt_pairs.jsonl", output_rows)
    write_jsonl(out_dir / "rejected_unmatched_fragments.jsonl", rejected_rows)
    write_jsonl(out_dir / "cut_manifest.jsonl", cut_manifest)
    write_jsonl(out_dir / "join_decisions.jsonl", join_decisions)
    write_jsonl(out_dir / "global_join_manifest.jsonl", group_manifest)

    train_seconds = round(sum(float(row.get("duration") or 0.0) for row in output_rows), 2)
    rejected_seconds = round(sum(float(row.get("duration") or 0.0) for row in rejected_rows), 2)
    summary = {
        "source_prompt_dir": str(prompt_dir),
        "total_input_clips": len(rows),
        "word_boundary_split_input_parts": len(parts),
        "word_boundary_selected_cuts": sum(len(item["selected_cuts"]) for item in cut_manifest),
        "global_join_groups": len(groups),
        "paired_clips": len(output_rows),
        "paired_seconds": train_seconds,
        "train_clips": len(output_rows),
        "train_seconds": train_seconds,
        "unpaired_clips": 0,
        "discarded_unmatched_fragments": len(rejected_rows),
        "discarded_unmatched_seconds": rejected_seconds,
        "copied_prompt_clips": len(prompt_copies),
        "settings": {
            "word_boundary_context_seconds": args.word_boundary_context_seconds,
            "word_boundary_speaker_threshold": args.word_boundary_speaker_threshold,
            "global_join_strict_distance": args.global_join_strict_distance,
            "global_join_medium_distance": args.global_join_medium_distance,
            "global_join_strong_known_distance": args.global_join_strong_known_distance,
            "global_join_unknownish_distance": args.global_join_unknownish_distance,
            "tiny_unmatched_seconds": args.tiny_unmatched_seconds,
            "tiny_unmatched_known_distance": args.tiny_unmatched_known_distance,
            "max_output_prompt_distance": args.max_output_prompt_distance,
        },
        "outputs": {
            "train_with_prompts": str(out_dir / "train_with_prompts.jsonl"),
            "paired_with_prompts": str(out_dir / "paired_with_prompts.jsonl"),
            "voice_prompt_pairs": str(out_dir / "voice_prompt_pairs.jsonl"),
            "prompts": str(prompts_dir),
            "clips": str(clips_dir),
            "rejected_unmatched_fragments": str(rejected_dir),
            "rejected_unmatched_manifest": str(out_dir / "rejected_unmatched_fragments.jsonl"),
            "join_decisions": str(out_dir / "join_decisions.jsonl"),
            "cut_manifest": str(out_dir / "cut_manifest.jsonl"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply guarded ASR-boundary splits and global same-speaker joins after strict prompt pairing.")
    parser.add_argument("prompt_dir", type=Path, help="Strict voice prompt directory containing train_with_prompts.jsonl.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--out-dir-name", default="strict_voice_prompts_t034_p034_training_guarded_global_join")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--speechbrain-model-dir", default="models/speechbrain-spkrec-ecapa-voxceleb")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clip-pad-seconds", type=float, default=0.08)
    parser.add_argument("--min-embedding-seconds", type=float, default=0.75)
    parser.add_argument("--word-boundary-context-seconds", type=float, default=0.75)
    parser.add_argument("--word-boundary-speaker-threshold", type=float, default=0.55)
    parser.add_argument("--word-boundary-min-side-seconds", type=float, default=0.25)
    parser.add_argument("--word-boundary-min-gap-seconds", type=float, default=0.60)
    parser.add_argument(
        "--word-boundary-require-terminal-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--global-join-max-gap-seconds", type=float, default=0.75)
    parser.add_argument("--global-join-strict-distance", type=float, default=0.42)
    parser.add_argument("--global-join-medium-distance", type=float, default=0.57)
    parser.add_argument("--global-join-strong-known-distance", type=float, default=0.35)
    parser.add_argument("--global-join-unknownish-distance", type=float, default=0.50)
    parser.add_argument("--tiny-unmatched-seconds", type=float, default=0.80)
    parser.add_argument("--tiny-unmatched-known-distance", type=float, default=0.60)
    parser.add_argument("--max-output-prompt-distance", type=float, default=0.62)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.speechbrain_model_dir).is_absolute():
        args.speechbrain_model_dir = str((args.project_root / args.speechbrain_model_dir).resolve())
    summary = apply_guarded_global_joins(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
