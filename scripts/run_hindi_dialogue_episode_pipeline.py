#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ENV = {
    **os.environ,
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=ENV, check=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_url_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if args.url:
        items.append({"url": args.url, "id": args.video_id, "title": args.title})
    if args.url_list:
        with args.url_list.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("url") or item.get("source_path"):
                    items.append(item)
    if not items:
        raise SystemExit("Pass --url or --url-list.")

    start = max(0, args.start)
    stop = None if args.limit is None else start + args.limit
    return items[start:stop]


def episode_name(index: int, item: dict[str, Any]) -> str:
    if item.get("slug"):
        return str(item["slug"])
    video_id = item.get("id") or item.get("video_id")
    if not video_id:
        if item.get("url"):
            url = str(item["url"]).rstrip("/")
            video_id = url.split("v=")[-1].split("&")[0].split("/")[-1]
        elif item.get("source_path"):
            video_id = Path(str(item["source_path"])).stem
        else:
            video_id = f"item_{index:03d}"
    return f"episode_{index:03d}_{video_id}"


def download_audio(item: dict[str, Any], episode_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    source_dir = episode_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = source_dir / "source_metadata.json"
    write_json(metadata_path, item)

    if item.get("source_path"):
        source_path = Path(str(item["source_path"])).expanduser()
        if not source_path.is_absolute():
            source_path = project_root / source_path
        source_path = source_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Local source_path does not exist: {source_path}")
        return source_path

    existing = sorted(source_dir.glob("*.m4a")) + sorted(source_dir.glob("*.webm")) + sorted(source_dir.glob("*.mp4"))
    if not existing:
        run(
            [
                args.ytdlp,
                "-f",
                "bestaudio[ext=m4a]/bestaudio",
                "-o",
                str(source_dir / "%(id)s.%(ext)s"),
                str(item["url"]),
            ],
            cwd=project_root,
        )
        existing = sorted(source_dir.glob("*.m4a")) + sorted(source_dir.glob("*.webm")) + sorted(source_dir.glob("*.mp4"))
    if not existing:
        raise FileNotFoundError(f"No downloaded audio found in {source_dir}")
    return existing[0]


def extract_work_wav(
    source_audio: Path,
    episode_dir: Path,
    args: argparse.Namespace,
    project_root: Path,
) -> Path:
    out = episode_dir / "work_raw_audio.wav"
    if not out.exists():
        command = ["ffmpeg", "-y"]
        if args.source_start_seconds > 0:
            command.extend(["-ss", str(args.source_start_seconds)])
        command.extend(["-i", str(source_audio)])
        if args.source_duration_seconds is not None:
            command.extend(["-t", str(args.source_duration_seconds)])
        command.extend(["-vn", "-ac", "2", "-ar", "44100", str(out)])
        run(command, cwd=project_root)
    return out


def separate_vocals(work_wav: Path, episode_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    if args.first_pass_separator == "mlx-roformer":
        vocals = episode_dir / "roformer_mps" / "work_raw_audio_vocals.wav"
        if not vocals.exists():
            run(
                [
                    args.python,
                    str(project_root / "scripts" / "separate_vocals_mlx_roformer.py"),
                    str(work_wav),
                    "--out-wav",
                    str(vocals),
                    "--project-root",
                    str(project_root),
                    "--model-dir",
                    str(project_root / "models" / "mlx-audio-separator"),
                    "--model-filename",
                    args.roformer_model_filename,
                    "--chunk-duration",
                    str(args.roformer_chunk_duration),
                    "--mono-channel",
                    args.roformer_mono_channel,
                ],
                cwd=project_root,
            )
        return vocals

    vocals = episode_dir / "demucs_mps" / "htdemucs" / "work_raw_audio" / "vocals.wav"
    if not vocals.exists():
        run(
            [
                args.demucs,
                "--two-stems=vocals",
                "-n",
                "htdemucs",
                "-d",
                args.demucs_device,
                "-o",
                str(episode_dir / "demucs_mps"),
                str(work_wav),
            ],
            cwd=project_root,
        )
    return vocals


def phrase_dialogue(vocals: Path, episode_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    out_dir = episode_dir / "phrase_dialogue"
    natural = out_dir / "dialogue_phrase_compact_natural.wav"
    if not natural.exists():
        run(
            [
                args.python,
                str(project_root / "scripts" / "smooth_dialogue_mask.py"),
                str(vocals),
                "--out-dir",
                str(out_dir),
                "--threshold",
                "0.50",
                "--min-speech-ms",
                "160",
                "--min-silence-ms",
                "420",
                "--speech-pad-ms",
                "220",
                "--merge-gap-ms",
                "520",
                "--fade-ms",
                "90",
                "--compact-gap-ms",
                "140",
                "--export-rejected-segments",
            ],
            cwd=project_root,
        )
    eq = out_dir / "dialogue_phrase_compact_eq_16k.wav"
    if not eq.exists():
        filters = ",".join(
            [
                "highpass=f=110",
                "lowpass=f=7600",
                "equalizer=f=105:t=q:w=1.0:g=-5",
                "equalizer=f=180:t=q:w=1.0:g=-2",
            ]
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(natural),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-af",
                filters,
                str(eq),
            ],
            cwd=project_root,
        )
    boosted = out_dir / "dialogue_phrase_compact_eq_quiet_boost.wav"
    if not boosted.exists():
        run(
            [
                args.python,
                str(project_root / "scripts" / "upward_loudness_normalize.py"),
                str(eq),
                str(boosted),
                "--target-rms-db",
                "-25",
                "--noise-floor-db",
                "-48",
                "--max-gain-db",
                "9",
            ],
            cwd=project_root,
        )
    return boosted


def filter_sentences(input_wav: Path, episode_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    out_dir = episode_dir / "audible_turns_audiofirst_no_songs_mlx_ecapa_mps_turnfirst_t032"
    if not (out_dir / "accepted_sentences.jsonl").exists():
        command = [
            args.python,
            str(project_root / "scripts" / "filter_audible_sentences.py"),
            str(input_wav),
            "--out-dir",
            str(out_dir),
            "--language",
            "hi",
            "--asr-backend",
            args.asr_backend,
            "--mlx-model",
            args.mlx_model,
            "--chunking",
            "speaker",
            "--clip-unit",
            "turn_with_sentence_fallback",
            "--speaker-boundary-mode",
            "audio_first",
            "--speaker-feature-backend",
            "speechbrain",
            "--speechbrain-model-dir",
            args.speechbrain_model_dir,
            "--speaker-embedding-device",
            args.device,
            "--speaker-embedding-batch-size",
            "64",
            "--speaker-split-threshold",
            "0.32",
            "--speaker-split-peak-neighborhood-words",
            "4",
            "--no-speaker-split-snap-to-utterance",
            "--speaker-split-min-side-words",
            "3",
            "--speaker-split-min-side-duration",
            "0.9",
            "--audio-speaker-min-turn-duration",
            "0.9",
            "--audio-speaker-min-boundary-gap-seconds",
            "0.9",
            "--audio-speaker-peak-neighborhood-seconds",
            "0.9",
            "--max-duration",
            "30",
            "--min-avg-word-prob",
            "0.80",
            "--min-word-prob",
            "0.45",
            "--low-word-prob",
            "0.60",
            "--max-low-word-fraction",
            "0.15",
            "--min-low-words-for-many-uncertain",
            "2",
            "--very-low-word-prob",
            "0.40",
            "--max-very-low-word-fraction",
            "0.02",
            "--max-compression-ratio",
            "2.35",
            "--min-avg-logprob",
            "-0.45",
            "--min-unique-word-ratio",
            "0.50",
            "--max-repeated-ngram-fraction",
            "0.25",
            "--song-acoustic-backend",
            "torch",
            "--song-device",
            args.device,
        ]
        run(command, cwd=project_root)
    return out_dir


def sentence_output_dir(episode_dir: Path) -> Path:
    return episode_dir / "audible_turns_audiofirst_no_songs_mlx_ecapa_mps_turnfirst_t032"


def phrase_output_wav(episode_dir: Path) -> Path:
    return episode_dir / "phrase_dialogue" / "dialogue_phrase_compact_eq_quiet_boost.wav"


def cleanup_large_intermediates(episode_dir: Path, *, include_phrase_audio: bool) -> None:
    paths = [
        episode_dir / "work_raw_audio.wav",
        episode_dir / "demucs_mps" / "htdemucs" / "work_raw_audio" / "no_vocals.wav",
        episode_dir / "demucs_mps" / "htdemucs" / "work_raw_audio" / "vocals.wav",
    ]
    paths.extend(episode_dir.glob("**/*.part"))
    if include_phrase_audio:
        phrase_dir = episode_dir / "phrase_dialogue"
        paths.extend(
            [
                phrase_dir / "dialogue_phrase_masked_timeline.wav",
                phrase_dir / "dialogue_phrase_compact_natural.wav",
                phrase_dir / "dialogue_phrase_compact_eq_16k.wav",
                phrase_dir / "dialogue_phrase_compact_eq_quiet_boost.wav",
                phrase_dir / "vad_detection_mild.wav",
            ]
        )
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                print(f"removed intermediate {path}", flush=True)
        except OSError as exc:
            print(f"could not remove intermediate {path}: {exc}", flush=True)


def assign_prompts(sentence_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    out_dir = sentence_dir / "strict_voice_prompts_t034_p034_training"
    if not (out_dir / "summary.json").exists():
        run(
            [
                args.python,
                str(project_root / "scripts" / "assign_strict_voice_prompts.py"),
                str(sentence_dir / "accepted_sentences.jsonl"),
                "--project-root",
                str(project_root),
                "--out-dir",
                str(out_dir),
                "--device",
                args.device,
                "--model-dir",
                args.speechbrain_model_dir,
                "--batch-size",
                "16",
                "--same-speaker-threshold",
                "0.34",
                "--prompt-pair-threshold",
                "0.34",
                "--no-holdout-prompts",
            ],
            cwd=project_root,
        )
    return out_dir


def apply_guarded_global_joins(prompt_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    if not args.guarded_global_join:
        return prompt_dir
    out_dir = prompt_dir.parent / "strict_voice_prompts_t034_p034_training_guarded_global_join"
    if not (out_dir / "summary.json").exists():
        run(
            [
                args.python,
                str(project_root / "scripts" / "apply_guarded_global_joins.py"),
                str(prompt_dir),
                "--project-root",
                str(project_root),
                "--out-dir",
                str(out_dir),
                "--device",
                args.device,
                "--speechbrain-model-dir",
                args.speechbrain_model_dir,
            ],
            cwd=project_root,
        )
    return out_dir


def apply_final_music_guard(prompt_dir: Path, args: argparse.Namespace, project_root: Path) -> Path:
    if not args.music_guard:
        return prompt_dir
    out_dir = prompt_dir.parent / "strict_voice_prompts_t034_p034_training_guarded_global_join_music_guard"
    if not (out_dir / "summary.json").exists():
        command = [
            args.python,
            str(project_root / "scripts" / "apply_final_music_guard.py"),
            str(prompt_dir),
            "--project-root",
            str(project_root),
            "--out-dir",
            str(out_dir),
            "--mlx-model",
            args.mlx_model,
            "--language",
            "hi",
            "--roformer-model-filename",
            args.roformer_model_filename,
            "--mono-channel",
            args.roformer_mono_channel,
        ]
        if args.music_guard_exclude_uncertain:
            command.append("--exclude-uncertain")
        run(command, cwd=project_root)
    return out_dir


def recover_background_music_dialogues(
    sentence_dir: Path,
    source_wav: Path | None,
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Any]:
    if not args.recover_background_music:
        return {"recovered_clips": 0, "skipped": "disabled"}
    if source_wav is None or not source_wav.exists():
        return {"recovered_clips": 0, "skipped": "missing_source_wav"}
    summary_path = sentence_dir / "background_music_recovery" / "summary.json"
    command = [
        args.python,
        str(project_root / "scripts" / "recover_background_music_dialogues.py"),
        str(sentence_dir),
        "--source-wav",
        str(source_wav),
        "--project-root",
        str(project_root),
        "--demucs",
        args.demucs,
        "--demucs-device",
        args.demucs_device,
        "--mlx-model",
        args.mlx_model,
        "--language",
        "hi",
    ]
    run(command, cwd=project_root)
    if summary_path.exists():
        return read_json(summary_path)
    return {"recovered_clips": 0, "skipped": "missing_summary"}


def clean_accepted_dialogues_with_roformer(
    sentence_dir: Path,
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Any]:
    if not args.clean_accepted_with_roformer:
        return {"updated_clips": 0, "skipped": "disabled"}
    summary_path = sentence_dir / "accepted_roformer_cleanup" / "summary.json"
    command = [
        args.python,
        str(project_root / "scripts" / "clean_accepted_dialogues_roformer.py"),
        str(sentence_dir),
        "--project-root",
        str(project_root),
        "--mlx-model",
        args.mlx_model,
        "--language",
        "hi",
        "--roformer-model-filename",
        args.roformer_model_filename,
        "--mono-channel",
        args.roformer_mono_channel,
    ]
    run(command, cwd=project_root)
    if summary_path.exists():
        return read_json(summary_path)
    return {"updated_clips": 0, "skipped": "missing_summary"}


def prune_to_paired_audio(sentence_dir: Path, prompt_dir: Path, args: argparse.Namespace, project_root: Path) -> None:
    run(
        [
            args.python,
            str(project_root / "scripts" / "prune_to_paired_audio.py"),
            str(sentence_dir),
            "--project-root",
            str(project_root),
            "--prompt-dir-name",
            prompt_dir.name,
            "--remove-other-prompt-dirs",
        ],
        cwd=project_root,
    )


def export_rejected_review_clips(
    sentence_dir: Path,
    prompt_dir: Path,
    source_wav: Path | None,
    args: argparse.Namespace,
    project_root: Path,
) -> None:
    command = [
        args.python,
        str(project_root / "scripts" / "export_rejected_review_clips.py"),
        "--sentence-dir",
        str(sentence_dir),
        "--prompt-dir",
        str(prompt_dir),
        "--project-root",
        str(project_root),
    ]
    if source_wav is not None and source_wav.exists():
        command.extend(["--source-wav", str(source_wav)])
    run(command, cwd=project_root)


def summarize_episode(episode_dir: Path, item: dict[str, Any], sentence_dir: Path, prompt_dir: Path) -> dict[str, Any]:
    sentence_summary = read_json(sentence_dir / "summary.json")
    prompt_summary = read_json(prompt_dir / "summary.json")
    phrase_summary_path = episode_dir / "phrase_dialogue" / "dialogue_phrase_summary.json"
    loudness_path = episode_dir / "phrase_dialogue" / "dialogue_phrase_compact_eq_quiet_boost.summary.json"
    summary = {
        "title": item.get("title"),
        "url": item.get("url"),
        "video_id": item.get("id") or item.get("video_id"),
        "episode_dir": str(episode_dir),
        "source_duration_seconds": item.get("duration"),
        "source_start_seconds": item.get("source_start_seconds"),
        "source_duration_seconds_requested": item.get("source_duration_seconds_requested"),
        "first_pass_separator": item.get("first_pass_separator"),
        "phrase_dialogue": read_json(phrase_summary_path) if phrase_summary_path.exists() else None,
        "loudness": read_json(loudness_path) if loudness_path.exists() else None,
        "accepted_clips": sentence_summary.get("accepted_sentences"),
        "accepted_seconds": sentence_summary.get("accepted_seconds"),
        "rejected_seconds": sentence_summary.get("rejected_seconds"),
        "accepted_roformer_cleanup": sentence_summary.get("accepted_roformer_cleanup"),
        "paired_clips": prompt_summary.get("paired_clips"),
        "paired_seconds": prompt_summary.get("paired_seconds"),
        "unpaired_clips": prompt_summary.get("unpaired_clips"),
        "train_clips": prompt_summary.get("train_clips"),
        "train_seconds": prompt_summary.get("train_seconds"),
        "music_guard": read_json(prompt_dir / "summary.json") if "music_guard" in prompt_dir.name else None,
        "outputs": {
            "accepted_manifest": str(sentence_dir / "accepted_sentences.jsonl"),
            "rejected_manifest": str(sentence_dir / "rejected_sentences.jsonl"),
            "accepted_compact": str(sentence_dir / "accepted_sentences_compact.wav"),
            "voice_prompt_pairs": str(prompt_dir / "voice_prompt_pairs.jsonl"),
            "paired_with_prompts": str(prompt_dir / "paired_with_prompts.jsonl"),
            "train_with_prompts": str(prompt_dir / "train_with_prompts.jsonl"),
            "rejected_review": str(sentence_dir / "rejected_review"),
            "prompts": str(prompt_dir / "prompts"),
            "music_guard_rejected": str(prompt_dir / "music_guard_review" / "rejected_music_high_confidence")
            if "music_guard" in prompt_dir.name
            else None,
        },
    }
    write_json(episode_dir / "pipeline_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hindi dialogue dataset pipeline on YouTube/local episodes.")
    parser.add_argument("--url")
    parser.add_argument("--video-id")
    parser.add_argument("--title")
    parser.add_argument("--url-list", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ytdlp", default=str(Path(sys.executable).with_name("yt-dlp")))
    parser.add_argument("--demucs", default=str(Path(sys.executable).with_name("demucs")))
    parser.add_argument("--demucs-device", default="mps")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--asr-backend", choices=["mlx", "openai"], default="mlx")
    parser.add_argument("--mlx-model", default="mlx-community/whisper-large-v3-turbo")
    parser.add_argument("--speechbrain-model-dir", default="models/speechbrain-spkrec-ecapa-voxceleb")
    parser.add_argument(
        "--first-pass-separator",
        choices=["mlx-roformer", "demucs"],
        default="demucs",
        help="Separator used for the initial full-source vocal/dialogue stem.",
    )
    parser.add_argument("--roformer-model-filename", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--roformer-chunk-duration", type=float, default=120.0)
    parser.add_argument(
        "--roformer-mono-channel",
        choices=["loudest", "left", "right", "quietest", "average"],
        default="left",
    )
    parser.add_argument(
        "--source-start-seconds",
        type=float,
        default=0.0,
        help="Start offset for source audio extraction.",
    )
    parser.add_argument(
        "--source-duration-seconds",
        type=float,
        default=None,
        help="Only process this many seconds from the source.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--recover-background-music",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a second Demucs vocals pass on clips rejected only for background music/noise and add recovered dialogue back to accepted clips.",
    )
    parser.add_argument(
        "--clean-accepted-with-roformer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MLX RoFormer over accepted dialogue clips to remove missed background music before voice-prompt pairing.",
    )
    parser.add_argument(
        "--prune-paired-only",
        action="store_true",
        help="After prompt assignment, delete unpaired/rejected review artifacts and keep only strict prompt-paired training rows.",
    )
    parser.add_argument(
        "--guarded-global-join",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After strict prompt pairing, split on ASR word-boundary speaker changes, rejoin guarded same-speaker neighbors, and discard tiny unmatched fragments.",
    )
    parser.add_argument(
        "--music-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After final prompt pairing, run a high-precision music-leakage guard and quarantine only high-confidence rejected clips.",
    )
    parser.add_argument(
        "--music-guard-exclude-uncertain",
        action="store_true",
        help="Also remove uncertain music-guard clips from training. Default keeps uncertain clips to avoid false positives.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if args.device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this Mac; refusing to fall back to CPU.")
    if args.mlx_model == "mlx-community/whisper-large-v3-turbo":
        local_mlx_model = project_root / "models" / "mlx-whisper-large-v3-turbo"
        args.mlx_model = str(local_mlx_model)
    if not Path(args.speechbrain_model_dir).is_absolute():
        args.speechbrain_model_dir = str((project_root / args.speechbrain_model_dir).resolve())

    items = read_url_items(args)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for offset, item in enumerate(items, start=args.start + 1):
        item = dict(item)
        item["source_start_seconds"] = args.source_start_seconds
        item["source_duration_seconds_requested"] = args.source_duration_seconds
        item["first_pass_separator"] = args.first_pass_separator
        episode_dir = out_root / episode_name(offset, item)
        episode_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Processing {episode_dir.name}: {item.get('title') or item.get('url')} ===", flush=True)
        try:
            sentence_dir = sentence_output_dir(episode_dir)
            review_source_wav = phrase_output_wav(episode_dir)
            if not (sentence_dir / "accepted_sentences.jsonl").exists():
                dialogue_wav = phrase_output_wav(episode_dir)
                if not dialogue_wav.exists():
                    vocals = episode_dir / "demucs_mps" / "htdemucs" / "work_raw_audio" / "vocals.wav"
                    if not vocals.exists():
                        source_audio = download_audio(item, episode_dir, args, project_root)
                        work_wav = extract_work_wav(source_audio, episode_dir, args, project_root)
                        vocals = separate_vocals(work_wav, episode_dir, args, project_root)
                    dialogue_wav = phrase_dialogue(vocals, episode_dir, args, project_root)
                    cleanup_large_intermediates(episode_dir, include_phrase_audio=False)
                review_source_wav = dialogue_wav
                sentence_dir = filter_sentences(dialogue_wav, episode_dir, args, project_root)
            recovery_summary = recover_background_music_dialogues(
                sentence_dir,
                review_source_wav,
                args,
                project_root,
            )
            cleanup_summary = clean_accepted_dialogues_with_roformer(
                sentence_dir,
                args,
                project_root,
            )
            prompt_dir_path = sentence_dir / "strict_voice_prompts_t034_p034_training"
            if (
                recovery_summary.get("recovered_clips")
                or cleanup_summary.get("updated_clips")
                or cleanup_summary.get("rejected_after_cleanup")
            ) and prompt_dir_path.exists():
                shutil.rmtree(prompt_dir_path)
            guarded_prompt_dir_path = sentence_dir / "strict_voice_prompts_t034_p034_training_guarded_global_join"
            if (
                recovery_summary.get("recovered_clips")
                or cleanup_summary.get("updated_clips")
                or cleanup_summary.get("rejected_after_cleanup")
            ) and guarded_prompt_dir_path.exists():
                shutil.rmtree(guarded_prompt_dir_path)
            prompt_dir = assign_prompts(sentence_dir, args, project_root)
            export_rejected_review_clips(sentence_dir, prompt_dir, review_source_wav, args, project_root)
            prompt_dir = apply_guarded_global_joins(prompt_dir, args, project_root)
            prompt_dir = apply_final_music_guard(prompt_dir, args, project_root)
            if args.prune_paired_only:
                prune_to_paired_audio(sentence_dir, prompt_dir, args, project_root)
            summaries.append(summarize_episode(episode_dir, item, sentence_dir, prompt_dir))
            cleanup_large_intermediates(episode_dir, include_phrase_audio=True)
        except Exception as exc:
            cleanup_large_intermediates(episode_dir, include_phrase_audio=False)
            failure = {
                "title": item.get("title"),
                "url": item.get("url"),
                "video_id": item.get("id") or item.get("video_id"),
                "episode_dir": str(episode_dir),
                "error": repr(exc),
            }
            failures.append(failure)
            write_json(episode_dir / "pipeline_failure.json", failure)
            print(json.dumps({"failed": failure}, ensure_ascii=False, indent=2), flush=True)
            if not args.continue_on_error:
                raise

    combined = {
        "episodes": len(summaries),
        "failures": len(failures),
        "accepted_seconds": round(sum(float(item.get("accepted_seconds") or 0.0) for item in summaries), 2),
        "paired_seconds": round(sum(float(item.get("paired_seconds") or 0.0) for item in summaries), 2),
        "train_seconds": round(sum(float(item.get("train_seconds") or 0.0) for item in summaries), 2),
        "summaries": summaries,
        "failure_details": failures,
    }
    write_json(out_root / "batch_summary.json", combined)
    print(json.dumps(combined, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
