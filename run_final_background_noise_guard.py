#!/usr/bin/env python3
"""Run the final background-music/noise guard on paired dialogue clips.

This is the shareable wrapper for the last cleanup step. It expects one or more
final paired prompt folders, each containing `train_with_prompts.jsonl`, `clips/`,
and `prompts/`. It writes a new sibling folder named
`strict_voice_prompts_t034_p034_training_guarded_global_join_music_guard` by
default, with cleaned `train_with_prompts.jsonl`, `clips/`, `prompts/`, and
review folders for rejected clips.

Examples:
  python run_final_background_noise_guard.py \
    data/movie/audible_turns_audiofirst_no_songs_mlx_ecapa_mps_turnfirst_t032/strict_voice_prompts_t034_p034_training_guarded_global_join \
    --force

  python run_final_background_noise_guard.py \
    --discover data/licensed_hindi_new_mixed_batch_full_clean_guarded \
    --force

  python run_final_background_noise_guard.py \
    --discover data/batch_root \
    --out-root data/final_music_guard_outputs \
    --force

The script is intentionally MPS-only. It sets PYTORCH_ENABLE_MPS_FALLBACK=0 and
fails if Apple Silicon GPU/MPS is unavailable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any


BOOTSTRAP_ENV = "FINAL_BACKGROUND_NOISE_GUARD_BOOTSTRAPPED"
ORIGINAL_CWD_ENV = "FINAL_BACKGROUND_NOISE_GUARD_ORIGINAL_CWD"
MIN_PYTHON = (3, 10)
MLX_WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
DEFAULT_OUT_DIR_NAME = "strict_voice_prompts_t034_p034_training_guarded_global_join_music_guard"

PYPI_PACKAGES = [
    "numpy",
    "soundfile",
    "librosa",
    "torch",
    "mlx-whisper",
    "mlx-audio-separator==0.1.4",
    "huggingface_hub",
    "transformers",
]
POST_INSTALL_PACKAGES = [
    "mlx==0.31.2",
    "mlx-metal==0.31.2",
]
REQUIRED_MODULES = [
    "numpy",
    "soundfile",
    "librosa",
    "torch",
    "mlx_whisper",
    "mlx_audio_separator",
    "huggingface_hub",
    "transformers",
]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def python_is_modern(python: Path) -> bool:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)",
            ],
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def candidate_modern_pythons() -> list[Path]:
    candidates = [
        Path("/opt/homebrew/bin/python3.12"),
        Path("/opt/homebrew/bin/python3.11"),
        Path("/opt/homebrew/bin/python3.10"),
        Path("/opt/homebrew/bin/python3"),
        Path("/usr/local/bin/python3.12"),
        Path("/usr/local/bin/python3.11"),
        Path("/usr/local/bin/python3.10"),
        Path("/usr/local/bin/python3"),
    ]
    return [path for path in candidates if path.exists() and python_is_modern(path)]


def ensure_modern_python(project_root: Path) -> None:
    if sys.version_info >= MIN_PYTHON:
        return
    for python in candidate_modern_pythons():
        env = dict(os.environ)
        env.setdefault(ORIGINAL_CWD_ENV, str(Path.cwd()))
        os.execve(
            str(python),
            [str(python), str(project_root / "run_final_background_noise_guard.py"), *sys.argv[1:]],
            env,
        )

    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError("Python 3.10+ is required. Install Homebrew/Python 3.11, then rerun.")
    run([brew, "install", "python@3.11"])
    for python in candidate_modern_pythons():
        env = dict(os.environ)
        env.setdefault(ORIGINAL_CWD_ENV, str(Path.cwd()))
        os.execve(
            str(python),
            [str(python), str(project_root / "run_final_background_noise_guard.py"), *sys.argv[1:]],
            env,
        )
    raise RuntimeError("Installed python@3.11, but no Python 3.10+ executable was found.")


def module_missing() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def create_or_update_venv(venv_dir: Path) -> Path:
    python = venv_dir / "bin" / "python"
    if python.exists() and not python_is_modern(python):
        print(f"Recreating Python environment with Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+: {venv_dir}", flush=True)
        shutil.rmtree(venv_dir)
    if not python.exists():
        print(f"Creating Python environment: {venv_dir}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python), "-m", "pip", "install", "--upgrade", *PYPI_PACKAGES])
    # mlx-audio-separator can pull an older MLX. Re-pin to the runtime that
    # works on this Apple Silicon setup.
    run([str(python), "-m", "pip", "install", "--upgrade", "--force-reinstall", *POST_INSTALL_PACKAGES])
    return python


def maybe_reexec_in_bootstrap_venv(args: argparse.Namespace, project_root: Path) -> None:
    if args.no_install or os.environ.get(BOOTSTRAP_ENV) == "1":
        return
    missing = module_missing()
    if not missing:
        return

    print("Missing Python packages: " + ", ".join(missing), flush=True)
    python = create_or_update_venv(args.venv.resolve())
    env = dict(os.environ)
    env[BOOTSTRAP_ENV] = "1"
    env.setdefault(ORIGINAL_CWD_ENV, str(Path.cwd()))
    os.execve(
        str(python),
        [str(python), str(project_root / "run_final_background_noise_guard.py"), *sys.argv[1:]],
        env,
    )


def ensure_brew_command(command: str, brew_package: str) -> None:
    if shutil.which(command):
        return
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError(f"{command} is required. Install Homebrew, then rerun.")
    run([brew, "install", brew_package])
    if not shutil.which(command):
        raise RuntimeError(f"{command} still was not found after installing {brew_package}.")


def ensure_acceleration() -> None:
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Refusing CPU fallback.")


def ensure_mlx_whisper_model(project_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    mlx_dir = project_root / "models" / "mlx-whisper-large-v3-turbo"
    if not (mlx_dir / "weights.safetensors").exists():
        print(f"Downloading MLX Whisper model to {mlx_dir}", flush=True)
        mlx_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=MLX_WHISPER_REPO, local_dir=mlx_dir)
    return mlx_dir.resolve()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:100] or "prompt_dir"


def is_final_guard_dir(path: Path) -> bool:
    return path.name == DEFAULT_OUT_DIR_NAME or path.name.endswith("_music_guard")


def coerce_prompt_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        if path.name != "train_with_prompts.jsonl":
            raise ValueError(f"Expected train_with_prompts.jsonl file, got: {path}")
        return path.parent.resolve()
    if (path / "train_with_prompts.jsonl").exists():
        return path.resolve()
    raise ValueError(f"Missing train_with_prompts.jsonl under: {path}")


def discover_prompt_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    prompt_dirs: list[Path] = []
    for manifest in root.rglob("train_with_prompts.jsonl"):
        candidate = manifest.parent
        if is_final_guard_dir(candidate):
            continue
        if candidate.name.endswith("_guarded_global_join") or candidate.name.startswith("strict_voice_prompts"):
            prompt_dirs.append(candidate)
    return sorted(set(prompt_dirs))


def infer_slug(prompt_dir: Path) -> str:
    for parent in prompt_dir.parents:
        if parent.name.startswith("audible_") and parent.parent.name:
            return slugify(parent.parent.name)
    if prompt_dir.parent.name:
        return slugify(prompt_dir.parent.name)
    return slugify(prompt_dir.name)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_inputs(args: argparse.Namespace, original_cwd: Path) -> list[Path]:
    raw_paths: list[Path] = []
    for value in args.inputs:
        path = Path(value)
        if not path.is_absolute():
            path = original_cwd / path
        raw_paths.append(path)

    if args.input_list:
        list_path = args.input_list
        if not list_path.is_absolute():
            list_path = original_cwd / list_path
        for line in list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                path = Path(line)
                raw_paths.append(path if path.is_absolute() else original_cwd / path)

    prompt_dirs: list[Path] = []
    for discover_root in args.discover:
        root = discover_root if discover_root.is_absolute() else original_cwd / discover_root
        prompt_dirs.extend(discover_prompt_dirs(root))

    for path in raw_paths:
        if (path / "train_with_prompts.jsonl").exists() or path.name == "train_with_prompts.jsonl":
            prompt_dirs.append(coerce_prompt_dir(path))
        else:
            prompt_dirs.extend(discover_prompt_dirs(path))

    unique: list[Path] = []
    seen: set[Path] = set()
    for prompt_dir in prompt_dirs:
        resolved = prompt_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def output_dir_for(prompt_dir: Path, args: argparse.Namespace, used: set[Path]) -> Path | None:
    if args.out_dir:
        if len(args.prompt_dirs) != 1:
            raise ValueError("--out-dir can only be used with exactly one prompt directory.")
        return args.out_dir.resolve()

    if args.out_root:
        args.out_root.mkdir(parents=True, exist_ok=True)
        base = args.out_root / infer_slug(prompt_dir)
        out_dir = base
        suffix = 2
        while out_dir in used or out_dir.exists():
            out_dir = Path(f"{base}_{suffix}")
            suffix += 1
        used.add(out_dir)
        return out_dir

    return None


def run_guard_for_prompt_dir(
    prompt_dir: Path,
    out_dir: Path | None,
    args: argparse.Namespace,
    project_root: Path,
    mlx_model: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(project_root / "scripts" / "apply_final_music_guard.py"),
        str(prompt_dir),
        "--project-root",
        str(project_root),
        "--mlx-model",
        str(mlx_model),
        "--language",
        args.language,
        "--roformer-model-filename",
        args.roformer_model_filename,
        "--mono-channel",
        args.mono_channel,
        "--classifier-device",
        "mps",
        "--post-recovery-song-device",
        "mps",
    ]
    if out_dir is not None:
        command.extend(["--out-dir", str(out_dir)])
    else:
        out_dir = prompt_dir.parent / DEFAULT_OUT_DIR_NAME
    if args.roformer_model_dir:
        command.extend(["--roformer-model-dir", str(args.roformer_model_dir.resolve())])
    if args.classifier_model_dir:
        command.extend(["--classifier-model-dir", str(args.classifier_model_dir.resolve())])
    if args.exclude_uncertain:
        command.append("--exclude-uncertain")
    if args.force:
        command.append("--force")
    if not args.validate_asr:
        command.append("--no-validate-asr")

    env = {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    print(f"\n===== FINAL BACKGROUND NOISE GUARD: {prompt_dir} =====", flush=True)
    run(command, cwd=project_root, env=env)

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary was not created: {summary_path}")
    summary = read_json(summary_path)
    return {
        "prompt_dir": str(prompt_dir),
        "out_dir": str(out_dir),
        "input_clips": summary.get("input_clips", 0),
        "input_seconds": summary.get("input_seconds", 0),
        "train_clips": summary.get("train_clips", 0),
        "train_seconds": summary.get("train_seconds", 0),
        "rejected_music_high_confidence_clips": summary.get("rejected_music_high_confidence_clips", 0),
        "rejected_music_high_confidence_seconds": summary.get("rejected_music_high_confidence_seconds", 0),
        "uncertain_music_clips": summary.get("uncertain_music_clips", 0),
        "uncertain_music_seconds": summary.get("uncertain_music_seconds", 0),
        "accepted_music_removed_clips": (summary.get("status_counts") or {}).get("accepted_music_removed", 0),
        "summary_json": str(summary_path),
        "train_with_prompts": str(out_dir / "train_with_prompts.jsonl"),
        "clips": str(out_dir / "clips"),
        "prompts": str(out_dir / "prompts"),
        "review_rejected": str(out_dir / "music_guard_review" / "rejected_music_high_confidence"),
        "review_uncertain": str(
            out_dir
            / "music_guard_review"
            / ("uncertain_music_rejected" if args.exclude_uncertain else "uncertain_music_kept")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final MPS-only background-music/noise clip guard on paired dialogue manifests."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Prompt dirs, train_with_prompts.jsonl files, or roots to discover recursively.",
    )
    parser.add_argument("--discover", type=Path, action="append", default=[], help="Recursively discover prompt dirs.")
    parser.add_argument("--input-list", type=Path, help="Text file with one prompt dir/root per line.")
    parser.add_argument("--out-dir", type=Path, help="Explicit output dir. Only valid for one input.")
    parser.add_argument("--out-root", type=Path, help="Write each result under this folder using inferred slugs.")
    parser.add_argument("--summary-json", type=Path, default=Path("final_background_noise_guard_summary.json"))
    parser.add_argument("--venv", type=Path, default=Path(".venv_dialogue_dataset"))
    parser.add_argument("--no-install", action="store_true", help="Do not create a venv or install missing packages.")
    parser.add_argument("--force", action="store_true", help="Delete and recreate existing output guard folders.")
    parser.add_argument(
        "--exclude-uncertain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude uncertain music/noise suspects from training. Default: true.",
    )
    parser.add_argument(
        "--validate-asr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate cleaned clips with MLX Whisper before keeping them. Default: true.",
    )
    parser.add_argument("--language", default="hi")
    parser.add_argument("--mono-channel", choices=["loudest", "left", "right", "quietest", "average"], default="left")
    parser.add_argument("--roformer-model-filename", default="vocals_mel_band_roformer.ckpt")
    parser.add_argument("--roformer-model-dir", type=Path)
    parser.add_argument("--classifier-model-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    original_cwd = Path(os.environ.get(ORIGINAL_CWD_ENV, Path.cwd())).resolve()
    os.chdir(project_root)

    ensure_modern_python(project_root)
    maybe_reexec_in_bootstrap_venv(args, project_root)
    ensure_brew_command("ffmpeg", "ffmpeg")
    ensure_acceleration()
    mlx_model = ensure_mlx_whisper_model(project_root)

    prompt_dirs = collect_inputs(args, original_cwd)
    if not prompt_dirs:
        raise SystemExit("No prompt dirs found. Pass a prompt dir or use --discover <batch-root>.")
    args.prompt_dirs = prompt_dirs

    used_outputs: set[Path] = set()
    results: list[dict[str, Any]] = []
    for prompt_dir in prompt_dirs:
        out_dir = output_dir_for(prompt_dir, args, used_outputs)
        results.append(run_guard_for_prompt_dir(prompt_dir, out_dir, args, project_root, mlx_model))

    summary = {
        "prompt_dirs": len(results),
        "input_clips": sum(int(item["input_clips"] or 0) for item in results),
        "input_seconds": round(sum(float(item["input_seconds"] or 0) for item in results), 2),
        "train_clips": sum(int(item["train_clips"] or 0) for item in results),
        "train_seconds": round(sum(float(item["train_seconds"] or 0) for item in results), 2),
        "rejected_music_high_confidence_clips": sum(
            int(item["rejected_music_high_confidence_clips"] or 0) for item in results
        ),
        "rejected_music_high_confidence_seconds": round(
            sum(float(item["rejected_music_high_confidence_seconds"] or 0) for item in results), 2
        ),
        "uncertain_music_clips": sum(int(item["uncertain_music_clips"] or 0) for item in results),
        "uncertain_music_seconds": round(sum(float(item["uncertain_music_seconds"] or 0) for item in results), 2),
        "accepted_music_removed_clips": sum(int(item["accepted_music_removed_clips"] or 0) for item in results),
        "exclude_uncertain": bool(args.exclude_uncertain),
        "results": results,
    }

    summary_path = args.summary_json
    if not summary_path.is_absolute():
        summary_path = original_cwd / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nDONE", flush=True)
    print(f"prompt_dirs: {summary['prompt_dirs']}", flush=True)
    print(f"train_clips: {summary['train_clips']}", flush=True)
    print(f"train_seconds: {summary['train_seconds']}", flush=True)
    print(f"music_rejects: {summary['rejected_music_high_confidence_clips']}", flush=True)
    print(f"uncertain_rejects: {summary['uncertain_music_clips']}", flush=True)
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
