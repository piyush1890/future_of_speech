#!/usr/bin/env python3
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
from urllib.parse import parse_qs, urlparse


BOOTSTRAP_ENV = "DIALOGUE_DATASET_BOOTSTRAPPED"
ORIGINAL_CWD_ENV = "DIALOGUE_DATASET_ORIGINAL_CWD"
PYPI_PACKAGES = [
    "numpy",
    "soundfile",
    "librosa",
    "torch",
    "torchaudio",
    "torchcodec",
    "silero-vad",
    "speechbrain",
    "mlx-whisper",
    "mlx-audio-separator==0.1.4",
    "yt-dlp",
    "demucs",
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
    "torchcodec",
    "silero_vad",
    "speechbrain",
    "mlx_whisper",
    "mlx_audio_separator",
    "yt_dlp",
    "demucs",
    "huggingface_hub",
    "transformers",
]
MLX_WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
SPEECHBRAIN_REPO = "speechbrain/spkrec-ecapa-voxceleb"
MIN_PYTHON = (3, 10)


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
            [str(python), str(project_root / "run_dialogue_dataset.py"), *sys.argv[1:]],
            env,
        )

    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError(
            "Python 3.10+ is required and Homebrew is not installed. Install Homebrew, then rerun this same command."
        )
    run([brew, "install", "python@3.11"])
    for python in candidate_modern_pythons():
        env = dict(os.environ)
        env.setdefault(ORIGINAL_CWD_ENV, str(Path.cwd()))
        os.execve(
            str(python),
            [str(python), str(project_root / "run_dialogue_dataset.py"), *sys.argv[1:]],
            env,
        )
    raise RuntimeError("Installed python@3.11, but no Python 3.10+ executable was found.")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def module_missing() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def bin_dir_for_python(python: Path) -> Path:
    return python.parent


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
    # mlx-audio-separator currently pins an older mlx in its dependency graph,
    # but the native audio wheel on this Mac needs the current MLX runtime.
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
    os.execve(str(python), [str(python), str(project_root / "run_dialogue_dataset.py"), *sys.argv[1:]], env)


def ensure_brew_command(command: str, brew_package: str) -> None:
    if shutil.which(command):
        return
    brew = shutil.which("brew")
    if not brew:
        raise RuntimeError(
            f"{command} is required and Homebrew is not installed. Install Homebrew, then rerun this same command."
        )
    run([brew, "install", brew_package])
    if not shutil.which(command):
        raise RuntimeError(f"{command} still was not found after installing {brew_package}.")


def ensure_acceleration() -> None:
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Refusing to fall back; run this on an Apple Silicon Mac with MPS.")


def ensure_models(project_root: Path) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    mlx_dir = project_root / "models" / "mlx-whisper-large-v3-turbo"
    if not (mlx_dir / "weights.safetensors").exists():
        print(f"Downloading MLX Whisper model to {mlx_dir}", flush=True)
        mlx_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=MLX_WHISPER_REPO, local_dir=mlx_dir)

    ecapa_dir = project_root / "models" / "speechbrain-spkrec-ecapa-voxceleb"
    if not (ecapa_dir / "hyperparams.yaml").exists():
        print(f"Downloading SpeechBrain ECAPA model to {ecapa_dir}", flush=True)
        ecapa_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=SPEECHBRAIN_REPO, local_dir=ecapa_dir)

    return mlx_dir.resolve(), ecapa_dir.resolve()


def resolve_executable(name: str) -> str:
    beside_python = bin_dir_for_python(Path(sys.executable)) / name
    if beside_python.exists():
        return str(beside_python)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"Required executable not found after installation: {name}")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:80] or "dialogue_source"


def youtube_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/") or None
    query_id = parse_qs(parsed.query).get("v", [None])[0]
    if query_id:
        return query_id
    parts = [part for part in parsed.path.split("/") if part]
    return parts[-1] if parts else None


def write_manifest(args: argparse.Namespace, project_root: Path, input_base: Path) -> Path:
    out_root = args.out_root.resolve()
    source_dir = out_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    input_value = args.input
    if is_url(input_value):
        source_id = youtube_id(input_value) or "youtube"
        slug = args.slug or slugify(f"youtube_{source_id}")
        item = {
            "slug": slug,
            "title": args.title or input_value,
            "url": input_value,
            "id": source_id,
        }
    else:
        source_path = Path(input_value).expanduser()
        if not source_path.is_absolute():
            source_path = (input_base / source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {source_path}")
        slug = args.slug or slugify(source_path.stem)
        item = {
            "slug": slug,
            "title": args.title or source_path.stem,
            "source_path": str(source_path),
        }

    manifest = source_dir / f"{item['slug']}.jsonl"
    manifest.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def print_outputs(out_root: Path) -> None:
    batch_summary = out_root / "batch_summary.json"
    if not batch_summary.exists():
        print(f"Batch finished, but summary was not found at {batch_summary}", flush=True)
        return
    data = json.loads(batch_summary.read_text(encoding="utf-8"))
    print("\nDONE", flush=True)
    print(f"episodes: {data.get('episodes')}", flush=True)
    print(f"failures: {data.get('failures')}", flush=True)
    print(f"paired_seconds: {data.get('paired_seconds')}", flush=True)
    for summary in data.get("summaries", []):
        outputs = summary.get("outputs", {})
        print(f"clips: {Path(outputs.get('accepted_manifest', '')).parent / 'clips'}", flush=True)
        print(f"jsonl: {outputs.get('train_with_prompts')}", flush=True)
        print(f"rejected_review: {outputs.get('rejected_review')}", flush=True)
        print(f"prompts: {outputs.get('prompts')}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command Hindi dialogue dataset builder: YouTube/local video -> paired JSONL + audio clips."
    )
    parser.add_argument("input", help="YouTube URL or local video/audio file path.")
    parser.add_argument("--out-root", type=Path, default=Path("data/dialogue_dataset_runs"))
    parser.add_argument("--slug", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--venv", type=Path, default=Path(".venv_dialogue_dataset"))
    parser.add_argument("--no-install", action="store_true", help="Do not create a venv or install missing packages.")
    parser.add_argument(
        "--music-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the optional final high-precision music guard and write rejected clips for review.",
    )
    parser.add_argument(
        "--music-guard-exclude-uncertain",
        action="store_true",
        help="When --music-guard is enabled, remove suspicious clips whose cleanup validation is uncertain.",
    )
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
    mlx_model, speechbrain_model = ensure_models(project_root)

    manifest = write_manifest(args, project_root, original_cwd)
    out_root = args.out_root.resolve()
    ytdlp = resolve_executable("yt-dlp")
    demucs = resolve_executable("demucs")

    command = [
        sys.executable,
        str(project_root / "scripts" / "run_hindi_dialogue_episode_pipeline.py"),
        "--url-list",
        str(manifest),
        "--out-root",
        str(out_root),
        "--project-root",
        str(project_root),
        "--python",
        sys.executable,
        "--ytdlp",
        ytdlp,
        "--demucs",
        demucs,
        "--device",
        "mps",
        "--demucs-device",
        "mps",
        "--asr-backend",
        "mlx",
        "--mlx-model",
        str(mlx_model),
        "--speechbrain-model-dir",
        str(speechbrain_model),
        "--continue-on-error",
    ]
    if args.music_guard:
        command.append("--music-guard")
    if args.music_guard_exclude_uncertain:
        command.append("--music-guard-exclude-uncertain")
    run(command, cwd=project_root)
    print_outputs(out_root)


if __name__ == "__main__":
    main()
