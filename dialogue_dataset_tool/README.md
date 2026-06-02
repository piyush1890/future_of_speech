# Dialogue Dataset Tool

One-command Hindi dialogue dataset builder.

## Run

```bash
/usr/bin/python3 run_dialogue_dataset.py "YOUTUBE_URL_OR_LOCAL_VIDEO_PATH" --out-root data/output_dataset
```

The script creates `.venv_dialogue_dataset`, installs Python 3.10+ with Homebrew
if the system Python is too old, downloads required models into `models/`, and
writes clips, JSONL files, strict voice-prompt pairs, and rejected-review folders
into the selected output folder.

Speaker turns are detected on audio first with ECAPA on MPS, then ASR runs inside
those speaker turns. The final output applies guarded ASR word-boundary speaker
splits, strict same-speaker rejoining, and tiny unmatched fragment rejection.

The bootstrap installs Python packages such as PyTorch, SpeechBrain, MLX Whisper,
MLX audio separator, Demucs, yt-dlp, librosa, soundfile, and Hugging Face Hub when
they are missing. It also downloads MLX Whisper and SpeechBrain ECAPA weights on
first run; RoFormer separator weights are initialized by `mlx-audio-separator`
into `models/mlx-audio-separator` when first used. GPU/MPS is required; the tool
refuses CPU fallback.

Zip only this folder when sharing the tool:

```bash
zip -r dialogue_dataset_tool.zip dialogue_dataset_tool -x "*/__pycache__/*" "*.pyc"
```
