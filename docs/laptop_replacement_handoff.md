# Laptop Replacement Handoff

Date: 2026-06-02

This repo's source, scripts, and lightweight documentation should be kept in git.
Large local artifacts are intentionally ignored because several files exceed
GitHub's normal 100 MB file limit.

## Git Remote

- Repo path: `/Users/piyush/projects/articulatory-tts`
- Branch: `main`
- Remote: `origin git@github.com:piyush1890/future_of_speech.git`

After cloning on a replacement machine, verify:

```bash
git status --short --branch
git log -1 --oneline --decorate
```

## Important Project Docs

- `SESSION_SUMMARY.md`: project history, architecture, experiment results, and continuation commands.
- `ARCHITECTURE.md`: detailed architecture and pipeline notes.
- `v8/README.md`: v8 design notes.
- `docs/*.excalidraw`: dataset and VibeVoice pipeline diagrams.
- `dialogue_dataset_tool/README.md`: one-command dialogue dataset builder notes.
- `data/*training_upload*/README.md`: small tracked notes for ignored dataset
  upload packages; the dataset payloads remain intentionally ignored.

## Local Artifacts Not In Git

These are important, but they are local/generated and should be restored from
source data, training runs, or model downloads rather than committed:

- `data/`: raw/processed datasets, extracted audio, generated clips, logs.
- `checkpoints*/` and `v*/checkpoints/`: model checkpoints.
- `outputs/` and `v*/outputs/`: generated audio samples.
- `backups/`: local backup tarballs.
- `.venv_dialogue_dataset/`: dialogue dataset tool virtualenv.
- `models/mlx-audio-separator/`: RoFormer separator cache.
- `models/mlx-whisper-large-v3-turbo/`: MLX Whisper cache.
- `models/speechbrain-spkrec-ecapa-voxceleb/`: SpeechBrain ECAPA cache.

The dialogue dataset tool bootstrap recreates `.venv_dialogue_dataset/` and
downloads the model caches into `models/` on first use.

## Dialogue Dataset Tool

Run from the repo root:

```bash
/usr/bin/python3 run_dialogue_dataset.py "YOUTUBE_URL_OR_LOCAL_VIDEO_PATH" --out-root data/output_dataset
```

The shareable package is `dialogue_dataset_tool.zip`, generated from:

```bash
zip -r dialogue_dataset_tool.zip dialogue_dataset_tool -x "*/__pycache__/*" "*.pyc"
```

## Machine Notes

This laptop currently prints a shell startup warning:

```text
/Users/piyush/.zprofile:2: no such file or directory: /opt/homebrew/bin/brew
```

On a replacement machine, either install Homebrew at `/opt/homebrew` or update
`~/.zprofile` so shell startup is clean.
