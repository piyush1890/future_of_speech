"""
Extract articulatory features from LibriSpeech audio using SPARC.
Saves per-utterance .npz files with: ema (T,12), pitch (T,), loudness (T,), spk_emb (64,)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def find_flac_files(data_dir: Path) -> list[Path]:
    """Find all .flac files in LibriSpeech directory structure."""
    return sorted(data_dir.rglob("*.flac"))


def preprocess(data_dir: str, output_dir: str, device: str = "cpu", model_name: str = "en", max_files: int = 0):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SPARC
    from sparc import load_model
    print(f"Loading SPARC model '{model_name}' on {device}...")
    coder = load_model(model_name, device=device)
    print("SPARC loaded.")

    flac_files = find_flac_files(data_dir)
    print(f"Found {len(flac_files)} .flac files")

    if max_files > 0 and len(flac_files) > max_files:
        flac_files = flac_files[:max_files]
        print(f"Limiting to first {max_files} files")

    # Track per-speaker embeddings for averaging later
    speaker_embs = {}
    stats = {"total": len(flac_files), "success": 0, "failed": 0}

    for flac_path in tqdm(flac_files, desc="Encoding"):
        # LibriSpeech structure: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
        utt_id = flac_path.stem  # e.g., "1272-128104-0000"
        speaker_id = utt_id.split("-")[0]

        out_path = output_dir / f"{utt_id}.npz"
        if out_path.exists():
            stats["success"] += 1
            continue

        try:
            code = coder.encode(str(flac_path))

            # SPARC returns numpy arrays directly
            ema = np.asarray(code["ema"], dtype=np.float32)           # (T_ema, 12)
            pitch = np.asarray(code["pitch"], dtype=np.float32).squeeze(-1)    # (T,) from (T,1)
            loudness = np.asarray(code["loudness"], dtype=np.float32).squeeze(-1)  # (T,)
            spk_emb = np.asarray(code["spk_emb"], dtype=np.float32)  # (64,)

            # Align lengths (EMA and pitch/loudness may differ by 1 frame)
            min_len = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
            ema = ema[:min_len]
            pitch = pitch[:min_len]
            loudness = loudness[:min_len]

            np.savez_compressed(
                out_path,
                ema=ema,
                pitch=pitch,
                loudness=loudness,
                spk_emb=spk_emb,
            )

            # Collect speaker embeddings
            if speaker_id not in speaker_embs:
                speaker_embs[speaker_id] = []
            speaker_embs[speaker_id].append(spk_emb)

            stats["success"] += 1
        except Exception as e:
            print(f"FAILED {utt_id}: {e}")
            stats["failed"] += 1

    # Save average speaker embeddings
    avg_spk_embs = {}
    for spk_id, embs in speaker_embs.items():
        avg_spk_embs[spk_id] = np.mean(embs, axis=0).tolist()

    with open(output_dir / "speaker_embeddings.json", "w") as f:
        json.dump(avg_spk_embs, f)

    # Save feature normalization stats (for VQ training)
    all_features = []
    for npz_path in sorted(output_dir.glob("*.npz")):
        d = np.load(npz_path)
        p = d["pitch"]
        l = d["loudness"]
        if p.ndim == 1:
            p = p[:, None]
        if l.ndim == 1:
            l = l[:, None]
        combined = np.concatenate([d["ema"], p, l], axis=-1)
        all_features.append(combined)

    all_features = np.concatenate(all_features, axis=0)  # (total_frames, 14)
    feat_mean = all_features.mean(axis=0)
    feat_std = all_features.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0  # avoid div by zero

    np.savez(
        output_dir / "norm_stats.npz",
        mean=feat_mean.astype(np.float32),
        std=feat_std.astype(np.float32),
    )

    print(f"\nDone! {stats['success']} succeeded, {stats['failed']} failed")
    print(f"Features saved to {output_dir}")
    print(f"Normalization stats: mean shape={feat_mean.shape}, std shape={feat_std.shape}")
    print(f"Speaker embeddings: {len(avg_spk_embs)} speakers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to LibriSpeech dir (e.g., data/LibriSpeech/dev-clean)")
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory for .npz files")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or mps")
    parser.add_argument("--model", type=str, default="en", help="SPARC model name")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files (0 = all)")
    args = parser.parse_args()

    preprocess(args.data_dir, args.output_dir, args.device, args.model, args.max_files)
