"""
After training the VQ tokenizer, quantize all features to discrete token indices.
Creates per-utterance .npy files with token IDs for transformer training.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer import ArticulatoryVQTokenizer


def tokenize_all(args):
    device = torch.device(args.device)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization stats
    stats = np.load(features_dir / "norm_stats.npz")
    feat_mean = stats["mean"]
    feat_std = stats["std"]

    # Load trained VQ model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_args = ckpt["args"]

    model = ArticulatoryVQTokenizer(
        codebook_size=model_args["codebook_size"],
        latent_dim=model_args["latent_dim"],
        hidden_dim=model_args["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded VQ model from {args.checkpoint} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Process all features
    npz_files = sorted(features_dir.glob("*.npz"))
    count = 0

    with torch.no_grad():
        for npz_path in tqdm(npz_files, desc="Tokenizing"):
            if npz_path.stem == "norm_stats":
                continue

            data = np.load(npz_path)
            ema = data["ema"]
            pitch = data["pitch"]
            loudness = data["loudness"]

            features = np.concatenate([ema, pitch[:, None], loudness[:, None]], axis=-1)
            features = (features - feat_mean) / feat_std

            x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
            _, indices, _ = model.encode(x)
            tokens = indices.squeeze(0).cpu().numpy()  # (T,)

            np.save(output_dir / f"{npz_path.stem}.npy", tokens)
            count += 1

    print(f"Tokenized {count} utterances → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vq_best.pt")
    parser.add_argument("--output-dir", type=str, default="data/vq_tokens")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    tokenize_all(args)
