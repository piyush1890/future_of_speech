"""
Hierarchical RVQ inference pipeline: text → phonemes → hier transformer → RVQ decode → SPARC.

Two lessons baked in from diagnostics:
  1. Training data has no <sil> between words (MFA-style). g2p_en inserting one per space
     creates a listy, start-stop delivery. We keep only a leading <sil>.
  2. Predicted pitch has ~2.6× GT frame-to-frame jitter, causing vocal tremor. A 5-frame
     median filter on pitch removes it with no audible blur. Enabled by default; disable
     via --no-smooth-pitch.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


class ArticulatoryTTSRVQHier:
    def __init__(
        self,
        rvq_checkpoint: str,
        transformer_checkpoint: str,
        vocab_path: str,
        norm_stats_path: str,
        sparc_model: str = "en",
        device: str = "cpu",
        smooth_pitch: bool = True,
        smooth_window: int = 5,
    ):
        self.device = torch.device(device)
        self.smooth_pitch = smooth_pitch
        self.smooth_window = smooth_window

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"]
        self.feat_std = stats["std"]

        self.vocab = PhonemeVocab(vocab_path)
        self.g2p = G2p()

        rvq_ckpt = torch.load(rvq_checkpoint, map_location=self.device, weights_only=True)
        ra = rvq_ckpt["args"]
        self.rvq = ArticulatoryRVQTokenizer(
            codebook_size=ra["codebook_size"],
            num_quantizers=ra["num_quantizers"],
            latent_dim=ra["latent_dim"],
            hidden_dim=ra["hidden_dim"],
        ).to(self.device)
        self.rvq.load_state_dict(rvq_ckpt["model_state_dict"])
        self.rvq.eval()

        tf_ckpt = torch.load(transformer_checkpoint, map_location=self.device, weights_only=True)
        ta = tf_ckpt["args"]
        self.transformer = ArticulatoryTTSModelRVQHier(
            vocab_size=tf_ckpt["vocab_size"],
            codebook_size=ta["codebook_size"],
            num_quantizers=ta["num_quantizers"],
            d_model=ta["d_model"],
            nhead=ta["nhead"],
            num_encoder_layers=ta["num_layers"],
            num_decoder_layers=ta["num_layers"],
            d_ff=ta["d_ff"],
            dropout=ta.get("dropout", 0.1),
            speaker_emb_dim=64,
        ).to(self.device)
        self.transformer.load_state_dict(tf_ckpt["model_state_dict"])
        self.transformer.eval()

        from sparc import load_model as load_sparc
        self.sparc = load_sparc(sparc_model, device="cpu")

        print(f"Hierarchical RVQ TTS loaded (epoch {tf_ckpt.get('epoch','?')}, "
              f"val={tf_ckpt.get('val_loss', float('nan')):.4f}). "
              f"Smooth pitch: {self.smooth_pitch} (window={self.smooth_window}).")

    def text_to_phonemes(self, text):
        """MFA-style: drop spaces and punctuation, single leading <sil>."""
        raw = self.g2p(text)
        ph = [p for p in raw if p and p[0].isalpha() and p[0].isupper()]
        return ["<sil>"] + ph

    @torch.no_grad()
    def synthesize(self, text, speaker_emb=None, duration_scale=1.0):
        phonemes = self.text_to_phonemes(text)
        indices = self.vocab.encode(phonemes, add_bos_eos=True)
        phoneme_ids = torch.tensor([indices], dtype=torch.long, device=self.device)

        if speaker_emb is None:
            speaker_emb = np.zeros(64, dtype=np.float32)
        spk = torch.from_numpy(speaker_emb).unsqueeze(0).to(self.device)

        token_ids, _ = self.transformer.generate(phoneme_ids, spk, duration_scale)

        features_norm = self.rvq.decode_indices(token_ids)
        features_norm = features_norm.squeeze(0).cpu().numpy()
        features = features_norm * self.feat_std + self.feat_mean

        if self.smooth_pitch and self.smooth_window > 1:
            from scipy.ndimage import median_filter
            features[:, 12] = median_filter(
                features[:, 12], size=self.smooth_window, mode="nearest"
            )

        ema = features[:, :12]
        pitch = features[:, 12]
        loudness = features[:, 13]

        waveform = self.sparc.decode(ema, pitch, loudness, speaker_emb)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().squeeze().cpu().numpy()

        return waveform, self.sparc.sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--rvq-checkpoint", type=str, default="checkpoints_rvq/rvq_best.pt")
    parser.add_argument("--transformer-checkpoint", type=str,
                        default="checkpoints_rvq_hier/transformer_best.pt")
    parser.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    parser.add_argument("--norm-stats", type=str, default="data/features_merged/norm_stats.npz")
    parser.add_argument("--speaker-emb", type=str, default=None,
                        help="Path to .npy (64-dim). Defaults to first speaker in features_merged.")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=str, default="outputs/output_rvq_hier.wav")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-smooth-pitch", dest="smooth_pitch", action="store_false",
                        help="Disable median-5 pitch smoothing (on by default)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Median filter window on pitch (frames, 50 Hz)")
    parser.set_defaults(smooth_pitch=True)
    args = parser.parse_args()

    tts = ArticulatoryTTSRVQHier(
        rvq_checkpoint=args.rvq_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        vocab_path=args.vocab_path,
        norm_stats_path=args.norm_stats,
        device=args.device,
        smooth_pitch=args.smooth_pitch,
        smooth_window=args.smooth_window,
    )

    spk_emb = None
    if args.speaker_emb:
        spk_emb = np.load(args.speaker_emb)
    else:
        import json
        with open("data/features_merged/speaker_embeddings.json") as f:
            spk_embs = json.load(f)
        spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

    waveform, sr = tts.synthesize(args.text, spk_emb, args.duration_scale)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({len(waveform)/sr:.2f}s)")


if __name__ == "__main__":
    main()
