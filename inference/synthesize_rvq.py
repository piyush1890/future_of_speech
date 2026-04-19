"""
Inference pipeline for Residual VQ model: text → phonemes → multi-codebook tokens → audio.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.transformer_rvq import ArticulatoryTTSModelRVQ
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


class ArticulatoryTTSRVQ:
    def __init__(
        self,
        rvq_checkpoint: str,
        transformer_checkpoint: str,
        vocab_path: str,
        norm_stats_path: str,
        sparc_model: str = "en",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"]
        self.feat_std = stats["std"]

        self.vocab = PhonemeVocab(vocab_path)
        self.g2p = G2p()

        # Load RVQ
        rvq_ckpt = torch.load(rvq_checkpoint, map_location=self.device, weights_only=True)
        rvq_args = rvq_ckpt["args"]
        self.rvq = ArticulatoryRVQTokenizer(
            codebook_size=rvq_args["codebook_size"],
            num_quantizers=rvq_args["num_quantizers"],
            latent_dim=rvq_args["latent_dim"],
            hidden_dim=rvq_args["hidden_dim"],
        ).to(self.device)
        self.rvq.load_state_dict(rvq_ckpt["model_state_dict"])
        self.rvq.eval()

        # Load transformer
        tf_ckpt = torch.load(transformer_checkpoint, map_location=self.device, weights_only=True)
        tf_args = tf_ckpt["args"]
        self.transformer = ArticulatoryTTSModelRVQ(
            vocab_size=tf_ckpt["vocab_size"],
            codebook_size=tf_args["codebook_size"],
            num_quantizers=tf_args["num_quantizers"],
            d_model=tf_args["d_model"],
            nhead=tf_args["nhead"],
            num_encoder_layers=tf_args["num_layers"],
            num_decoder_layers=tf_args["num_layers"],
            d_ff=tf_args["d_ff"],
            dropout=tf_args.get("dropout", 0.1),
            speaker_emb_dim=64,
        ).to(self.device)
        self.transformer.load_state_dict(tf_ckpt["model_state_dict"])
        self.transformer.eval()

        from sparc import load_model as load_sparc
        self.sparc = load_sparc(sparc_model, device="cpu")

        print("Articulatory TTS (RVQ) loaded.")

    def text_to_phonemes(self, text):
        raw = self.g2p(text)
        return ["<sil>" if p == " " else p for p in raw if p.strip() or p == " "]

    @torch.no_grad()
    def synthesize(self, text, speaker_emb=None, duration_scale=1.0):
        phonemes = self.text_to_phonemes(text)
        indices = self.vocab.encode(phonemes, add_bos_eos=True)
        phoneme_ids = torch.tensor([indices], dtype=torch.long, device=self.device)

        if speaker_emb is None:
            speaker_emb = np.zeros(64, dtype=np.float32)
        spk = torch.from_numpy(speaker_emb).unsqueeze(0).to(self.device)

        token_ids, _ = self.transformer.generate(phoneme_ids, spk, duration_scale)  # (1, T, K)

        features_norm = self.rvq.decode_indices(token_ids)
        features_norm = features_norm.squeeze(0).cpu().numpy()
        features = features_norm * self.feat_std + self.feat_mean

        ema = features[:, :12]
        pitch = features[:, 12]
        loudness = features[:, 13]

        waveform = self.sparc.decode(ema, pitch, loudness, speaker_emb)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()

        return waveform, self.sparc.sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--rvq-checkpoint", type=str, default="checkpoints_rvq/rvq_best.pt")
    parser.add_argument("--transformer-checkpoint", type=str, default="checkpoints_rvq/transformer_best.pt")
    parser.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    parser.add_argument("--norm-stats", type=str, default="data/features_all/norm_stats.npz")
    parser.add_argument("--speaker-emb", type=str, default=None)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=str, default="outputs/output_rvq.wav")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tts = ArticulatoryTTSRVQ(
        rvq_checkpoint=args.rvq_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        vocab_path=args.vocab_path,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )

    spk_emb = None
    if args.speaker_emb:
        spk_emb = np.load(args.speaker_emb)

    waveform, sr = tts.synthesize(args.text, spk_emb, args.duration_scale)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({len(waveform)/sr:.2f}s)")


if __name__ == "__main__":
    main()
