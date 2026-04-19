"""
Full text-to-speech pipeline: text → phonemes → articulatory tokens → waveform.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer import ArticulatoryVQTokenizer
from models.transformer import ArticulatoryTTSModel
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


class ArticulatoryTTS:
    """End-to-end articulatory TTS system."""

    def __init__(
        self,
        vq_checkpoint: str,
        transformer_checkpoint: str,
        vocab_path: str,
        norm_stats_path: str,
        sparc_model: str = "en",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # Load normalization stats
        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"]
        self.feat_std = stats["std"]

        # Load vocabulary
        self.vocab = PhonemeVocab(vocab_path)

        # Load g2p
        self.g2p = G2p()

        # Load VQ tokenizer
        vq_ckpt = torch.load(vq_checkpoint, map_location=self.device, weights_only=True)
        vq_args = vq_ckpt["args"]
        self.vq = ArticulatoryVQTokenizer(
            codebook_size=vq_args["codebook_size"],
            latent_dim=vq_args["latent_dim"],
            hidden_dim=vq_args["hidden_dim"],
        ).to(self.device)
        self.vq.load_state_dict(vq_ckpt["model_state_dict"])
        self.vq.eval()

        # Load transformer (autoregressive or non-autoregressive)
        tf_ckpt = torch.load(transformer_checkpoint, map_location=self.device, weights_only=True)
        tf_args = tf_ckpt["args"]
        is_ar = tf_ckpt.get("autoregressive", False)

        if is_ar:
            from models.transformer_ar import ArticulatoryTTSModelAR
            self.transformer = ArticulatoryTTSModelAR(
                vocab_size=tf_ckpt["vocab_size"],
                codebook_size=tf_args["codebook_size"],
                d_model=tf_args["d_model"],
                nhead=tf_args["nhead"],
                num_encoder_layers=tf_args["num_layers"],
                num_decoder_layers=tf_args["num_layers"],
                d_ff=tf_args["d_ff"],
            ).to(self.device)
        else:
            self.transformer = ArticulatoryTTSModel(
                vocab_size=tf_ckpt["vocab_size"],
                codebook_size=tf_args["codebook_size"],
                d_model=tf_args["d_model"],
                nhead=tf_args["nhead"],
                num_encoder_layers=tf_args["num_layers"],
                num_decoder_layers=tf_args["num_layers"],
                d_ff=tf_args["d_ff"],
            ).to(self.device)
        self.transformer.load_state_dict(tf_ckpt["model_state_dict"])
        self.transformer.eval()

        # Load SPARC vocoder
        from sparc import load_model as load_sparc
        self.sparc = load_sparc(sparc_model, device="cpu")  # vocoder on CPU for stability

        print("Articulatory TTS loaded.")

    def text_to_phonemes(self, text: str) -> list[str]:
        """Convert text to ARPAbet phoneme list."""
        raw = self.g2p(text)
        phonemes = []
        for p in raw:
            if p == " ":
                phonemes.append("<sil>")
            elif p.strip():
                phonemes.append(p)
        return phonemes

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speaker_emb: np.ndarray = None,
        duration_scale: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """
        Synthesize speech from text.
        Args:
            text: input text
            speaker_emb: (64,) speaker embedding. If None, uses zeros (neutral voice).
            duration_scale: speed control (>1 = slower, <1 = faster)
        Returns:
            waveform: (samples,) numpy array
            sample_rate: int
        """
        # Text → phonemes → indices
        phonemes = self.text_to_phonemes(text)
        indices = self.vocab.encode(phonemes, add_bos_eos=True)
        phoneme_ids = torch.tensor([indices], dtype=torch.long, device=self.device)

        # Speaker embedding
        if speaker_emb is None:
            speaker_emb = np.zeros(64, dtype=np.float32)
        spk = torch.from_numpy(speaker_emb).unsqueeze(0).to(self.device)

        # Generate articulatory token IDs
        token_ids, durations = self.transformer.generate(phoneme_ids, spk, duration_scale)
        token_ids = token_ids.squeeze(0)  # (T,)

        # Decode tokens → continuous features (normalized)
        features_norm = self.vq.decode_indices(token_ids.unsqueeze(0))  # (1, T, 14)
        features_norm = features_norm.squeeze(0).cpu().numpy()  # (T, 14)

        # Denormalize
        features = features_norm * self.feat_std + self.feat_mean

        # Split into EMA, pitch, loudness
        ema = features[:, :12]
        pitch = features[:, 12]
        loudness = features[:, 13]

        # SPARC vocoder expects numpy arrays (not tensors), positional args
        waveform = self.sparc.decode(ema, pitch, loudness, speaker_emb)

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()

        return waveform, self.sparc.sr


def main():
    parser = argparse.ArgumentParser(description="Articulatory TTS synthesis")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--vq-checkpoint", type=str, default="checkpoints/vq_best.pt")
    parser.add_argument("--transformer-checkpoint", type=str, default="checkpoints/transformer_best.pt")
    parser.add_argument("--vocab-path", type=str, default="data/processed/vocab.json")
    parser.add_argument("--norm-stats", type=str, default="data/features/norm_stats.npz")
    parser.add_argument("--speaker-emb", type=str, default=None, help="Path to .npy speaker embedding")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=str, default="outputs/output.wav")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tts = ArticulatoryTTS(
        vq_checkpoint=args.vq_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        vocab_path=args.vocab_path,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )

    # Load speaker embedding if provided
    spk_emb = None
    if args.speaker_emb:
        spk_emb = np.load(args.speaker_emb)

    waveform, sr = tts.synthesize(args.text, spk_emb, args.duration_scale)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({len(waveform)/sr:.2f}s, {sr}Hz)")


if __name__ == "__main__":
    main()
