"""
Hierarchical RVQ inference: text → phonemes → hier transformer → RVQ decode → SPARC.

Prosody comes entirely from the model:
  • A reference utterance's SPARC features are passed through the trained StyleEncoder
    to produce a 256-dim style_vec.
  • That style_vec is added to the phoneme encoder input alongside the speaker embedding.
  • The duration predictor and frame decoder both read from this combined encoding.

There is no post-hoc emotion preset, no pitch multiplier, no smoothness filter.
If output is monotone or jittery, treat it as a training-side problem.

Default reference is an ESD-Happy utterance (0011_Happy_0011_000927) — gives lively
prosody. Override via `--reference path/to/some.npz` for a different prosody target.
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
from models.style_encoder import StyleEncoder
from models.phoneme_vocab import PhonemeVocab
from g2p_en import G2p


DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"


def load_reference_features(path: str) -> np.ndarray:
    """Load a reference .npz (SPARC output) into a (T, 14) feature array.
    Pitch is converted to log(Hz+1) to match training-feature normalization."""
    f = np.load(path)
    T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
    pitch = f["pitch"][:T].astype(np.float32)
    if pitch.max() > 20:
        # Raw Hz — convert to log space (training features are stored as log(Hz+1))
        pitch = np.log(pitch + 1.0)
    return np.concatenate([
        f["ema"][:T].astype(np.float32),
        pitch[:, None],
        f["loudness"][:T, None].astype(np.float32),
    ], axis=1)


class ArticulatoryTTSRVQHier:
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

        if "style_encoder_state_dict" not in tf_ckpt:
            raise RuntimeError(
                f"Transformer checkpoint {transformer_checkpoint} has no "
                f"style_encoder_state_dict. This inference path requires a v3+ "
                f"checkpoint trained with the StyleEncoder branch."
            )
        self.style_encoder = StyleEncoder(
            input_dim=14, style_dim=ta["d_model"], hidden=128, n_conv_layers=4,
        ).to(self.device)
        self.style_encoder.load_state_dict(tf_ckpt["style_encoder_state_dict"])
        self.style_encoder.eval()

        from sparc import load_model as load_sparc
        self.sparc = load_sparc(sparc_model, device="cpu")

        print(f"Hierarchical RVQ TTS loaded (epoch {tf_ckpt.get('epoch','?')}, "
              f"val={tf_ckpt.get('val_loss', float('nan')):.4f}).")

    def text_to_phonemes(self, text):
        """MFA-style: drop spaces and punctuation. <sil> at start, between sentences,
        and at end — every MFA training utterance is bracketed by <sil>, and the
        bidirectional encoder reads global structure off these markers. Without
        the trailing <sil>, the duration predictor flattens to a near-uniform output
        across the whole utterance."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        phonemes = ["<sil>"]
        for si, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            if si > 0:
                phonemes.append("<sil>")
            for p in self.g2p(sentence):
                if p and p[0].isalpha() and p[0].isupper():
                    phonemes.append(p)
        phonemes.append("<sil>")
        return phonemes

    def encode_reference(self, reference_features: np.ndarray) -> torch.Tensor:
        """Encode a (T, 14) reference into a (1, style_dim) style_vec.
        Features are passed RAW (un-normalized) — that's how training fed them
        (see dataset_rvq.py: style_features = unnormalized concat). Normalizing
        here puts StyleEncoder off-manifold and the transformer-conditioning gets
        garbled (sounds 'choked' regardless of which reference you pick)."""
        ref_t = torch.from_numpy(reference_features.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.style_encoder(ref_t)

    @torch.no_grad()
    def synthesize(self, text, speaker_emb, reference_features, duration_scale=1.0):
        """
        text:                str
        speaker_emb:         (64,) float32 — target voice timbre
        reference_features:  (T, 14) float32 — SPARC features of a reference utterance
                             whose prosody (pitch contour, energy, tempo) we want to mimic
        duration_scale:      pre-model tempo knob (>1 slower, <1 faster)
        """
        phonemes = self.text_to_phonemes(text)
        indices = self.vocab.encode(phonemes, add_bos_eos=True)
        phoneme_ids = torch.tensor([indices], dtype=torch.long, device=self.device)
        phoneme_mask = phoneme_ids != 0

        spk = torch.from_numpy(speaker_emb).unsqueeze(0).to(self.device)
        style_vec = self.encode_reference(reference_features)

        enc = self.transformer.encode_phonemes(
            phoneme_ids, spk, phoneme_mask, style_vec=style_vec,
        )
        pred_dur = self.transformer.duration_predictor(enc, phoneme_mask)
        durations = (pred_dur * duration_scale).round().clamp(min=1)
        T = int(durations.sum().item())

        decoded, _ = self.transformer._decode_frames(enc, durations, T, phoneme_mask)
        logits = self.transformer._run_hierarchical_heads(decoded, target_tokens=None)
        token_ids = logits.argmax(dim=-1)

        features_norm = self.rvq.decode_indices(token_ids).squeeze(0).cpu().numpy()
        features = features_norm * self.feat_std + self.feat_mean
        features = features[:T]

        # Features store pitch as log(Hz + 1) — convert back to Hz for SPARC
        features[:, 12] = np.exp(features[:, 12]) - 1.0

        ema = features[:, :12]
        pitch = features[:, 12]
        pitch[pitch < 30] = 0.0   # voicing floor — 30 Hz is below all real F0
        loudness = features[:, 13]

        waveform = self.sparc.decode(ema, pitch, loudness, speaker_emb)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().squeeze().cpu().numpy()
        return waveform, self.sparc.sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    parser.add_argument("--rvq-checkpoint", type=str,
                        default="checkpoints_rvq_logpitch_v2/rvq_best.pt")
    parser.add_argument("--transformer-checkpoint", type=str,
                        default="checkpoints_rvq_logpitch_hier_v4/transformer_best.pt")
    parser.add_argument("--vocab-path", type=str, default="data/processed_all/vocab_mfa.json")
    parser.add_argument("--norm-stats", type=str,
                        default="data/features_merged_logpitch_v2/norm_stats.npz")
    parser.add_argument("--speaker-emb", type=str, default=None,
                        help="Path to .npy (64-dim) for target voice. "
                             "If omitted, uses the speaker_emb from --reference.")
    parser.add_argument("--reference", type=str, default=DEFAULT_REFERENCE,
                        help="Path to .npz with SPARC features of a reference utterance — "
                             "its prosody (pitch contour, energy, tempo) is what we mimic.")
    parser.add_argument("--duration-scale", type=float, default=1.0,
                        help="Tempo knob: >1 slower, <1 faster. Pre-model.")
    parser.add_argument("--output", "-o", type=str, default="outputs/output_rvq_hier.wav")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    tts = ArticulatoryTTSRVQHier(
        rvq_checkpoint=args.rvq_checkpoint,
        transformer_checkpoint=args.transformer_checkpoint,
        vocab_path=args.vocab_path,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )

    ref_features = load_reference_features(args.reference)
    print(f"Reference: {args.reference}  ({ref_features.shape[0]} frames, "
          f"{ref_features.shape[0]/50:.2f}s)")

    if args.speaker_emb:
        spk_emb = np.load(args.speaker_emb).astype(np.float32)
    else:
        spk_emb = np.load(args.reference)["spk_emb"].astype(np.float32)
        print(f"Speaker emb taken from reference utterance.")

    waveform, sr = tts.synthesize(
        args.text, spk_emb, ref_features, duration_scale=args.duration_scale,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({len(waveform)/sr:.2f}s)")


if __name__ == "__main__":
    main()
