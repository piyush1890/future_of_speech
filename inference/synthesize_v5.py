"""
v5 inference pipeline: text + knobs (or reference) → per-phoneme style codes →
transformer → RVQ → SPARC → audio.

Two paths, picked at synthesize() time:

  KNOB PATH    (when planner checkpoint is loaded and no reference is provided):
                text + speaker + (emotion, style, intensity) → planner → style_codes
                style_codes → style_codebook.embed_codes → per-phoneme embeddings
                → transformer.encode_phonemes(..., style_emb=embeds)

  REFERENCE PATH  (when a reference utterance's features are provided):
                reference_features + reference_durations → per-phoneme style encoder
                → continuous z → style_codebook (quantize) → style_codes
                → style_codebook.embed_codes → per-phoneme embeddings
                → transformer.encode_phonemes(..., style_emb=embeds)

Both paths produce per-phoneme style codes from the same codebook the frame decoder
was trained against. The planner just learns to predict what the encoder would
have given for a real utterance with these (text, knobs).

No post-hoc smoothing, no emotion presets, no waveform scaling. All prosody from
the model. log-pitch → Hz conversion happens before SPARC as the only "post"
processing (and that's a feature-format conversion, not a quality filter).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vq_tokenizer_rvq import ArticulatoryRVQTokenizer
from models.transformer_rvq_hier import ArticulatoryTTSModelRVQHier
from models.per_phoneme_style_encoder import PerPhonemeStyleEncoder
from models.style_codebook import StyleCodebook, PAD_CODE
from models.style_planner import StylePlanner
from models.phoneme_vocab import PhonemeVocab
from training.dataset_rvq import EMOTION_TO_ID, STYLE_TO_ID, N_EMOTIONS, N_STYLES
from g2p_en import G2p


DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"


def load_reference_features(path: str):
    """Load a reference .npz into (features (T, 14), spk_emb (64,)).
    Pitch is converted to log(Hz+1) to match training-feature normalization."""
    f = np.load(path)
    T = min(f["ema"].shape[0], f["pitch"].shape[0], f["loudness"].shape[0])
    pitch = f["pitch"][:T].astype(np.float32)
    if pitch.max() > 20:
        pitch = np.log(pitch + 1.0)
    feats = np.concatenate([
        f["ema"][:T].astype(np.float32),
        pitch[:, None],
        f["loudness"][:T, None].astype(np.float32),
    ], axis=1)
    spk = f["spk_emb"].astype(np.float32) if "spk_emb" in f.files else None
    return feats, spk


class ArticulatoryTTSv5:
    def __init__(
        self,
        stage1_checkpoint: str,
        planner_checkpoint: str | None = None,    # optional
        rvq_checkpoint: str | None = None,         # for SPARC decoding via RVQ.decode_indices
        vocab_path: str = "data/processed_all/vocab_mfa.json",
        norm_stats_path: str = "data/features_merged_logpitch_v2/norm_stats.npz",
        sparc_model: str = "en",
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        stats = np.load(norm_stats_path)
        self.feat_mean = stats["mean"]
        self.feat_std  = stats["std"]
        self.vocab = PhonemeVocab(vocab_path)
        self.g2p = G2p()

        # ─── stage 1 modules ──────────────────────────────────
        s1 = torch.load(stage1_checkpoint, map_location=self.device, weights_only=True)
        sa = s1["args"]
        self.s1_args = sa

        # RVQ tokenizer (for decoding token indices → features)
        rvq_path = rvq_checkpoint or sa["rvq_checkpoint"]
        rvq_ckpt = torch.load(rvq_path, map_location=self.device, weights_only=True)
        ra = rvq_ckpt["args"]
        self.rvq = ArticulatoryRVQTokenizer(
            codebook_size=ra["codebook_size"], num_quantizers=ra["num_quantizers"],
            latent_dim=ra["latent_dim"], hidden_dim=ra["hidden_dim"],
        ).to(self.device)
        self.rvq.load_state_dict(rvq_ckpt["model_state_dict"])
        self.rvq.eval()

        self.transformer = ArticulatoryTTSModelRVQHier(
            vocab_size=s1["vocab_size"],
            codebook_size=sa["codebook_size"], num_quantizers=sa["num_quantizers"],
            d_model=sa["d_model"], nhead=sa["nhead"],
            num_encoder_layers=sa["num_layers"], num_decoder_layers=sa["num_layers"],
            d_ff=sa["d_ff"], dropout=sa.get("dropout", 0.1), speaker_emb_dim=64,
            # v6: honour tied-output if present in saved args. The frozen_codebooks
            # buffer is saved within the state_dict so loading restores it.
            tied_output=sa.get("tied_output", False),
            codebook_latent_dim=sa.get("codebook_latent_dim", 64),
        ).to(self.device)
        self.transformer.load_state_dict(s1["model_state_dict"])
        self.transformer.eval()

        self.style_encoder = PerPhonemeStyleEncoder(
            input_dim=14, style_dim=sa["d_model"], hidden=128, n_conv_layers=3,
        ).to(self.device)
        self.style_encoder.load_state_dict(s1["style_encoder_state_dict"])
        self.style_encoder.eval()

        self.style_codebook = StyleCodebook(
            latent_dim=sa["d_model"], codebook_size=sa["style_codebook_size"],
        ).to(self.device)
        self.style_codebook.load_state_dict(s1["style_codebook_state_dict"])
        self.style_codebook.eval()

        # ─── optional stage 2: planner ─────────────────────────
        self.planner = None
        if planner_checkpoint and Path(planner_checkpoint).exists():
            p2 = torch.load(planner_checkpoint, map_location=self.device, weights_only=True)
            pa = p2["args"]
            self.planner = StylePlanner(
                vocab_size=s1["vocab_size"],
                style_codebook_size=sa["style_codebook_size"],
                n_emotions=N_EMOTIONS, n_styles=N_STYLES, speaker_emb_dim=64,
                d_model=pa["d_model"], nhead=pa["nhead"],
                num_encoder_layers=pa["num_layers"], num_decoder_layers=pa["num_layers"],
                d_ff=pa["d_ff"], dropout=pa.get("dropout", 0.1), knob_dropout=0.0,
            ).to(self.device)
            self.planner.load_state_dict(p2["planner_state_dict"])
            self.planner.eval()
            print(f"Loaded planner: epoch={p2.get('epoch')}, val_loss={p2.get('val_loss', float('nan')):.4f}")

        from sparc import load_model as load_sparc
        self.sparc = load_sparc(sparc_model, device="cpu")

        print(f"v5 TTS loaded (stage1 epoch={s1.get('epoch','?')}, "
              f"val={s1.get('val_loss', float('nan')):.4f})  "
              f"planner={'yes' if self.planner else 'no'}")

    # ─── helpers ─────────────────────────────────────────────

    def text_to_phonemes(self, text: str) -> list[str]:
        """MFA-style: drop punctuation + spaces. <sil> at start, between sentences,
        and at end — matches training distribution."""
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

    @torch.no_grad()
    def codes_from_reference(
        self, reference_features: np.ndarray, reference_durations: np.ndarray | None = None,
        target_n_phonemes: int = None,
    ) -> torch.Tensor:
        """Encode a reference utterance into per-phoneme style codes.

        If reference_durations is provided, we use them to slice the reference's
        features per phoneme (matches the training pipeline). If not, we evenly
        split the reference's frames across `target_n_phonemes` slots — usable
        when we don't have phoneme alignments for the reference but we know the
        target text's phoneme count.

        Returns: (1, N) long tensor of codes.
        """
        T_ref = reference_features.shape[0]

        if reference_durations is not None:
            durs = reference_durations
            N = len(durs)
        elif target_n_phonemes is not None:
            # Even split. Keep BOS=0, EOS=0 if caller-supplied phonemes already
            # have those wrapping <sil>s baked in; here we just need any int-dur
            # set summing to T_ref, length = target_n_phonemes.
            N = target_n_phonemes
            base = T_ref // N
            rem = T_ref - base * N
            durs = np.full(N, base, dtype=np.int64)
            durs[:rem] += 1
        else:
            raise ValueError("provide either reference_durations or target_n_phonemes")

        feat_t = torch.from_numpy(reference_features.astype(np.float32)).unsqueeze(0).to(self.device)
        durs_t = torch.from_numpy(np.asarray(durs, dtype=np.int64)).unsqueeze(0).to(self.device)
        mask_t = torch.ones_like(durs_t, dtype=torch.bool)
        z = self.style_encoder(feat_t, durs_t.float(), mask_t)            # (1, N, D)
        _, codes, _ = self.style_codebook(z, mask_t)                       # (1, N)
        return codes

    @torch.no_grad()
    def codes_from_planner(
        self, phoneme_ids: torch.Tensor, phoneme_mask: torch.Tensor,
        speaker_emb: torch.Tensor, emotion: str, style: str, intensity: float,
        temperature: float = 0.7, top_k: int = 0, cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the AR planner from text + knobs → per-phoneme codes."""
        if self.planner is None:
            raise RuntimeError("Planner not loaded; pass planner_checkpoint at __init__")
        emo_id = torch.tensor([EMOTION_TO_ID.get(emotion, 0)], dtype=torch.long, device=self.device)
        sty_id = torch.tensor([STYLE_TO_ID.get(style,   0)], dtype=torch.long, device=self.device)
        intens = torch.tensor([float(intensity)], device=self.device)
        codes = self.planner.generate(
            phoneme_ids, phoneme_mask, speaker_emb, emo_id, sty_id, intens,
            temperature=temperature, top_k=top_k, cfg_scale=cfg_scale,
        )
        return codes

    # ─── main API ────────────────────────────────────────────

    @torch.no_grad()
    def synthesize(
        self, text: str, speaker_emb: np.ndarray,
        *,
        # exactly one of these:
        reference_features: np.ndarray | None = None,
        reference_durations: np.ndarray | None = None,
        emotion: str = "neutral", style: str = "default", intensity: float = 1.0,
        # general
        duration_scale: float = 1.0,
        # planner sampling
        sampling_temperature: float = 0.7, top_k: int = 0, cfg_scale: float = 1.0,
    ):
        """Synthesize text. If reference_features is provided → reference path.
        Otherwise → planner path (requires planner loaded). Returns (waveform, sr)."""
        phonemes = self.text_to_phonemes(text)
        # Phoneme ids include BOS + EOS (so length = N+2)
        indices = self.vocab.encode(phonemes, add_bos_eos=True)
        phoneme_ids = torch.tensor([indices], dtype=torch.long, device=self.device)
        phoneme_mask = phoneme_ids != 0

        spk = torch.from_numpy(speaker_emb.astype(np.float32)).unsqueeze(0).to(self.device)

        # ─── pick path: get per-phoneme codes ────────────────
        N_total = phoneme_ids.shape[1]   # BOS + N_phonemes + EOS
        if reference_features is not None:
            codes = self.codes_from_reference(
                reference_features, reference_durations,
                target_n_phonemes=N_total,
            )
        elif self.planner is not None:
            codes = self.codes_from_planner(
                phoneme_ids, phoneme_mask, spk, emotion, style, intensity,
                temperature=sampling_temperature, top_k=top_k, cfg_scale=cfg_scale,
            )
            # Planner only emits N codes for the phonemes; we need codes for
            # BOS + N + EOS. Pad BOS/EOS with PAD_CODE (= zero embedding via
            # embed_codes) so they don't influence the encoder.
            if codes.shape[1] != N_total:
                # Planner returned N codes for the body; BOS/EOS take PAD
                pad = torch.full((codes.shape[0], 1), PAD_CODE, dtype=codes.dtype, device=self.device)
                codes = torch.cat([pad, codes, pad], dim=1)
                # Trim/pad to N_total
                if codes.shape[1] > N_total:
                    codes = codes[:, :N_total]
                elif codes.shape[1] < N_total:
                    extra = torch.full((codes.shape[0], N_total - codes.shape[1]),
                                       PAD_CODE, dtype=codes.dtype, device=self.device)
                    codes = torch.cat([codes, extra], dim=1)
        else:
            # Analogue of v4 fix: skip cross-utterance style entirely. Use PAD_CODE
            # everywhere → embed_codes returns zeros → no style added by the encoder.
            codes = torch.full((1, N_total), PAD_CODE, dtype=torch.long, device=self.device)

        # ─── codes → embeddings → transformer ────────────────
        style_emb = self.style_codebook.embed_codes(codes)   # (1, N_total, D)

        enc = self.transformer.encode_phonemes(
            phoneme_ids, spk, phoneme_mask, style_emb=style_emb,
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

        # log-pitch → Hz
        features[:, 12] = np.exp(features[:, 12]) - 1.0

        ema = features[:, :12]
        pitch = features[:, 12]
        pitch[pitch < 30] = 0.0
        loudness = features[:, 13]

        waveform = self.sparc.decode(ema, pitch, loudness, speaker_emb)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().squeeze().cpu().numpy()
        return waveform, self.sparc.sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", type=str)
    p.add_argument("--stage1-checkpoint", default="checkpoints_v5_stage1/transformer_best.pt")
    p.add_argument("--planner-checkpoint", default="checkpoints_v5_planner/planner_best.pt")
    p.add_argument("--rvq-checkpoint", default="checkpoints_rvq_logpitch_v2/rvq_best.pt")
    p.add_argument("--vocab-path", default="data/processed_all/vocab_mfa.json")
    p.add_argument("--norm-stats", default="data/features_merged_logpitch_v2/norm_stats.npz")
    p.add_argument("--reference", default=None,
                   help="Path to .npz with SPARC features. If set, uses reference path; else planner.")
    p.add_argument("--speaker-emb", default=None,
                   help="Path to .npy spk_emb (64,). Defaults to reference's spk_emb if reference given, "
                        "else first LibriSpeech speaker.")
    p.add_argument("--emotion", default="neutral", choices=list(EMOTION_TO_ID))
    p.add_argument("--style",   default="default", choices=list(STYLE_TO_ID))
    p.add_argument("--intensity", type=float, default=1.0)
    p.add_argument("--duration-scale", type=float, default=1.0)
    p.add_argument("--sampling-temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--cfg-scale", type=float, default=1.0,
                   help="Classifier-free guidance scale on planner. 1.0 = off, >1 sharpens knob effect.")
    p.add_argument("--output", "-o", default="outputs/v5.wav")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Planner is optional — only loaded if checkpoint exists
    planner_ckpt = args.planner_checkpoint if Path(args.planner_checkpoint).exists() else None
    if planner_ckpt is None:
        print(f"(no planner checkpoint at {args.planner_checkpoint} — knob path disabled, "
              f"reference path required)")

    tts = ArticulatoryTTSv5(
        stage1_checkpoint=args.stage1_checkpoint,
        planner_checkpoint=planner_ckpt,
        rvq_checkpoint=args.rvq_checkpoint,
        vocab_path=args.vocab_path,
        norm_stats_path=args.norm_stats,
        device=args.device,
    )

    # Load reference if provided (or use it as default for spk_emb fallback)
    ref_feats, ref_spk = (None, None)
    if args.reference:
        ref_feats, ref_spk = load_reference_features(args.reference)
        print(f"Reference: {args.reference}  ({ref_feats.shape[0]} frames)")

    # Speaker emb resolution: explicit > reference > default
    if args.speaker_emb:
        spk_emb = np.load(args.speaker_emb).astype(np.float32)
    elif ref_spk is not None:
        spk_emb = ref_spk
    else:
        spk_emb = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
        print(f"(using default speaker_emb from {DEFAULT_REFERENCE})")

    waveform, sr = tts.synthesize(
        args.text, spk_emb,
        reference_features=ref_feats,
        emotion=args.emotion, style=args.style, intensity=args.intensity,
        duration_scale=args.duration_scale,
        sampling_temperature=args.sampling_temperature,
        top_k=args.top_k, cfg_scale=args.cfg_scale,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sr)
    print(f"Saved {args.output} ({len(waveform)/sr:.2f}s)")


if __name__ == "__main__":
    main()
