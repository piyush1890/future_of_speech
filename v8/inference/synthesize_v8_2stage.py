"""
v8 two-stage inference: planner predicts z, stage1 model uses z to render audio.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from g2p_en import G2p

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v8.models.phoneme_tts import PhonemeTTSv8
from v8.models.v8_planner import V8Planner, V8CodebookPlanner
from v8.models.phoneme_classes import build_render_class_table
from models.phoneme_vocab import PhonemeVocab
from models.style_codebook import StyleCodebook


DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"


def text_to_phonemes(text, g2p):
    """MFA-style: drop punctuation + spaces. <sil> at start, end, and ALL
    punctuation boundaries (.!?,;:— …). Short comma silences vs long sentence
    silences emerge from the duration predictor's context-conditional output."""
    # Split at any of these punctuation marks followed by whitespace
    parts = re.split(r"(?<=[.!?,;:—–])\s+", text.strip())
    phs = ["<sil>"]
    for si, part in enumerate(parts):
        if not part.strip():
            continue
        if si > 0:
            phs.append("<sil>")
        for p in g2p(part):
            if p and p[0].isalpha() and p[0].isupper():
                phs.append(p)
    phs.append("<sil>")
    return phs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str)
    ap.add_argument("--stage1-checkpoint", default="v8/checkpoints/stage1_z/best.pt")
    ap.add_argument("--planner-checkpoint", default="v8/checkpoints/stage2_planner/best.pt")
    ap.add_argument("--vocab", default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--speaker-emb", default=None)
    ap.add_argument("--valence",   type=float, default=0.5)
    ap.add_argument("--arousal",   type=float, default=0.5)
    ap.add_argument("--dominance", type=float, default=0.5)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--sampling-std", type=float, default=0.0,
                    help="Gaussian noise std for continuous AR (V8Planner)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for codebook AR sampling (0 = argmax)")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--duration-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", "-o", default="v8/outputs/v8_2stage.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Stage 1 (encoder + heads) ────
    s1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    sa = s1["args"]
    print(f"v8 stage1: epoch={s1['epoch']} val={s1['val_loss']:.4f}")
    vocab = PhonemeVocab(args.vocab)
    rmode = sa.get("render_mode", "hybrid")
    rct = build_render_class_table(vocab) if rmode == "hybrid" else None
    stage1 = PhonemeTTSv8(
        vocab_size=s1["vocab_size"],
        feature_dim=sa.get("feature_dim", 14),
        d_model=sa["d_model"], nhead=sa["nhead"],
        num_layers=sa["num_layers"], d_ff=sa["d_ff"],
        dropout=sa.get("dropout", 0.0),
        speaker_emb_dim=64,
        style_dim=sa["d_model"],
        render_mode=rmode,
        render_class_table=rct,
    ).to(device)
    stage1.load_state_dict(s1["model"]); stage1.eval()

    # ── Stage 2 (planner) ────
    s2 = torch.load(args.planner_checkpoint, map_location=device, weights_only=False)
    pa = s2["args"]
    print(f"v8 stage2 planner: epoch={s2['epoch']} val={s2['val_loss']:.4f}")
    is_codebook = "codebook_size" in s2

    if is_codebook:
        # Codebook AR planner — load v5 codebook entries, build planner with them
        v5_ckpt = torch.load(pa["v5_checkpoint"], map_location=device, weights_only=False)
        cb = StyleCodebook(latent_dim=v5_ckpt["args"]["d_model"],
                           codebook_size=v5_ckpt["args"]["style_codebook_size"]).to(device)
        cb.load_state_dict(v5_ckpt["style_codebook_state_dict"])
        cb.eval()
        cb_entries = cb.vq._codebook.embed.detach().clone()
        if cb_entries.dim() == 3:
            cb_entries = cb_entries.squeeze(0)
        cb_entries = cb_entries.to(device)
        planner = V8CodebookPlanner(
            vocab_size=s2["vocab_size"],
            codebook_entries=cb_entries,
            d_model=pa["planner_d_model"], nhead=pa["planner_nhead"],
            num_layers=pa["planner_layers"], d_ff=pa["planner_d_ff"],
            dropout=pa.get("dropout", 0.0),
            knob_dim=3, speaker_emb_dim=64,
            knob_dropout=0.0,
            max_context=pa.get("max_context", 100),
        ).to(device)
        planner.load_state_dict(s2["model"]); planner.eval()
        print(f"  codebook planner: {s2['codebook_size']} codes, style_dim={s2['style_dim']}")
    else:
        planner = V8Planner(
            vocab_size=s2["vocab_size"],
            d_model=pa["planner_d_model"], nhead=pa["planner_nhead"],
            num_layers=pa["planner_layers"], d_ff=pa["planner_d_ff"],
            dropout=pa.get("dropout", 0.0),
            knob_dim=3, speaker_emb_dim=64,
            style_dim=pa["style_dim"],
            knob_dropout=0.0,
            causal=pa.get("causal", False),
        ).to(device)
        planner.load_state_dict(s2["model"]); planner.eval()

    g2p = G2p()
    phonemes = text_to_phonemes(args.text, g2p)
    phoneme_ids = torch.tensor([vocab.encode(phonemes, add_bos_eos=True)], dtype=torch.long, device=device)
    phoneme_mask = phoneme_ids != 0

    if args.speaker_emb:
        spk = np.load(args.speaker_emb).astype(np.float32)
    else:
        spk = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
        print("(default speaker)")
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)
    knobs = torch.tensor([[args.valence, args.arousal, args.dominance]],
                         dtype=torch.float32, device=device)

    with torch.no_grad():
        if is_codebook:
            # Codebook AR planner: sample code IDs, return codebook-embedded z
            code_ids, z = planner.generate(
                phoneme_ids, spk_t, knobs, phoneme_mask,
                temperature=args.temperature, top_k=args.top_k,
            )
            print(f"sampled code ids (first 10): {code_ids[0, :10].tolist()}")
        elif planner.causal:
            if args.cfg_scale != 1.0:
                uncond = planner.generate(phoneme_ids, spk_t, knobs, phoneme_mask,
                                          sampling_std=args.sampling_std,
                                          force_drop_knobs=True)
                cond   = planner.generate(phoneme_ids, spk_t, knobs, phoneme_mask,
                                          sampling_std=args.sampling_std,
                                          force_drop_knobs=False)
                z = uncond + args.cfg_scale * (cond - uncond)
            else:
                z = planner.generate(phoneme_ids, spk_t, knobs, phoneme_mask,
                                     sampling_std=args.sampling_std)
        else:
            if args.cfg_scale != 1.0:
                uncond = planner(phoneme_ids, spk_t, knobs, phoneme_mask, force_drop_knobs=True)
                cond   = planner(phoneme_ids, spk_t, knobs, phoneme_mask, force_drop_knobs=False)
                z = uncond + args.cfg_scale * (cond - uncond)
            else:
                z = planner(phoneme_ids, spk_t, knobs, phoneme_mask)
        # Render
        frames, durations, frame_mask = stage1.generate(
            phoneme_ids, spk_t, style_emb=z, duration_scale=args.duration_scale,
        )

    feats_logp = frames.squeeze(0).cpu().numpy()
    feats_logp = feats_logp[frame_mask.squeeze(0).cpu().numpy()]
    feats = feats_logp.copy()
    feats[:, 12] = np.exp(feats[:, 12]) - 1.0
    ema = feats[:, :12]
    pitch = feats[:, 12].copy(); pitch[pitch < 30] = 0.0
    loud = feats[:, 13]

    print(f"Phonemes: {len(phonemes)}  Frames: {feats.shape[0]}  Duration: {feats.shape[0]/50:.2f}s")

    print("Loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    wav = sparc.decode(ema, pitch, loud, spk)
    if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, sparc.sr)
    print(f"Saved {args.output} ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
