"""
v8 inference: text → phonemes → encoder → 3 anchors per phoneme + duration
            → 3-pt linear interpolation → 50Hz frame stream → SPARC → audio.
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
from models.phoneme_vocab import PhonemeVocab


DEFAULT_REFERENCE = "data/features_merged_logpitch_v2/0011_Happy_0011_000927.npz"


def text_to_phonemes(text: str, g2p) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    phs = ["<sil>"]
    for si, s in enumerate(sents):
        if not s.strip(): continue
        if si > 0: phs.append("<sil>")
        for p in g2p(s):
            if p and p[0].isalpha() and p[0].isupper():
                phs.append(p)
    phs.append("<sil>")
    return phs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", type=str)
    ap.add_argument("--checkpoint", default="v8/checkpoints/stage1/best.pt")
    ap.add_argument("--vocab", default="data/processed_all/vocab_mfa.json")
    ap.add_argument("--speaker-emb", default=None,
                    help="Path to .npy with spk_emb. Defaults to project default.")
    ap.add_argument("--duration-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", "-o", default="v8/outputs/v8_test.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    c = torch.load(args.checkpoint, map_location=device, weights_only=True)
    saved_args = c["args"]
    print(f"v8 checkpoint: epoch={c['epoch']} val={c['val_loss']:.4f}")

    model = PhonemeTTSv8(
        vocab_size=c["vocab_size"],
        feature_dim=saved_args.get("feature_dim", 14),
        d_model=saved_args["d_model"],
        nhead=saved_args["nhead"],
        num_layers=saved_args["num_layers"],
        d_ff=saved_args["d_ff"],
        dropout=saved_args.get("dropout", 0.0),
        speaker_emb_dim=64,
    ).to(device)
    model.load_state_dict(c["model"])
    model.eval()

    vocab = PhonemeVocab(args.vocab)
    g2p = G2p()
    phonemes = text_to_phonemes(args.text, g2p)
    indices = vocab.encode(phonemes, add_bos_eos=True)
    phoneme_ids = torch.tensor([indices], dtype=torch.long, device=device)

    if args.speaker_emb:
        spk = np.load(args.speaker_emb).astype(np.float32)
    else:
        spk = np.load(DEFAULT_REFERENCE)["spk_emb"].astype(np.float32)
        print(f"(using default speaker_emb)")
    spk_t = torch.from_numpy(spk).unsqueeze(0).to(device)

    with torch.no_grad():
        frames, durations, frame_mask = model.generate(
            phoneme_ids, spk_t, style_emb=None, duration_scale=args.duration_scale,
        )
    feats_logp = frames.squeeze(0).cpu().numpy()    # (T, 14)
    feats_logp = feats_logp[frame_mask.squeeze(0).cpu().numpy()]  # drop padding

    # Convert log-pitch → Hz
    feats = feats_logp.copy()
    feats[:, 12] = np.exp(feats[:, 12]) - 1.0
    ema = feats[:, :12]
    pitch = feats[:, 12].copy()
    pitch[pitch < 30] = 0.0
    loud = feats[:, 13]

    print(f"Phonemes: {len(phonemes)}  Total frames: {feats.shape[0]}  "
          f"Duration: {feats.shape[0]/50:.2f}s")

    print("Loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")

    wav = sparc.decode(ema, pitch, loud, spk)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().squeeze().cpu().numpy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav, sparc.sr)
    print(f"Saved {out_path} ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
