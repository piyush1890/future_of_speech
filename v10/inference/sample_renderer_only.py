"""Sample audio from the v10 renderer alone (no planner).
Style codes come from running the trained style encoder on a real utterance's frames.
This tests: given correct phonemes, speaker, emotion, and ORACLE style codes,
does the renderer generate intelligible frame tokens?"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import torch

from v10.models.v10_tokenizer import V10Tokenizer
from v10.models.v10_renderer import V10Renderer
from v10.models.v10_style import V10StyleEncoder
from v10.training.dataset_v10 import V10Dataset, collate_v10_renderer
from v10.inference.synthesize_v10 import renderer_generate

EMOTION_TO_ID = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "surprise": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="0011_Neutral_0011_000313",
                    help="UID of a real utterance to extract style codes + speaker from")
    ap.add_argument("--tokenizer-checkpoint", default="v10/checkpoints/tokenizer/best.pt")
    ap.add_argument("--stage1-checkpoint", default="v10/checkpoints/stage1_renderer/best.pt")
    ap.add_argument("--norm-stats", default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--output", default="v10/outputs/renderer_e1.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    tc = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False)
    ta = tc["args"]
    print(f"Tokenizer epoch={tc['epoch']} val_mse={tc.get('val_mse', float('nan')):.4f}")
    tokenizer = V10Tokenizer(
        d_model=ta.get("d_model", 256),
        num_encoder_layers=ta.get("enc_layers", 4),
        num_decoder_layers=ta.get("dec_layers", 4),
        codebook_size=ta.get("codebook_size", 1024),
        num_quantizers=ta.get("num_quantizers", 4),
        max_frames=ta.get("max_frames", 800) + 16,
    ).to(device)
    tokenizer.load_state_dict(tc["model"]); tokenizer.eval()

    s1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    s1a = s1["args"]
    print(f"Renderer epoch={s1['epoch']} val_ce={s1.get('val_ce', float('nan')):.4f}")
    renderer = V10Renderer(
        codebook_size=s1a.get("codebook_size", 1024),
        num_quantizers=s1a.get("num_quantizers", 4),
        style_codebook_size=s1a.get("style_codebook_size", 64),
        d_model=s1a.get("d_model", 256),
        num_encoder_layers=s1a.get("enc_layers", 4),
        num_decoder_layers=s1a.get("dec_layers", 6),
        knob_dim=6,
        knob_dropout=0.0,
        max_phonemes=s1a.get("max_phonemes", 200) + 4,
        max_frames=s1a.get("max_frames", 800) + 16,
    ).to(device)
    renderer.load_state_dict(s1["renderer"]); renderer.eval()

    style_enc = V10StyleEncoder(codebook_size=s1a.get("style_codebook_size", 64)).to(device)
    style_enc.load_state_dict(s1["style_enc"]); style_enc.eval()

    # Load one real utterance via dataset (gives us frames + phonemes + EOP + speaker + knobs)
    print(f"\nLoading utterance: {args.uid}")
    ds = V10Dataset(max_frames=args.max_frames, knob_source="emotion", preload=False)
    if args.uid not in ds.utt_ids:
        print(f"FATAL: {args.uid} not in dataset")
        return
    item = ds._load(args.uid)
    batch = collate_v10_renderer([item])
    for k in ("phoneme_ids", "phoneme_mask", "spk_emb", "knobs", "frames",
             "frame_mask", "frame_to_enc_pos", "eop", "body_durations"):
        batch[k] = batch[k].to(device)

    n_total = batch["phoneme_ids"].shape[1]
    print(f"  N_total={n_total}  body_phonemes={int(batch['body_durations'].numel() and (batch['body_durations']>0).sum().item())}  T_frames_GT={int(batch['frame_mask'].sum().item())}")

    # Style codes from style encoder (oracle)
    with torch.no_grad():
        style_out = style_enc(batch["frames"], batch["frame_mask"],
                              batch["frame_to_enc_pos"], n_total=n_total)
    style_codes = style_out["codes"]
    print(f"  oracle style codes (first 10 body): {style_codes[0, 1:11].tolist()}")

    # Renderer AR: phonemes + speaker + knobs + oracle style codes → frame tokens
    print(f"\nGenerating frame tokens (T_max={args.max_frames}, temp={args.temperature}, cfg={args.cfg_scale})...")
    gen = renderer_generate(
        renderer, batch["phoneme_ids"], style_codes, batch["spk_emb"], batch["knobs"],
        batch["phoneme_mask"], max_frames=args.max_frames,
        temperature=args.temperature, cfg_scale=args.cfg_scale,
    )
    n_frames = gen["n_frames"]
    eops_fired = int(gen["eop"].sum().item())
    print(f"  generated {n_frames} frames  EOPs fired: {eops_fired}  (expected ~{int((batch['body_durations']>0).sum().item())})")
    print(f"  first 10 frame_to_enc_pos: {gen['frame_to_enc_pos'][0, :10].tolist()}")

    # Decode tokens → frames
    frame_mask_gen = torch.ones(1, n_frames, dtype=torch.bool, device=device)
    with torch.no_grad():
        feats_norm_50 = tokenizer.tokens_to_frames(gen["frame_codes"], frame_mask_gen)
    feats_norm_50 = feats_norm_50[0].cpu().numpy()

    # SPARC native is 50 Hz; tokens are at 50 Hz too (frame_stride=1). No upsample.
    feats_norm = feats_norm_50

    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)
    feats = feats_norm * feat_std + feat_mean
    feats[:, 12] = np.exp(feats[:, 12]) - 1.0
    feats[feats[:, 12] < 30, 12] = 0.0
    print(f"  frames: {feats.shape[0]} @ 100Hz  duration: {feats.shape[0]/100:.2f}s")

    spk = batch["spk_emb"][0].cpu().numpy()
    print("\nLoading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    wav = sparc.decode(feats[:, :12], feats[:, 12], feats[:, 13], spk)
    if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, wav, sparc.sr)
    print(f"saved {out_path} ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
