"""Demo: chunked rendering — render N short utterances independently and
concatenate the resulting articulator-feature streams with brief silence,
then vocode in one SPARC pass.

This proves the chunking approach works even before Stage 2 planner is
trained. In production, the planner would replace the "oracle style codes"
step (use planner.generate(text) per chunk).
"""
import argparse
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


def render_one(tokenizer, renderer, style_enc, item, device, eop_threshold, temperature):
    """Render a single utterance → 14-d feature stream (T, 14) at 50 Hz."""
    batch = collate_v10_renderer([item])
    for k in ("phoneme_ids","phoneme_mask","spk_emb","knobs","frames",
              "frame_mask","frame_to_enc_pos"):
        batch[k] = batch[k].to(device)
    n_total = batch["phoneme_ids"].shape[1]
    with torch.no_grad():
        style_out = style_enc(batch["frames"], batch["frame_mask"],
                              batch["frame_to_enc_pos"], n_total=n_total)
        gen = renderer_generate(
            renderer, batch["phoneme_ids"], style_out["codes"], batch["spk_emb"],
            batch["knobs"], batch["phoneme_mask"],
            max_frames=400, temperature=temperature, cfg_scale=1.0,
            eop_threshold=eop_threshold,
        )
        n_frames = gen["n_frames"]
        frame_mask = torch.ones(1, n_frames, dtype=torch.bool, device=device)
        feats_norm = tokenizer.tokens_to_frames(gen["frame_codes"], frame_mask)[0].cpu().numpy()
    return feats_norm, n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="+", default=[
        "0011_Neutral_0011_000313",
        "0011_Neutral_0011_000040",
        "0011_Neutral_0011_000176",
    ])
    ap.add_argument("--tokenizer-checkpoint", default="v10/checkpoints/tokenizer/best.pt")
    ap.add_argument("--stage1-checkpoint", default="v10/checkpoints/stage1_renderer/best.pt")
    ap.add_argument("--norm-stats", default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eop-threshold", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--silence-frames", type=int, default=10,
                    help="Zero-articulator frames between chunks at 50 Hz (10 = 200ms).")
    ap.add_argument("--output", default="v10/outputs/demo_chunked.wav")
    args = ap.parse_args()

    device = torch.device(args.device)

    tc = torch.load(args.tokenizer_checkpoint, map_location=device, weights_only=False); ta = tc["args"]
    tok = V10Tokenizer(d_model=ta["d_model"], num_encoder_layers=ta["enc_layers"],
                       num_decoder_layers=ta["dec_layers"], codebook_size=ta["codebook_size"],
                       num_quantizers=ta["num_quantizers"], max_frames=ta["max_frames"]+16).to(device)
    tok.load_state_dict(tc["model"]); tok.eval()
    print(f"tokenizer epoch={tc['epoch']} val_mse={tc.get('val_mse', float('nan')):.4f}")

    s1 = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False); s1a = s1["args"]
    print(f"renderer epoch={s1['epoch']} val_ce={s1.get('val_ce', float('nan')):.4f}")
    renderer = V10Renderer(
        codebook_size=s1a["codebook_size"], num_quantizers=s1a["num_quantizers"],
        style_codebook_size=s1a["style_codebook_size"], d_model=s1a["d_model"],
        num_encoder_layers=s1a["enc_layers"], num_decoder_layers=s1a["dec_layers"],
        knob_dim=6, knob_dropout=0.0,
        max_phonemes=s1a["max_phonemes"]+4, max_frames=s1a["max_frames"]+16,
    ).to(device)
    renderer.load_state_dict(s1["renderer"]); renderer.eval()
    style_enc = V10StyleEncoder(codebook_size=s1a["style_codebook_size"]).to(device)
    style_enc.load_state_dict(s1["style_enc"]); style_enc.eval()

    ds = V10Dataset(max_frames=800, knob_source="emotion", preload=False)

    chunks = []
    feat_dim = 14
    for i, uid in enumerate(args.uids):
        if uid not in ds.utt_ids:
            print(f"  WARN: {uid} not in dataset; skipping")
            continue
        item = ds._load(uid)
        feats_norm, n_frames = render_one(
            tok, renderer, style_enc, item, device,
            args.eop_threshold, args.temperature,
        )
        gt_T = item["frames"].shape[0]
        print(f"  chunk {i+1}: {uid}  {n_frames} frames @ 50Hz ({n_frames/50:.2f}s)  GT was {gt_T}f ({gt_T/50:.2f}s)")
        chunks.append(feats_norm)
        if i < len(args.uids) - 1:
            chunks.append(np.zeros((args.silence_frames, feat_dim), dtype=np.float32))
    full_norm = np.concatenate(chunks, axis=0)
    print(f"\nconcatenated: {full_norm.shape[0]} frames ({full_norm.shape[0]/50:.2f}s)")

    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)
    full = full_norm * feat_std + feat_mean
    full[:, 12] = np.exp(full[:, 12]) - 1.0
    full[full[:, 12] < 30, 12] = 0.0

    print("loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    # use first chunk's speaker embedding
    item0 = ds._load(args.uids[0])
    spk = item0["spk_emb"]
    wav = sparc.decode(full[:, :12], full[:, 12], full[:, 13], spk)
    if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, sparc.sr)
    print(f"saved {args.output}  ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
