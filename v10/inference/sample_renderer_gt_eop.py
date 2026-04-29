"""Render audio using renderer's predicted RVQ tokens but GT phoneme positions.

This isolates the question: are the renderer's TOKEN predictions usable, or
is EOP timing the only thing dragging audio down?

Pipeline:
  - Real utterance gives GT phonemes, GT speaker, GT emotion, GT frame_to_enc_pos, GT length
  - Style encoder runs on GT frames → oracle style codes
  - Renderer AR generates tokens for T_GT frames using GT positions (no EOP sampling)
  - Tokenizer decodes → SPARC → audio
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


def sample_logits(logits, temperature, top_k):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / max(1e-6, temperature)
    if top_k > 0:
        v, _ = torch.topk(logits, top_k, dim=-1)
        thr = v[..., -1:].expand_as(logits)
        logits = torch.where(logits < thr, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def renderer_generate_gt_pos(renderer, phoneme_ids, style_codes, spk_emb, knobs,
                             phoneme_mask, gt_frame_to_enc_pos, T,
                             temperature=1.0, top_k=0):
    """AR-generate tokens using GT phoneme positions for each frame.
    Uses self-generated previous tokens (true AR), but supplies the correct
    phoneme assignment so cross-attention always queries the right phoneme."""
    B, _ = phoneme_ids.shape
    device = phoneme_ids.device
    K = renderer.K

    enc = renderer.encode(phoneme_ids, style_codes, spk_emb, knobs, phoneme_mask,
                          force_drop_knobs=False)

    gen_codes = torch.zeros(B, T, K, dtype=torch.long, device=device)
    for f in range(T):
        codes_so_far = gen_codes[:, :f + 1, :]
        pos_so_far = gt_frame_to_enc_pos[:, :f + 1]
        dec_inp = renderer._make_decoder_input(codes_so_far, pos_so_far)
        causal = torch.triu(torch.full((f + 1, f + 1), float("-inf"), device=device),
                            diagonal=1)
        h = renderer.decoder(
            dec_inp, enc, tgt_mask=causal,
            memory_key_padding_mask=~phoneme_mask,
        )
        h_f = h[:, f, :]
        for k in range(K):
            cl = renderer.heads.step_logits_one(h_f, k, gen_codes[:, f, :k])
            gen_codes[:, f, k] = sample_logits(cl, temperature, top_k)
    return gen_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="0011_Neutral_0011_000313")
    ap.add_argument("--tokenizer-checkpoint", default="v10/checkpoints/tokenizer/best.pt")
    ap.add_argument("--stage1-checkpoint", default="v10/checkpoints/stage1_renderer/best.pt")
    ap.add_argument("--norm-stats", default="data/features_merged_logpitch_v2/norm_stats.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output", default="v10/outputs/renderer_e1_gt_eop.wav")
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
        knob_dim=6, knob_dropout=0.0,
        max_phonemes=s1a.get("max_phonemes", 200) + 4,
        max_frames=s1a.get("max_frames", 800) + 16,
    ).to(device)
    renderer.load_state_dict(s1["renderer"]); renderer.eval()

    style_enc = V10StyleEncoder(codebook_size=s1a.get("style_codebook_size", 64)).to(device)
    style_enc.load_state_dict(s1["style_enc"]); style_enc.eval()

    print(f"\nLoading utterance: {args.uid}")
    ds = V10Dataset(max_frames=800, knob_source="emotion", preload=False)
    if args.uid not in ds.utt_ids:
        print(f"FATAL: {args.uid} not in dataset"); return
    item = ds._load(args.uid)
    batch = collate_v10_renderer([item])
    for k in ("phoneme_ids", "phoneme_mask", "spk_emb", "knobs", "frames",
             "frame_mask", "frame_to_enc_pos", "eop", "body_durations"):
        batch[k] = batch[k].to(device)

    n_total = batch["phoneme_ids"].shape[1]
    T_gt = int(batch["frame_mask"].sum().item())
    print(f"  N_total={n_total}  T_GT={T_gt}")

    with torch.no_grad():
        style_out = style_enc(batch["frames"], batch["frame_mask"],
                              batch["frame_to_enc_pos"], n_total=n_total)
    style_codes = style_out["codes"]
    print(f"  oracle style codes (first 10): {style_codes[0, 1:11].tolist()}")

    # Generate tokens with GT positions, GT length
    gt_pos = batch["frame_to_enc_pos"][:, :T_gt]
    print(f"\nGenerating {T_gt} frame tokens with GT phoneme positions...")
    gen_codes = renderer_generate_gt_pos(
        renderer, batch["phoneme_ids"], style_codes, batch["spk_emb"], batch["knobs"],
        batch["phoneme_mask"], gt_pos, T_gt, temperature=args.temperature,
    )
    print(f"  generated shape: {gen_codes.shape}  range: [{gen_codes.min().item()}, {gen_codes.max().item()}]")

    # Decode tokens → frames
    frame_mask_gen = torch.ones(1, T_gt, dtype=torch.bool, device=device)
    with torch.no_grad():
        feats_norm = tokenizer.tokens_to_frames(gen_codes, frame_mask_gen)[0].cpu().numpy()

    stats = np.load(args.norm_stats)
    feat_mean = stats["mean"].astype(np.float32); feat_std = stats["std"].astype(np.float32)
    feats = feats_norm * feat_std + feat_mean
    feats[:, 12] = np.exp(feats[:, 12]) - 1.0
    feats[feats[:, 12] < 30, 12] = 0.0
    print(f"\nframes: {feats.shape[0]} @ 50Hz  duration: {feats.shape[0]/50:.2f}s")

    spk = batch["spk_emb"][0].cpu().numpy()
    print("Loading SPARC...")
    from sparc import load_model as load_sparc
    sparc = load_sparc("en", device="cpu")
    wav = sparc.decode(feats[:, :12], feats[:, 12], feats[:, 13], spk)
    if isinstance(wav, torch.Tensor): wav = wav.detach().squeeze().cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, wav, sparc.sr)
    print(f"saved {args.output} ({len(wav)/sparc.sr:.2f}s)")


if __name__ == "__main__":
    main()
