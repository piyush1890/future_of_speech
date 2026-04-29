"""
v9 tokenizer smoke test (batched API): pack one utterance's body phonemes
into a single batch, forward through the tokenizer, sanity-check shapes
and gradients. Times CPU and MPS to confirm GPU is now faster.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v9.models.phoneme_rvq import PhonemeRVQTokenizer
from v9.training.dataset_v9 import V9PhonemeBlocksDataset


def pack_utterance(item):
    """Pack body phonemes from one utterance dict into batched tensors."""
    frames    = item["frames"]
    offsets   = item["frame_offsets"]
    phoneme_ids = item["phoneme_ids"]
    blocks, ph_ids, lens = [], [], []
    for p_idx in range(1, len(phoneme_ids) - 1):
        s = int(offsets[p_idx]); e = int(offsets[p_idx + 1])
        if e <= s: continue
        blocks.append(frames[s:e])
        ph_ids.append(int(phoneme_ids[p_idx]))
        lens.append(e - s)
    F_max = max(lens)
    B = len(blocks)
    padded = np.zeros((B, F_max, 14), dtype=np.float32)
    for i, b in enumerate(blocks):
        padded[i, :len(b)] = b
    return (torch.from_numpy(padded), torch.tensor(ph_ids, dtype=torch.long),
            torch.tensor(lens, dtype=torch.long))


def time_one_step(model, frames, ph_ids, lengths, device, n=5):
    model.train()
    # Warm up
    for _ in range(2):
        recon, info = model.forward_batch(frames, ph_ids, lengths)
        mask = info["valid_mask"].unsqueeze(-1).float()
        loss = (((recon - frames) ** 2) * mask).sum() / (mask.sum().clamp(min=1) * 14) \
               + 0.25 * info["commit_loss"]
        loss.backward()
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
    if str(device) == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n):
        recon, info = model.forward_batch(frames, ph_ids, lengths)
        mask = info["valid_mask"].unsqueeze(-1).float()
        loss = (((recon - frames) ** 2) * mask).sum() / (mask.sum().clamp(min=1) * 14) \
               + 0.25 * info["commit_loss"]
        loss.backward()
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
    if str(device) == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0
    return elapsed / n


def main():
    print("Loading dataset (no preload, one item)...")
    ds = V9PhonemeBlocksDataset(preload=False, knob_source="none")
    print(f"  total utterances: {len(ds)}")
    pick_idx = next(i for i, uid in enumerate(ds.utt_ids)
                    if 30 < len(ds.phon_data[uid]['indices']) < 60)
    uid = ds.utt_ids[pick_idx]
    item = ds._load(uid)
    frames_cpu, ph_ids_cpu, lens_cpu = pack_utterance(item)
    print(f"  uid={uid}  B={frames_cpu.shape[0]} phonemes  F_max={frames_cpu.shape[1]}  "
          f"lens range=[{lens_cpu.min().item()}, {lens_cpu.max().item()}]")

    # Sanity: forward + backward on CPU
    print("\nCPU forward + backward...")
    m_cpu = PhonemeRVQTokenizer(vocab_size=73)
    m_cpu.train()
    recon, info = m_cpu.forward_batch(frames_cpu, ph_ids_cpu, lens_cpu)
    assert recon.shape == frames_cpu.shape, f"shape: {recon.shape} vs {frames_cpu.shape}"
    assert not torch.isnan(recon).any(), "NaN in recon"
    assert info["start_idx"].shape == (frames_cpu.shape[0], 4)
    assert info["end_idx"].shape   == (frames_cpu.shape[0], 4)
    print(f"  recon shape ok, start_idx[0]={info['start_idx'][0].tolist()}  "
          f"end_idx[0]={info['end_idx'][0].tolist()}  commit={info['commit_loss'].item():.4f}")
    mask = info["valid_mask"].unsqueeze(-1).float()
    loss = (((recon - frames_cpu) ** 2) * mask).sum() / (mask.sum() * 14) + 0.25 * info["commit_loss"]
    loss.backward()
    has_grad = sum(1 for p in m_cpu.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_p  = sum(1 for _ in m_cpu.parameters())
    print(f"  loss={loss.item():.4f}  params with grads: {has_grad}/{total_p}")
    assert has_grad > 0

    # Speed comparison
    print("\nSpeed: 1 batched 'step' = forward(B={} phonemes) + backward".format(frames_cpu.shape[0]))
    t_cpu = time_one_step(m_cpu, frames_cpu, ph_ids_cpu, lens_cpu, "cpu")
    print(f"  CPU: {t_cpu*1000:.1f} ms/step")

    if torch.backends.mps.is_available():
        m_mps = PhonemeRVQTokenizer(vocab_size=73).to("mps")
        f_mps  = frames_cpu.to("mps")
        ph_mps = ph_ids_cpu.to("mps")
        ln_mps = lens_cpu.to("mps")
        t_mps = time_one_step(m_mps, f_mps, ph_mps, ln_mps, "mps")
        print(f"  MPS: {t_mps*1000:.1f} ms/step  (speedup over CPU: {t_cpu/t_mps:.1f}×)")

    print("\nSmoke test passed ✓")


if __name__ == "__main__":
    main()
