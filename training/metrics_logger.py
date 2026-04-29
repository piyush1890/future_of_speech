"""
Training metrics logger for v5.

Writes a JSONL file (`<checkpoint_dir>/metrics.jsonl`) with one record per
event. Three event kinds:
  - "step"      — every N training steps; loss components + gradient ratios
  - "epoch"     — end of each epoch; full validation metrics
  - "audit"     — periodic codebook + acceleration audit (every K epochs)

Designed to be cheap on the hot path (no extra forward passes) but rich enough
that we can plot and detect "is the smooth loss actually biting" / "is the style
codebook collapsing" / "is jitter going down" over the course of training,
*without* parsing stdout.

Gradient ratio measurement (the expensive bit) requires a separate backward pass
per loss component — we only do this every `grad_ratio_every` steps to amortize
the cost (~1% overhead at default cadence of 200 steps).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


class MetricsLogger:
    def __init__(
        self,
        log_path: str | Path,
        grad_ratio_every: int = 200,        # measure CE-vs-smooth gradient ratio every N steps
        scalar_every: int = 50,             # write a "step" record every N steps (lightweight)
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so resumed runs add to the existing log
        self.f = open(self.log_path, "a", buffering=1)   # line-buffered
        self.grad_ratio_every = grad_ratio_every
        self.scalar_every = scalar_every
        self._t_started = time.time()
        self._epoch_start = None

    # ───────────────────── public API ───────────────────── #

    def step(self, *, step: int, lr: float, losses: dict[str, float],
             grad_norms: dict[str, float] | None = None,
             grad_ratios: dict[str, float] | None = None,
             extras: dict | None = None):
        """Record a per-step metric. Call every step but it only writes every
        `scalar_every`. The grad-ratio fields should only be passed on the
        steps where you actually computed them (every `grad_ratio_every`)."""
        if step % self.scalar_every != 0 and not grad_ratios:
            return
        rec = {
            "type": "step",
            "step": int(step),
            "lr": float(lr),
            "losses": {k: float(v) for k, v in losses.items()},
            "wall": round(time.time() - self._t_started, 1),
        }
        if grad_norms:
            rec["grad_norms"] = {k: float(v) for k, v in grad_norms.items()}
        if grad_ratios:
            rec["grad_ratios"] = {k: float(v) for k, v in grad_ratios.items()}
        if extras:
            rec["extras"] = extras
        self._write(rec)

    def epoch_start(self, *, epoch: int):
        self._epoch_start = time.time()
        self._write({"type": "epoch_start", "epoch": int(epoch),
                     "wall": round(time.time() - self._t_started, 1)})

    def epoch_end(self, *, epoch: int, train_metrics: dict, val_metrics: dict,
                  best_val_loss: float | None = None):
        elapsed = time.time() - (self._epoch_start or time.time())
        rec = {
            "type": "epoch",
            "epoch": int(epoch),
            "elapsed_sec": round(elapsed, 1),
            "wall": round(time.time() - self._t_started, 1),
            "train": _to_floats(train_metrics),
            "val":   _to_floats(val_metrics),
        }
        if best_val_loss is not None:
            rec["best_val_loss"] = float(best_val_loss)
        self._write(rec)

    def audit(self, *, epoch: int, name: str, payload: dict):
        """For periodic deep audits: codebook usage, acceleration histograms,
        per-emotion val CE, etc. `name` distinguishes audit kinds."""
        self._write({"type": "audit", "epoch": int(epoch), "name": name,
                     "wall": round(time.time() - self._t_started, 1),
                     "payload": _to_floats(payload, allow_lists=True)})

    def close(self):
        try:
            self.f.flush(); self.f.close()
        except Exception:
            pass

    # ───────────────────── helpers ───────────────────── #

    def _write(self, rec: dict):
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ───────── computation helpers callers can use ───────── #

def grad_norm_total(parameters: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm over all .grad tensors in the param list. Caller is responsible
    for zeroing grads + calling backward(retain_graph=True) before this."""
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return math.sqrt(sq)


def measure_grad_ratios(loss_components: dict[str, torch.Tensor],
                        params: list[torch.nn.Parameter],
                        zero_grad_fn) -> dict[str, dict]:
    """For each loss component, do an isolated backward and measure ||grad||.
    Returns: {component_name: {"norm": float, "ratio_to_total": float}}.

    Pass `zero_grad_fn` as a closure that resets ALL relevant grads (model +
    style_encoder + any other modules in the optimizer's param groups).

    NOTE: this is expensive — pays one extra backward per component plus one
    for the total. Only call every `grad_ratio_every` steps.
    """
    norms = {}
    # First: per-component
    for name, loss in loss_components.items():
        zero_grad_fn()
        loss.backward(retain_graph=True)
        norms[name] = grad_norm_total(params)
    # Then total (sum of all)
    total_loss = sum(loss_components.values())
    zero_grad_fn()
    total_loss.backward(retain_graph=True)
    total_norm = grad_norm_total(params)

    out = {}
    for name, n in norms.items():
        out[name] = {"norm": n, "ratio_to_total": n / max(1e-12, total_norm)}
    out["_total"] = {"norm": total_norm}
    return out


def codebook_usage(indices: torch.Tensor, codebook_size: int,
                   pad_value: int = -1) -> dict:
    """Compute discrete-codebook usage stats from a batch of integer indices.
    Returns: active count, perplexity (exp of token entropy), top-K coverage."""
    flat = indices.reshape(-1)
    flat = flat[flat != pad_value].cpu().numpy()
    if flat.size == 0:
        return {"active": 0, "perplexity": 0.0, "top10_coverage": 0.0,
                "n_observations": 0}
    counts = np.bincount(flat, minlength=codebook_size)
    p = counts / counts.sum()
    nz = p[p > 0]
    perplexity = float(np.exp(-(nz * np.log(nz)).sum()))
    top10 = float(np.sort(counts)[-10:].sum() / counts.sum())
    return {
        "active": int((counts > 0).sum()),
        "perplexity": perplexity,
        "top10_coverage": top10,
        "n_observations": int(flat.size),
    }


def _to_floats(d: dict, allow_lists: bool = False):
    """Recursively coerce tensors/np-scalars to Python floats so JSON dumps cleanly."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _to_floats(v, allow_lists=allow_lists)
        elif isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out[k] = float(v.item())
            elif allow_lists:
                out[k] = v.detach().cpu().tolist()
            else:
                out[k] = float(v.float().mean().item())
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                out[k] = float(v.item())
            elif allow_lists:
                out[k] = v.tolist()
            else:
                out[k] = float(v.mean())
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = repr(v)
    return out


if __name__ == "__main__":
    # Quick smoke test
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    logger = MetricsLogger(log_path=path, grad_ratio_every=2, scalar_every=1)
    for s in range(5):
        logger.step(step=s, lr=3e-4,
                    losses={"ce": 4.0 - s*0.1, "smooth": 0.5, "dur": 0.1},
                    grad_norms={"ce": 1.0, "smooth": 0.05} if s % 2 == 0 else None,
                    grad_ratios={"smooth/ce": 0.05} if s % 2 == 0 else None)
    logger.epoch_start(epoch=1)
    logger.epoch_end(epoch=1,
                     train_metrics={"ce": 3.5, "dur": 0.08},
                     val_metrics={"ce": 3.6, "perplexity_L0": 478.0},
                     best_val_loss=3.6)
    logger.audit(epoch=1, name="codebook_usage",
                 payload={"L0": codebook_usage(torch.randint(0, 512, (100, 50)), 512)})
    logger.close()

    print(f"Wrote {path}; first 5 lines:")
    for line in Path(path).read_text().split("\n")[:5]:
        print(" ", line)
    print(f"Total lines: {len(Path(path).read_text().strip().split(chr(10)))}")
