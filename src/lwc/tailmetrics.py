"""Metrics beyond MSE.

EXPERIMENT_LOG E11 showed why these are needed: a codec with 1.3x worse NMSE than INT4 had
150x worse error on the top 0.1% of weights and 98x worse perplexity, and its activation NMSE
moved by only 1.7x. Aggregate error hides clipping; the tail is what the model notices.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


@torch.no_grad()
def tail_report(W: torch.Tensor, Wh: torch.Tensor,
                fracs=(0.01, 0.001, 0.0001)) -> Dict[str, float]:
    """Weight-space error, aggregate and restricted to the largest-magnitude weights."""
    W = W.float()
    Wh = Wh.float()
    E = W - Wh
    w2 = W.pow(2).sum()
    out = {
        "nmse": float(E.pow(2).sum() / w2.clamp_min(1e-30)),
        "max_abs_err": float(E.abs().max()),
        "max_err_over_max_w": float(E.abs().max() / W.abs().max().clamp_min(1e-30)),
    }
    a = W.abs().reshape(-1)
    e = E.reshape(-1)
    wf = W.reshape(-1)
    n = a.numel()
    for f in fracs:
        k = max(1, int(round(f * n)))
        idx = torch.topk(a, k).indices
        num = e[idx].pow(2).sum()
        den = wf[idx].pow(2).sum().clamp_min(1e-30)
        tag = f"top{f:g}".replace("0.", "")
        out[f"relerr_{tag}"] = float(num / den)
        # a sign flip on a large weight is a qualitative failure, not a small error
        out[f"signflip_{tag}"] = float(
            (torch.sign(wf[idx]) != torch.sign(wf[idx] - e[idx])).float().mean())
    return out


@torch.no_grad()
def hessian_proxy(W: torch.Tensor, Wh: torch.Tensor, H: torch.Tensor) -> float:
    """tr(dW H dW^T) / tr(W H W^T).

    With H = E[x x^T] this is exactly the activation NMSE ||(W-What)X||^2 / ||WX||^2, so it
    doubles as a cross-check on the activation measurement.
    """
    dW = (W - Wh).float()
    Wf = W.float()
    num = float((dW @ H * dW).sum())
    den = float((Wf @ H * Wf).sum())
    return num / max(den, 1e-30)


@torch.no_grad()
def act_error_quantiles(W: torch.Tensor, Wh: torch.Tensor, X: torch.Tensor,
                        chunk: int = 2048) -> Dict[str, float]:
    """Per-output-channel distribution of the activation error, not just its mean."""
    dW = (W - Wh).float()
    Wf = W.float()
    num = torch.zeros(W.shape[0], device=W.device)
    den = torch.zeros(W.shape[0], device=W.device)
    for i in range(0, X.shape[1], chunk):
        Xc = X[:, i:i + chunk].float()
        num += (dW @ Xc).pow(2).sum(1)
        den += (Wf @ Xc).pow(2).sum(1)
    r = num / den.clamp_min(1e-30)
    return {
        "act_nmse": float(num.sum() / den.sum().clamp_min(1e-30)),
        "act_nmse_ch_p50": float(r.quantile(0.50)),
        "act_nmse_ch_p95": float(r.quantile(0.95)),
        "act_nmse_ch_max": float(r.max()),
    }
