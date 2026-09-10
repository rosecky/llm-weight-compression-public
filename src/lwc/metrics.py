"""Error metrics at three levels: weight, activation-space, end-to-end (see eval_ppl)."""
from __future__ import annotations

import torch


def nmse(W: torch.Tensor, Wh: torch.Tensor) -> float:
    """||W - Wh||_F^2 / ||W||_F^2"""
    W = W.float()
    Wh = Wh.float()
    return float(((W - Wh) ** 2).sum() / (W**2).sum().clamp_min(1e-30))


def rel_fro(W: torch.Tensor, Wh: torch.Tensor) -> float:
    return float(nmse(W, Wh) ** 0.5)


@torch.no_grad()
def act_nmse_from_X(W: torch.Tensor, Wh: torch.Tensor, X: torch.Tensor, chunk: int = 4096) -> float:
    """||(W-Wh) X||_F^2 / ||W X||_F^2 with X of shape (in_features, n_samples).

    Computed in column chunks so memory stays bounded.
    """
    dW = (W - Wh).float()
    Wf = W.float()
    num = 0.0
    den = 0.0
    for i in range(0, X.shape[1], chunk):
        Xc = X[:, i : i + chunk].float()
        num += float((dW @ Xc).pow(2).sum())
        den += float((Wf @ Xc).pow(2).sum())
    return num / max(den, 1e-30)


@torch.no_grad()
def gram(X: torch.Tensor) -> torch.Tensor:
    """H = X X^T, (in, in). Only for small in_features."""
    Xf = X.float()
    return Xf @ Xf.T


@torch.no_grad()
def act_nmse_from_H(W: torch.Tensor, Wh: torch.Tensor, H: torch.Tensor) -> float:
    dW = (W - Wh).float()
    Wf = W.float()
    num = float((dW @ H * dW).sum())
    den = float((Wf @ H * Wf).sum())
    return num / max(den, 1e-30)


def snr_db(nmse_val: float) -> float:
    import math

    return -10.0 * math.log10(max(nmse_val, 1e-30))
