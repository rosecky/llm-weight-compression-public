"""Functional graphs over channels: construction, structure metrics, null models, and the
balanced partitioning that turns a graph into a permutation.

Design decisions that matter for whether the answer is trustworthy:

* **Correlation, not covariance.** Raw covariance edges are dominated by channel scale, and
  "channels with large activations cluster together" is first-order anisotropy that this
  project has already measured to death (kill criterion G7). The primary graph is therefore
  the correlation matrix, with the covariance graph kept only as a comparison.
* **Off-diagonal energy.** Every structure metric zeroes the diagonal first. Otherwise any
  partition scores near 1.0 simply because `C_ii` is large, which would make the whole study
  look successful for free.
* **The partition is an ordering.** What a deployable method needs is a *permutation*, so the
  primitive here is spectral sequencing (Fiedler ordering) and equal-size contiguous blocks
  cut from it -- not a free-form clustering that would then have to be turned into a
  permutation after the fact.
* **Nulls are matrices, not scores.** Each null is a surrogate matrix that the *entire*
  pipeline is re-run on, so the null gets the same optimisation the real graph gets.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .gptq import guard_alloc


# ---------------------------------------------------------------- graph construction


def to_correlation(C: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    d = torch.diagonal(C).clamp_min(eps).sqrt()
    return C / (d.reshape(-1, 1) * d.reshape(1, -1))


def to_partial_correlation(C: torch.Tensor, percdamp: float = 0.01) -> torch.Tensor:
    """Normalised negative precision matrix: coupling with all other channels regressed out."""
    n = C.shape[0]
    Cd = C.clone().float()
    dead = torch.diagonal(Cd) <= 0
    if bool(dead.any()):
        Cd[dead, dead] = 1.0
    Cd[range(n), range(n)] += percdamp * torch.diagonal(Cd).mean()
    P = torch.cholesky_inverse(torch.linalg.cholesky(Cd))
    d = torch.diagonal(P).clamp_min(1e-30).sqrt()
    return -P / (d.reshape(-1, 1) * d.reshape(1, -1))


def affinity(C: torch.Tensor, kind: str = "corr") -> torch.Tensor:
    """Non-negative symmetric affinity with a zero diagonal."""
    if kind == "cov":
        S = C.abs()
    elif kind == "corr":
        S = to_correlation(C).abs()
    elif kind == "pcorr":
        S = to_partial_correlation(C).abs()
    else:
        raise ValueError(kind)
    S = 0.5 * (S + S.T)
    S.fill_diagonal_(0.0)
    return S.clamp_min(0.0)


def topk_sparsify(S: torch.Tensor, k: int) -> torch.Tensor:
    """Keep each vertex's k strongest edges; symmetrise by union (max)."""
    if k <= 0 or k >= S.shape[0]:
        return S
    thresh = S.topk(k, dim=1).values[:, -1:]
    M = (S >= thresh).float()
    M = torch.maximum(M, M.T)
    return S * M


# ---------------------------------------------------------------- structure metrics


def _offdiag_energy(S: torch.Tensor) -> torch.Tensor:
    return S.pow(2).sum() - torch.diagonal(S).pow(2).sum()


@torch.no_grad()
def block_energy_ratio(S: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of total off-diagonal edge energy that falls inside blocks.

    A random balanced partition into blocks of size b scores about (b-1)/(n-1), so the number
    to look at is always this divided by that baseline.
    """
    n = S.shape[0]
    tot = _offdiag_energy(S)
    if float(tot) <= 0:
        return float("nan")
    inside = torch.zeros((), device=S.device)
    for c in torch.unique(labels):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        sub = S[idx][:, idx]
        inside += sub.pow(2).sum() - torch.diagonal(sub).pow(2).sum()
    return float(inside / tot)


@torch.no_grad()
def modularity(S: torch.Tensor, labels: torch.Tensor) -> float:
    """Newman weighted modularity on a non-negative graph with zero diagonal."""
    d = S.sum(1)
    m2 = float(d.sum())
    if m2 <= 0:
        return float("nan")
    q = 0.0
    for c in torch.unique(labels):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        q += float(S[idx][:, idx].sum()) / m2 - (float(d[idx].sum()) / m2) ** 2
    return q


@torch.no_grad()
def normalized_cut(S: torch.Tensor, labels: torch.Tensor) -> float:
    d = S.sum(1)
    tot = 0.0
    for c in torch.unique(labels):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        vol = float(d[idx].sum())
        if vol <= 0:
            continue
        within = float(S[idx][:, idx].sum())
        tot += (vol - within) / vol
    return tot / float(torch.unique(labels).numel())


@torch.no_grad()
def spectral_analyze(S: torch.Tensor, n_modes: int = 16, n_vec: int = 1):
    """One eigendecomposition, used for both the ordering and the spectral diagnostics.

    The normalised Laplacian is `L = I - Sn` with `Sn = D^-1/2 S D^-1/2`, so `eig(L)` is
    `1 - eig(Sn)` and a single `eigh(Sn)` gives the Fiedler ordering *and* the spectrum.

    Localisation is reported as an effective support size `1 / sum(v_i^4)`: n for a fully
    delocalised mode, small for a mode living on a few channels. Delocalised modes are
    precisely the case where a block-diagonal (graph-community) description cannot work.
    """
    n = S.shape[0]
    d = S.sum(1).clamp_min(1e-30)
    Dm = d.rsqrt()
    Sn = Dm.reshape(-1, 1) * S * Dm.reshape(1, -1)
    w, V = torch.linalg.eigh(Sn)                  # ascending
    lam = 1.0 - w.flip(0)                         # Laplacian eigenvalues, ascending
    f = V[:, -2] * Dm                             # Fiedler vector of L
    if n_vec > 1:
        for k in range(3, 2 + n_vec):
            f = f + (V[:, -k] * Dm) * (10.0 ** (-3 * (k - 2)))
    order = torch.argsort(f)
    top = V[:, -n_modes:]
    eff = 1.0 / top.pow(4).sum(0).clamp_min(1e-30)
    summary = {
        "lap_lambda2": float(lam[1]),
        "lap_gap": float(lam[1] - lam[0]),
        "eff_support_top": float(eff[-1]),
        "eff_support_mean16": float(eff.mean()),
        "eff_support_frac": float(eff.mean() / n),
    }
    return order, summary


# ---------------------------------------------------------------- null models


def random_orthogonal(n: int, seed: int, device: str) -> Tuple[torch.Tensor, str]:
    """Haar orthogonal by QR of a Gaussian, with a structured fallback for large n."""
    if n <= 3072:
        guard_alloc((n, n), torch.float32, "random orthogonal", limit_mib=512.0)
        g = torch.Generator(device=device).manual_seed(seed)
        Q, R = torch.linalg.qr(torch.randn(n, n, generator=g, device=device))
        return Q * torch.sign(torch.diagonal(R)).reshape(1, -1), "haar"
    from .rotate import Rotation
    rot = Rotation(n, seed=seed, device=device)
    return rot.apply(torch.eye(n, device=device)), "randomized_hadamard"


@torch.no_grad()
def make_graph_null(C: torch.Tensor, kind: str, seed: int = 0) -> torch.Tensor:
    """Surrogate covariance/affinity with a specific property preserved.

    permute   vertex relabelling; keeps every scalar graph statistic, destroys alignment
    config    weighted configuration model: expected edge weight d_i d_j / 2m, i.e. exactly
              the null modularity is measured against, with the strength sequence preserved
    rewire    the observed edge-weight multiset reassigned to pairs drawn with probability
              proportional to d_i d_j -- a randomised degree-preserving rewiring
    spectral  Q L Q^T with the eigenvalues of C and a random orthogonal Q: the matrix keeps
              its entire spectrum and loses only *which channels* the eigenmodes live on.
              This is the null that separates "strong eigenmodes" from "exploitable topology".
    """
    n = C.shape[0]
    g = torch.Generator(device=C.device).manual_seed(seed)
    if kind == "real":
        return C
    if kind == "permute":
        p = torch.randperm(n, generator=g, device=C.device)
        return C[p][:, p]
    if kind == "spectral":
        w, _ = torch.linalg.eigh(C.float())
        Q, _ = random_orthogonal(n, seed, str(C.device))
        return (Q * w.reshape(1, -1)) @ Q.T
    S = C.abs().clone()
    S.fill_diagonal_(0.0)
    d = S.sum(1).clamp_min(1e-30)
    if kind == "config":
        m2 = d.sum()
        out = torch.outer(d, d) / m2
        out.fill_diagonal_(0.0)
        return out
    if kind == "rewire":
        iu = torch.triu_indices(n, n, offset=1, device=C.device)
        w = S[iu[0], iu[1]]
        p = (d[iu[0]] * d[iu[1]])
        p = p / p.sum()
        keep = int((w > 0).sum())
        pick = torch.multinomial(p, keep, replacement=False, generator=g)
        vals = w[w > 0][torch.randperm(keep, generator=g, device=C.device)]
        out = torch.zeros_like(S)
        out[iu[0][pick], iu[1][pick]] = vals
        return out + out.T
    raise ValueError(kind)


# ---------------------------------------------------------------- partitioning


@torch.no_grad()
def spectral_order(S: torch.Tensor, n_vec: int = 1, seed: int = 0) -> torch.Tensor:
    """Fiedler ordering: the classic spectral sequencing.

    Returns a permutation. Contiguous chunks of it are the balanced communities, which is
    what a deployable method needs anyway -- a physical reorder, not a label set.
    """
    return spectral_analyze(S, n_vec=n_vec)[0]


@torch.no_grad()
def balanced_labels(order: torch.Tensor, block: int) -> torch.Tensor:
    """Equal-size communities as contiguous chunks of an ordering."""
    n = order.numel()
    lab = torch.empty(n, dtype=torch.long, device=order.device)
    lab[order] = torch.arange(n, device=order.device) // block
    return lab


@torch.no_grad()
def scale_order(C: torch.Tensor) -> torch.Tensor:
    """The strong simple baseline: order channels by their own second moment."""
    return torch.argsort(torch.diagonal(C), descending=True)


@torch.no_grad()
def random_order(n: int, seed: int, device: str) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randperm(n, generator=g, device=device)
