"""Phase 1 structural diagnostics.

Central idea
------------
Everything is reduced to ONE comparable scalar: bits saved per weight relative to just
coding the tile directly. For a codebook of K prototypes over d-dimensional tiles that
leaves relative residual energy rho:

    rate spent  = log2(K) / d                       bits/weight
    rate saved  = 0.5 * log2(1 / rho)               bits/weight  (Gaussian high-rate R-D)
    net_gain    = rate_saved - rate_spent

For an i.i.d. source, net_gain is ~0 by the rate-distortion theorem: any bits you spend on a
prototype buy you back exactly the same number of bits on the residual. So net_gain is only
positive if the tiles carry exploitable dependence structure.

To separate the possible sources we run the identical pipeline on null models:
  * "real"     the true weight matrix
  * "shuffle"  all entries randomly permuted -> exact same marginal distribution,
               zero dependence structure
  * "gauss"    i.i.d. Gaussian, matched variance -> no marginal shape gain either

    real - shuffle  = exploitable DEPENDENCE between weights (what this project needs)
    shuffle - gauss = marginal SHAPE gain (classic VQ gain, already used by AQLM/QuIP#)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch

from .tiles import to_tiles


# ---------------------------------------------------------------- null models

def make_null(W: torch.Tensor, kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=W.device).manual_seed(seed)
    if kind == "real":
        return W
    if kind == "shuffle":
        flat = W.reshape(-1)
        perm = torch.randperm(flat.numel(), generator=g, device=W.device)
        return flat[perm].reshape(W.shape)
    if kind == "shuffle_rows":
        # preserve per-row scale + marginal, destroy within-row arrangement
        idx = torch.argsort(torch.rand(W.shape, generator=g, device=W.device), dim=1)
        return torch.gather(W, 1, idx)
    if kind == "shuffle_cols":
        idx = torch.argsort(torch.rand(W.shape, generator=g, device=W.device), dim=0)
        return torch.gather(W, 0, idx)
    if kind == "gauss":
        return torch.randn(W.shape, generator=g, device=W.device, dtype=W.dtype) * W.std()
    if kind == "gauss_rowcol":
        # i.i.d. Gaussian with matched per-row and per-column RMS (keeps outlier-channel
        # scale structure, destroys everything else)
        G = torch.randn(W.shape, generator=g, device=W.device, dtype=W.dtype)
        for _ in range(4):
            rw = W.pow(2).mean(1, keepdim=True).sqrt()
            rg = G.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
            G = G * (rw / rg)
            cw = W.pow(2).mean(0, keepdim=True).sqrt()
            cg = G.pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-12)
            G = G * (cw / cg)
        return G
    raise ValueError(kind)


# ---------------------------------------------------------------- k-means (GPU, chunked)

@torch.no_grad()
def kmeans(X: torch.Tensor, K: int, iters: int = 25, seed: int = 0,
           chunk: int = 16384) -> torch.Tensor:
    """Lloyd iterations. X:(N,d) -> centroids (K,d). Memory bounded by `chunk`."""
    N, d = X.shape
    K = min(K, N)
    g = torch.Generator(device=X.device).manual_seed(seed)
    idx = torch.randperm(N, generator=g, device=X.device)[:K]
    C = X[idx].clone()
    for _ in range(iters):
        sums = torch.zeros_like(C)
        cnts = torch.zeros(K, device=X.device, dtype=X.dtype)
        Cn = (C * C).sum(1)
        for i in range(0, N, chunk):
            Xc = X[i:i + chunk]
            dist = Cn.unsqueeze(0) - 2.0 * (Xc @ C.T)   # drop the constant ||x||^2 term
            a = dist.argmin(1)
            sums.index_add_(0, a, Xc)
            cnts.index_add_(0, a, torch.ones(Xc.shape[0], device=X.device, dtype=X.dtype))
        empty = cnts == 0
        C = sums / cnts.clamp_min(1).unsqueeze(1)
        if bool(empty.any()):                            # re-seed dead centroids
            ridx = torch.randperm(N, generator=g, device=X.device)[: int(empty.sum())]
            C[empty] = X[ridx]
    return C


@torch.no_grad()
def assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    Cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]
        out[i:i + chunk] = (Cn.unsqueeze(0) - 2.0 * (Xc @ C.T)).argmin(1)
    return out


@torch.no_grad()
def residual_energy(X: torch.Tensor, C: torch.Tensor, chunk: int = 16384) -> float:
    """Relative residual energy after nearest-prototype substitution."""
    Cn = (C * C).sum(1)
    num = 0.0
    den = float((X * X).sum())
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]
        a = (Cn.unsqueeze(0) - 2.0 * (Xc @ C.T)).argmin(1)
        num += float((Xc - C[a]).pow(2).sum())
    return num / max(den, 1e-30)


# ---------------------------------------------------------------- PCA / low-rank

@torch.no_grad()
def pca_spectrum(X: torch.Tensor, center: bool = False) -> torch.Tensor:
    """Eigenvalues (descending) of the d x d second-moment matrix of tiles X:(N,d)."""
    Xf = X.float()
    if center:
        Xf = Xf - Xf.mean(0, keepdim=True)
    Cov = (Xf.T @ Xf) / Xf.shape[0]
    ev = torch.linalg.eigvalsh(Cov.double())
    return ev.flip(0).clamp_min(0).float()


def rank_for_energy(ev: torch.Tensor, frac: float) -> int:
    c = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-30)
    return int((c < frac).sum().item()) + 1


# ---------------------------------------------------------------- the reported scalar

def net_gain_bits(rho: float, K: int, d: int, extra_bits_per_tile: float = 0.0) -> float:
    """bits/weight saved by the prototype term, net of what the code costs."""
    rate_spent = (math.log2(max(K, 1)) + extra_bits_per_tile) / d
    rate_saved = 0.5 * math.log2(1.0 / max(rho, 1e-12))
    return rate_saved - rate_spent


def _sample_tiles(W: torch.Tensor, th: int, tw: int, max_tiles: int, seed: int) -> torch.Tensor:
    T = to_tiles(W, th, tw).float()
    if T.shape[0] > max_tiles:
        g = torch.Generator(device=T.device).manual_seed(seed)
        T = T[torch.randperm(T.shape[0], generator=g, device=T.device)[:max_tiles]]
    return T.contiguous()


def collect_tiles(mats, th: int, tw: int, variant: str = "real", seed: int = 0,
                  max_tiles: int = 400_000) -> torch.Tensor:
    """Pool tiles from several matrices (the realistic shared-codebook setting).

    The null model is applied per matrix BEFORE tiling, so `shuffle` destroys structure
    within each matrix while leaving the pooled set the same size and marginal.
    """
    per = max(max_tiles // max(len(mats), 1), 1)
    parts = []
    for i, W in enumerate(mats):
        Wv = make_null(W, variant, seed=seed + 17 * i)
        parts.append(_sample_tiles(Wv, th, tw, per, seed + i))
    return torch.cat(parts, 0).contiguous()


def _split(T: torch.Tensor, holdout: float, seed: int, min_train: int = 0):
    n = T.shape[0]
    g = torch.Generator(device=T.device).manual_seed(seed + 991)
    perm = torch.randperm(n, generator=g, device=T.device)
    n_tr = max(int(n * (1.0 - holdout)), min(n, min_train))
    Ttr, Tte = T[perm[:n_tr]].contiguous(), T[perm[n_tr:]].contiguous()
    if Tte.shape[0] < 256:
        Tte = Ttr
    return Ttr, Tte


def codebook_bpw(K: int, d: int, cb_bits: float, n_weights_served: int) -> float:
    """Amortized cost of storing the codebook, in bits per served weight."""
    return K * d * cb_bits / max(n_weights_served, 1)


@torch.no_grad()
def probe_vq_tiles(T: torch.Tensor, K: int, d_hint: int | None = None, seed: int = 0,
                   iters: int = 25, normalized: bool = False, scale_bits: float = 8.0,
                   holdout: float = 0.5, cb_bits: float = 16.0,
                   n_weights_served: int | None = None, **tag) -> Dict:
    """Fit a k-means codebook on TRAIN tiles, measure residual energy on HELD-OUT tiles.

    The held-out split matters: with K comparable to the tile count, in-sample residual
    energy is memorisation and would fake a structure signal. `net_gain` always uses
    the held-out rho.
    """
    d = d_hint or T.shape[1]
    extra = 0.0
    if normalized:
        T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
        extra = scale_bits                      # per-tile scale must be paid for
    Ttr, Tte = _split(T, holdout, seed, min_train=K)
    C = kmeans(Ttr, K, iters=iters, seed=seed)
    rho_tr = residual_energy(Ttr, C)
    rho = residual_energy(Tte, C)
    rate_spent = (math.log2(K) + extra) / d
    rate_saved = 0.5 * math.log2(1.0 / max(rho, 1e-12))
    served = n_weights_served if n_weights_served is not None else T.numel()
    cb = codebook_bpw(K, d, cb_bits, served)
    out = dict(probe="vq", d=d, K=K, normalized=normalized, rho=rho, rho_train=rho_tr,
               rate_spent=rate_spent, rate_saved=rate_saved,
               net_gain=rate_saved - rate_spent, n_tiles=int(T.shape[0]),
               codebook_bpw=cb, net_gain_with_cb=rate_saved - rate_spent - cb)
    out.update(tag)
    return out


@torch.no_grad()
def probe_pca_tiles(T: torch.Tensor, r: int, coef_bits: float = 8.0, seed: int = 0,
                    holdout: float = 0.5, **tag) -> Dict:
    """Keep r principal components per tile; coefficients cost `coef_bits` each.

    Basis estimated on TRAIN tiles, residual energy measured on HELD-OUT tiles.
    """
    d = T.shape[1]
    Ttr, Tte = _split(T, holdout, seed)
    Cov = (Ttr.T @ Ttr).double() / Ttr.shape[0]
    evals, evecs = torch.linalg.eigh(Cov)
    order = torch.argsort(evals, descending=True)
    ev = evals[order].clamp_min(0).float()
    U = evecs[:, order][:, :r].float()                   # (d, r)
    proj = Tte @ U
    rho = float((Tte.pow(2).sum() - proj.pow(2).sum()) / Tte.pow(2).sum().clamp_min(1e-30))
    rho = max(rho, 1e-12)
    rate_spent = r * coef_bits / d
    rate_saved = 0.5 * math.log2(1.0 / rho)
    out = dict(probe="pca", d=d, r=r, coef_bits=coef_bits, rho=rho, rate_spent=rate_spent,
               rate_saved=rate_saved, net_gain=rate_saved - rate_spent,
               ev_top1_frac=float(ev[0] / ev.sum()),
               r90=rank_for_energy(ev, 0.90), r99=rank_for_energy(ev, 0.99),
               n_tiles=int(T.shape[0]))
    out.update(tag)
    return out


@torch.no_grad()
def vq_probe(W: torch.Tensor, th: int, tw: int, K: int, variant: str = "real",
             seed: int = 0, max_tiles: int = 400_000, **kw) -> Dict:
    """Convenience wrapper: single matrix -> pooled-tile VQ probe."""
    T = collect_tiles([W], th, tw, variant=variant, seed=seed, max_tiles=max_tiles)
    return probe_vq_tiles(T, K, seed=seed, n_weights_served=W.numel(),
                          variant=variant, th=th, tw=tw, **kw)


@torch.no_grad()
def pca_probe(W: torch.Tensor, th: int, tw: int, r: int, variant: str = "real",
              seed: int = 0, max_tiles: int = 400_000, **kw) -> Dict:
    T = collect_tiles([W], th, tw, variant=variant, seed=seed, max_tiles=max_tiles)
    return probe_pca_tiles(T, r, seed=seed, variant=variant, th=th, tw=tw, **kw)


@torch.no_grad()
def nn_distance_stats(W: torch.Tensor, th: int, tw: int, variant: str = "real",
                      n_query: int = 4096, n_cand: int = 65536, seed: int = 0,
                      normalized: bool = True) -> Dict:
    """Nearest-neighbour distance among tiles (sampled, NOT all-pairs).

    Reports E[min_j ||x_i - x_j||^2] / E[||x_i||^2]; for i.i.d. high-dim data this is close
    to 2 (for unit-norm tiles) because everything is nearly orthogonal.
    """
    Wv = make_null(W, variant, seed=seed)
    T = _sample_tiles(Wv, th, tw, n_cand + n_query, seed)
    if normalized:
        T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q, c = T[:n_query], T[n_query:]
    if c.shape[0] == 0:
        return dict(probe="nn", variant=variant, th=th, tw=tw, nn_rel=float("nan"))
    Cn = (c * c).sum(1)
    best = torch.full((q.shape[0],), float("inf"), device=T.device)
    for i in range(0, c.shape[0], 16384):
        cc = c[i:i + 16384]
        dist = (q * q).sum(1, keepdim=True) + (cc * cc).sum(1).unsqueeze(0) - 2.0 * (q @ cc.T)
        best = torch.minimum(best, dist.min(1).values)
    return dict(probe="nn", variant=variant, th=th, tw=tw,
                nn_rel=float(best.mean() / (q * q).sum(1).mean()),
                cos_max=float(1.0 - best.mean() / 2.0) if normalized else float("nan"))
