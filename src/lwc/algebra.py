"""Kill criterion R2 -- does composition actually add expressiveness?

Two independent measurements per transform family:

1. **closure residual.** Fit a *single* member of the family to the action of a depth-2
   composition. If the family is a group closed under composition, the fit is exact
   (residual ~ 0) and depth is algebraically redundant: `T_b(T_a(B))` is just `T_c(B)`.

2. **distinct-operator growth.** A depth-d path costs `d*log2(K)` bits, which is only honest
   if the `K^d` paths reach `K^d` *distinct* operators. We hash each composed operator by its
   action on a random probe and count how many survive. The ratio

       log2(distinct operators) / (d * log2 K)

   is the fraction of the path rate that carries information. Anything well below 1 means the
   code is paying for paths that collide.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from .codecs.recursive import FAMILIES


@torch.no_grad()
def _fit_family_one_step(family: str, B: torch.Tensor, Y: torch.Tensor,
                         th: int, tw: int) -> float:
    """Best single-step fit of the family to the map B -> Y. Returns relative residual."""
    n = B.shape[0]
    if family == "affine":
        x, y = B.reshape(-1), Y.reshape(-1)
        xm, ym = x.mean(), y.mean()
        a = ((x - xm) * (y - ym)).sum() / (x - xm).pow(2).sum().clamp_min(1e-30)
        b = ym - a * xm
        return float((y - (a * x + b)).pow(2).sum() / y.pow(2).sum().clamp_min(1e-30))
    if family in ("diag", "lowrank_add"):
        Bm = B.reshape(n, th, tw)
        Ym = Y.reshape(n, th, tw)
        if family == "diag":
            r = torch.ones(th, 1, device=B.device)
            c = torch.ones(1, tw, device=B.device)
            for _ in range(60):                       # alternating least squares
                A = Bm * c
                r = ((A * Ym).sum((0, 2)) /
                     A.pow(2).sum((0, 2)).clamp_min(1e-30)).reshape(th, 1)
                A2 = Bm * r
                c = ((A2 * Ym).sum((0, 1)) /
                     A2.pow(2).sum((0, 1)).clamp_min(1e-30)).reshape(1, tw)
            P = Bm * r * c
        else:
            R = (Ym - Bm).mean(0)                     # best shared additive term
            P = Bm + R
        return float((P - Ym).pow(2).sum() / Ym.pow(2).sum().clamp_min(1e-30))
    if family in ("signperm", "blockrot"):
        # best pair of dense linear maps Y = A B C is hard; use the strictly easier test of a
        # single dense left map and a single dense right map fitted by alternating least squares
        Bm = B.reshape(n, th, tw)
        Ym = Y.reshape(n, th, tw)
        A = torch.eye(th, device=B.device)
        C = torch.eye(tw, device=B.device)
        for _ in range(40):
            X = torch.einsum("nij,jk->nik", Bm, C).reshape(-1, th).T      # (th, n*tw)
            T = Ym.permute(0, 2, 1).reshape(-1, th).T
            A = torch.linalg.lstsq(X.T, T.T).solution.T
            X2 = torch.einsum("ij,njk->nik", A, Bm).reshape(-1, tw)
            T2 = Ym.reshape(-1, tw)
            C = torch.linalg.lstsq(X2, T2).solution
        P = torch.einsum("ij,njk,kl->nil", A, Bm, C)
        return float((P - Ym).pow(2).sum() / Ym.pow(2).sum().clamp_min(1e-30))
    raise ValueError(family)


@torch.no_grad()
def closure_residual(family: str, K: int = 16, th: int = 8, tw: int = 8,
                     n_probe: int = 512, seed: int = 0, device: str = "cuda") -> float:
    """Relative error of approximating a random depth-2 composition by one family member."""
    fam = FAMILIES[family](K, th, tw, seed=seed, device=device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    B = torch.randn(n_probe, th * tw, generator=g, device=device)
    ka = int(torch.randint(0, K, (1,), generator=g, device=device))
    kb = int(torch.randint(0, K, (1,), generator=g, device=device))
    Y = fam.apply_all(fam.apply_all(B)[:, ka, :])[:, kb, :]
    return _fit_family_one_step(family, B, Y, th, tw)


@torch.no_grad()
def distinct_operators(family: str, K: int = 16, depth: int = 3, th: int = 8, tw: int = 8,
                       n_paths: int = 20000, n_probe: int = 3, seed: int = 0,
                       device: str = "cuda", tol_digits: int = 4) -> Dict:
    """Count distinct operators reachable by `depth` compositions, via their action on probes."""
    fam = FAMILIES[family](K, th, tw, seed=seed, device=device)
    g = torch.Generator(device=device).manual_seed(seed + 7)
    P = torch.randn(n_probe, th * tw, generator=g, device=device)
    total_paths = K ** depth
    if total_paths <= n_paths:
        paths = torch.cartesian_prod(*[torch.arange(K, device=device)] * depth) \
            if depth > 1 else torch.arange(K, device=device).unsqueeze(1)
        paths = paths.reshape(-1, depth)
    else:
        paths = torch.randint(0, K, (n_paths, depth), generator=g, device=device)
    sigs = []
    for i in range(0, paths.shape[0], 2048):
        chunk = paths[i:i + 2048]
        X = P.unsqueeze(0).expand(chunk.shape[0], -1, -1).reshape(-1, th * tw)
        for lvl in range(depth):
            allk = fam.apply_all(X)                              # (m, K, d)
            sel = chunk[:, lvl].repeat_interleave(n_probe)
            X = allk[torch.arange(X.shape[0], device=device), sel]
        sigs.append(X.reshape(chunk.shape[0], -1))
    S = torch.cat(sigs, 0)
    S = S / S.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q = torch.round(S * 10 ** tol_digits).to(torch.int64)
    uniq = torch.unique(q, dim=0).shape[0]
    sampled = paths.shape[0]
    return dict(family=family, K=K, depth=depth, paths_evaluated=sampled,
                distinct_operators=uniq,
                distinct_frac=uniq / sampled,
                path_rate_bits=depth * math.log2(K),
                informative_rate_bits=math.log2(max(uniq, 1)) if total_paths <= n_paths
                else depth * math.log2(K) * (uniq / sampled),
                exhaustive=total_paths <= n_paths)


def analyze_all(families: List[str], K: int = 16, depths=(1, 2, 3), th: int = 8, tw: int = 8,
                device: str = "cuda", seed: int = 0) -> List[Dict]:
    out = []
    for f in families:
        try:
            cr = closure_residual(f, K=K, th=th, tw=tw, seed=seed, device=device)
        except Exception as e:                        # pragma: no cover - diagnostic path
            cr = float("nan")
            print(f"[warn] closure_residual({f}) failed: {type(e).__name__}: {e}")
        for d in depths:
            rec = distinct_operators(f, K=K, depth=d, th=th, tw=tw, seed=seed, device=device)
            rec["closure_residual_depth2"] = cr
            rec["closed_under_composition"] = bool(cr < 1e-6) if cr == cr else None
            out.append(rec)
    return out
