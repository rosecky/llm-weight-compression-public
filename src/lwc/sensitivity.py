"""Sensitivity maps: several cheap estimates of "how expensive is it to damage W_ij", and
the machinery to check which of them actually predicts functional damage.

The maps
--------
    S1  |W_ij|                       magnitude, the naive baseline
    S2  W_ij^2                       squared magnitude
    S3  W_ij^2 * E[x_j^2]            activation-weighted (AWQ-style)
    S4  H_jj                         pure cost multiplier per input channel; does not depend
                                     on W at all, so it says a channel is expensive to damage
                                     regardless of what is stored there
    S5  H_jj * (W_ij - Q(W_ij))^2    the actual local quantization damage under a baseline
                                     quantizer -- the diagonal second-order approximation
    S5c same, with the GPTQ-compensated error
    S7  W_ij^2 / [H^-1]_jj           OBS saliency: the cost of removing W_ij once every other
                                     weight is allowed to move to compensate

S1-S4 are *predictions*; S5 is the quantity the oracle allocator actually minimises. They are
not interchangeable, and section 1 of the report is about which ones survive contact with
measured damage.

Ground truth
------------
`exact_unit_damage` computes, for every unit of a partition, the exact layer-output error
||E_u X||^2 that perturbing only that unit produces -- exact because it uses the full Hessian,
not its diagonal. Comparing it with the diagonal prediction measures precisely what the
off-diagonal terms are worth. `stratified_units` then picks a spread of units for the much
more expensive end-to-end check (perturb one unit, measure the model's NLL).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------- the maps


@torch.no_grad()
def build_maps(W: torch.Tensor, H: torch.Tensor, Wq_naive: Optional[torch.Tensor] = None,
               Wq_gptq: Optional[torch.Tensor] = None,
               percdamp: float = 0.01) -> Dict[str, torch.Tensor]:
    """All maps for one matrix. Each is (out, in) and non-negative."""
    hdiag = torch.diagonal(H).clamp_min(0.0)
    maps: Dict[str, torch.Tensor] = {
        "S1_absw": W.abs(),
        "S2_w2": W.pow(2),
        "S3_act": W.pow(2) * hdiag.reshape(1, -1),
        "S4_hdiag": hdiag.reshape(1, -1).expand_as(W).contiguous(),
    }
    if Wq_naive is not None:
        maps["S5_damage"] = (W - Wq_naive).pow(2) * hdiag.reshape(1, -1)
    if Wq_gptq is not None:
        maps["S5c_damage_gptq"] = (W - Wq_gptq).pow(2) * hdiag.reshape(1, -1)
    n = H.shape[0]
    Hd = H.clone().float()
    dead = torch.diagonal(Hd) == 0
    Hd[dead, dead] = 1.0
    Hd[range(n), range(n)] += percdamp * torch.diagonal(Hd).mean()
    hinv_diag = torch.diagonal(torch.cholesky_inverse(torch.linalg.cholesky(Hd)))
    maps["S7_obs"] = W.pow(2) / hinv_diag.clamp_min(1e-30).reshape(1, -1)
    return maps


# ---------------------------------------------------------------- unit partitions


def unit_index(shape: Tuple[int, int], unit: str, group: int = 128, tile_rows: int = 128):
    """Returns (n_units, unit_id_map (out,in) int64) for one partition."""
    out_f, in_f = shape
    ng = in_f // group
    dev = "cpu"
    r = torch.arange(out_f).reshape(-1, 1)
    c = torch.arange(in_f).reshape(1, -1)
    if unit == "group":
        idx = r * ng + (c // group)
        return out_f * ng, idx.expand(out_f, in_f)
    if unit == "row":
        return out_f, r.expand(out_f, in_f)
    if unit == "col":
        return in_f, c.expand(out_f, in_f)
    if unit == "colblk":
        return ng, (c // group).expand(out_f, in_f)
    if unit == "tile":
        nr = out_f // tile_rows
        return nr * ng, ((r // tile_rows) * ng + (c // group)).expand(out_f, in_f)
    raise ValueError(unit)


@torch.no_grad()
def reduce_by_unit(M: torch.Tensor, unit: str, group: int = 128,
                   tile_rows: int = 128, how: str = "sum") -> torch.Tensor:
    """Aggregate a map over the units of a partition. Returns (n_units,)."""
    out_f, in_f = M.shape
    ng = in_f // group
    if unit == "group":
        return M.reshape(out_f, ng, group).sum(-1).reshape(-1) if how == "sum" \
            else M.reshape(out_f, ng, group).amax(-1).reshape(-1)
    if unit == "row":
        return M.sum(1) if how == "sum" else M.amax(1)
    if unit == "col":
        return M.sum(0) if how == "sum" else M.amax(0)
    if unit == "colblk":
        B = M.reshape(out_f, ng, group)
        return B.sum(-1).sum(0) if how == "sum" else B.amax(-1).amax(0)
    if unit == "tile":
        nr = out_f // tile_rows
        B = M.reshape(nr, tile_rows, ng, group)
        return B.sum(-1).sum(1).reshape(-1) if how == "sum" else B.amax(-1).amax(1).reshape(-1)
    raise ValueError(unit)


# ---------------------------------------------------------------- exact damage


@torch.no_grad()
def exact_unit_damage(E: torch.Tensor, H: torch.Tensor, unit: str, group: int = 128,
                      tile_rows: int = 128) -> torch.Tensor:
    """Exact ||E_u X||^2 for every unit u, i.e. sum_r e_r^T H e_r restricted to u's columns.

    This is what would happen if *only* unit u were perturbed. It uses the full Hessian, so
    the gap to the diagonal prediction is exactly the off-diagonal contribution.
    """
    out_f, in_f = E.shape
    ng = in_f // group
    if unit in ("group", "colblk", "tile"):
        vals = []
        for gi in range(ng):
            c0, c1 = gi * group, (gi + 1) * group
            Eg = E[:, c0:c1]
            per_row = ((Eg @ H[c0:c1, c0:c1]) * Eg).sum(1)          # (out,)
            vals.append(per_row)
        V = torch.stack(vals, 1)                                    # (out, ng)
        if unit == "group":
            return V.reshape(-1)
        if unit == "colblk":
            return V.sum(0)
        nr = out_f // tile_rows
        return V.reshape(nr, tile_rows, ng).sum(1).reshape(-1)
    if unit == "row":
        return ((E @ H) * E).sum(1)
    if unit == "col":
        return (E.pow(2) * torch.diagonal(H).reshape(1, -1)).sum(0)  # a column is 1-dim in H
    raise ValueError(unit)


# ---------------------------------------------------------------- agreement statistics


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks, so ties do not get an arbitrary order.

    This matters: a map that is constant along the axis being ranked (S4 = H_jj summed over a
    row is the same for every row) is *entirely* ties, and naive argsort ranking turns that
    into a large spurious correlation with whatever the argsort order happens to be.
    """
    order = torch.argsort(x)
    xs = x[order]
    r = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
    # average the positions inside each run of equal values
    newgrp = torch.ones_like(r, dtype=torch.bool)
    newgrp[1:] = xs[1:] != xs[:-1]
    gid = torch.cumsum(newgrp.long(), 0) - 1
    ng = int(gid[-1].item()) + 1
    s = torch.zeros(ng, dtype=torch.float64, device=x.device).index_add_(0, gid, r)
    c = torch.zeros(ng, dtype=torch.float64, device=x.device).index_add_(
        0, gid, torch.ones_like(r))
    avg = (s / c)[gid]
    out = torch.empty_like(r)
    out[order] = avg
    return out


@torch.no_grad()
def agreement(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-30) -> Dict[str, float]:
    """How well does a cheap map predict measured damage? Rank and log-linear agreement,
    plus the practically decisive quantity: overlap of the top-1% sets."""
    p = pred.double().reshape(-1)
    t = truth.double().reshape(-1)
    keep = torch.isfinite(p) & torch.isfinite(t) & (t > 0)
    p, t = p[keep], t[keep]
    if p.numel() < 8:
        return {"spearman": float("nan"), "log_pearson": float("nan"), "top1_overlap": float("nan")}
    rp, rt = _rank(p), _rank(t)
    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    lp, lt = torch.log(p.clamp_min(eps)), torch.log(t.clamp_min(eps))
    k = max(1, int(round(0.01 * p.numel())))
    sp = set(torch.topk(p, k).indices.tolist())
    st = set(torch.topk(t, k).indices.tolist())
    return {"spearman": corr(rp, rt), "log_pearson": corr(lp, lt),
            "top1_overlap": len(sp & st) / k}


def stratified_units(score: torch.Tensor, n: int = 24, seed: int = 0) -> List[int]:
    """Pick `n` units spread across the whole range of a score, for the expensive check."""
    order = torch.argsort(score.reshape(-1), descending=True)
    m = order.numel()
    pos = [int(round(x)) for x in torch.linspace(0, m - 1, n).tolist()]
    return [int(order[p]) for p in dict.fromkeys(pos)]
