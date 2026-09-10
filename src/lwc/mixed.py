"""Mixed-precision scalar quantization driven by an explicit bit map, and the oracle
allocator that produces those maps.

This is the machinery for the *layout* question. Nothing here tries to represent weight
values better -- the representation is the same asymmetric min-max group-wise scalar
quantizer used in the final diagnostic. The only new degree of freedom is **where the bits
go**.

Storage model
-------------
A group of `group` consecutive input channels of one output row stores `lo` and `hi`
(2 x 16 bits, exactly as before). A weight with `b` bits then uses step
`(hi - lo) / (2^b - 1)`; the step is *derived* from the group parameters and the weight's own
bit width, so a mixed-precision group costs no extra scale storage. `b = 16` keeps the weight
verbatim (the outlier class), `b = 0` reconstructs the group midpoint.

What the oracle is allowed to do
--------------------------------
`allocate_lagrangian` minimises total weighted squared error subject to a total bit budget,
with NO charge for describing the resulting map. That is the point: it is an upper bound on
what any layout, however clever, could buy. `unit` controls the granularity of the decision
(per weight, per group, per column, per row, per tile), which is how the "arbitrary oracle vs
channel-only oracle" comparison is made.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .gptq import Quantizer, guard_alloc

BIT_OPTIONS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 16)


# ---------------------------------------------------------------- the quantizer


class MixedScalarQuantizer(Quantizer):
    """Asymmetric min-max group-wise RTN with a per-weight bit width.

    `bits_w` is an (out, in) integer tensor. Setting it to a constant reproduces
    `ScalarQuantizer` exactly, which is checked in the runner.
    """

    atom = 1

    def __init__(self, bits_w: torch.Tensor, group: int = 128, meta_bits: int = 16):
        self.bits_w = bits_w
        self.group = group
        self.meta_bits = meta_bits
        self.lo: Optional[torch.Tensor] = None
        self.rng: Optional[torch.Tensor] = None

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        lo = W_group.amin(dim=1, keepdim=True)
        hi = W_group.amax(dim=1, keepdim=True)
        self.lo = lo
        self.rng = (hi - lo).clamp_min(1e-12)

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        b = self.bits_w[:, col].to(w.dtype).unsqueeze(1)          # (out, 1)
        qmax = (2.0 ** b - 1.0)
        s = self.rng / qmax.clamp_min(1.0)
        q = torch.clamp(torch.round((w - self.lo) / s.clamp_min(1e-12)),
                        torch.zeros_like(qmax), qmax)
        out = q * s + self.lo
        out = torch.where(b >= 16, w, out)                        # verbatim / outlier class
        out = torch.where(b <= 0, self.lo + 0.5 * self.rng, out)  # 0 bits -> group midpoint
        return out

    # ------------------------------------------------------------ honest accounting
    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        bw = self.bits_w
        codes = float(bw.sum().item())
        ng = (in_f + self.group - 1) // self.group
        blocks = bw[:, :ng * self.group].reshape(out_f, ng, self.group)
        needs_scale = (blocks < 16).any(dim=2)                    # a pure-fp16 group needs none
        return {"codes": codes,
                "scales": float(needs_scale.sum().item()) * 2 * self.meta_bits}


# ---------------------------------------------------------------- distortion tables


@torch.no_grad()
def quant_at_bits(W: torch.Tensor, b: int, group: int) -> torch.Tensor:
    """Uniform group-wise reconstruction at `b` bits -- the same map the codec would use."""
    out_f, in_f = W.shape
    ng = in_f // group
    Wg = W[:, :ng * group].reshape(out_f, ng, group)
    lo = Wg.amin(-1, keepdim=True)
    hi = Wg.amax(-1, keepdim=True)
    if b >= 16:
        return W.clone()
    if b <= 0:
        R = (0.5 * (lo + hi)).expand_as(Wg)
    else:
        qmax = 2 ** b - 1
        s = ((hi - lo) / qmax).clamp_min(1e-12)
        R = torch.clamp(torch.round((Wg - lo) / s), 0, qmax) * s + lo
    return R.reshape(out_f, ng * group).contiguous()


@torch.no_grad()
def per_weight_distortion(W: torch.Tensor, colw: torch.Tensor, group: int,
                          bit_options: Sequence[int] = BIT_OPTIONS) -> torch.Tensor:
    """D[b, i, j] = colw[j] * (W_ij - Q_b(W_ij))^2 -- sensitivity map S5, per bit width.

    `colw` is the per-input-channel cost multiplier (diag(H), or 1/diag(H^-1)).
    """
    out_f, in_f = W.shape
    guard_alloc((len(bit_options), out_f, in_f), torch.float32, "distortion table",
                limit_mib=768.0)
    D = torch.empty(len(bit_options), out_f, in_f, device=W.device, dtype=torch.float32)
    cw = colw.reshape(1, -1).float()
    for k, b in enumerate(bit_options):
        D[k] = (W - quant_at_bits(W, b, group)).pow(2) * cw
    return D


# ---------------------------------------------------------------- unit grids


def unit_grid(shape: Tuple[int, int], unit: str, group: int, tile_rows: int = 128):
    """Returns (reduce_fn, expand_fn, n_units) for one allocation granularity."""
    out_f, in_f = shape
    ng = in_f // group

    if unit == "weight":
        return (lambda D: D.reshape(D.shape[0], -1),
                lambda ch: ch.reshape(out_f, in_f),
                out_f * in_f)
    if unit == "group":                     # (row, column block) -- the natural codec group
        return (lambda D: D.reshape(D.shape[0], out_f, ng, group).sum(-1).reshape(D.shape[0], -1),
                lambda ch: ch.reshape(out_f, ng, 1).expand(out_f, ng, group).reshape(out_f, in_f),
                out_f * ng)
    if unit == "col":                       # one input channel, shared by every row
        return (lambda D: D.sum(1),
                lambda ch: ch.reshape(1, in_f).expand(out_f, in_f),
                in_f)
    if unit == "row":                       # one output channel
        return (lambda D: D.sum(2),
                lambda ch: ch.reshape(out_f, 1).expand(out_f, in_f),
                out_f)
    if unit == "colblk":                    # one column block, shared by every row
        return (lambda D: D.reshape(D.shape[0], out_f, ng, group).sum(-1).sum(1),
                lambda ch: ch.reshape(1, ng, 1).expand(out_f, ng, group).reshape(out_f, in_f),
                ng)
    if unit == "tile":                      # tile_rows x group block
        nr = out_f // tile_rows
        assert nr * tile_rows == out_f, "out=%d not divisible by %d" % (out_f, tile_rows)
        return (lambda D: D.reshape(D.shape[0], nr, tile_rows, ng, group)
                           .sum(-1).sum(2).reshape(D.shape[0], -1),
                lambda ch: ch.reshape(nr, 1, ng, 1)
                            .expand(nr, tile_rows, ng, group).reshape(out_f, in_f),
                nr * ng)
    raise ValueError(unit)


# ---------------------------------------------------------------- the allocator


@torch.no_grad()
def allocate_lagrangian(D: torch.Tensor, weights_per_unit: torch.Tensor,
                        bit_options: Sequence[int], target_bits: float,
                        iters: int = 80) -> Tuple[torch.Tensor, float]:
    """Convex-hull-optimal bit allocation: min sum_u D[b_u, u] s.t. sum_u R[b_u, u] <= budget.

    Standard BFOS Lagrangian sweep. `D` is (B, U); `weights_per_unit` is (U,). Returns the
    chosen option index per unit and the total code bits used.
    """
    B, U = D.shape
    bo = torch.tensor([float(b) for b in bit_options], device=D.device).reshape(B, 1)
    R = bo * weights_per_unit.reshape(1, U).float()
    scale = float(D.mean().clamp_min(1e-30))
    Dn = D / scale                                      # keep lambda near O(1)

    lo, hi = 1e-14, 1e14
    best = None
    for _ in range(iters):
        lam = math.sqrt(lo * hi)
        ch = (Dn + lam * R).argmin(0)
        tot = float(R.gather(0, ch.unsqueeze(0)).sum())
        if tot > target_bits:
            lo = lam
        else:
            hi = lam
            best = (ch, tot)
    if best is None:                                    # even all-minimum exceeds the budget
        ch = torch.zeros(U, dtype=torch.long, device=D.device)
        best = (ch, float(R.gather(0, ch.unsqueeze(0)).sum()))
    return best


@torch.no_grad()
def oracle_bits_map(D_w: torch.Tensor, shape: Tuple[int, int], unit: str, group: int,
                    target_bits: float, bit_options: Sequence[int] = BIT_OPTIONS,
                    tile_rows: int = 128) -> torch.Tensor:
    """Per-weight bit map from an oracle allocation at `unit` granularity."""
    reduce_fn, expand_fn, n_units = unit_grid(shape, unit, group, tile_rows)
    D_u = reduce_fn(D_w)
    wpu = torch.full((n_units,), float(shape[0] * shape[1]) / n_units, device=D_w.device)
    ch, _ = allocate_lagrangian(D_u, wpu, bit_options, target_bits)
    bo = torch.tensor([int(b) for b in bit_options], device=D_w.device, dtype=torch.int16)
    return expand_fn(bo[ch]).contiguous()
