"""Joint / non-greedy optimization of *quantization decisions* at a fixed storage format.

The question this module exists to answer is not "can we invent a better representation" --
four previous studies in this repository said no -- but the sharper one:

    given exactly the codebook, the scales, the group size and the bit width that the baseline
    already ships, is the particular integer code array that greedy GPTQ picks anywhere near
    the best one?

So every object here stores precisely what a group-wise INT-b checkpoint stores: an integer
code per weight plus one (scale, zero) pair per (output row, input group). Nothing else. An
optimizer is allowed to be arbitrarily expensive offline; it is not allowed to add a single
bit.

Three levels of optimizer, in increasing strength and cost:

`cd_refine`       exact cyclic coordinate descent on the true quadratic objective, vectorised
                  over all output rows. Each coordinate is set to its *exact* conditional
                  minimiser over the integer grid, so the objective is monotone by
                  construction. This is the "iterative revisiting GPTQ" arm (J1/J2).

`cd_subset_exact` block coordinate descent where each block of a few dozen decisions is solved
                  to optimality by a box-constrained sphere decoder -- higher-order moves (J3).

`sphere_decode`   exact closest-vector search on a small subproblem, i.e. the oracle (J6). If
                  it terminates inside its node budget the answer is *the* optimum, not a good
                  solution, which is what makes the oracle gap a measurement.

## Why the objective is a quadratic form and what `G` means

For `y = W x` with input second moment `A = E[x x^T]` the layer's own output error is
`tr(dW A dW^T)`. If we care about an error *further downstream* -- through the rest of the MLP,
the transformer block, the next two blocks, or the model's NLL -- then to first order
`dz = J dy`, so the downstream squared error is `tr(dW A dW^T G)` with `G = E[J^T J]`. That is
the only thing that changes when the optimization scope widens: `G = I` is exactly GPTQ's
scope, and a downstream `G` couples output rows that GPTQ treats as independent problems.
Random probes estimate `E[J^T J]` at one backward pass per probe (`scopeg.block_G`).

`G` is used *densely*. A truncated `diag + low rank` surrogate was tried first and is actively
harmful as an objective: the optimizer walks into whatever null space the truncation leaves, and
minimising a rank-64 surrogate of the Fisher made the true Fisher damage 32x worse than not
optimising at all. `damp_dense` adds the trust region that makes even the exact empirical metric
a well-posed objective, since an empirical `G` from N token samples is itself singular.

## Monotonicity

With `G = I` the rows of a column update are exactly independent, so a whole column can be set
to its joint conditional minimum in one vectorised step and the objective cannot increase.
With a non-diagonal `G` the rows in a column interact, so the exact change

    dD = 2 d^T M[:, j] + A_jj (d^T G d)

is evaluated for the proposed column move and, if it is not a decrease, the move is backed off
by halving the accepted row set. Every optimizer here is therefore monotone in the objective it
is given, which makes "it got worse end to end" unambiguously a statement about the objective
rather than about the search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .gptq import ScalarQuantizer, gptq


# ====================================================================== the stored state


@dataclass
class QuantState:
    """Exactly what a group-wise INT-b checkpoint holds. Nothing else is allowed."""

    codes: torch.Tensor          # (out, in) integer levels in [0, 2^bits - 1]
    scale: torch.Tensor          # (out, n_groups)
    zero: torch.Tensor           # (out, n_groups)
    bits: int
    group: int

    @property
    def qmax(self) -> int:
        return 2 ** self.bits - 1

    def expand(self) -> Tuple[torch.Tensor, torch.Tensor]:
        in_f = self.codes.shape[1]
        s = self.scale.repeat_interleave(self.group, dim=1)[:, :in_f]
        z = self.zero.repeat_interleave(self.group, dim=1)[:, :in_f]
        return s, z

    def dequant(self) -> torch.Tensor:
        s, z = self.expand()
        return self.codes.float() * s + z

    def clone(self) -> "QuantState":
        return QuantState(self.codes.clone(), self.scale.clone(), self.zero.clone(),
                          self.bits, self.group)

    def storage_bits(self, meta_bits: int = 16) -> Dict[str, float]:
        out_f, in_f = self.codes.shape
        return {"codes": float(out_f * in_f * self.bits),
                "scales": float(out_f * self.scale.shape[1] * 2 * meta_bits)}

    def n_differ(self, other: "QuantState") -> Tuple[int, float]:
        d = self.codes != other.codes
        return int(d.sum()), float(d.float().mean())


class RecordingScalarQuantizer(ScalarQuantizer):
    """`ScalarQuantizer` that also keeps the (scale, zero) it chose for each group.

    Subclassing rather than editing `gptq.py` keeps the canonical engine untouched, so the
    baseline being compared against is byte-for-byte the one every earlier experiment used.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        super().find_params(W_group, col)
        self.rec[col] = (self.scale.squeeze(1).clone(), self.zero.squeeze(1).clone())


def quantize_with_state(W: torch.Tensor, H: Optional[torch.Tensor], bits: int, group: int,
                        percdamp: float = 0.01) -> Tuple[torch.Tensor, QuantState]:
    """Run the canonical engine (naive when `H is None`, GPTQ otherwise) and recover the
    integer state it implies. `dequant()` of the result reproduces `What` exactly."""
    out_f, in_f = W.shape
    q = RecordingScalarQuantizer(bits=bits, group=group)
    What = gptq(W, H, q, percdamp=percdamp)
    n_groups = (in_f + group - 1) // group
    scale = torch.zeros(out_f, n_groups, device=W.device, dtype=torch.float32)
    zero = torch.zeros_like(scale)
    for col, (s, z) in q.rec.items():
        scale[:, col // group] = s
        zero[:, col // group] = z
    st = QuantState(torch.zeros(out_f, in_f, device=W.device, dtype=torch.int16),
                    scale, zero, bits, group)
    s_full, z_full = st.expand()
    st.codes = torch.round((What - z_full) / s_full).clamp(0, st.qmax).to(torch.int16)
    return What, st


# ====================================================================== the output metric


class OutputMetric:
    """`G` as `diag(d) + U diag(lam) U^T`, or the identity.

    `apply(X)` is `G X` for `X` of shape (out, k). The identity case short-circuits, so the
    `G = I` arm costs nothing extra and is numerically identical to the plain layer objective.
    """

    def __init__(self, diag: Optional[torch.Tensor] = None,
                 U: Optional[torch.Tensor] = None, lam: Optional[torch.Tensor] = None):
        self.diag = diag
        self.U = U
        self.lam = lam

    @property
    def is_identity(self) -> bool:
        return self.diag is None and self.U is None

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return X
        out = X * self.diag.unsqueeze(1) if self.diag is not None else torch.zeros_like(X)
        if self.U is not None:
            out = out + self.U @ (self.lam.unsqueeze(1) * (self.U.T @ X))
        return out

    def diagonal(self, n: int, device) -> torch.Tensor:
        if self.is_identity:
            return torch.ones(n, device=device)
        d = self.diag.clone() if self.diag is not None else torch.zeros(n, device=device)
        if self.U is not None:
            d = d + (self.U ** 2) @ self.lam
        return d.clamp_min(1e-20)

    def to(self, device) -> "OutputMetric":
        return OutputMetric(None if self.diag is None else self.diag.to(device),
                            None if self.U is None else self.U.to(device),
                            None if self.lam is None else self.lam.to(device))

    def rank(self) -> int:
        return 0 if self.U is None else int(self.U.shape[1])


def metric_from_dense(G: torch.Tensor, rank: int = 0) -> OutputMetric:
    """Top-`rank` eigenmodes plus whatever diagonal they leave over (`rank<=0` keeps all).

    Keeping the diagonal exact matters: it is the denominator of every coordinate update, so an
    approximation that got the diagonal wrong would move weights to the wrong levels even where
    the low-rank part is irrelevant. The residual diagonal is clamped at zero, which keeps the
    approximation PSD (a sum of two PSD terms) and the objective therefore bounded below.
    """
    G = 0.5 * (G + G.T)
    lam, U = torch.linalg.eigh(G)
    if rank <= 0 or rank >= G.shape[0]:
        return OutputMetric(torch.zeros(G.shape[0], device=G.device), U, lam.clamp_min(0.0))
    lam_r = lam[-rank:].clamp_min(0.0)
    U_r = U[:, -rank:].contiguous()
    resid = (torch.diagonal(G) - (U_r ** 2) @ lam_r).clamp_min(0.0)
    return OutputMetric(resid, U_r, lam_r)


def metric_identity() -> OutputMetric:
    return OutputMetric()


class DenseMetric:
    """`G` held densely. Exact, and the reference the cheaper forms are checked against."""

    def __init__(self, G: torch.Tensor):
        self.G = G

    @property
    def is_identity(self) -> bool:
        return False

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        return self.G @ X

    def diagonal(self, n: int, device) -> torch.Tensor:
        return torch.diagonal(self.G).clamp_min(1e-20)

    def rank(self) -> int:
        return int(self.G.shape[0])


def damp_dense(G: torch.Tensor, rel: float) -> torch.Tensor:
    """`G + rel * mean(diag G) * I` -- a trust region on the widened objective.

    An empirical `G = E[g g^T]` estimated from N token samples has rank at most N, and a
    truncated eigendecomposition has rank at most whatever was kept. Either way the metric has
    a large null space, and an optimizer handed a singular objective will move weights
    arbitrarily far along directions the objective declares free: measured here, minimising a
    rank-64 surrogate of the Fisher made the *true* Fisher damage 32x worse than not optimising
    at all. Damping is the standard cure and it is also the honest interpolation -- `rel` large
    recovers the layer's own objective, `rel -> 0` the raw downstream one -- so the scope
    question is asked along a controlled path rather than at one arbitrary point.
    """
    n = G.shape[0]
    c = float(torch.diagonal(G).mean())
    return G + (rel * c) * torch.eye(n, device=G.device, dtype=G.dtype)


# ====================================================================== objective


@torch.no_grad()
def damage(D: torch.Tensor, A: torch.Tensor, G: Optional[OutputMetric] = None,
           chunk: int = 2048) -> float:
    """tr(D A D^T G) in row chunks, so no (out, out) intermediate is ever formed."""
    if G is None or G.is_identity:
        tot = 0.0
        for i in range(0, D.shape[0], chunk):
            Dc = D[i:i + chunk]
            tot += float(((Dc @ A) * Dc).sum())
        return tot
    GD = G.apply(D)
    tot = 0.0
    for i in range(0, D.shape[0], chunk):
        tot += float(((D[i:i + chunk] @ A) * GD[i:i + chunk]).sum())
    return tot


@torch.no_grad()
def rel_damage(W: torch.Tensor, Wh: torch.Tensor, A: torch.Tensor,
               G: Optional[OutputMetric] = None) -> float:
    return damage(W - Wh, A, G) / max(damage(W, A, G), 1e-30)


# ====================================================================== coordinate descent


@torch.no_grad()
def cd_refine(W: torch.Tensor, A: torch.Tensor, st: QuantState,
              G: Optional[OutputMetric] = None, sweeps: int = 4, block: int = 128,
              order: str = "forward", seed: int = 0,
              mask: Optional[torch.Tensor] = None, backoff_steps: int = 12,
              tol: float = 1e-6, verbose: bool = False) -> Tuple[torch.Tensor, Dict]:
    """Exact cyclic coordinate descent over the integer codes; scales are held fixed.

    `mask` (out, in) bool restricts which decisions may move -- the ambiguity-restricted arm,
    where only the least confident decisions are reopened.

    Returns the refined `What` and a stats dict. `st` is modified in place.
    """
    out_f, in_f = W.shape
    dev = W.device
    ident = G is None or G.is_identity
    gdiag = torch.ones(out_f, device=dev) if ident else G.diagonal(out_f, dev)
    adiag = torch.diagonal(A).clamp_min(1e-12)
    s_full, z_full = st.expand()
    codes_f = st.codes.float()
    Dcur = W - (codes_f * s_full + z_full)

    # The update written in code units. Since `Delta = W - (c s + z)` exactly, the exact
    # conditional minimiser `Delta - M / (G_rr A_jj)` becomes, in the integer variable,
    #
    #     c_new = round(c + M / (G_rr A_jj s))        and     dDelta = -(c_new - c) s
    #
    # which removes `W`, `z` and `Delta` from the inner loop entirely: one multiply-add and a
    # round, instead of five arithmetic kernels and a strided read of three more tensors. That
    # matters because this loop runs once per input column per sweep -- about 1.5 million times
    # for a full model -- and at that size the cost is kernel launches, not arithmetic.
    inv_ds = 1.0 / (gdiag.unsqueeze(1) * adiag.unsqueeze(0) * s_full).clamp_min(1e-30)
    neg_s = -s_full

    d0 = damage(Dcur, A, G)
    hist = [d0]
    n_moves = 0
    n_backoff = 0
    n_moved_dev = torch.zeros((), device=dev, dtype=torch.long)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    for sw in range(sweeps):
        cols = torch.arange(in_f)
        if order == "backward":
            cols = cols.flip(0)
        elif order == "random":
            cols = torch.randperm(in_f, generator=gen)
        elif order == "alternate":
            cols = cols.flip(0) if sw % 2 else cols
        M = Dcur @ A                       # M = G Delta A, current for every column touched
        if not ident:
            M = G.apply(M)
        n_moved_dev.zero_()

        for b0 in range(0, in_f, block):
            sel = cols[b0:b0 + block]
            nb = sel.numel()
            # Transposed block buffers: the inner loop then reads and writes contiguous rows
            # instead of gathering strided columns out of an (out, in) tensor.
            Mb = M[:, sel].T.contiguous()
            Cb = codes_f[:, sel].T.contiguous()
            IDb = inv_ds[:, sel].T.contiguous()
            NSb = neg_s[:, sel].T.contiguous()
            Mkb = mask[:, sel].T.contiguous() if mask is not None else None
            Ab = A[sel][:, sel].contiguous()
            Ddb = torch.zeros(nb, out_f, device=dev)

            # This path is deliberately free of device syncs: a `bool(...)` or `int(...)` per
            # column costs about a millisecond, which across a full model is 1.5M columns and
            # dominates everything else. Move counts accumulate on the device and are read
            # once per sweep.
            for t in range(nb):
                cj = Cb[t]
                c_new = torch.round(cj + Mb[t] * IDb[t]).clamp_(0, st.qmax)
                if Mkb is not None:
                    c_new = torch.where(Mkb[t], c_new, cj)
                dd = (c_new - cj) * NSb[t]
                if not ident:
                    j = int(sel[t])
                    # Rows in one column are only independent when G is diagonal. Each row's
                    # proposal is its exact conditional minimum, so every individual move has
                    # `gain < 0`, but their sum need not: the cross term
                    # `A_jj sum_{r != r'} d_r G_rr' d_r'` can outweigh it. So the *exact* joint
                    # change is evaluated and the accepted set halved until it is a decrease,
                    # with a final fallback to the single best row -- which cannot fail,
                    # because a one-row move has no cross term at all. The objective is
                    # therefore monotone here exactly as it is in the identity case.
                    keep = (dd != 0) & torch.isfinite(dd)
                    gain = 2.0 * dd * Mb[t] + adiag[j] * gdiag * dd * dd
                    gain = torch.where(keep, gain, torch.zeros_like(gain))
                    if bool(keep.any()):
                        ok = False
                        for _ in range(backoff_steps):
                            d_try = torch.where(keep, dd, torch.zeros_like(dd))
                            Gd = G.apply(d_try.unsqueeze(1)).squeeze(1)
                            exact = float(2.0 * (d_try * Mb[t]).sum()
                                          + adiag[j] * (d_try * Gd).sum())
                            if exact < 0.0:
                                ok = True
                                break
                            n_backoff += 1
                            if int(keep.sum()) <= 1:
                                break
                            thr = torch.quantile(gain[keep].float(), 0.5)
                            keep = keep & (gain <= thr)
                        if not ok:
                            best_r = int(torch.argmin(gain))
                            keep = torch.zeros_like(keep)
                            if float(gain[best_r]) < 0.0:
                                keep[best_r] = True
                    dd = torch.where(keep, dd, torch.zeros_like(dd))
                    c_new = torch.where(keep, c_new, cj)
                Cb[t] = c_new
                Ddb[t] = dd
                n_moved_dev += (dd != 0).sum()
                upd = dd if ident else G.apply(dd.unsqueeze(1)).squeeze(1)
                Mb += Ab[t].unsqueeze(1) * upd.unsqueeze(0)

            dd_blk = Ddb.T.contiguous()
            codes_f[:, sel] = Cb.T
            Dcur[:, sel] += dd_blk
            M += (dd_blk if ident else G.apply(dd_blk)) @ A[sel, :]
            M[:, sel] = Mb.T

        moved_this_sweep = int(n_moved_dev)
        n_moves += moved_this_sweep
        d = damage(Dcur, A, G)
        hist.append(d)
        if verbose:
            print("    sweep %d: damage %.6e (%.4f of start), %d moves"
                  % (sw, d, d / max(d0, 1e-30), moved_this_sweep))
        if moved_this_sweep == 0 or (hist[-2] - d) <= tol * max(abs(d0), 1e-30):
            break

    st.codes = codes_f.to(torch.int16)
    return codes_f * s_full + z_full, dict(
        damage_start=d0, damage_end=hist[-1], history=hist, sweeps_run=len(hist) - 1,
        moves=n_moves, backoffs=n_backoff, frac_moved=n_moves / float(out_f * in_f))


@torch.no_grad()
def refit_scales(W: torch.Tensor, A: torch.Tensor, st: QuantState) -> None:
    """Weighted least-squares refit of (scale, zero) per (row, group) with codes fixed.

    The storage format is unchanged -- a scale and a zero are stored either way -- so this
    stays inside the rules of the critical experiment. Weights are `A_jj`, the part of the
    objective a per-group closed form can see.
    """
    in_f = W.shape[1]
    a = torch.diagonal(A).clamp_min(1e-12)
    for gi in range(st.scale.shape[1]):
        j0, j1 = gi * st.group, min((gi + 1) * st.group, in_f)
        c = st.codes[:, j0:j1].float()
        w = W[:, j0:j1]
        aw = a[j0:j1].unsqueeze(0)
        s_a = aw.sum()
        s_c = (aw * c).sum(1)
        s_cc = (aw * c * c).sum(1)
        s_w = (aw * w).sum(1)
        s_cw = (aw * c * w).sum(1)
        det = s_cc * s_a - s_c * s_c
        ok = det.abs() > 1e-20
        scale = (s_cw * s_a - s_c * s_w) / torch.where(ok, det, torch.ones_like(det))
        zero = (s_cc * s_w - s_c * s_cw) / torch.where(ok, det, torch.ones_like(det))
        good = ok & (scale.abs() > 1e-12)
        st.scale[:, gi] = torch.where(good, scale, st.scale[:, gi])
        st.zero[:, gi] = torch.where(good, zero, st.zero[:, gi])


@torch.no_grad()
def additive_change(W: torch.Tensor, A: torch.Tensor, ref: QuantState, new: QuantState,
                    G: Optional[OutputMetric] = None) -> Dict[str, float]:
    """Are the decisions an optimizer changed cooperative, or independently good?

    Section 9.1 asks for `I(a, b) = D(a+b) - D(a) - D(b)` over quantization changes. For a
    quadratic objective this does not have to be sampled: the interaction of two single-code
    moves is exactly `2 d_a d_b G_{r_a r_b} A_{j_a j_b}`, and there are *no* third- or
    higher-order terms at all -- the objective is quadratic, so its Taylor expansion in the
    changes stops at two. So the whole question reduces to comparing two numbers: the damage
    change actually achieved, and the sum of what each changed decision would have achieved on
    its own from the same starting point.

        additive / actual >> 1   the changes are cooperative: each one is only good because
                                 the others are made too
        additive / actual ~ 1    the changes are essentially independent improvements

    Both are computed in closed form from `M = G dW A` at the reference state.
    """
    s_r, z_r = ref.expand()
    s_n, z_n = new.expand()
    D0 = W - (ref.codes.float() * s_r + z_r)
    D1 = W - (new.codes.float() * s_n + z_n)
    dd = D1 - D0
    M0 = D0 @ A
    if G is not None and not G.is_identity:
        M0 = G.apply(M0)
    gdiag = (torch.ones(W.shape[0], device=W.device) if G is None or G.is_identity
             else G.diagonal(W.shape[0], W.device))
    adiag = torch.diagonal(A)
    add = float(2.0 * (dd * M0).sum()
                + ((dd * dd) * (gdiag.unsqueeze(1) * adiag.unsqueeze(0))).sum())
    d0 = damage(D0, A, G)
    d1 = damage(D1, A, G)
    actual = d1 - d0
    return dict(actual_change=actual, additive_change=add,
                interaction=actual - add,
                additive_over_actual=(add / actual) if actual < 0 else None)


@torch.no_grad()
def margins(W: torch.Tensor, A: torch.Tensor, st: QuantState,
            G: Optional[OutputMetric] = None) -> torch.Tensor:
    """Functional rounding ambiguity: the objective cost of the best alternative level.

    With everything else fixed the objective is a parabola in one code, so moving one level
    costs `2 dd M[r,j] + A_jj G_rr dd^2` with `dd = -+ s`. The margin is the cheaper of the two
    directions; a small margin means the decision is nearly a coin flip. At a coordinate-wise
    optimum every margin is >= 0.
    """
    dev = W.device
    out_f, in_f = W.shape
    ident = G is None or G.is_identity
    gdiag = torch.ones(out_f, device=dev) if ident else G.diagonal(out_f, dev)
    adiag = torch.diagonal(A).clamp_min(1e-12)
    s_full, z_full = st.expand()
    Dcur = W - (st.codes.float() * s_full + z_full)
    M = Dcur @ A
    if not ident:
        M = G.apply(M)
    q = adiag.unsqueeze(0) * gdiag.unsqueeze(1)
    best = torch.full_like(Dcur, float("inf"))
    codes = st.codes.float()
    for sgn in (-1.0, 1.0):                 # code moves by sgn, so Delta moves by -sgn * s
        newc = codes + sgn
        dd = -sgn * s_full
        cost = 2.0 * dd * M + q * dd * dd
        ok = (newc >= 0) & (newc <= st.qmax)
        best = torch.minimum(best, torch.where(ok, cost, torch.full_like(cost, float("inf"))))
    return best


# ====================================================================== exact small solver


def _cvp_cost(R: np.ndarray, x: np.ndarray, c: np.ndarray) -> float:
    r = R @ (x - c)
    return float(r @ r)


def sphere_decode(R: torch.Tensor, x: torch.Tensor, lo: np.ndarray, hi: np.ndarray,
                  c_init: torch.Tensor, node_budget: int = 400_000
                  ) -> Tuple[np.ndarray, float, int, bool]:
    """Box-constrained closest-vector search: minimise ||R (x - c)||^2 over integer c in box.

    Depth-first Schnorr-Euchner enumeration with the incumbent from `c_init` as the initial
    radius, so the search can only improve on whatever the caller already had. Returns
    (best c, best cost, nodes visited, proved_optimal). `proved_optimal` is False only if the
    node budget ran out.
    """
    Rn = R.double().cpu().numpy() if torch.is_tensor(R) else np.asarray(R, dtype=np.float64)
    xn = x.double().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)
    n = Rn.shape[0]
    lo = lo.astype(np.int64)
    hi = hi.astype(np.int64)

    c = (c_init.cpu().numpy() if torch.is_tensor(c_init)
         else np.asarray(c_init)).astype(np.float64).copy()
    best_c = c.copy()
    best = _cvp_cost(Rn, xn, c)

    acc = np.zeros(n)              # acc[k] = sum_{j>k} R[k,j] (x_j - c_j) for the current path
    cost_above = np.zeros(n + 1)
    centre = np.zeros(n)
    base = np.zeros(n, dtype=np.int64)
    stepc = np.zeros(n, dtype=np.int64)
    nemit = np.zeros(n, dtype=np.int64)
    ndir = np.ones(n, dtype=np.int64)
    width = (hi - lo + 1)

    def open_level(k: int) -> None:
        centre[k] = xn[k] + acc[k] / Rn[k, k]
        b = int(np.rint(centre[k]))
        base[k] = min(max(b, int(lo[k])), int(hi[k]))
        ndir[k] = 1 if centre[k] >= base[k] else -1
        stepc[k] = 0
        nemit[k] = 0

    def next_cand(k: int) -> bool:
        while nemit[k] < width[k]:
            s = int(stepc[k])
            stepc[k] += 1
            d = (s + 1) // 2
            sgn = ndir[k] if (s % 2 == 1) else -ndir[k]
            cand = int(base[k]) + (int(sgn) * d if s > 0 else 0)
            if lo[k] <= cand <= hi[k]:
                nemit[k] += 1
                c[k] = cand
                return True
        return False

    k = n - 1
    open_level(k)
    if not next_cand(k):
        return best_c, best, 0, True
    nodes = 0
    proved = True

    while True:
        nodes += 1
        if nodes > node_budget:
            proved = False
            break
        term = Rn[k, k] * (centre[k] - c[k])
        tc = cost_above[k] + term * term
        if tc < best:
            if k == 0:
                best = tc
                best_c = c.copy()
            else:
                acc[:k] += Rn[:k, k] * (xn[k] - c[k])
                cost_above[k - 1] = tc
                k -= 1
                open_level(k)
                if next_cand(k):
                    continue
                # level k has no candidate at all (empty box); undo and go up
                k += 1
                acc[:k] -= Rn[:k, k] * (xn[k] - c[k])
        while not next_cand(k):
            k += 1
            if k == n:
                return best_c, best, nodes, proved
            acc[:k] -= Rn[:k, k] * (xn[k] - c[k])

    return best_c, best, nodes, proved


def _np(*ts):
    """Row-level search is inherently sequential and scalar: every iteration is an O(n) update
    guarded by a comparison. On the GPU each of those is a kernel launch plus a device sync,
    which measured 1.2 ms per coordinate -- 68 s for one 896-wide row. In float64 numpy the same
    update is a few microseconds, so these solvers run on the CPU and only the interface is
    torch."""
    return [t.detach().double().cpu().numpy() if torch.is_tensor(t) else t for t in ts]


@torch.no_grad()
def subproblem_form(w: torch.Tensor, A: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
                    c: torch.Tensor, idx: torch.Tensor, sort: bool = True):
    """Reduce one row's problem, restricted to coordinates `idx`, to a plain CVP in code units.

    `D(c) = y^T B y` with `y = t - c`, `t = (w - z)/s`, `B = diag(s) A diag(s)`. Freezing the
    complement leaves `y_S^T B_SS y_S + 2 y_S^T b + const` with `b = B_{S,Sbar} y_Sbar`;
    completing the square gives the CVP target `x = t_S + B_SS^{-1} b`.

    `sort` orders the free coordinates by increasing conditional precision, so the sphere
    decoder -- which walks the Cholesky factor from the last coordinate backwards -- commits to
    the best-determined decisions first. Without it, enumeration on this geometry rarely proved
    optimality inside any sane node budget; with it, most subproblems close in a few thousand
    nodes. Returns the (possibly reordered) index vector alongside the CVP.
    """
    wn, An, sn, zn, cn = _np(w, A, s, z, c)
    ix = idx.cpu().numpy().astype(np.int64)
    t = (wn - zn) / sn
    y = t - cn
    yc = y.copy()
    yc[ix] = 0.0
    Bfull_rows = An[ix] * sn[ix][:, None] * sn[None, :]
    b = Bfull_rows @ yc
    Bss = An[np.ix_(ix, ix)] * sn[ix][:, None] * sn[ix][None, :]
    Bss = 0.5 * (Bss + Bss.T)
    if sort:
        order = np.argsort(np.diagonal(Bss))
        ix = ix[order]
        b = b[order]
        Bss = Bss[np.ix_(order, order)]
    ev = np.linalg.eigvalsh(Bss)
    if ev.min() <= 1e-12:
        Bss = Bss + (1e-12 - ev.min() + 1e-12) * np.eye(Bss.shape[0])
    ystar = np.linalg.solve(Bss, -b)
    x = t[ix] - ystar
    R = np.linalg.cholesky(Bss).T          # upper factor: R^T R = Bss
    return R, x, torch.tensor(ix, dtype=torch.long, device=w.device)


@torch.no_grad()
def row_damage(w: torch.Tensor, A: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
               c: torch.Tensor) -> float:
    d = w - (c.float() * s + z)
    return float(d @ (A @ d))


def cd_row(w: torch.Tensor, A: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
           c: torch.Tensor, qmax: int, sweeps: int = 60,
           idx: Optional[torch.Tensor] = None, order_seed: Optional[int] = None
           ) -> torch.Tensor:
    """Exact coordinate descent for a single row, optionally restricted to `idx`."""
    wn, An, sn, zn, cn = _np(w, A, s, z, c)
    n = wn.size
    free = np.arange(n) if idx is None else idx.cpu().numpy().astype(np.int64)
    d = wn - (cn * sn + zn)
    Md = An @ d
    adiag = np.maximum(np.diagonal(An), 1e-12)
    rng = np.random.default_rng(order_seed) if order_seed is not None else None
    for _ in range(sweeps):
        moved = 0
        seq = free if rng is None else rng.permutation(free)
        for j in seq:
            d_star = d[j] - Md[j] / adiag[j]
            cc = min(max(round((wn[j] - d_star - zn[j]) / sn[j]), 0), qmax)
            if cc == cn[j]:
                continue
            dn = wn[j] - (cc * sn[j] + zn[j])
            Md += An[j] * (dn - d[j])        # A is symmetric, so row j is column j
            d[j] = dn
            cn[j] = cc
            moved += 1
        if moved == 0:
            break
    return torch.tensor(cn, dtype=torch.float32, device=w.device)


@torch.no_grad()
def cd_subset_exact(w: torch.Tensor, A: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
                    c: torch.Tensor, qmax: int, sub: int = 24, rounds: int = 3,
                    seed: int = 0, node_budget: int = 200_000,
                    blocks: str = "contig") -> Tuple[torch.Tensor, Dict]:
    """Block coordinate descent whose blocks are solved *exactly* by the sphere decoder.

    This is the J3 arm: it makes joint moves over `sub` decisions at once, so if the loss of
    greedy quantization were caused by higher-order coupling that single-coordinate moves
    cannot see, this is the optimizer that would find it.
    """
    n = w.numel()
    c = c.clone().float()
    g = torch.Generator(device="cpu").manual_seed(seed)
    total_nodes, n_proved, n_blocks = 0, 0, 0
    for _ in range(rounds):
        perm = torch.randperm(n, generator=g) if blocks == "random" else torch.arange(n)
        for b0 in range(0, n - sub + 1, sub):
            idx = perm[b0:b0 + sub].to(w.device)
            R, x, idx = subproblem_form(w, A, s, z, c, idx)
            lo = np.zeros(idx.numel(), dtype=np.int64)
            hi = np.full(idx.numel(), qmax, dtype=np.int64)
            best_c, _, nodes, proved = sphere_decode(R, x, lo, hi, c[idx],
                                                     node_budget=node_budget)
            c[idx] = torch.tensor(best_c, dtype=torch.float32, device=w.device)
            total_nodes += nodes
            n_proved += int(proved)
            n_blocks += 1
    return c, dict(nodes=total_nodes, blocks=n_blocks,
                   frac_proved=n_proved / max(n_blocks, 1))


def anneal_row(w: torch.Tensor, A: torch.Tensor, s: torch.Tensor, z: torch.Tensor,
               c: torch.Tensor, qmax: int, iters: int = 200_000, t0: float = 3e-2,
               t1: float = 1e-5, seed: int = 0) -> torch.Tensor:
    """Simulated annealing over one row's codes, with exact incremental objective updates.

    A deliberately different search family from coordinate descent: if CD were stopping at a
    poor local optimum, an annealer given a large budget from the same start would escape it.
    Temperatures are expressed as a fraction of the starting objective, so the schedule means
    the same thing on every matrix.
    """
    wn, An, sn, zn, cn = _np(w, A, s, z, c)
    n = wn.size
    d = wn - (cn * sn + zn)
    Md = An @ d
    adiag = np.maximum(np.diagonal(An), 1e-12)
    cur = float(d @ Md)
    best_c, best = cn.copy(), cur
    rng = np.random.default_rng(seed)
    js = rng.integers(0, n, iters)
    steps = rng.integers(0, 2, iters) * 2 - 1
    us = rng.random(iters)
    scale0 = max(abs(cur), 1e-30)
    for i in range(iters):
        j = int(js[i])
        cc = cn[j] + steps[i]
        if cc < 0 or cc > qmax:
            continue
        dn = wn[j] - (cc * sn[j] + zn[j])
        dd = dn - d[j]
        delta = 2.0 * dd * Md[j] + adiag[j] * dd * dd
        T = scale0 * t0 * (t1 / t0) ** (i / max(iters - 1, 1))
        if delta <= 0 or us[i] < np.exp(-delta / T):
            Md += An[j] * dd
            d[j] = dn
            cn[j] = cc
            cur += delta
            if cur < best:
                best, best_c = cur, cn.copy()
    return torch.tensor(best_c, dtype=torch.float32, device=w.device)
