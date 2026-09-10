"""The simplest vector-quantized representation that can isolate the functional-scope effect.

Phase question: does the module-scope gain measured for the scalar quantizer survive (or grow)
when the local representation is a modern low-bit vector code? To answer it cleanly the VQ
here is deliberately minimal -- GPTVQ-like, nothing more:

    * d consecutive input channels of one output row form one vector,
    * one codebook of K entries per matrix (K=32, d=2 -> 5 bits per 2 weights = 2.5 bpw),
    * a per-(row, group) amplitude (variant A, GPTVQ-style) or per-row amplitude (variant B,
      testing how far Hadamard lets scale metadata be amortised),
    * codebook fitted by diagonal-Hessian-weighted k-means on the amplitude-normalised
      vectors of that matrix.

Two deliberate choices carry earlier findings forward rather than re-litigating them:

    * Amplitudes are **frozen from the original weights** before any compensation runs. The
      scale-feedback study (EXPERIMENT_LOG E27b/c) showed working-weight scales lose up to
      24-45% at 2 bits, and freezing is also what published pre-sweep grid methods do.
    * The wider scope enters ONLY through the dense output metric `G` measured inside the
      sequential pipeline -- the same machinery the scalar module-scope result used. No
      low-rank surrogate (its null space is exploitable; measured 32x worse).

Storage is auditable per matrix: indices + scales + the codebook itself, no hidden state.

`vq_refine` is exact coordinate descent over codeword indices: for one vector position the
objective `tr(D A D^T G)` restricted to a change in row r, columns (j1..jd) is an explicit
quadratic, so all K candidates are scored exactly and in parallel across rows. Under identity
G rows are independent and every improving move is exact; under a dense G simultaneous row
moves interact, so accepted rows go through the same verify-and-backoff (with a guaranteed
single-best-row fallback) that `cd_refine` uses.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from .joint import OutputMetric, metric_identity


# ====================================================================== codebook


@torch.no_grad()
def weighted_kmeans(X: torch.Tensor, w: torch.Tensor, K: int, iters: int = 25,
                    seed: int = 0) -> torch.Tensor:
    """K-means under a per-sample diagonal metric.

    X: (N, d) samples, w: (N, d) nonnegative weights. Distance of sample n to centroid k is
    sum_c w[n,c] (X[n,c] - C[k,c])^2, and the optimal centroid update is the per-coordinate
    weighted mean -- so the algorithm stays exact Lloyd, just anisotropic.
    """
    N, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(N, generator=g)[:K].to(X.device)].clone()
    wX = w * X
    for _ in range(iters):
        # (N,K) weighted distances, chunked so N x K never gets large
        idx = torch.empty(N, dtype=torch.long, device=X.device)
        step = max(1, (1 << 22) // K)
        for i in range(0, N, step):
            xs, ws = X[i:i + step], w[i:i + step]
            # sum_c w (x - c)^2 = sum_c w x^2 - 2 sum_c (w x) c + sum_c w c^2
            dist = (ws @ (C * C).T) - 2.0 * ((ws * xs) @ C.T)
            idx[i:i + step] = dist.argmin(1)
        num = torch.zeros(K, d, device=X.device).index_add_(0, idx, wX)
        den = torch.zeros(K, d, device=X.device).index_add_(0, idx, w)
        empty = den.sum(1) == 0
        C = num / den.clamp_min(1e-12)
        if bool(empty.any()):
            # reseed empty clusters on the currently worst-fit samples
            far = ((X - C[idx]) ** 2 * w).sum(1)
            C[empty] = X[far.topk(int(empty.sum())).indices]
    return C


# ====================================================================== state


class VQState:
    """Indices + frozen amplitudes + codebook: the entire stored representation."""

    def __init__(self, codes: torch.Tensor, scale: torch.Tensor, codebook: torch.Tensor,
                 d: int, group: int, meta_bits: int = 16):
        self.codes = codes                  # (out, nvec) int16
        self.scale = scale                  # (out, n_groups) fp32
        self.C = codebook                   # (K, d)
        self.d = d
        self.group = group                  # columns per scale group (row-mode: group = in_f)
        self.meta_bits = meta_bits
        self.K = codebook.shape[0]

    def clone(self) -> "VQState":
        return VQState(self.codes.clone(), self.scale.clone(), self.C.clone(),
                       self.d, self.group, self.meta_bits)

    def scale_for_cols(self, in_f: int) -> torch.Tensor:
        """(out, in) expansion of the per-group amplitude."""
        return self.scale.repeat_interleave(self.group, dim=1)[:, :in_f]

    def dequant(self) -> torch.Tensor:
        out_f, nvec = self.codes.shape
        w = self.C[self.codes.long()]                       # (out, nvec, d)
        w = w.reshape(out_f, nvec * self.d)
        return w * self.scale_for_cols(nvec * self.d)

    def n_differ(self, other: "VQState") -> Tuple[int, float]:
        n = int((self.codes != other.codes).sum())
        return n, n / self.codes.numel()

    def storage_bits(self) -> Dict[str, float]:
        out_f, nvec = self.codes.shape
        return {"codes": float(out_f * nvec * math.log2(self.K)),
                "scales": float(self.scale.numel() * self.meta_bits),
                "codebook": float(self.C.numel() * self.meta_bits)}


@torch.no_grad()
def build_vq(W: torch.Tensor, adiag: torch.Tensor, d: int = 2, K: int = 32,
             group: int = 128, scale_mode: str = "group", seed: int = 0,
             kmeans_iters: int = 25) -> VQState:
    """Fit amplitudes (frozen, from the given W), codebook, and the naive assignment."""
    out_f, in_f = W.shape
    assert in_f % d == 0
    if scale_mode == "row":
        group = in_f
    assert in_f % group == 0 and group % d == 0
    ng = in_f // group
    scale = W.reshape(out_f, ng, group).pow(2).mean(2).sqrt().clamp_min(1e-12)

    s_full = scale.repeat_interleave(group, dim=1)
    V = (W / s_full).reshape(out_f, in_f // d, d).reshape(-1, d)
    # reconstruction error in weight space is s^2 * a_j * (v - c)^2 per coordinate
    wgt = (adiag.reshape(1, in_f).expand(out_f, in_f) * s_full ** 2) \
        .reshape(-1, d).clamp_min(1e-12)
    C = weighted_kmeans(V, wgt, K, iters=kmeans_iters, seed=seed)

    dist = (wgt @ (C * C).T) - 2.0 * ((wgt * V) @ C.T)
    codes = dist.argmin(1).reshape(out_f, in_f // d).to(torch.int16)
    return VQState(codes, scale, C, d, group)


# ====================================================================== engine adapter


class VQFrozenQuantizer:
    """Adapter so the canonical `gptq()` engine can drive the VQ code with the engine's own
    atom mechanism (error is propagated between atoms, never inside one). Scales are the
    frozen ones from `VQState`; the in-atom metric is the diagonal Hessian of the atom's
    columns, which is what the engine's compensation geometry sees for a fixed atom."""

    group = 0

    def __init__(self, st: VQState, adiag: torch.Tensor):
        self.st = st
        self.atom = st.d
        self.adiag = adiag
        self.s_full = st.scale_for_cols(st.codes.shape[1] * st.d)

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        pass                                # amplitudes are frozen by design

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        s = self.s_full[:, col:col + self.atom]
        a = self.adiag[col:col + self.atom].clamp_min(1e-12)
        v = w / s
        wgt = (a.unsqueeze(0) * s * s)
        C = self.st.C
        dist = (wgt @ (C * C).T) - 2.0 * ((wgt * v) @ C.T)
        idx = dist.argmin(1)
        self.st.codes[:, col // self.atom] = idx.to(torch.int16)
        return C[idx] * s

    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        return self.st.storage_bits()


# ====================================================================== refinement


@torch.no_grad()
def vq_refine(W: torch.Tensor, A: torch.Tensor, st: VQState,
              G: Optional[OutputMetric] = None, sweeps: int = 6,
              order: str = "forward", seed: int = 0,
              cb_slice: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """Exact coordinate descent over codeword indices under `tr(D A D^T G)`.

    `cb_slice` (nvec, 2) int, optional: per column-vector the half-open index range
    `[lo, hi)` of `st.C` that vector may use -- for formats with one codebook per column
    block stored concatenated (e.g. the peer's per-256-column codebooks). Codes are always
    written as indices into the full concatenated `st.C`. `None` = every vector may use
    every centroid (the original behaviour).

    Returns the dequantised weights and an info dict mirroring `cd_refine`.
    """
    dev = W.device
    out_f, in_f = W.shape
    d, K = st.d, st.K
    nvec = in_f // d
    G = G if G is not None else metric_identity()
    ident = G.is_identity
    gdiag = torch.ones(out_f, device=dev) if ident else G.diagonal(out_f, dev)

    s_full = st.scale_for_cols(in_f)
    What = st.dequant()
    D = W - What
    M = (D if ident else G.apply(D)) @ A
    d_start = float(((D @ A) * (D if ident else G.apply(D))).sum())

    cols = torch.arange(in_f, device=dev).reshape(nvec, d)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_moved = 0

    for sw in range(sweeps):
        moved = 0
        seq = range(nvec) if order == "forward" else \
            torch.randperm(nvec, generator=g).tolist()
        for v in seq:
            jj = cols[v]                                     # (d,) column indices
            s = s_full[:, jj[0]].unsqueeze(1)                # (out,1) group scale
            Asub = A[jj][:, jj]                              # (d,d)
            Wv = W[:, jj]                                    # (out,d)
            Dv = D[:, jj]
            Mv = M[:, jj]
            if cb_slice is None:
                lo, Cv = 0, st.C
            else:
                lo, hi = int(cb_slice[v, 0]), int(cb_slice[v, 1])
                Cv = st.C[lo:hi]
            cand = s.unsqueeze(2) * Cv.unsqueeze(0)          # (out,K',d)
            delta = (Wv.unsqueeze(1) - cand) - Dv.unsqueeze(1)   # (out,K',d) change in D
            lin = 2.0 * (delta * Mv.unsqueeze(1)).sum(2)
            quad = torch.einsum("okc,cd,okd->ok", delta, Asub, delta)
            obj = lin + gdiag.unsqueeze(1) * quad
            gain, kloc = obj.min(1)                          # kloc indexes Cv / delta
            kbest = kloc + lo                                # kbest indexes st.C
            cur = st.codes[:, v].long()
            better = (gain < -1e-12) & (kbest != cur)
            if not bool(better.any()):
                continue

            keep = better.clone()
            if not ident:
                # joint moves across rows interact through G off-diagonals: verify the exact
                # joint change and back off by gain rank, with the single best row as the
                # guaranteed fallback (a single-row move has no cross term).
                ok = False
                for _ in range(6):
                    if not bool(keep.any()):
                        break
                    dD = torch.zeros(out_f, d, device=dev)
                    rows = keep.nonzero(as_tuple=True)[0]
                    dD[rows] = delta[rows, kloc[rows]]
                    joint = float(2.0 * (dD * Mv).sum()
                                  + (G.apply(dD) * (dD @ Asub)).sum())
                    if joint < 0:
                        ok = True
                        break
                    thr = gain[keep].median()
                    keep = keep & (gain <= thr) & (gain < 0)
                if not ok:
                    best_r = int(torch.argmin(gain))
                    keep = torch.zeros_like(keep)
                    if float(gain[best_r]) < 0.0:
                        keep[best_r] = True
                    if not bool(keep.any()):
                        continue

            rows = keep.nonzero(as_tuple=True)[0]
            dD = torch.zeros(out_f, d, device=dev)
            dD[rows] = delta[rows, kloc[rows]]
            st.codes[rows, v] = kbest[rows].to(torch.int16)
            D[:, jj] += dD
            M += (dD if ident else G.apply(dD)) @ A[jj, :]
            moved += int(rows.numel())
        n_moved += moved
        if moved == 0:
            break

    d_end = float(((D @ A) * (D if ident else G.apply(D))).sum())
    return W - D, dict(damage_start=d_start, damage_end=d_end,
                       n_moved=n_moved, frac_moved=n_moved / max(st.codes.numel(), 1))
