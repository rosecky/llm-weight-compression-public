"""GPTQ / OBS-style error compensation, with a pluggable quantizer.

The point of this module is the *factorial* design: the identical compensation machinery must
drive both the scalar quantizer and the vector quantizer, so that "GPTQ - naive" isolates the
compensation effect and nothing else.

Method. For a linear layer with calibration inputs `x`, the local loss is
`||W x - What x||^2`, whose Hessian is `H = 2 E[x x^T]`. Quantize input columns left to right;
after fixing column `j`, redistribute its error into the *not-yet-quantized* columns along the
inverse-Hessian direction, which is the second-order-optimal update:

    err_j        = (w_j - q_j) / [Hinv]_jj
    W[:, j+1:]  -= err_j (x) [Hinv]_{j, j+1:}

`Hinv` here is the upper Cholesky factor of the inverse Hessian, the standard GPTQ
factorisation that makes the update a single triangular row.

**Atoms.** A quantizer may need to see several columns at once (the vector quantizer codes
8 consecutive input channels of a row jointly). `Quantizer.atom` declares that width. Error is
then propagated only to columns *after* the atom, never inside it, because the atom's
quantization is already jointly determined. Setting `atom=1` recovers textbook GPTQ. Running
the scalar quantizer at `atom=8` gives a control that isolates the atom width from the
representation.

Nothing here is a heuristic residual correction: the update uses the activation second moment,
and with `atom=1, group=0` it reduces exactly to the published algorithm.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def guard_alloc(shape, dtype=torch.float32, name: str = "tensor", limit_mib: float = 1024.0):
    """Refuse to silently allocate a huge tensor. Returns the size in MiB."""
    n = 1
    for s in shape:
        n *= int(s)
    mib = n * torch.tensor([], dtype=dtype).element_size() / 2 ** 20
    if mib > limit_mib:
        raise MemoryError(
            f"{name} would need {mib:.0f} MiB for shape {tuple(shape)} "
            f"(limit {limit_mib:.0f} MiB). Chunk it instead.")
    return mib


# ====================================================================== quantizers


class Quantizer:
    """Interface. `atom` columns are quantized jointly; `group` columns share parameters."""

    atom: int = 1
    group: int = 0                       # 0 = parameters are global, no per-group refresh

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        """Called at each group boundary with the CURRENT (error-updated) columns.

        `col` is the index of the group's first input column, so a quantizer whose
        parameters vary per group (e.g. a mixed-precision bit map) knows where it is.
        """

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        raise NotImplementedError

    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        raise NotImplementedError


class ScalarQuantizer(Quantizer):
    """Asymmetric min-max group-wise RTN.

    Min-max is deliberate: it spans the observed range of every group, so the largest weight in
    a group is never clipped. The earlier Lloyd-Max/RMS baseline reconstructed the largest
    weight with up to 74% error and cost 98x perplexity (see EXPERIMENT_LOG E11); this codec
    cannot do that by construction.
    """

    def __init__(self, bits: int = 3, group: int = 128, atom: int = 1, meta_bits: int = 16):
        self.bits, self.group, self.atom, self.meta_bits = bits, group, atom, meta_bits
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None
        self.qmax = 2 ** bits - 1

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        lo = W_group.amin(dim=1, keepdim=True)
        hi = W_group.amax(dim=1, keepdim=True)
        self.scale = ((hi - lo) / self.qmax).clamp_min(1e-12)
        self.zero = lo

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        q = torch.clamp(torch.round((w - self.zero) / self.scale), 0, self.qmax)
        return q * self.scale + self.zero

    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        n_groups = out_f * ((in_f + self.group - 1) // self.group)
        return {"codes": float(out_f * in_f * self.bits),
                "scales": float(n_groups * 2 * self.meta_bits)}


class Tile2DQuantizer(Quantizer):
    """Asymmetric min-max over a 2-D tile: `row_block` output channels x `col_block` inputs.

    `ScalarQuantizer` shares one scale along a row only, so the *order of output channels is
    irrelevant to it*. A 2-D tile is the smallest change that makes an output-channel
    partition mean anything, which is what the graph study needs in order to test whether
    output communities buy anything. With `row_block * col_block = 128` the scale budget is
    identical to `group=128`, so the comparison is at matched bits.

    `row_block=1` reduces exactly to `ScalarQuantizer` at the same group size.
    """

    atom = 1

    def __init__(self, bits: int = 3, row_block: int = 16, col_block: int = 8,
                 meta_bits: int = 16):
        self.bits, self.rb, self.group, self.meta_bits = bits, row_block, col_block, meta_bits
        self.qmax = 2 ** bits - 1
        self.scale: Optional[torch.Tensor] = None
        self.zero: Optional[torch.Tensor] = None

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        out_f, g = W_group.shape
        nb = out_f // self.rb
        B = W_group[:nb * self.rb].reshape(nb, self.rb * g)
        lo = B.amin(1)
        hi = B.amax(1)
        sc = ((hi - lo) / self.qmax).clamp_min(1e-12)
        self.zero = lo.repeat_interleave(self.rb).reshape(-1, 1)
        self.scale = sc.repeat_interleave(self.rb).reshape(-1, 1)

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        q = torch.clamp(torch.round((w - self.zero) / self.scale), 0, self.qmax)
        return q * self.scale + self.zero

    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        n_tiles = (out_f // self.rb) * ((in_f + self.group - 1) // self.group)
        return {"codes": float(out_f * in_f * self.bits),
                "scales": float(n_tiles * 2 * self.meta_bits)}


class VQQuantizer(Quantizer):
    """Residual (additive) VQ over `atom` consecutive input channels of each row.

    This is the best-performing representation from the earlier study: realistically
    decodable (M lookups + M-1 adds per weight), no hidden storage (the codebook is
    64-256 KiB, i.e. 0.0015-0.006 bits/weight amortised over the model), and no permutation
    or unranking overhead. No new representation is being developed here.
    """

    group = 0

    def __init__(self, codebooks, cb_bits: int = 16):
        self.codebooks = codebooks                    # list of (K, d) tensors
        self.atom = codebooks[0].shape[1]
        self.K = codebooks[0].shape[0]
        self.stages = len(codebooks)
        self.cb_bits = cb_bits

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        acc = torch.zeros_like(w)
        cur = w
        for C in self.codebooks:
            Cn = (C * C).sum(1)
            idx = (Cn.unsqueeze(0) - 2.0 * (cur @ C.T)).argmin(1)
            sel = C[idx]
            acc = acc + sel
            cur = cur - sel
        return acc

    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        n_tiles = out_f * ((in_f + self.atom - 1) // self.atom)
        return {"codes": float(n_tiles * self.stages * math.log2(self.K))}

    def shared_bits(self) -> float:
        return float(sum(C.numel() for C in self.codebooks) * self.cb_bits)


# ====================================================================== the engine


@torch.no_grad()
def cholesky_inverse_upper(H: torch.Tensor, percdamp: float = 0.01) -> torch.Tensor:
    """Upper Cholesky factor of H^-1, with dead columns handled and Tikhonov damping."""
    n = H.shape[0]
    H = H.clone().float()
    dead = torch.diagonal(H) == 0
    if bool(dead.any()):
        H[dead, dead] = 1.0
    H = 0.5 * (H + H.T)
    base = torch.diagonal(H).mean()
    damp = percdamp * base
    H[range(n), range(n)] += damp
    # An empirical second moment can lose positive-definiteness to accumulation error or a
    # lossy cache round trip. Escalate the damping rather than crashing; a path that already
    # succeeded at `percdamp` is unaffected.
    for attempt in range(8):
        try:
            L = torch.linalg.cholesky(H)
            break
        except Exception:
            extra = base * percdamp * (10.0 ** attempt)
            H[range(n), range(n)] += extra
    else:
        raise RuntimeError("Hessian not positive-definite even after damping")
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True), dead


def actorder_bits(in_f: int, out_f: int) -> float:
    """Act-order permutes input columns, so the decoder needs to know the order.

    Real implementations ship a per-column group index (`g_idx`); we charge the conservative
    ceil(log2(in_features)) bits per column. At in=4864, out=896 this is 0.0145 bits/weight --
    small, but it is not zero and it is counted.
    """
    return float(in_f * math.ceil(math.log2(max(in_f, 2))))


@torch.no_grad()
def gptq(W: torch.Tensor, H: Optional[torch.Tensor], quant: Quantizer,
         blocksize: int = 128, percdamp: float = 0.01,
         actorder: bool = False) -> torch.Tensor:
    """Quantize W (out, in) with second-order error compensation. Returns What.

    `H = None` disables compensation entirely and reduces to naive independent quantization
    with the identical quantizer -- that is the `naive` arm of the factorial.
    """
    out_f, in_f = W.shape
    W = W.clone().float()
    Q = torch.zeros_like(W)
    a = quant.atom

    if H is None:                                  # ---- naive arm
        for i1 in range(0, in_f, max(quant.group or blocksize, a)):
            i2 = min(i1 + max(quant.group or blocksize, a), in_f)
            if quant.group:
                quant.find_params(W[:, i1:i2], i1)
            for j in range(i1, i2, a):
                Q[:, j:j + a] = quant.quantize(W[:, j:j + a], j)
        return Q

    guard_alloc((in_f, in_f), torch.float32, "inverse Hessian", limit_mib=1024.0)
    perm = inv_perm = None
    if actorder:
        perm = torch.argsort(torch.diagonal(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        inv_perm = torch.argsort(perm)

    Hinv, dead = cholesky_inverse_upper(H, percdamp)
    W[:, dead] = 0.0

    blocksize = max(blocksize, a)
    blocksize = (blocksize // a) * a                # keep atoms inside a block
    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        width = i2 - i1
        for j in range(0, width, a):
            col = i1 + j
            if quant.group and col % quant.group == 0:
                # Fit the FULL group, crossing the block boundary if the group is wider than
                # the block (columns beyond the block come from the working W, which carries
                # cross-block compensation -- the original GPTQ's semantics). Fitting only the
                # in-block half was measured to clip the unfitted half and cost up to 3x at
                # b4/g256 (see gptq_scalefeed.py / EXPERIMENT_LOG E27b).
                if col + quant.group > i2:
                    Wg = torch.cat([W1[:, j:width], W[:, i2:col + quant.group]], 1)
                    quant.find_params(Wg[:, :quant.group], col)
                else:
                    quant.find_params(W1[:, j:j + quant.group], col)
            aa = min(a, width - j)
            w = W1[:, j:j + aa]
            q = quant.quantize(w, col)
            Q1[:, j:j + aa] = q
            for t in range(aa):
                d = Hinv1[j + t, j + t]
                err = (w[:, t] - q[:, t]) / d
                E1[:, j + t] = err
                if j + a < width:                   # never compensate inside the atom
                    W1[:, j + a:] -= err.unsqueeze(1) * Hinv1[j + t, j + a:].unsqueeze(0)
        Q[:, i1:i2] = Q1
        if i2 < in_f:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    if actorder:
        Q = Q[:, inv_perm]
    return Q


class HessianAccumulator:
    """H = sum_t x_t x_t^T, accumulated in fp32 with bounded memory."""

    def __init__(self, in_f: int, device: str = "cuda"):
        guard_alloc((in_f, in_f), torch.float32, "Hessian", limit_mib=1024.0)
        self.H = torch.zeros(in_f, in_f, device=device, dtype=torch.float32)
        self.n = 0

    @torch.no_grad()
    def add(self, X: torch.Tensor, chunk: int = 8192) -> None:
        """X: (n_samples, in_features)."""
        X = X.reshape(-1, X.shape[-1])
        for i in range(0, X.shape[0], chunk):
            xc = X[i:i + chunk].float()
            self.H += xc.T @ xc
            self.n += xc.shape[0]

    def finalize(self) -> torch.Tensor:
        return self.H * (2.0 / max(self.n, 1))
