"""Baselines 3/4 and Variants A/B -- the vector-quantization family.

All members share the same skeleton so the comparison is apples-to-apples:

    tile W into (th x tw) blocks -> optional per-tile normalisation ->
    M stages of residual VQ against shared codebooks -> optional cheap per-tile transform ->
    optional scalar-quantized residual

Residual (additive) VQ is used rather than a single giant codebook because that is the only
form that reaches 2-4 bits/weight with a codebook small enough to sit in shared memory --
a single codebook would need K = 2^(bpw*d) entries. Decoding is M table lookups plus
(M-1) adds per weight, which is kernel-friendly.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import torch

from ..structure import kmeans
from ..tiles import from_tiles, to_tiles
from .base import Codec, DecodeCost, MatrixResult, register
from .scalar import quantize_groupwise


@torch.no_grad()
def _assign(X: torch.Tensor, C: torch.Tensor, chunk: int = 0) -> torch.Tensor:
    """Nearest centroid. The chunk adapts to K so the distance matrix stays ~64 MB
    regardless of codebook size (a fixed chunk of 8192 needs 537 MB at K=16384)."""
    K = C.shape[0]
    if chunk <= 0:
        chunk = int(max(512, min(8192, (1 << 24) // max(K, 1))))
    Cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = (Cn.unsqueeze(0) - 2.0 * (X[i:i + chunk] @ C.T)).argmin(1)
    return out


class _TileVQBase(Codec):
    """Shared machinery: normalisation modes, residual-VQ codebooks, bit accounting."""

    name = "_tilevq"
    needs_fit = True

    def __init__(self, th: int = 8, tw: int = 8, K: int = 256, stages: int = 2,
                 norm: str = "none", scale_bits: int = 8, cb_bits: int = 16,
                 kmeans_iters: int = 20, seed: int = 0):
        assert norm in ("none", "scale", "affine", "rank1")
        self.th, self.tw, self.K, self.stages = th, tw, K, stages
        self.norm, self.scale_bits, self.cb_bits = norm, scale_bits, cb_bits
        self.kmeans_iters, self.seed = kmeans_iters, seed
        self.codebooks: List[torch.Tensor] = []
        self.d = th * tw

    # ---------------------------------------------------------------- normalisation
    def _normalize(self, T: torch.Tensor):
        """Return (normalized tiles, params, extra bits per tile)."""
        if self.norm == "none":
            return T, None, 0.0
        if self.norm == "scale":
            a = T.norm(dim=1, keepdim=True).clamp_min(1e-12) / math.sqrt(self.d)
            return T / a, (a,), float(self.scale_bits)
        if self.norm == "affine":
            b = T.mean(dim=1, keepdim=True)
            a = (T - b).norm(dim=1, keepdim=True).clamp_min(1e-12) / math.sqrt(self.d)
            return (T - b) / a, (a, b), float(2 * self.scale_bits)
        if self.norm == "rank1":
            # Variant B-lite: remove a rank-1 row/col outer product per tile, cheap to decode
            M = T.reshape(-1, self.th, self.tw)
            u = M.pow(2).mean(2).sqrt()                       # (n, th)
            v = M.pow(2).mean(1).sqrt()                       # (n, tw)
            sc = (u.unsqueeze(2) * v.unsqueeze(1)).clamp_min(1e-12)
            g = (M / sc).reshape(-1, self.d)
            return g, (u, v), float(self.scale_bits * (self.th + self.tw))
        raise ValueError(self.norm)

    def _renormalize(self, T: torch.Tensor, params) -> torch.Tensor:
        """Normalise with already-quantized parameters (encoder/decoder consistency)."""
        if self.norm == "none":
            return T
        if self.norm == "scale":
            return T / params[0]
        if self.norm == "affine":
            return (T - params[1]) / params[0]
        if self.norm == "rank1":
            u, v = params
            M = T.reshape(-1, self.th, self.tw)
            return (M / (u.unsqueeze(2) * v.unsqueeze(1)).clamp_min(1e-12)).reshape(-1, self.d)
        raise ValueError(self.norm)

    def _denormalize(self, R: torch.Tensor, params) -> torch.Tensor:
        if self.norm == "none":
            return R
        if self.norm == "scale":
            return R * params[0]
        if self.norm == "affine":
            return R * params[0] + params[1]
        if self.norm == "rank1":
            u, v = params
            M = R.reshape(-1, self.th, self.tw) * (u.unsqueeze(2) * v.unsqueeze(1))
            return M.reshape(-1, self.d)

    @staticmethod
    def _fake_quant(x: torch.Tensor, bits: int, log_domain: bool = False) -> torch.Tensor:
        """Round tile parameters to `bits`, so the declared bits are actually paid.

        Positive scale parameters are quantized in the log domain, which is what any real
        implementation does and what keeps small-scale tiles from collapsing.
        """
        if bits >= 16:
            return x
        y = torch.log2(x.clamp_min(1e-12)) if log_domain else x
        lo, hi = y.amin(), y.amax()
        s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
        q = torch.round((y - lo) / s) * s + lo
        return torch.exp2(q) if log_domain else q

    def _quant_params(self, params):
        """Scales are log-quantized, the affine offset is linear-quantized."""
        if self.norm == "affine":
            return (self._fake_quant(params[0], self.scale_bits, log_domain=True),
                    self._fake_quant(params[1], self.scale_bits))
        return tuple(self._fake_quant(p, self.scale_bits, log_domain=True) for p in params)

    # ---------------------------------------------------------------- fit / encode
    def fit(self, tile_stream: Iterable[torch.Tensor]) -> None:
        buf = [t for t in tile_stream]
        T = torch.cat(buf, 0) if len(buf) > 1 else buf[0]
        Tn, _, _ = self._normalize(T.float())
        self.codebooks = []
        R = Tn
        for s in range(self.stages):
            C = kmeans(R, self.K, iters=self.kmeans_iters, seed=self.seed + s)
            self.codebooks.append(C)
            R = R - C[_assign(R, C)]

    def _vq_reconstruct(self, Tn: torch.Tensor) -> torch.Tensor:
        R = Tn
        acc = torch.zeros_like(Tn)
        for C in self.codebooks:
            a = _assign(R, C)
            acc += C[a]
            R = R - C[a]
        return acc

    def shared_bits(self) -> Dict[str, float]:
        return {"codebooks": sum(C.numel() * self.cb_bits for C in self.codebooks)}

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        out_f, in_f = W.shape
        T = to_tiles(W, self.th, self.tw).float()
        n_tiles = T.shape[0]
        _, params, extra = self._normalize(T)
        if params is not None:
            # quantize the tile parameters FIRST, then normalise with the quantized values, so
            # encoder and decoder see the same scale and the declared bits are really paid
            params = self._quant_params(params)
            Tn = self._renormalize(T, params)
        else:
            Tn = T
        rec = self._vq_reconstruct(Tn)
        rec = self._denormalize(rec, params)
        Wh = from_tiles(rec, out_f, in_f, self.th, self.tw)
        bits = {
            "codes": float(n_tiles * self.stages * math.log2(self.K)),
            "tile_params": float(n_tiles * extra),
        }
        return MatrixResult(Wh, bits)

    def decode_cost(self) -> DecodeCost:
        extra = {"none": 0.0, "scale": 1.0, "affine": 2.0, "rank1": 2.0}[self.norm]
        return DecodeCost(
            flops_per_weight=(self.stages - 1) + extra,
            bytes_streamed_per_weight=(self.stages * math.log2(self.K) + {
                "none": 0.0, "scale": self.scale_bits, "affine": 2 * self.scale_bits,
                "rank1": self.scale_bits * (self.th + self.tw)}[self.norm]) / 8.0 / self.d,
            shared_state_bytes=sum(C.numel() for C in self.codebooks) * 2,
            random_lookups_per_weight=self.stages / self.d,
            notes=f"{self.stages}-stage residual VQ, d={self.d}, K={self.K}, norm={self.norm}")


@register
class TileVQ(_TileVQBase):
    """Baseline 3 -- W_t ~= sum_s B_s[c_t^s]."""

    name = "vq"


@register
class TileVQAffine(_TileVQBase):
    """Variant A -- W_t ~= a_t * B[c_t] + b_t (per-tile affine transform)."""

    name = "vq_affine"

    def __init__(self, **kw):
        kw.setdefault("norm", "affine")
        super().__init__(**kw)


@register
class TileVQRank1(_TileVQBase):
    """Variant B-lite -- per-tile rank-1 row/column scaling around a shared codebook."""

    name = "vq_rank1"

    def __init__(self, **kw):
        kw.setdefault("norm", "rank1")
        super().__init__(**kw)


@register
class TileVQResidual(_TileVQBase):
    """Baseline 4 / Variant D-style -- VQ prototype plus a scalar-quantized residual.

    W_t ~= B[c_t] + Q_b(W_t - B[c_t])
    """

    name = "vq_resid"

    def __init__(self, resid_bits: int = 2, resid_group: int = 64, **kw):
        self.resid_bits, self.resid_group = resid_bits, resid_group
        super().__init__(**kw)

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        base = super().compress_matrix(W)
        R = W - base.recon
        Rq = quantize_groupwise(R, self.resid_bits, self.resid_group)
        out_f, in_f = W.shape
        n_groups = out_f * ((in_f + self.resid_group - 1) // self.resid_group)
        bits = dict(base.bits)
        bits["resid_codes"] = float(W.numel() * self.resid_bits)
        bits["resid_scales"] = float(n_groups * 32)
        return MatrixResult(base.recon + Rq, bits)

    def decode_cost(self) -> DecodeCost:
        dc = super().decode_cost()
        dc.flops_per_weight += 3.0                       # dequant (mul+add) + add to prototype
        dc.bytes_streamed_per_weight += self.resid_bits / 8.0 + 4.0 / self.resid_group
        dc.notes += f" + INT{self.resid_bits} residual g{self.resid_group}"
        return dc
