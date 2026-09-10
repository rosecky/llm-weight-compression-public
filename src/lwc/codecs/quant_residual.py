"""Variant D -- low-bit quantization plus a structured correction.

    W ~= Q_b(W) + correction(W - Q_b(W))

Correction families, all with honest bit accounting:

* ``lowrank``  rank-r factorisation of the residual, factors at 16 bit.
               NOT tile-local -- flagged accordingly in DecodeCost.
* ``vq``       residual VQ over tiles of the residual, shared codebook.
* ``sparse``   keep the top-p fraction of residual entries, value + index.
* ``none``     plain quantization (identical to the ScalarQuantCodec, kept for symmetry).

The point of the variant is the hypothesis that scalar quantization handles the high-entropy
bulk while a cheap procedural term mops up structured error. Phase 1D measures whether such
structured error exists at all; this codec measures what it costs to chase it anyway.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch

from ..structure import kmeans
from ..tiles import from_tiles, to_tiles
from .base import Codec, DecodeCost, MatrixResult, register
from .scalar import groupwise_bits, quantize_groupwise
from .vq import _assign


@register
class QuantPlusCorrection(Codec):
    name = "quant_plus"
    needs_fit = False

    def __init__(self, bits: int = 3, group: int = 128, correction: str = "lowrank",
                 rank: int = 32, sparsity: float = 0.01, th: int = 8, tw: int = 8,
                 K: int = 256, stages: int = 1, cb_bits: int = 16, seed: int = 0):
        self.bits, self.group, self.correction = bits, group, correction
        self.rank, self.sparsity = rank, sparsity
        self.th, self.tw, self.K, self.stages = th, tw, K, stages
        self.cb_bits, self.seed = cb_bits, seed
        self.codebooks: List[torch.Tensor] = []
        self.needs_fit = correction == "vq"

    def fit(self, tile_stream: Iterable[torch.Tensor]) -> None:
        """Codebooks are fit on RESIDUAL tiles supplied by the runner."""
        T = torch.cat([t for t in tile_stream], 0).float()
        self.codebooks = []
        R = T
        for s in range(self.stages):
            C = kmeans(R, self.K, iters=20, seed=self.seed + s)
            self.codebooks.append(C)
            R = R - C[_assign(R, C)]

    def shared_bits(self) -> Dict[str, float]:
        if self.correction != "vq":
            return {}
        return {"codebooks": sum(C.numel() * self.cb_bits for C in self.codebooks)}

    @torch.no_grad()
    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        Q = quantize_groupwise(W, self.bits, self.group)
        bits = groupwise_bits(W, self.bits, self.group, symmetric=False)
        R = W - Q
        out_f, in_f = W.shape

        if self.correction == "none":
            return MatrixResult(Q, bits)

        if self.correction == "lowrank":
            r = min(self.rank, min(W.shape))
            U, S, Vh = torch.linalg.svd(R, full_matrices=False)
            A = U[:, :r] * S[:r].sqrt()
            B = S[:r].sqrt().unsqueeze(1) * Vh[:r]
            bits["corr_factors"] = float((A.numel() + B.numel()) * 16)
            return MatrixResult(Q + A @ B, bits)

        if self.correction == "sparse":
            k = max(int(self.sparsity * R.numel()), 1)
            flat = R.reshape(-1)
            vals, idx = torch.topk(flat.abs(), k)
            S_ = torch.zeros_like(flat)
            S_[idx] = flat[idx]
            # value at fp16 + index at log2(numel) bits (a real format would use CSR deltas)
            bits["corr_values"] = float(k * 16)
            bits["corr_indices"] = float(k * math.log2(R.numel()))
            return MatrixResult(Q + S_.reshape(W.shape), bits)

        if self.correction == "vq":
            T = to_tiles(R, self.th, self.tw).float()
            acc = torch.zeros_like(T)
            cur = T
            for C in self.codebooks:
                a = _assign(cur, C)
                acc += C[a]
                cur = cur - C[a]
            bits["corr_codes"] = float(T.shape[0] * self.stages * math.log2(self.K))
            return MatrixResult(Q + from_tiles(acc, out_f, in_f, self.th, self.tw), bits)

        raise ValueError(self.correction)

    def decode_cost(self) -> DecodeCost:
        base = 2.0
        if self.correction == "none":
            return DecodeCost(flops_per_weight=base, bytes_streamed_per_weight=self.bits / 8,
                              notes=f"INT{self.bits} only")
        if self.correction == "lowrank":
            return DecodeCost(flops_per_weight=base + 2.0 * self.rank, tile_local=False,
                              notes=f"INT{self.bits} + global rank-{self.rank} correction; the "
                                    "correction is a second GEMM, not a tile-local decode")
        if self.correction == "sparse":
            return DecodeCost(flops_per_weight=base + 1.0, branch_free=False,
                              random_lookups_per_weight=self.sparsity,
                              notes=f"INT{self.bits} + {self.sparsity:.1%} sparse outliers; "
                                    "irregular gather, needs a separate SpMM path")
        return DecodeCost(
            flops_per_weight=base + self.stages,
            bytes_streamed_per_weight=self.bits / 8 + self.stages * math.log2(self.K)
            / 8.0 / (self.th * self.tw),
            shared_state_bytes=sum(C.numel() for C in self.codebooks) * 2,
            random_lookups_per_weight=self.stages / (self.th * self.tw),
            notes=f"INT{self.bits} + {self.stages}-stage residual VQ correction")
