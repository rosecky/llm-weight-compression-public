"""Baseline 2 -- low-rank approximation, whole-matrix and per-tile."""
from __future__ import annotations

import math
from typing import Dict

import torch

from ..tiles import from_tiles, to_tiles
from .base import Codec, DecodeCost, MatrixResult, register


@register
class GlobalLowRank(Codec):
    """W ~= U V, rank r, factors stored at `factor_bits`."""

    name = "lowrank"

    def __init__(self, rank: int = 64, factor_bits: int = 16):
        self.rank, self.factor_bits = rank, factor_bits

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        r = min(self.rank, min(W.shape))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        A = U[:, :r] * S[:r].sqrt()
        B = S[:r].sqrt().unsqueeze(1) * Vh[:r]
        if self.factor_bits < 16:
            A = _fq(A, self.factor_bits)
            B = _fq(B, self.factor_bits)
        bits = {"factors": float((A.numel() + B.numel()) * self.factor_bits)}
        return MatrixResult(A @ B, bits)

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(flops_per_weight=2.0 * self.rank, tile_local=False,
                          notes="reconstruction needs a full rank-r product; not tile-local "
                                "unless materialised, so it is a storage baseline only")


@register
class TileLowRank(Codec):
    """Per-tile rank-r factorisation: tile-local, but the rate is brutal for square tiles."""

    name = "tile_lowrank"

    def __init__(self, th: int = 16, tw: int = 16, rank: int = 1, factor_bits: int = 8):
        self.th, self.tw, self.rank, self.factor_bits = th, tw, rank, factor_bits

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        out_f, in_f = W.shape
        T = to_tiles(W, self.th, self.tw).float().reshape(-1, self.th, self.tw)
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)
        r = min(self.rank, min(self.th, self.tw))
        A = U[:, :, :r] * S[:, None, :r].sqrt()
        B = S[:, :r, None].sqrt() * Vh[:, :r, :]
        if self.factor_bits < 16:
            A, B = _fq(A, self.factor_bits), _fq(B, self.factor_bits)
        rec = torch.bmm(A, B).reshape(-1, self.th * self.tw)
        n_tiles = rec.shape[0]
        bits = {"factors": float(n_tiles * r * (self.th + self.tw) * self.factor_bits)}
        return MatrixResult(from_tiles(rec, out_f, in_f, self.th, self.tw), bits)

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(flops_per_weight=2.0 * self.rank,
                          bytes_streamed_per_weight=self.rank * (self.th + self.tw)
                          * self.factor_bits / 8.0 / (self.th * self.tw),
                          notes=f"per-tile rank-{self.rank} outer product")


def _fq(x: torch.Tensor, bits: int) -> torch.Tensor:
    lo, hi = x.amin(), x.amax()
    s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
    return torch.round((x - lo) / s) * s + lo
