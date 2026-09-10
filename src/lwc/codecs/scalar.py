"""Baseline 0 / 1 -- floating point and group-wise scalar quantization.

Group-wise round-to-nearest with per-group scale+zero, groups laid out along the input
dimension (the layout every INT4 GEMM kernel already uses). This is the reference every
fancier codec has to beat.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import Codec, DecodeCost, MatrixResult, register


@torch.no_grad()
def quantize_groupwise(W: torch.Tensor, bits: int, group: int = 128,
                       symmetric: bool = False) -> torch.Tensor:
    """Round-to-nearest, per-group affine quantization along the input dimension."""
    out_f, in_f = W.shape
    g = min(group, in_f)
    pad = (-in_f) % g
    Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
    G = Wp.reshape(out_f, -1, g)
    qmax = 2 ** bits - 1
    if symmetric:
        s = G.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1)
        s = s.clamp_min(1e-12)
        Q = torch.clamp(torch.round(G / s), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        R = Q * s
    else:
        lo = G.amin(-1, keepdim=True)
        hi = G.amax(-1, keepdim=True)
        s = ((hi - lo) / qmax).clamp_min(1e-12)
        Q = torch.clamp(torch.round((G - lo) / s), 0, qmax)
        R = Q * s + lo
    R = R.reshape(out_f, -1)[:, :in_f]
    return R.contiguous()


def groupwise_bits(W: torch.Tensor, bits: int, group: int, symmetric: bool,
                   meta_bits: int = 16) -> Dict[str, float]:
    out_f, in_f = W.shape
    n_groups = out_f * ((in_f + group - 1) // group)
    per_group = meta_bits * (1 if symmetric else 2)
    return {"codes": float(out_f * in_f * bits), "scales": float(n_groups * per_group)}


@register
class FPCodec(Codec):
    name = "fp"

    def __init__(self, bits: int = 16):
        self.bits = bits

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        if self.bits == 16:
            R = W.to(torch.float16).float()
        elif self.bits == 8:                       # e4m3-ish, per-tensor scaled
            s = W.abs().amax() / 448.0
            R = (W / s.clamp_min(1e-12)).to(torch.float8_e4m3fn).float() * s
        else:
            R = W.clone()
        return MatrixResult(R, {"codes": float(W.numel() * self.bits)})

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(flops_per_weight=0.0, bytes_streamed_per_weight=self.bits / 8,
                          shared_state_bytes=0, notes="no decode; reference point")


@register
class ScalarQuantCodec(Codec):
    name = "int"

    def __init__(self, bits: int = 4, group: int = 128, symmetric: bool = False):
        self.bits, self.group, self.symmetric = bits, group, symmetric

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        R = quantize_groupwise(W, self.bits, self.group, self.symmetric)
        return MatrixResult(R, groupwise_bits(W, self.bits, self.group, self.symmetric))

    def decode_cost(self) -> DecodeCost:
        # dequant is one multiply + one add per weight, scale is broadcast within a group
        return DecodeCost(flops_per_weight=2.0,
                          bytes_streamed_per_weight=self.bits / 8 + 4.0 / self.group,
                          shared_state_bytes=0.0, random_lookups_per_weight=0.0,
                          notes=f"INT{self.bits} g{self.group}; this is what a Marlin-class "
                                "kernel already does")
