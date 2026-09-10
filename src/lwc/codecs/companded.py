"""Baseline 1b -- Lloyd-Max / NF-style scalar quantization.

Plain min-max RTN is a weak low-bit baseline: it spreads uniform levels over the group's
observed range, which for Gaussian-ish data wastes most of its levels on the tails. At 2 bits
that costs a factor of ~2 in MSE against the optimal scalar quantizer, and comparing a vector
quantizer against it would overstate the VQ gain.

This codec uses levels that are optimal for a unit Gaussian (Lloyd-Max, computed once at
import), with a per-group scale. It is the honest scalar baseline: any remaining VQ advantage
over it is genuine space-filling + shape gain, not an artifact of a badly chosen quantizer.
Decoding is a tiny LUT lookup plus one multiply, i.e. exactly as kernel-friendly as INT.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict

import torch

from .base import Codec, DecodeCost, MatrixResult, register


@lru_cache(maxsize=None)
def lloyd_max_gaussian(bits: int, iters: int = 200, n: int = 1 << 21, seed: int = 0):
    """Lloyd-Max levels for a unit Gaussian, found by Lloyd iterations on a large sample."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g).cuda() if torch.cuda.is_available() else \
        torch.randn(n, generator=g)
    K = 2 ** bits
    # init with uniform quantiles so the iteration starts sane
    q = torch.linspace(0.5 / K, 1 - 0.5 / K, K, device=x.device)
    lv = torch.distributions.Normal(0.0, 1.0).icdf(q).to(x.device)
    for _ in range(iters):
        idx = torch.bucketize(x, (lv[1:] + lv[:-1]) / 2)
        s = torch.zeros_like(lv).index_add_(0, idx, x)
        c = torch.zeros_like(lv).index_add_(0, idx, torch.ones_like(x))
        lv = torch.where(c > 0, s / c.clamp_min(1), lv)
    return lv.cpu()


@torch.no_grad()
def quantize_companded(W: torch.Tensor, bits: int, group: int = 128,
                       scale_mode: str = "rms") -> torch.Tensor:
    out_f, in_f = W.shape
    gsz = min(group, in_f)
    pad = (-in_f) % gsz
    Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
    G = Wp.reshape(out_f, -1, gsz)
    if scale_mode == "rms":
        s = G.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    else:
        s = G.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 3.0
    lv = lloyd_max_gaussian(bits).to(W.device, W.dtype)
    Gn = G / s
    idx = torch.bucketize(Gn.reshape(-1), (lv[1:] + lv[:-1]) / 2)
    R = (lv[idx].reshape(G.shape) * s).reshape(out_f, -1)[:, :in_f]
    return R.contiguous()


@register
class CompandedQuant(Codec):
    name = "nf"

    def __init__(self, bits: int = 3, group: int = 128, scale_mode: str = "rms"):
        self.bits, self.group, self.scale_mode = bits, group, scale_mode

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        R = quantize_companded(W, self.bits, self.group, self.scale_mode)
        out_f, in_f = W.shape
        n_groups = out_f * ((in_f + self.group - 1) // self.group)
        bits = {"codes": float(W.numel() * self.bits),
                "scales": float(n_groups * 16)}     # one scale per group, no zero point
        return MatrixResult(R, bits)

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(
            flops_per_weight=1.0,                   # LUT lookup + one multiply by the scale
            bytes_streamed_per_weight=self.bits / 8 + 2.0 / self.group,
            shared_state_bytes=2 ** self.bits * 2,  # the level table, tens of bytes
            random_lookups_per_weight=0.0,          # a 4-16 entry LUT lives in registers
            notes=f"Lloyd-Max {self.bits}-bit levels, per-group {self.scale_mode} scale; "
                  "NF4-class baseline, same kernel shape as INT")
