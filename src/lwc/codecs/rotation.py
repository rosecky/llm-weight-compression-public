"""Strong modern baseline -- random Hadamard rotation (incoherence processing) + INT.

QuIP#/QuaRot/SpinQuant style. W' = S_L H W H S_R with H a Hadamard matrix and S random signs.
The rotation makes the weight marginal closer to Gaussian and removes outlier channels, which
is what actually buys the bits at 2-3 bpw.

Note the direction of the effect, which matters for this project's thesis: the rotation
*destroys* structure and *improves* compressibility. It is included as the baseline that the
procedural codecs have to beat.

In deployment the rotations fold into neighbouring ops (the activation side pays an
O(n log n) transform per token), so decode of the weight itself stays a plain dequant.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from .base import Codec, DecodeCost, MatrixResult, register
from .scalar import groupwise_bits, quantize_groupwise


def _is_pow2(n: int) -> bool:
    return n & (n - 1) == 0 and n > 0


@torch.no_grad()
def hadamard_transform(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Fast Walsh-Hadamard transform along `dim`; length must be a power of two.

    Normalised so the transform is orthonormal.
    """
    X = X.transpose(dim, -1).contiguous()
    shape = X.shape
    n = shape[-1]
    assert _is_pow2(n), f"FWHT needs a power-of-two length, got {n}"
    Y = X.reshape(-1, n)
    h = 1
    while h < n:
        Y = Y.reshape(-1, n // (2 * h), 2, h)
        a = Y[:, :, 0, :]
        b = Y[:, :, 1, :]
        Y = torch.stack([a + b, a - b], dim=2)
        h *= 2
    Y = Y.reshape(shape) / math.sqrt(n)
    return Y.transpose(dim, -1).contiguous()


def _largest_pow2_divisor(n: int, cap: int = 256) -> int:
    b = 1
    while n % (2 * b) == 0 and 2 * b <= cap:
        b *= 2
    return b


@torch.no_grad()
def _block_hadamard(X: torch.Tensor, dim: int, block: int) -> torch.Tensor:
    """Block-diagonal FWHT along `dim`. Handles non-power-of-two lengths, which is the normal
    case in real models (Qwen2.5-0.5B is 896 x 4864 -- neither side is a power of two)."""
    X = X.transpose(dim, -1).contiguous()
    shape = X.shape
    n = shape[-1]
    Y = X.reshape(-1, n // block, block)
    Y = hadamard_transform(Y, dim=-1)
    Y = Y.reshape(shape)
    return Y.transpose(dim, -1).contiguous()


@torch.no_grad()
def random_rotate(W: torch.Tensor, seed: int = 0, both_sides: bool = True,
                  rounds: int = 2, cap: int = 256):
    """Sign flip + block Hadamard + permutation, repeated, on each side.

    Two rounds with a permutation in between mix across block boundaries, so the result
    behaves like a dense random rotation while costing O(n log b) per token on the
    activation side. The whole thing is described by a seed, so it costs no storage.
    """
    g = torch.Generator(device=W.device).manual_seed(seed)
    out_f, in_f = W.shape
    R = W
    ops = []
    for side, n, dim in (("right", in_f, 1), ("left", out_f, 0)):
        if side == "left" and not both_sides:
            continue
        b = _largest_pow2_divisor(n, cap)
        if b < 8:                                   # not enough structure to be worth it
            continue
        for _ in range(rounds):
            s = (torch.randint(0, 2, (n,), generator=g, device=W.device) * 2 - 1).float()
            p = torch.randperm(n, generator=g, device=W.device)
            R = R * (s if dim == 1 else s.unsqueeze(1))
            R = _block_hadamard(R, dim, b)
            R = R[:, p] if dim == 1 else R[p, :]
            ops.append((dim, b, s, p))
    return R.contiguous(), ops


@torch.no_grad()
def undo_rotate(R: torch.Tensor, ops) -> torch.Tensor:
    W = R
    for (dim, b, s, p) in reversed(ops):
        inv = torch.empty_like(p)
        inv[p] = torch.arange(p.numel(), device=p.device)
        W = W[:, inv] if dim == 1 else W[inv, :]
        W = _block_hadamard(W, dim, b)              # Hadamard is its own inverse (orthonormal)
        W = W * (s if dim == 1 else s.unsqueeze(1))
    return W


@register
class RotatedScalarQuant(Codec):
    """Hadamard-rotated group-wise INT quantization."""

    name = "rht_int"

    def __init__(self, bits: int = 3, group: int = 128, seed: int = 0, both_sides: bool = True):
        self.bits, self.group, self.seed, self.both_sides = bits, group, seed, both_sides

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        R, inv = random_rotate(W, self.seed, self.both_sides)
        if not inv:                                   # no power-of-two side; fall back
            Rq = quantize_groupwise(W, self.bits, self.group)
            return MatrixResult(Rq, groupwise_bits(W, self.bits, self.group, False))
        Rq = quantize_groupwise(R, self.bits, self.group)
        Wh = undo_rotate(Rq, inv)
        bits = groupwise_bits(W, self.bits, self.group, symmetric=False)
        bits["rotation_seed"] = 32.0                  # the rotation is a seed, not a matrix
        return MatrixResult(Wh, bits)

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(
            flops_per_weight=2.0,
            bytes_streamed_per_weight=self.bits / 8 + 4.0 / self.group,
            shared_state_bytes=0.0,
            notes=f"INT{self.bits} g{self.group} in a rotated basis; weight decode is a plain "
                  "dequant, the Hadamard cost is paid on the activation side "
                  "(O(n log n) per token, amortised over the batch)")
