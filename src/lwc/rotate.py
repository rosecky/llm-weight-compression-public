"""Randomized Hadamard (incoherence) transforms, with the Hessian transformed consistently.

A linear layer computes `y = W x`. Introduce orthogonal `Q` on the input dimension and
orthogonal `P` on the output dimension:

    W' = P W Q          =>      W = P^T W' Q^T

so `y = P^T W' (Q^T x)`: the rotated layer consumes rotated activations `x' = Q^T x` and
produces rotated outputs. The layer is therefore *functionally unchanged* -- we quantize `W'`
and reconstruct `What = P^T Q(W') Q^T`, which is a drop-in replacement for `W`. Nothing else
in the model has to change, so the end-to-end network before quantization is exactly
equivalent (verified numerically by `roundtrip_error`).

**The Hessian must be rotated too.** GPTQ's local loss uses `H = E[x x^T]`. In the rotated
basis the layer sees `x' = Q^T x`, so

    H' = E[x' x'^T] = Q^T H Q

Using this rather than `H` is the difference between GPTQ actually compensating error in the
rotated coordinates and silently optimising the wrong objective. `RightRotation.rotate_hessian`
does it without ever forming `Q` densely.

The left rotation `P` does not touch `H` at all: GPTQ treats output rows independently and `H`
is indexed by input channels only.

**Deployment caveat (this is a diagnostic, not a kernel).** Here we rotate, quantize, and
rotate back, so the stored object is `What` in native coordinates and the rate-distortion
geometry is measured exactly. A real QuIP#/QuaRot deployment instead absorbs `Q` and `P` into
neighbouring ops so the runtime applies them to *activations* (O(n log n) per token, amortised
over the batch) and the weight decode stays a plain dequant. That changes the decode cost, not
the distortion, which is what this experiment measures.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from .codecs.rotation import _largest_pow2_divisor, hadamard_transform


def _block_hadamard(X: torch.Tensor, block: int) -> torch.Tensor:
    """Block-diagonal FWHT along the LAST dimension (length need not be a power of two)."""
    shape = X.shape
    n = shape[-1]
    Y = X.reshape(-1, n // block, block)
    Y = hadamard_transform(Y, dim=-1)
    return Y.reshape(shape)


class Rotation:
    """An orthogonal operator on one axis, stored as a seed-cheap op list.

    Each round is: multiply by random signs, block Hadamard, random permutation. Two rounds
    mix across block boundaries so the result behaves like a dense random rotation while
    costing O(n log b) per vector. Storage is a seed, i.e. 32 bits, not an n x n matrix.
    """

    def __init__(self, n: int, seed: int = 0, rounds: int = 2, cap: int = 256,
                 device: str = "cuda"):
        self.n = n
        self.block = _largest_pow2_divisor(n, cap)
        self.enabled = self.block >= 8
        self.ops: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if not self.enabled:
            return
        g = torch.Generator(device=device).manual_seed(seed)
        for _ in range(rounds):
            s = (torch.randint(0, 2, (n,), generator=g, device=device) * 2 - 1).float()
            p = torch.randperm(n, generator=g, device=device)
            self.ops.append((s, p))

    # ---------------------------------------------------------------- forward / inverse
    def apply(self, X: torch.Tensor) -> torch.Tensor:
        """X (..., n) -> X @ Q. Operates on the last axis."""
        if not self.enabled:
            return X
        Y = X
        for (s, p) in self.ops:
            Y = _block_hadamard(Y * s, self.block)
            Y = Y[..., p]
        return Y.contiguous()

    def apply_inv(self, X: torch.Tensor) -> torch.Tensor:
        """X (..., n) -> X @ Q^T."""
        if not self.enabled:
            return X
        Y = X
        for (s, p) in reversed(self.ops):
            inv = torch.empty_like(p)
            inv[p] = torch.arange(p.numel(), device=p.device)
            Y = Y[..., inv]
            Y = _block_hadamard(Y, self.block) * s   # FWHT is its own inverse (orthonormal)
        return Y.contiguous()

    # ---------------------------------------------------------------- weights
    def apply_right(self, W: torch.Tensor) -> torch.Tensor:
        """W (out, in) -> W Q."""
        return self.apply(W)

    def apply_right_inv(self, W: torch.Tensor) -> torch.Tensor:
        """W (out, in) -> W Q^T."""
        return self.apply_inv(W)

    def apply_left(self, W: torch.Tensor) -> torch.Tensor:
        """W (out, in) -> P W, with P this rotation acting on the output axis."""
        return self.apply(W.T).T.contiguous()

    def apply_left_inv(self, W: torch.Tensor) -> torch.Tensor:
        return self.apply_inv(W.T).T.contiguous()

    # ---------------------------------------------------------------- Hessian
    def rotate_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """H (in, in) -> Q^T H Q, without ever forming Q.

        `apply` right-multiplies by Q, so:  apply(apply(H).T) = apply(Q^T H^T) = Q^T H Q
        (H is symmetric). The result is symmetric, so no final transpose is needed.
        """
        if not self.enabled:
            return H
        return self.apply(self.apply(H).T.contiguous())


class RotationPair:
    """Left (output-axis) and right (input-axis) rotations applied together."""

    def __init__(self, out_f: int, in_f: int, seed: int = 0, both_sides: bool = True,
                 device: str = "cuda"):
        self.right = Rotation(in_f, seed=seed, device=device)
        self.left = Rotation(out_f, seed=seed + 1337, device=device) if both_sides else None
        self.enabled = self.right.enabled or (self.left is not None and self.left.enabled)

    def forward_w(self, W: torch.Tensor) -> torch.Tensor:
        """W -> P W Q (the basis the quantizer sees)."""
        Wr = self.right.apply_right(W)
        return self.left.apply_left(Wr) if self.left is not None else Wr

    def inverse_w(self, Wq: torch.Tensor) -> torch.Tensor:
        """P^T Wq Q^T (back to native coordinates, a drop-in replacement for W)."""
        W = self.left.apply_left_inv(Wq) if self.left is not None else Wq
        return self.right.apply_right_inv(W)

    def forward_h(self, H: torch.Tensor) -> torch.Tensor:
        return self.right.rotate_hessian(H)

    def bits(self) -> float:
        """Storage: the rotation is a seed, not a matrix."""
        return 32.0 if self.enabled else 0.0


@torch.no_grad()
def roundtrip_error(W: torch.Tensor, rp: RotationPair) -> float:
    """Sanity check that the transform is exactly invertible (should be ~1e-13)."""
    return float((rp.inverse_w(rp.forward_w(W)) - W).pow(2).sum() / W.pow(2).sum())


@torch.no_grad()
def coordinate_stats(W: torch.Tensor) -> dict:
    """Incoherence diagnostics: what the rotation is supposed to change."""
    x = W.reshape(-1).float()
    mu, sd = x.mean(), x.std().clamp_min(1e-12)
    z = (x - mu) / sd
    k = int(max(1, round(0.0001 * x.numel())))
    return dict(
        kurtosis=float((z ** 4).mean()),
        max_over_rms=float(x.abs().max() / x.pow(2).mean().sqrt().clamp_min(1e-12)),
        top1e4_over_rms=float(x.abs().topk(k).values.mean()
                              / x.pow(2).mean().sqrt().clamp_min(1e-12)),
        row_rms_spread=float(W.pow(2).mean(1).sqrt().max()
                             / W.pow(2).mean(1).sqrt().mean().clamp_min(1e-12)),
        col_rms_spread=float(W.pow(2).mean(0).sqrt().max()
                             / W.pow(2).mean(0).sqrt().mean().clamp_min(1e-12)),
    )
