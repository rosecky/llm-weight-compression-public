"""Learned orthogonal changes of basis, as a *correction to Hadamard* rather than a rival to it.

The question is whether a fixed randomized Hadamard is unnecessarily generic: it is designed to
make coordinates incoherent in general, not to prepare a particular matrix for a particular
quantizer followed by a particular compensation algorithm. So can a learned basis do better at
the same bitrate?

## The design that makes the answer attributable

Every learned rotation here is parameterised so that **its initial value is exactly the
identity**, and it is composed *on top of* the existing `RotationPair`:

    W'' = P_learn (P_had W Q_had) Q_learn        A'' = Q_learn^T (Q_had^T A Q_had) Q_learn

At initialisation `P_learn = Q_learn = I`, so the arm is bit-for-bit the Hadamard baseline that
every earlier experiment in this repository used. Whatever the optimizer then buys is the value
of *learning*, measured against the strongest baseline rather than against a straw man. Setting
the fixed pair to identity instead gives the "can learning discover Hadamard from scratch" arm.

Identity at init is not free for the obvious parameterisation: a Householder reflection is never
close to the identity, whatever its vector. So the stack is used in pairs,

    R(V) = M(V) M(V0)^T,     M(V) = H(v_1) H(v_2) ... H(v_k),     H(v) = I - 2 v v^T / ||v||^2

which is exactly orthogonal for any `V`, exactly the identity at `V = V0`, and needs `k n`
parameters rather than `n^2`. Cayley and matrix-exponential parameterisations are identity-at-
zero too but need an `n x n` solve or exponential per step, which is not affordable at
`n = 4864`.

## Why the gradients are exact rather than straight-through

For a fixed group scale/zero, `q = round((w - z)/s) s + z` is piecewise constant in `w`, so
almost everywhere `d q / d w = 0` and the error `e = w - q` has `d e / d w = 1`. The gradient of
any smooth function of `e` is therefore available *exactly* by detaching the integer codes and
letting `s` and `z` carry gradient -- no straight-through estimator, no soft relaxation, no
temperature. Code changes are measure-zero events that the alternating outer loop handles by
re-running the real quantizer.

## What it costs

A Hadamard is a seed: 32 bits. A learned rotation is `k n` real numbers per axis, and that is
storage, so it is charged. `param_bits` returns it and the experiments amortise it over the
weights the transform serves -- per matrix for a per-matrix rotation, per block for a shared one.
The difference is large: `k=8` on a 896-wide input is 0.14 bits/weight for one 896x896 matrix and
0.02 for one shared across a block.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .rotate import RotationPair


class HouseholderRotation:
    """`R = M(V) M(V0)^T` as a product of `k` Householder reflections per factor.

    Orthogonal by construction at every point of the parameter space, so there is no
    orthogonality penalty to tune and no drift to monitor; and equal to the identity at
    initialisation, so composing it with a fixed transform starts exactly at that transform.
    """

    def __init__(self, n: int, k: int = 8, device: str = "cuda", seed: int = 0,
                 scale: float = 1.0):
        g = torch.Generator(device=device).manual_seed(seed)
        V0 = torch.randn(k, n, generator=g, device=device)
        V0 = V0 / V0.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.V0 = V0 * scale
        self.V = self.V0.clone().requires_grad_(True)
        self.n, self.k = n, k

    # -------------------------------------------------------------- core
    @staticmethod
    def _stack(X: torch.Tensor, V: torch.Tensor, reverse: bool) -> torch.Tensor:
        """`X @ M(V)` (or `X @ M(V)^T` when reversed). Operates on the last axis."""
        order = range(V.shape[0] - 1, -1, -1) if reverse else range(V.shape[0])
        for i in order:
            v = V[i]
            v = v / v.norm().clamp_min(1e-12)
            X = X - 2.0 * (X @ v).unsqueeze(-1) * v
        return X

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        """`X @ R`."""
        return self._stack(self._stack(X, self.V, False), self.V0, True)

    def apply_inv(self, X: torch.Tensor) -> torch.Tensor:
        """`X @ R^T`."""
        return self._stack(self._stack(X, self.V0, False), self.V, True)

    # -------------------------------------------------------------- weights / Hessian
    def apply_right(self, W: torch.Tensor) -> torch.Tensor:
        return self.apply(W)

    def apply_right_inv(self, W: torch.Tensor) -> torch.Tensor:
        return self.apply_inv(W)

    def apply_left(self, W: torch.Tensor) -> torch.Tensor:
        return self.apply(W.T).T

    def apply_left_inv(self, W: torch.Tensor) -> torch.Tensor:
        return self.apply_inv(W.T).T

    def rotate_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """`R^T H R`, formed the same way `rotate.Rotation` does it."""
        return self.apply(self.apply(H).T)

    # -------------------------------------------------------------- bookkeeping
    def params(self) -> List[torch.Tensor]:
        return [self.V]

    def param_bits(self, meta_bits: int = 16) -> float:
        return float(self.k * self.n * meta_bits)

    def drift(self) -> float:
        """How far the learned rotation has moved from the identity, in Frobenius terms."""
        with torch.no_grad():
            E = torch.eye(self.n, device=self.V.device)
            return float((self.apply(E) - E).norm() / math.sqrt(2.0 * self.n))


class IdentityRotation:
    """Placeholder with the same interface, so arms can be written uniformly."""

    def apply(self, X):
        return X

    apply_inv = apply
    apply_right = apply
    apply_right_inv = apply
    apply_left = apply
    apply_left_inv = apply

    def rotate_hessian(self, H):
        return H

    def params(self) -> List[torch.Tensor]:
        return []

    def param_bits(self, meta_bits: int = 16) -> float:
        return 0.0

    def drift(self) -> float:
        return 0.0


class StackedRotation:
    """Two rotations applied in sequence, `X -> (X A) B`, so a learnable correction can sit on
    top of an arbitrary fixed starting basis rather than only on top of a `RotationPair`."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def apply(self, X):
        return self.second.apply(self.first.apply(X))

    def apply_inv(self, X):
        return self.first.apply_inv(self.second.apply_inv(X))

    apply_right = apply
    apply_right_inv = apply_inv

    def apply_left(self, W):
        return self.apply(W.T).T

    def apply_left_inv(self, W):
        return self.apply_inv(W.T).T

    def rotate_hessian(self, H):
        return self.apply(self.apply(H).T)

    def params(self):
        return list(self.first.params()) + list(self.second.params())

    def param_bits(self, meta_bits: int = 16) -> float:
        return self.first.param_bits(meta_bits) + self.second.param_bits(meta_bits)

    def drift(self) -> float:
        return self.second.drift()


class HaarRotation(IdentityRotation):
    """A genuinely uniform random orthogonal matrix, from the QR of a Gaussian.

    The randomized Hadamard is *also* a random orthogonal transform, but a structured one, so
    "learned beats Hadamard" and "learned beats a random rotation" are different claims and both
    controls are run. Dense, so it is only used where `n` is small enough to afford it.
    """

    def __init__(self, n: int, device: str = "cuda", seed: int = 0):
        g = torch.Generator(device=device).manual_seed(seed)
        Q, R = torch.linalg.qr(torch.randn(n, n, generator=g, device=device))
        self.Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)   # Haar measure needs the fix
        self.n = n

    def apply(self, X):
        return X @ self.Q

    def apply_inv(self, X):
        return X @ self.Q.T

    apply_right = apply
    apply_right_inv = apply_inv

    def apply_left(self, W):
        return (W.T @ self.Q).T

    def apply_left_inv(self, W):
        return (W.T @ self.Q.T).T

    def rotate_hessian(self, H):
        return self.Q.T @ H @ self.Q

    def param_bits(self, meta_bits: int = 16) -> float:
        return float(self.n * self.n * meta_bits)


class ComposedTransform:
    """A fixed transform (Hadamard, Haar, or nothing) with a learned correction on top.

    `forward_w` is what the quantizer sees; `inverse_w` puts the quantized result back into
    native coordinates, so the stored checkpoint is a drop-in replacement for `W` and the
    full-precision model is never touched at all. Functional equivalence is therefore exact by
    construction, not approximate -- `roundtrip_error` measures only floating-point noise.
    """

    def __init__(self, fixed_pair: Optional[RotationPair],
                 left: Optional[object] = None, right: Optional[object] = None):
        self.fixed = fixed_pair
        self.left = left or IdentityRotation()
        self.right = right or IdentityRotation()

    def forward_w(self, W: torch.Tensor) -> torch.Tensor:
        W1 = self.fixed.forward_w(W) if self.fixed is not None else W
        return self.left.apply_left(self.right.apply_right(W1))

    def inverse_w(self, Wq: torch.Tensor) -> torch.Tensor:
        W1 = self.right.apply_right_inv(self.left.apply_left_inv(Wq))
        return self.fixed.inverse_w(W1) if self.fixed is not None else W1

    def forward_h(self, A: torch.Tensor) -> torch.Tensor:
        A1 = self.fixed.forward_h(A) if self.fixed is not None else A
        A1 = self.right.rotate_hessian(A1)
        return 0.5 * (A1 + A1.T)

    def params(self) -> List[torch.Tensor]:
        return self.left.params() + self.right.params()

    def param_bits(self, meta_bits: int = 16) -> float:
        fixed = 32.0 if self.fixed is not None and self.fixed.enabled else 0.0
        return fixed + self.left.param_bits(meta_bits) + self.right.param_bits(meta_bits)

    def drift(self) -> Tuple[float, float]:
        return self.left.drift(), self.right.drift()


# ====================================================================== differentiable pieces


def group_scales(W: torch.Tensor, bits: int, group: int):
    """Per-(row, group) min-max scale and zero, differentiable in `W`.

    Identical in value to `gptq.ScalarQuantizer.find_params`, so the differentiable path and the
    real quantizer agree exactly at every point rather than approximately.
    """
    out_f, in_f = W.shape
    if in_f % group:
        raise ValueError("group %d must divide in_features %d" % (group, in_f))
    B = W.reshape(out_f, in_f // group, group)
    lo, hi = B.amin(2), B.amax(2)
    s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
    return s, lo


def expand_groups(s: torch.Tensor, in_f: int, group: int) -> torch.Tensor:
    return s.repeat_interleave(group, dim=1)[:, :in_f]


def quant_error(W: torch.Tensor, bits: int, group: int,
                codes: Optional[torch.Tensor] = None):
    """`W - What` with exact almost-everywhere gradients.

    `round` is piecewise constant, so detaching the integer codes and letting the scales carry
    gradient gives the *true* derivative of the error almost everywhere. Passing `codes`
    externally is what makes the compensation-aware objective possible: those are the integers
    the real GPTQ + coordinate-descent pipeline committed to, held fixed while the basis moves.
    """
    in_f = W.shape[1]
    s, z = group_scales(W, bits, group)
    sf, zf = expand_groups(s, in_f, group), expand_groups(z, in_f, group)
    if codes is None:
        codes = torch.round((W - zf) / sf).clamp(0, 2 ** bits - 1).detach()
    return W - (codes.float() * sf + zf), codes


@torch.no_grad()
def grid_distance(W: torch.Tensor, bits: int, group: int) -> float:
    """Mean |w - nearest level| in units of the step: 0 is on-grid, 0.5 is worst possible."""
    in_f = W.shape[1]
    s, z = group_scales(W, bits, group)
    sf, zf = expand_groups(s, in_f, group), expand_groups(z, in_f, group)
    u = (W - zf) / sf
    return float((u - torch.round(u).clamp(0, 2 ** bits - 1)).abs().mean())


@torch.no_grad()
def interaction_decomposition(D: torch.Tensor, A: torch.Tensor) -> Dict[str, float]:
    """Split the achieved damage into what the errors would do alone and what they do together.

    `D_independent = sum_j A_jj ||D[:, j]||^2` is the damage if every input channel's error
    acted with no cross-terms; `D_joint = tr(D A D^T)` is what actually happens. The difference
    is the compensation the pipeline manufactured. The graph study measured this for GPTQ and
    found the additive damage to be 7.7-10.8x the achieved one; the question here is whether a
    learned basis can make that ratio larger still.
    """
    adiag = torch.diagonal(A)
    d_ind = float((adiag.unsqueeze(0) * D * D).sum())
    d_joint = float(((D @ A) * D).sum())
    return dict(d_independent=d_ind, d_joint=d_joint, interaction=d_joint - d_ind,
                additive_over_joint=d_ind / max(d_joint, 1e-30))
