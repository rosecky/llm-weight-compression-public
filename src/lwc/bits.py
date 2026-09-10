"""Honest bit accounting.

Every codec must declare *all* storage it needs:
codes + codebooks + decoder params + scales + indices + residuals + outliers + metadata.

Shared state (a codebook or decoder used by many matrices) is amortized by the runner over
the total number of original weights it serves -- never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BitBudget:
    """Bits, broken out by component. `n_weights` is the number of ORIGINAL weights covered."""

    n_weights: int = 0
    per_matrix: Dict[str, float] = field(default_factory=dict)   # scales with #weights
    shared: Dict[str, float] = field(default_factory=dict)       # amortized shared state

    def add_per_matrix(self, key: str, bits: float) -> None:
        self.per_matrix[key] = self.per_matrix.get(key, 0.0) + float(bits)

    def add_shared(self, key: str, bits: float) -> None:
        self.shared[key] = self.shared.get(key, 0.0) + float(bits)

    def merge(self, other: "BitBudget") -> "BitBudget":
        """Merge another budget in. Shared components are taken as a max-union (same shared
        state counted once), per-matrix components add up."""
        out = BitBudget(n_weights=self.n_weights + other.n_weights)
        out.per_matrix = dict(self.per_matrix)
        for k, v in other.per_matrix.items():
            out.per_matrix[k] = out.per_matrix.get(k, 0.0) + v
        out.shared = dict(self.shared)
        for k, v in other.shared.items():
            out.shared[k] = max(out.shared.get(k, 0.0), v)
        return out

    @property
    def total_bits(self) -> float:
        return sum(self.per_matrix.values()) + sum(self.shared.values())

    @property
    def bpw(self) -> float:
        """Effective bits per ORIGINAL weight, decoder and codebook included."""
        return self.total_bits / max(self.n_weights, 1)

    @property
    def shared_bpw(self) -> float:
        return sum(self.shared.values()) / max(self.n_weights, 1)

    def breakdown(self) -> Dict[str, float]:
        d = {f"pm/{k}": v / max(self.n_weights, 1) for k, v in self.per_matrix.items()}
        d.update({f"sh/{k}": v / max(self.n_weights, 1) for k, v in self.shared.items()})
        return d

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.breakdown().items()) if v > 1e-6)
        return f"BitBudget(bpw={self.bpw:.4f}, n={self.n_weights}, [{parts}])"


def bits_fp(n: int, dtype_bits: int = 16) -> float:
    return float(n * dtype_bits)


def bits_index(n: int, cardinality: int) -> float:
    """n indices into a set of `cardinality` symbols, at the entropy-free (fixed-width) rate."""
    import math

    if cardinality <= 1:
        return 0.0
    return float(n) * math.log2(cardinality)
