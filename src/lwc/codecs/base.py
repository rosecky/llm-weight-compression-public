"""Codec interface.

Design constraints that matter for the research question:

* ``fit`` sees only a *stream* of sampled tiles -> encoders scale linearly and stay in
  bounded memory, so the same code path is applicable to a 70B model layer-wise.
* ``compress_matrix`` handles one matrix at a time -> no all-pairs, no global optimisation.
* every codec must report a complete :class:`BitBudget` and a :class:`DecodeCost`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..bits import BitBudget


@dataclass
class DecodeCost:
    """Rough decoder complexity, per reconstructed weight unless stated otherwise."""

    flops_per_weight: float = 0.0
    bytes_streamed_per_weight: float = 0.0     # per-tile state read from global memory
    shared_state_bytes: float = 0.0            # codebook / decoder params (want <= ~64 KB)
    random_lookups_per_weight: float = 0.0     # dependent loads -> bad
    tile_local: bool = True
    branch_free: bool = True
    iterative: bool = False
    notes: str = ""

    def practical_verdict(self) -> str:
        """Screening rule, deliberately blunt (see docs/PHASE0_PRIOR_ART.md)."""
        if self.iterative or not self.tile_local:
            return "impractical"
        if self.shared_state_bytes > 256 * 1024:
            return "poor (state exceeds L1/shared memory)"
        if self.flops_per_weight > 32:
            return "poor (arithmetic dominates the memory saving)"
        if self.flops_per_weight > 8 or self.random_lookups_per_weight > 0.25:
            return "medium"
        return "good"


@dataclass
class MatrixResult:
    recon: torch.Tensor
    bits: Dict[str, float] = field(default_factory=dict)   # per-matrix bits, by component


class Codec:
    """Base class. Subclasses override ``fit`` (optional) and ``compress_matrix``."""

    name: str = "codec"
    needs_fit: bool = False

    def fit(self, tile_stream: Iterable[torch.Tensor]) -> None:
        """Build shared state (codebook / decoder) from a stream of tile batches."""

    def shared_bits(self) -> Dict[str, float]:
        """Absolute bits of state shared across all matrices this codec encodes."""
        return {}

    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        raise NotImplementedError

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(notes="unspecified")

    # -------------------------------------------------------------- helpers
    def budget_for(self, results: List[MatrixResult], n_weights: int) -> BitBudget:
        b = BitBudget(n_weights=n_weights)
        for r in results:
            for k, v in r.bits.items():
                b.add_per_matrix(k, v)
        for k, v in self.shared_bits().items():
            b.add_shared(k, v)
        return b


REGISTRY: Dict[str, type] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kw) -> Codec:
    if name not in REGISTRY:
        raise KeyError(f"unknown codec {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
