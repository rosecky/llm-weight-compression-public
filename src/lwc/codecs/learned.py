"""Variant C -- shared learned decoder.

    z_t = E(W_t)   [encode time only, cost does not matter]
    W_t ~= D(z_t)  [decode time, must be cheap]

Three decoder families, in increasing order of decode cost:

* ``linear``     D(z) = A z + b.  Mathematically this is a shared basis, i.e. exactly the PCA
                 codec. Included so the "is a learned decoder better than a basis?" question is
                 answered on the same footing.
* ``separable``  the tile is decoded as  Ahat = P Z Q  with Z a small (a x b) code matrix.
                 FLOPs per weight ~ a + b, and P/Q live in registers. Kernel-friendly.
* ``mlp``        one hidden layer with GELU. FLOPs per weight = (k*h + h*d)/d.

The bottleneck is uniformly quantized with a straight-through estimator during training, so the
declared ``code_bits`` are actually paid for. Decoder parameters are counted as shared storage.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from ..tiles import from_tiles, to_tiles
from .base import Codec, DecodeCost, MatrixResult, register


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, levels):
        y = torch.clamp(x, -1.0, 1.0)
        return torch.round((y + 1.0) * 0.5 * (levels - 1)) / (levels - 1) * 2.0 - 1.0

    @staticmethod
    def backward(ctx, g):
        return g, None


class _SeparableDecoder(nn.Module):
    """Ahat = P @ Z @ Q, with Z the (a x b) code."""

    def __init__(self, th, tw, a, b):
        super().__init__()
        self.th, self.tw, self.a, self.b = th, tw, a, b
        self.P = nn.Parameter(torch.randn(th, a) / math.sqrt(a))
        self.Q = nn.Parameter(torch.randn(b, tw) / math.sqrt(b))

    def forward(self, z):
        Z = z.reshape(-1, self.a, self.b)
        return (self.P @ Z @ self.Q).reshape(z.shape[0], self.th * self.tw)

    def n_params(self):
        return self.P.numel() + self.Q.numel()

    def flops_per_weight(self):
        # P Z: th*a*b muls, then (th x b) @ Q: th*b*tw muls -> per weight of the th*tw tile
        return 2.0 * (self.a * self.b / self.tw + self.b)


class _LinearDecoder(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.lin = nn.Linear(k, d)
        self.k, self.d = k, d

    def forward(self, z):
        return self.lin(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_weight(self):
        return 2.0 * self.k


class _MLPDecoder(nn.Module):
    def __init__(self, k, h, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, d))
        self.k, self.h, self.d = k, h, d

    def forward(self, z):
        return self.net(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_weight(self):
        return 2.0 * (self.k * self.h + self.h * self.d) / self.d


@register
class LearnedDecoderCodec(Codec):
    """Shared tiny decoder + per-tile quantized code + per-tile scale."""

    name = "learned"
    needs_fit = True

    def __init__(self, th: int = 8, tw: int = 8, code_dim: int = 8, code_bits: int = 4,
                 arch: str = "separable", hidden: int = 64, sep_a: int = 4, sep_b: int = 4,
                 steps: int = 1500, lr: float = 3e-3, batch: int = 8192,
                 param_bits: int = 16, scale_bits: int = 8, seed: int = 0,
                 device: str = "cuda"):
        self.th, self.tw, self.d = th, tw, th * tw
        self.arch, self.code_bits, self.param_bits = arch, code_bits, param_bits
        self.scale_bits, self.seed, self.device = scale_bits, seed, device
        self.steps, self.lr, self.batch = steps, lr, batch
        self.code_dim = sep_a * sep_b if arch == "separable" else code_dim
        self.sep_a, self.sep_b, self.hidden = sep_a, sep_b, hidden
        self.dec = None
        self.enc = None

    # ---------------------------------------------------------------- build / fit
    def _build(self):
        torch.manual_seed(self.seed)
        if self.arch == "linear":
            dec = _LinearDecoder(self.code_dim, self.d)
        elif self.arch == "mlp":
            dec = _MLPDecoder(self.code_dim, self.hidden, self.d)
        elif self.arch == "separable":
            dec = _SeparableDecoder(self.th, self.tw, self.sep_a, self.sep_b)
        else:
            raise ValueError(self.arch)
        enc = nn.Sequential(nn.Linear(self.d, 2 * self.code_dim), nn.GELU(),
                            nn.Linear(2 * self.code_dim, self.code_dim), nn.Tanh())
        return dec.to(self.device), enc.to(self.device)

    def fit(self, tile_stream: Iterable[torch.Tensor]) -> None:
        T = torch.cat([t for t in tile_stream], 0).float().to(self.device)
        # per-tile scale removed first; the scale is stored separately and paid for
        s = T.norm(dim=1, keepdim=True).clamp_min(1e-12) / math.sqrt(self.d)
        Tn = T / s
        self.dec, self.enc = self._build()
        opt = torch.optim.Adam(list(self.dec.parameters()) + list(self.enc.parameters()),
                               lr=self.lr)
        levels = 2 ** self.code_bits
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        for step in range(self.steps):
            idx = torch.randint(0, Tn.shape[0], (min(self.batch, Tn.shape[0]),),
                                generator=g, device=self.device)
            x = Tn[idx]
            z = _FakeQuant.apply(self.enc(x), levels)
            loss = (self.dec(z) - x).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self._final_loss = float(loss)
        for p in list(self.dec.parameters()) + list(self.enc.parameters()):
            p.requires_grad_(False)

    # ---------------------------------------------------------------- encode
    @torch.no_grad()
    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        out_f, in_f = W.shape
        T = to_tiles(W, self.th, self.tw).float()
        s = T.norm(dim=1, keepdim=True).clamp_min(1e-12) / math.sqrt(self.d)
        # per-tile scale is itself quantized (log-domain, 8 bit) so it is honestly paid for
        ls = torch.log2(s)
        lo, hi = ls.amin(), ls.amax()
        step = ((hi - lo) / (2 ** self.scale_bits - 1)).clamp_min(1e-12)
        s = torch.exp2(torch.round((ls - lo) / step) * step + lo)
        Tn = T / s
        levels = 2 ** self.code_bits
        rec = torch.empty_like(Tn)
        for i in range(0, Tn.shape[0], 65536):
            x = Tn[i:i + 65536]
            z = _FakeQuant.apply(self.enc(x), levels)
            rec[i:i + 65536] = self.dec(z)
        Wh = from_tiles(rec * s, out_f, in_f, self.th, self.tw)
        n_tiles = T.shape[0]
        bits = {"codes": float(n_tiles * self.code_dim * self.code_bits),
                "tile_scales": float(n_tiles * self.scale_bits)}
        return MatrixResult(Wh, bits)

    def shared_bits(self) -> Dict[str, float]:
        # only the DECODER ships; the encoder is an offline tool
        return {"decoder": float(self.dec.n_params() * self.param_bits)}

    def decode_cost(self) -> DecodeCost:
        return DecodeCost(
            flops_per_weight=self.dec.flops_per_weight() + 1.0,   # +1 for the scale multiply
            bytes_streamed_per_weight=(self.code_dim * self.code_bits + self.scale_bits)
            / 8.0 / self.d,
            shared_state_bytes=self.dec.n_params() * self.param_bits / 8.0,
            random_lookups_per_weight=0.0,
            notes=f"{self.arch} decoder, code_dim={self.code_dim}, {self.code_bits}b codes")
