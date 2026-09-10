"""Permutation modulation -- the strongest form of the "procedural codebook" idea.

The recursive-composition codec navigates the signed-permutation group with random generators
and a beam search, which is hopeless: that group's *optimal* element for a given tile has a
closed form. Sorting finds it directly.

    encode:  rank-order |x| (one sort), keep the signs, fit one scale
    decode:  shuffle a single stored magnitude profile into that rank order, apply the signs,
             multiply by the scale

The implicit codebook has `d! * 2^d` entries (10.3 M for d=8) while the stored state is ONE
vector of d numbers. If a computed codebook can ever beat a stored one, this is where.
What the code cannot express is how far each order statistic deviates from the profile;
`resid_bits` optionally spends a little scalar quantization on exactly that.

This is Slepian's permutation modulation (1965), the ancestor of QTIP-style computed
codebooks, and it is the honest instantiation of "small shared system of transformations +
short local address".
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch

from ..structure import kmeans
from ..tiles import from_tiles, to_tiles
from .base import Codec, DecodeCost, MatrixResult, register
from .scalar import quantize_groupwise


@register
class PermutationModulationCodec(Codec):
    name = "permmod"
    needs_fit = True

    def __init__(self, th: int = 1, tw: int = 8, n_roots: int = 1, scale_bits: int = 8,
                 rank_block: int = 0, resid_bits: int = 0, resid_group: int = 64,
                 profile_bits: int = 16, perm_encoding: str = "entropy",
                 seed: int = 0, device: str = "cuda"):
        assert perm_encoding in ("entropy", "direct")
        self.th, self.tw, self.d = th, tw, th * tw
        self.R, self.scale_bits = n_roots, scale_bits
        # rank_block > 1 transmits only a coarse rank (which block of ranks), lowering the rate
        self.rank_block = rank_block
        self.resid_bits, self.resid_group = resid_bits, resid_group
        self.profile_bits, self.seed, self.device = profile_bits, seed, device
        # How the permutation is actually stored, which is NOT free:
        #   "entropy" -- log2(d!) bits, the true entropy. Decoding needs either an unranking
        #                table (d! * d bytes: 320 KB for d=8, far past shared memory) or a
        #                sequential Lehmer-code loop of ~d divisions. Charged in decode_cost.
        #   "direct"  -- d * ceil(log2 d) bits, a plain index per position. Costs more bits but
        #                decodes as one contiguous gather, which is what a kernel would want.
        self.perm_encoding = perm_encoding
        self.profiles: Optional[torch.Tensor] = None      # (R, d) descending magnitudes

    # ---------------------------------------------------------------- fit
    def fit(self, tile_stream: Iterable[torch.Tensor]) -> None:
        T = torch.cat([t for t in tile_stream], 0).float().to(self.device)
        srt, _ = torch.sort(T.abs(), dim=1, descending=True)
        srt = srt / srt.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if self.R == 1:
            prof = srt.mean(0, keepdim=True)
        else:
            prof = kmeans(srt, self.R, iters=20, seed=self.seed)
        prof, _ = torch.sort(prof.clamp_min(0), dim=1, descending=True)
        self.profiles = self._q(prof, self.profile_bits)

    @staticmethod
    def _q(x: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 16:
            return x
        lo, hi = x.amin(), x.amax()
        s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
        return torch.round((x - lo) / s) * s + lo

    def _rank_bits(self) -> float:
        """Bits to transmit the rank order: a full permutation, or a coarse block assignment."""
        if self.rank_block and self.rank_block > 1:
            nb = max(self.d // self.rank_block, 1)
            return self.d * math.ceil(math.log2(nb)) if nb > 1 else 0.0
        if self.perm_encoding == "direct":
            return self.d * math.ceil(math.log2(self.d))   # one index per position
        return math.lgamma(self.d + 1) / math.log(2.0)     # log2(d!), entropy-coded

    # ---------------------------------------------------------------- encode
    @torch.no_grad()
    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        out_f, in_f = W.shape
        T = to_tiles(W, self.th, self.tw).float()
        n = T.shape[0]
        order = torch.argsort(T.abs(), dim=1, descending=True)
        rank = torch.empty_like(order)
        ar = torch.arange(self.d, device=T.device).unsqueeze(0).expand(n, -1)
        rank.scatter_(1, order, ar)                        # rank[i, j] = position of coord j
        if self.rank_block and self.rank_block > 1:
            rank = (rank // self.rank_block) * self.rank_block + (self.rank_block - 1) // 2
        sign = torch.sign(T)
        sign[sign == 0] = 1.0

        if self.R == 1:
            base = self.profiles[0][rank] * sign
        else:
            # Chunked over tiles: the (R, n, d) candidate tensor is 4.5 GB for a 4864x896
            # matrix at R=256, so it must never be materialised whole.
            base = torch.empty_like(T)
            step = max(1, (1 << 22) // max(self.R * self.d, 1))
            for i in range(0, n, step):
                rk, sg, t = rank[i:i + step], sign[i:i + step], T[i:i + step]
                cand = self.profiles[:, rk].permute(1, 0, 2) * sg.unsqueeze(1)   # (m, R, d)
                sim = torch.einsum("md,mrd->mr", t, cand) / cand.norm(dim=2).clamp_min(1e-12)
                pick = sim.abs().argmax(1)
                base[i:i + step] = cand[torch.arange(t.shape[0], device=T.device), pick]
                del cand, sim

        s = (T * base).sum(1, keepdim=True) / base.pow(2).sum(1, keepdim=True).clamp_min(1e-12)
        ls = torch.log2(s.abs().clamp_min(1e-12))
        lo, hi = ls.amin(), ls.amax()
        step = ((hi - lo) / (2 ** (self.scale_bits - 1) - 1)).clamp_min(1e-12)
        sq = torch.sign(s) * torch.exp2(torch.round((ls - lo) / step) * step + lo)
        Wh = from_tiles(base * sq, out_f, in_f, self.th, self.tw)

        bits = {"rank_codes": float(n * self._rank_bits()),
                "sign_codes": float(n * self.d),
                "tile_scales": float(n * self.scale_bits)}
        if self.R > 1:
            bits["profile_ids"] = float(n * math.log2(self.R))
        if self.resid_bits > 0:
            Rq = quantize_groupwise(W - Wh, self.resid_bits, self.resid_group)
            n_groups = out_f * ((in_f + self.resid_group - 1) // self.resid_group)
            bits["resid_codes"] = float(W.numel() * self.resid_bits)
            bits["resid_scales"] = float(n_groups * 32)
            Wh = Wh + Rq
        return MatrixResult(Wh, bits)

    def shared_bits(self) -> Dict[str, float]:
        return {"magnitude_profiles": float(self.R * self.d * self.profile_bits)}

    def decode_cost(self) -> DecodeCost:
        # The permutation is not free at decode time, and the two encodings pay differently.
        unrank_flops = 0.0
        table_bytes = 0.0
        note = ""
        if not (self.rank_block and self.rank_block > 1):
            if self.perm_encoding == "entropy":
                # Lehmer-code unranking: ~d steps of divide + mod + a compaction, per tile
                unrank_flops = 4.0
                table_bytes = math.exp(math.lgamma(self.d + 1)) * self.d  # the LUT alternative
                note = ("; the log2(d!) code must be UNRANKED at decode: either ~4 ops/weight "
                        f"of sequential Lehmer decoding (branchy) or a {table_bytes/1024:.0f} KiB "
                        "unranking table, which does not fit in shared memory")
            else:
                unrank_flops = 0.0
                note = "; permutation stored as one index per position -> a plain gather"
        return DecodeCost(
            flops_per_weight=2.0 + unrank_flops + (3.0 if self.resid_bits else 0.0),
            bytes_streamed_per_weight=(self._rank_bits() + self.d + self.scale_bits
                                       + (math.log2(self.R) if self.R > 1 else 0.0))
            / 8.0 / self.d + (self.resid_bits / 8.0 if self.resid_bits else 0.0),
            shared_state_bytes=self.R * self.d * self.profile_bits / 8.0,
            random_lookups_per_weight=1.0 / self.d,
            branch_free=(self.perm_encoding == "direct"),
            notes=f"permutation modulation d={self.d}, {self.R} profile(s) = "
                  f"{self.R * self.d} stored numbers; implicit codebook of "
                  f"{self.R} x {self.d}! x 2^{self.d} entries; decode is one in-register "
                  f"shuffle + sign flip + scale" + note)
