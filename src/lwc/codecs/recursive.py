"""IFS-inspired branch -- recursive / compositional operator representations.

    W_t ~= s_t * T_{k_d} ( ... T_{k_1} ( B_r ) ... )  +  R_t

A short per-tile *program* (root id + a sequence of transform ids + a scale) instead of a
direct dictionary index. The point is not that weights "look fractal". The point is the
storage arithmetic:

    reachable set size   = R * K^depth          (the effective codebook)
    stored parameters    = R*d + K*p            (roots + transform dictionary)

so a depth-2 grammar with R=256, K=16 reaches 65 536 codewords from ~18 k stored numbers --
a 227x amplification. The *rate* is identical to a direct codebook of the same reachable size
(log2(R) + depth*log2(K) bits), so recursion can never beat the source's rate-distortion
bound. What it can do is make a large effective codebook affordable to store. That is exactly
the trade QTIP makes with a computed codebook, and it is the hypothesis worth testing.

Two things are therefore measured for every family, and both are reported:

* **equal rate** -- against a trained codebook with the same number of codewords
  (which costs far more storage). Does composition lose distortion?
* **equal storage** -- against a trained codebook with the same stored parameters
  (which then affords far fewer codewords, i.e. a lower rate).

Algebraic-reduction warning (kill criterion R2): several natural families are *groups closed
under composition* -- scalar affine, diagonal scaling, sign flips, permutations. For those,
`T_b(T_a(B))` is just another member of the same one-step family, so depth adds no
expressiveness at all (though it may still add reachable-set coverage if the dictionary
generates a larger subgroup than it contains). `algebra.py` measures this directly.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..structure import kmeans
from ..tiles import from_tiles, to_tiles
from .base import Codec, DecodeCost, MatrixResult, register


# =============================================================== transform families


class TransformFamily:
    """A dictionary of K cheap transforms acting on d-dimensional tiles (th x tw)."""

    name = "base"
    orthogonal = False

    def __init__(self, K: int, th: int, tw: int, seed: int = 0, device: str = "cuda"):
        self.K, self.th, self.tw, self.d = K, th, tw, th * tw
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self._build()

    def _build(self):
        raise NotImplementedError

    def apply_all(self, X: torch.Tensor) -> torch.Tensor:
        """X: (n, d) -> (n, K, d): every transform applied to every input."""
        raise NotImplementedError

    def params_per_transform_bits(self) -> float:
        raise NotImplementedError

    def flops_per_weight_per_depth(self) -> float:
        raise NotImplementedError

    def notes(self) -> str:
        return ""


class SignPermFamily(TransformFamily):
    """T(B) = signed permutation of B's entries.

    Composition stays inside the signed-permutation group (order 2^d * d!), so depth cannot
    add expressiveness beyond the subgroup the K generators produce -- but that subgroup can
    be astronomically larger than K. This family is Slepian permutation modulation, a classic
    near-optimal source code for i.i.d. sources with essentially zero stored state.
    """

    name = "signperm"
    orthogonal = True

    def _build(self):
        self.perm = torch.stack([torch.randperm(self.d, generator=self.g, device=self.device)
                                 for _ in range(self.K)])
        self.sign = (torch.randint(0, 2, (self.K, self.d), generator=self.g,
                                   device=self.device) * 2 - 1).float()

    def apply_all(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, self.perm] * self.sign.unsqueeze(0)      # (n, K, d)

    def apply_one(self, X: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.gather(X, 1, self.perm[k]) * self.sign[k]

    def invert_one(self, X: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        Y = X * self.sign[k]
        out = torch.empty_like(Y)
        out.scatter_(1, self.perm[k], Y)
        return out

    def params_per_transform_bits(self) -> float:
        return self.d * math.log2(self.d) + self.d          # perm indices + sign bits

    def flops_per_weight_per_depth(self) -> float:
        return 1.0                                          # a shuffle + a sign flip

    def notes(self) -> str:
        return ("signed permutation; composes to a single signed permutation, so the DECODER "
                "can fold the whole path into one shuffle regardless of depth")


class DiagFamily(TransformFamily):
    """T(B) = D_r B D_c with diagonal row/column scalings. A commutative group: composition
    collapses to a single diagonal pair, so depth adds nothing (kill criterion R2)."""

    name = "diag"

    def _build(self):
        self.r = torch.exp(0.4 * torch.randn(self.K, self.th, generator=self.g,
                                             device=self.device))
        self.c = torch.exp(0.4 * torch.randn(self.K, self.tw, generator=self.g,
                                             device=self.device))

    def apply_all(self, X: torch.Tensor) -> torch.Tensor:
        M = X.reshape(-1, 1, self.th, self.tw)
        S = (self.r.unsqueeze(2) * self.c.unsqueeze(1)).unsqueeze(0)   # (1,K,th,tw)
        return (M * S).reshape(X.shape[0], self.K, self.d)

    def params_per_transform_bits(self) -> float:
        return (self.th + self.tw) * 16.0

    def flops_per_weight_per_depth(self) -> float:
        return 1.0

    def notes(self) -> str:
        return "diagonal row/col scaling; closed under composition -> depth is algebraically "\
               "redundant"


class BlockRotFamily(TransformFamily):
    """T(B) = A B C with A, C block-diagonal 2x2 Givens rotations on a shifted pairing.

    The shift makes successive transforms act on different coordinate pairs, so the family is
    NOT closed under composition -- depth genuinely enlarges the reachable operator set while
    each step still costs ~2 flops per weight. This is the butterfly-like case.
    """

    name = "blockrot"
    orthogonal = True

    def _build(self):
        self.ang_r = torch.rand(self.K, self.th // 2, generator=self.g,
                                device=self.device) * 2 * math.pi
        self.ang_c = torch.rand(self.K, self.tw // 2, generator=self.g,
                                device=self.device) * 2 * math.pi
        self.shift_r = torch.randint(0, max(self.th, 1), (self.K,), generator=self.g,
                                     device=self.device)
        self.shift_c = torch.randint(0, max(self.tw, 1), (self.K,), generator=self.g,
                                     device=self.device)

    def _rot_matrix(self, ang, shift, n):
        """Build the (K, n, n) block rotation with a per-transform cyclic shift of the pairing."""
        K = ang.shape[0]
        M = torch.zeros(K, n, n, device=self.device)
        idx = torch.arange(n, device=self.device)
        for k in range(K):
            p = (idx + int(shift[k])) % n
            M[k] = torch.eye(n, device=self.device)
            for j in range(n // 2):
                a, b = int(p[2 * j]), int(p[2 * j + 1])
                c, s = torch.cos(ang[k, j]), torch.sin(ang[k, j])
                M[k, a, a] = c
                M[k, a, b] = -s
                M[k, b, a] = s
                M[k, b, b] = c
        return M

    def build_matrices(self):
        if not hasattr(self, "_A"):
            self._A = self._rot_matrix(self.ang_r, self.shift_r, self.th)
            self._C = self._rot_matrix(self.ang_c, self.shift_c, self.tw)
        return self._A, self._C

    def apply_all(self, X: torch.Tensor) -> torch.Tensor:
        A, C = self.build_matrices()
        M = X.reshape(-1, self.th, self.tw)
        out = torch.einsum("kij,njm,kml->nkil", A, M, C)
        return out.reshape(X.shape[0], self.K, self.d)

    def params_per_transform_bits(self) -> float:
        return (self.th // 2 + self.tw // 2) * 16.0 + 2 * math.log2(max(self.th, 2))

    def flops_per_weight_per_depth(self) -> float:
        return 6.0                                   # two Givens passes, 3 flops each per entry

    def notes(self) -> str:
        return "block 2x2 Givens rotations with a per-transform pairing shift; NOT closed "\
               "under composition, so depth adds real expressiveness"


class AffineFamily(TransformFamily):
    """T(B) = a B + b. The textbook example of algebraic collapse: T_b(T_a(B)) is affine."""

    name = "affine"

    def _build(self):
        self.a = torch.exp(0.3 * torch.randn(self.K, generator=self.g, device=self.device))
        self.b = 0.2 * torch.randn(self.K, generator=self.g, device=self.device)

    def apply_all(self, X: torch.Tensor) -> torch.Tensor:
        return X.unsqueeze(1) * self.a.view(1, -1, 1) + self.b.view(1, -1, 1)

    def params_per_transform_bits(self) -> float:
        return 32.0

    def flops_per_weight_per_depth(self) -> float:
        return 2.0

    def notes(self) -> str:
        return "scalar affine; composition collapses exactly -> depth is pure waste"


class LowRankAddFamily(TransformFamily):
    """T(B) = B + u v^T. Composition gives B + sum of rank-1 terms, i.e. this is additive
    refinement in disguise -- expressiveness grows with depth but only as a rank-d update."""

    name = "lowrank_add"

    def _build(self):
        self.u = 0.5 * torch.randn(self.K, self.th, generator=self.g, device=self.device)
        self.v = 0.5 * torch.randn(self.K, self.tw, generator=self.g, device=self.device)

    def apply_all(self, X: torch.Tensor) -> torch.Tensor:
        M = X.reshape(-1, 1, self.th, self.tw)
        R = (self.u.unsqueeze(2) * self.v.unsqueeze(1)).unsqueeze(0)
        return (M + R).reshape(X.shape[0], self.K, self.d)

    def params_per_transform_bits(self) -> float:
        return (self.th + self.tw) * 16.0

    def flops_per_weight_per_depth(self) -> float:
        return 1.0

    def notes(self) -> str:
        return "additive rank-1 update; equivalent to additive refinement with rank-1 atoms"


FAMILIES = {f.name: f for f in [SignPermFamily, DiagFamily, BlockRotFamily,
                                AffineFamily, LowRankAddFamily]}


# =============================================================== the codec


@register
class RecursiveGrammarCodec(Codec):
    """Root + depth-d composition of shared transforms + optional per-tile scale."""

    name = "recursive"
    needs_fit = True

    def __init__(self, th: int = 8, tw: int = 8, n_roots: int = 256, n_transforms: int = 16,
                 depth: int = 2, family: str = "signperm", scale_bits: int = 8,
                 root_bits: int = 16, param_bits: int = 16, refine_iters: int = 2,
                 max_reachable: int = 1 << 20, beam: int = 16, seed: int = 0,
                 device: str = "cuda", entropy_code: bool = True):
        self.th, self.tw, self.d = th, tw, th * tw
        self.R, self.K, self.depth = n_roots, n_transforms, depth
        self.family_name, self.scale_bits = family, scale_bits
        self.root_bits, self.param_bits = root_bits, param_bits
        self.refine_iters, self.max_reachable, self.beam = refine_iters, max_reachable, beam
        self.seed, self.device, self.entropy_code = seed, device, entropy_code
        self.fam: Optional[TransformFamily] = None
        self.roots: Optional[torch.Tensor] = None
        self.reach: Optional[torch.Tensor] = None      # (n_cand, d) materialised reachable set
        self.paths: Optional[torch.Tensor] = None      # (n_cand, 1+depth) root id + transform ids
        self.stats: Dict = {}

    # ---------------------------------------------------------------- reachable set
    def _expand(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialise the reachable set. Encoding-time only -- the decoder never does this."""
        cur = self.roots                                            # (R, d)
        paths = torch.arange(self.R, device=self.device).unsqueeze(1)
        for _ in range(self.depth):
            nxt = self.fam.apply_all(cur)                           # (n, K, d)
            n = cur.shape[0]
            cur = nxt.reshape(n * self.K, self.d)
            ids = torch.arange(self.K, device=self.device).repeat(n).unsqueeze(1)
            paths = torch.repeat_interleave(paths, self.K, dim=0)
            paths = torch.cat([paths, ids], dim=1)
            if cur.shape[0] > self.max_reachable:                   # keep memory bounded
                g = torch.Generator(device=self.device).manual_seed(self.seed)
                sel = torch.randperm(cur.shape[0], generator=g,
                                     device=self.device)[:self.max_reachable]
                cur, paths = cur[sel].contiguous(), paths[sel].contiguous()
        return cur.contiguous(), paths.contiguous()

    # ---------------------------------------------------------------- fit
    def fit(self, tile_stream: Iterable[torch.Tensor]) -> None:
        T = torch.cat([t for t in tile_stream], 0).float().to(self.device)
        Tn = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.fam = FAMILIES[self.family_name](self.K, self.th, self.tw,
                                              seed=self.seed, device=self.device)
        self.roots = kmeans(Tn, self.R, iters=20, seed=self.seed)
        self.roots = self.roots / self.roots.norm(dim=1, keepdim=True).clamp_min(1e-12)

        for _ in range(max(self.refine_iters, 0)):
            if self.family_name != "signperm":
                break                                   # root refinement needs invertibility
            _, p = self._encode_beam(Tn)
            self._refine_roots(Tn, p)
        self._collect_stats(Tn)

    def _refine_roots(self, Tn: torch.Tensor, p: torch.Tensor) -> None:
        """Pull each assigned tile back through its (orthogonal) path and re-average the root.

        This is Lloyd's algorithm for the composite code: the transforms stay fixed, the roots
        move to the centroid of what their programs are actually asked to represent.
        """
        if self.family_name != "signperm":
            return
        back = Tn
        for lvl in range(self.depth, 0, -1):
            back = self.fam.invert_one(back, p[:, lvl])
        rid = p[:, 0]
        sums = torch.zeros_like(self.roots).index_add_(0, rid, back)
        cnt = torch.zeros(self.R, device=self.device).index_add_(
            0, rid, torch.ones(rid.shape[0], device=self.device))
        newr = sums / cnt.clamp_min(1).unsqueeze(1)
        keep = cnt > 0
        self.roots[keep] = (newr[keep] /
                            newr[keep].norm(dim=1, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def _encode_beam(self, Tn: torch.Tensor, chunk: int = 4096):
        """Beam search over programs. Returns (codeword vectors, paths).

        Cost is O(n_tiles * (R + depth * beam * K) * d) -- linear in the number of tiles, with
        no all-pairs search anywhere, so kill criterion R8 is satisfied by construction.
        Scoring uses |<x, c>| / ||c|| because the per-tile scale is fitted afterwards.
        """
        n, d = Tn.shape
        beam = self.beam
        outC = torch.empty(n, d, device=Tn.device)
        outP = torch.empty(n, 1 + self.depth, dtype=torch.long, device=Tn.device)
        Rn = self.roots / self.roots.norm(dim=1, keepdim=True).clamp_min(1e-12)
        for i in range(0, n, chunk):
            x = Tn[i:i + chunk]                                   # (m, d)
            m = x.shape[0]
            sc = (x @ Rn.T).abs()                                 # (m, R)
            b = min(beam, self.R)
            top = sc.topk(b, dim=1).indices                       # (m, b)
            cand = self.roots[top]                                # (m, b, d)
            paths = top.unsqueeze(2)                              # (m, b, 1)
            for _ in range(self.depth):
                exp = self.fam.apply_all(cand.reshape(m * cand.shape[1], d))
                exp = exp.reshape(m, -1, d)                       # (m, b*K, d)
                nrm = exp.norm(dim=2).clamp_min(1e-12)
                s = (torch.einsum("md,mcd->mc", x, exp) / nrm).abs()
                bb = min(beam, exp.shape[1])
                top = s.topk(bb, dim=1).indices                   # (m, bb)
                cand = torch.gather(exp, 1, top.unsqueeze(2).expand(-1, -1, d))
                prev = torch.gather(paths, 1,
                                    (top // self.K).unsqueeze(2).expand(-1, -1, paths.shape[2]))
                paths = torch.cat([prev, (top % self.K).unsqueeze(2)], dim=2)
            nrm = cand.norm(dim=2).clamp_min(1e-12)
            s = (torch.einsum("md,mcd->mc", x, cand) / nrm).abs()
            best = s.argmax(1)
            ar = torch.arange(m, device=Tn.device)
            outC[i:i + chunk] = cand[ar, best]
            outP[i:i + chunk] = paths[ar, best]
        return outC, outP

    # ---------------------------------------------------------------- grammar statistics
    @torch.no_grad()
    def _collect_stats(self, Tn: torch.Tensor) -> None:
        sub = Tn[: min(Tn.shape[0], 100_000)]
        _, p = self._encode_beam(sub)
        n_reach = self.R * (self.K ** self.depth)
        mult = torch.tensor([self.K ** (self.depth - i) for i in range(self.depth)],
                            device=p.device, dtype=torch.long) if self.depth else None
        idx = p[:, 0] * (self.K ** self.depth)
        if self.depth:
            idx = idx + (p[:, 1:] * mult.unsqueeze(0) // self.K).sum(1)
        used = torch.unique(idx)
        counts = torch.bincount(idx, minlength=n_reach).float()
        prob = counts[counts > 0] / counts.sum()
        path_entropy = float(-(prob * prob.log2()).sum())
        tf = p[:, 1:].reshape(-1)
        tcounts = torch.bincount(tf, minlength=self.K).float() if self.depth else \
            torch.zeros(self.K)
        tprob = tcounts / tcounts.sum().clamp_min(1)
        rcounts = torch.bincount(p[:, 0], minlength=self.R).float()
        stats = dict(
            n_reachable=int(n_reach),
            n_unique_used=int(used.numel()),
            frac_reachable_used=float(used.numel()) / float(n_reach),
            path_entropy_bits=path_entropy,
            fixed_width_bits=math.log2(self.R) + self.depth * math.log2(max(self.K, 1)),
            transform_usage_entropy=float(-(tprob[tprob > 0] *
                                            tprob[tprob > 0].log2()).sum()) if self.depth else 0.0,
            transform_usage_max_frac=float(tprob.max()) if self.depth else 0.0,
            root_reuse_mean=float(rcounts[rcounts > 0].mean()),
            root_reuse_median=float(rcounts[rcounts > 0].median()),
            codeword_reuse_mean=float(counts[counts > 0].mean()),
            n_probe_tiles=int(min(Tn.shape[0], 100_000)),
        )
        if self.depth >= 2:
            bg = p[:, 1] * self.K + p[:, 2] if self.depth >= 2 else None
            bc = torch.bincount(bg, minlength=self.K * self.K).float()
            bp = bc / bc.sum().clamp_min(1)
            stats["bigram_entropy_bits"] = float(-(bp[bp > 0] * bp[bp > 0].log2()).sum())
            stats["bigram_max_frac"] = float(bp.max())
            stats["bigram_uniform_bits"] = 2 * math.log2(max(self.K, 1))
        self.stats = stats

    # ---------------------------------------------------------------- encode
    @torch.no_grad()
    def compress_matrix(self, W: torch.Tensor) -> MatrixResult:
        out_f, in_f = W.shape
        T = to_tiles(W, self.th, self.tw).float()
        nrm = T.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Tn = T / nrm
        C, _ = self._encode_beam(Tn)
        cn2 = C.pow(2).sum(1, keepdim=True).clamp_min(1e-12)
        s = (T * C).sum(1, keepdim=True) / cn2                  # optimal signed scale
        # log-domain quantization of |s| plus one sign bit
        ls = torch.log2(s.abs().clamp_min(1e-12))
        lo, hi = ls.amin(), ls.amax()
        step = ((hi - lo) / (2 ** (self.scale_bits - 1) - 1)).clamp_min(1e-12)
        sq = torch.sign(s) * torch.exp2(torch.round((ls - lo) / step) * step + lo)
        rec = C * sq
        n_tiles = T.shape[0]
        code_bits = math.log2(self.R) + self.depth * math.log2(max(self.K, 1))
        bits = {"program_codes": float(n_tiles * code_bits),
                "tile_scales": float(n_tiles * self.scale_bits)}
        return MatrixResult(from_tiles(rec, out_f, in_f, self.th, self.tw), bits)

    def shared_bits(self) -> Dict[str, float]:
        return {"roots": float(self.R * self.d * self.root_bits),
                "transform_dict": float(self.K * self.fam.params_per_transform_bits())}

    def decode_cost(self) -> DecodeCost:
        per_depth = self.fam.flops_per_weight_per_depth()
        # signed permutations fold: the whole path is one shuffle at decode time
        folds = self.family_name in ("signperm", "diag", "affine")
        eff_depth = 1.0 if folds else float(self.depth)
        return DecodeCost(
            flops_per_weight=eff_depth * per_depth + 1.0,        # + the scale multiply
            bytes_streamed_per_weight=(math.log2(self.R)
                                       + self.depth * math.log2(max(self.K, 1))
                                       + self.scale_bits) / 8.0 / self.d,
            shared_state_bytes=(self.R * self.d * self.root_bits
                                + self.K * self.fam.params_per_transform_bits()) / 8.0,
            random_lookups_per_weight=1.0 / self.d,
            notes=f"depth-{self.depth} {self.family_name} grammar, R={self.R}, K={self.K}; "
                  + self.fam.notes()
                  + ("; path folds to one operator at decode time" if folds else ""))
