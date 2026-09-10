"""Phase 8: do the same low-bit codes want to mean different things in different groups?

Every group already stores `lo` and `hi`, so it already has its own scale and offset. The one
degree of freedom left is the *shape* of the reconstruction levels inside that range. So:
learn a small shared dictionary of `T` level-shapes, and let each group pick one, paying
log2(T) bits for the choice.

    w = lo_g + (hi_g - lo_g) * L[t_g][q]

`T = 1` with uniform levels is exactly the `ScalarQuantizer` baseline; `T = 1` with learned
levels is a global companded quantizer of the NF4 family; `T > 1` is the new thing -- locally
adaptive code semantics. The question is whether groups are heterogeneous enough for the
choice to be worth its log2(T)/group bits.

The matched-null arm answers the obvious objection: a shape dictionary will always fit
*something*, so it is also run on an i.i.d. Gaussian with matched row/column scales. Whatever
the null achieves is not structure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

from ..gptq import Quantizer, gptq
from ..metrics import nmse
from ..modelio import get_module, list_linear_layers, load_model, total_weights
from ..rotate import RotationPair
from ..structure import make_null
from ..tailmetrics import tail_report


# ---------------------------------------------------------------- the quantizer


class LocalShapeQuantizer(Quantizer):
    """Group-wise min-max range with a per-group choice from a shared dictionary of shapes."""

    atom = 1

    def __init__(self, shapes: torch.Tensor, group: int = 128, meta_bits: int = 16):
        self.L = shapes                                   # (T, K), sorted, in [0, 1]
        self.T, self.K = shapes.shape
        self.bits = math.log2(self.K)
        self.group = group
        self.meta_bits = meta_bits
        self.lo: Optional[torch.Tensor] = None
        self.rng: Optional[torch.Tensor] = None
        self.sel: Optional[torch.Tensor] = None
        self.Lsel: Optional[torch.Tensor] = None

    def find_params(self, W_group: torch.Tensor, col: int = 0) -> None:
        lo = W_group.amin(dim=1, keepdim=True)
        hi = W_group.amax(dim=1, keepdim=True)
        self.lo = lo
        self.rng = (hi - lo).clamp_min(1e-12)
        U = (W_group - lo) / self.rng                     # (out, g) in [0, 1]
        if self.T == 1:
            self.sel = torch.zeros(U.shape[0], dtype=torch.long, device=U.device)
        else:
            err = torch.empty(U.shape[0], self.T, device=U.device)
            for t in range(self.T):                       # loop over T keeps memory bounded
                d = (U.unsqueeze(-1) - self.L[t].reshape(1, 1, -1)).abs().amin(-1)
                err[:, t] = d.pow(2).sum(1)
            self.sel = err.argmin(1)
        self.Lsel = self.L[self.sel]                      # (out, K)

    def quantize(self, w: torch.Tensor, col: int) -> torch.Tensor:
        u = (w - self.lo) / self.rng                      # (out, 1)
        q = (u - self.Lsel).abs().argmin(1, keepdim=True)
        return self.lo + self.rng * self.Lsel.gather(1, q)

    def storage_bits(self, out_f: int, in_f: int) -> Dict[str, float]:
        n_groups = out_f * ((in_f + self.group - 1) // self.group)
        out = {"codes": float(out_f * in_f * self.bits),
               "scales": float(n_groups * 2 * self.meta_bits)}
        if self.T > 1:
            out["shape_ids"] = float(n_groups * math.log2(self.T))
        return out

    def shared_bits(self) -> float:
        return float(self.L.numel() * self.meta_bits)


# ---------------------------------------------------------------- fitting the dictionary


@torch.no_grad()
def lloyd_1d(x: torch.Tensor, K: int, iters: int = 30, init: Optional[torch.Tensor] = None):
    """Lloyd-Max levels for a 1-D sample, clamped to keep the extremes representable."""
    L = init.clone() if init is not None else torch.quantile(
        x, torch.linspace(0, 1, K, device=x.device))
    for _ in range(iters):
        a = (x.unsqueeze(1) - L.unsqueeze(0)).abs().argmin(1)
        s = torch.zeros(K, device=x.device).index_add_(0, a, x)
        c = torch.zeros(K, device=x.device).index_add_(0, a, torch.ones_like(x))
        newL = torch.where(c > 0, s / c.clamp_min(1), L)
        L = torch.sort(newL).values
    return L


@torch.no_grad()
def fit_shapes(U: torch.Tensor, K: int, T: int, iters: int = 8, seed: int = 0,
               sub: int = 200_000) -> torch.Tensor:
    """U: (n_groups, g) of normalised group values. Returns (T, K) shapes in [0, 1]."""
    dev = U.device
    g = torch.Generator(device=dev).manual_seed(seed)
    flat = U.reshape(-1)
    if flat.numel() > sub:
        flat = flat[torch.randperm(flat.numel(), generator=g, device=dev)[:sub]]
    base = lloyd_1d(flat, K)
    if T == 1:
        return base.reshape(1, K)
    # seed the dictionary by splitting groups on a shape statistic (how peaked they are),
    # then alternate assignment / refit
    stat = U.pow(2).mean(1)
    qs = torch.quantile(stat, torch.linspace(0, 1, T + 1, device=dev)[1:-1])
    assign = torch.bucketize(stat, qs)
    L = torch.stack([base] * T)
    for _ in range(iters):
        for t in range(T):
            sel = U[assign == t]
            if sel.numel() < 1000:
                continue
            f = sel.reshape(-1)
            if f.numel() > sub:
                f = f[torch.randperm(f.numel(), generator=g, device=dev)[:sub]]
            L[t] = lloyd_1d(f, K, iters=15, init=L[t])
        err = torch.empty(U.shape[0], T, device=dev)
        for t in range(T):
            err[:, t] = (U.unsqueeze(-1) - L[t].reshape(1, 1, -1)).abs().amin(-1).pow(2).sum(1)
        assign = err.argmin(1)
    return L


@torch.no_grad()
def collect_normalised(mats: List[torch.Tensor], group: int, max_groups: int = 60_000,
                       seed: int = 0) -> torch.Tensor:
    out = []
    g = torch.Generator(device=mats[0].device).manual_seed(seed)
    for W in mats:
        ng = W.shape[1] // group
        B = W[:, :ng * group].reshape(-1, group)
        lo = B.amin(1, keepdim=True)
        rng = (B.amax(1, keepdim=True) - lo).clamp_min(1e-12)
        out.append((B - lo) / rng)
    U = torch.cat(out, 0)
    if U.shape[0] > max_groups:
        U = U[torch.randperm(U.shape[0], generator=g, device=U.device)[:max_groups]]
    return U


# ---------------------------------------------------------------- runner


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--calib", default="cache/calib.pt")
    ap.add_argument("--bits", default="2,3")
    ap.add_argument("--ts", default="1,2,4,8,16")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--comps", default="naive,gptq")
    ap.add_argument("--nulls", default="real,gauss_rowcol")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/local_cb.jsonl")
    args = ap.parse_args()

    dev = args.device
    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)
    calib = torch.load(args.calib, map_location="cpu")
    refs = [r for r in all_refs if r.layer_idx in layers and r.proj in projs
            and r.name in calib]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for variant in args.nulls.split(","):
        Ws = {}
        for r in refs:
            W = make_null(get_module(model, r.name).weight.detach().to(dev, torch.float32),
                          variant, seed=args.seed)
            Ws[r.name] = W
        for coord in args.coords.split(","):
            rot = {}
            Wc = {}
            for r in refs:
                if coord == "hadamard":
                    rp = RotationPair(*Ws[r.name].shape, seed=args.seed, device=dev)
                    rot[r.name] = rp
                    Wc[r.name] = rp.forward_w(Ws[r.name])
                else:
                    rot[r.name] = None
                    Wc[r.name] = Ws[r.name]
            U = collect_normalised([Wc[r.name] for r in refs[:6]], args.group, seed=args.seed)
            for b in [int(x) for x in args.bits.split(",")]:
                K = 2 ** b
                uniform = torch.linspace(0, 1, K, device=dev).reshape(1, K)
                for T in [int(x) for x in args.ts.split(",")] + [0]:
                    shapes = uniform if T == 0 else fit_shapes(U, K, T, seed=args.seed)
                    for comp in args.comps.split(","):
                        num = den = bits = 0.0
                        nw = 0
                        tails = []
                        for r in refs:
                            W0 = Ws[r.name]
                            X = calib[r.name].to(dev).float()
                            H = X @ X.T * (2.0 / X.shape[1])
                            rp = rot[r.name]
                            Win = Wc[r.name]
                            Hin = rp.forward_h(H) if rp is not None else H
                            q = LocalShapeQuantizer(shapes, group=args.group)
                            Q = gptq(Win, Hin if comp == "gptq" else None, q,
                                     percdamp=args.percdamp)
                            Wh = rp.inverse_w(Q) if rp is not None else Q
                            dW = W0 - Wh
                            num += float((dW @ H * dW).sum())
                            den += float((W0 @ H * W0).sum())
                            bits += sum(q.storage_bits(*W0.shape).values()) \
                                + (32.0 if coord == "hadamard" else 0.0)
                            nw += W0.numel()
                            tails.append(tail_report(W0, Wh))
                            del X, H, Hin, Q, Wh, dW
                            if dev == "cuda":
                                torch.cuda.empty_cache()
                        shared = 0.0 if T == 0 else float(shapes.numel() * 16)
                        rec = dict(variant=variant, coord=coord, bits=b,
                                   T=("uniform" if T == 0 else T), comp=comp,
                                   group=args.group, cell="%s/%s" % (comp, coord),
                                   bpw=bits / nw + shared / amortize_over,
                                   act_nmse=num / max(den, 1e-30),
                                   n_matrices=len(refs), n_weights=nw)
                        for k in tails[0]:
                            rec[k] = float(sum(d[k] for d in tails) / len(tails))
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        print("%-13s %-9s b%d T=%-7s %-5s bpw=%.4f act=%.5f  %.0fs"
                              % (variant, coord, b, rec["T"], comp, rec["bpw"],
                                 rec["act_nmse"], time.time() - t0))
            del Wc, U
            if dev == "cuda":
                torch.cuda.empty_cache()
        del Ws
        if dev == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("%.0fs, peak %.0f MiB -> %s"
          % (time.time() - t0,
             torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else 0, args.out))


if __name__ == "__main__":
    main()
