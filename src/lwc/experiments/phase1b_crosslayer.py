"""Phase 1B -- cross-layer redundancy between the same projection in different layers.

Four increasingly generous notions of "layer j is predictable from layer i":

1. element-wise cosine / best affine  a*W_i + b
2. best diagonal row+column rescaling of W_i  (function-preserving-ish family)
3. best low-rank right-map                W_j ~= W_i @ M,  rank(M) = r
4. codebook transfer: fit a tile codebook on layer i, use it on layer j
   (compared against fitting on layer j itself, and against a shuffle null)

Every number is reported as relative residual energy ||W_j - pred||^2 / ||W_j||^2, so 1.0
means "no better than predicting zero" and 0.0 means perfect prediction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from ..modelio import get_module, list_linear_layers, load_model
from ..structure import collect_tiles, kmeans, residual_energy


@torch.no_grad()
def rel(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A - B).pow(2).sum() / B.pow(2).sum().clamp_min(1e-30))


@torch.no_grad()
def best_affine(Wi: torch.Tensor, Wj: torch.Tensor) -> float:
    """min_{a,b} ||W_j - (a W_i + b)||^2 / ||W_j||^2"""
    x, y = Wi.reshape(-1), Wj.reshape(-1)
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / (x - xm).pow(2).sum().clamp_min(1e-30)
    b = ym - a * xm
    return float((y - (a * x + b)).pow(2).sum() / y.pow(2).sum().clamp_min(1e-30))


@torch.no_grad()
def best_rowcol_scale(Wi: torch.Tensor, Wj: torch.Tensor, iters: int = 30) -> float:
    """min_{u,v} ||W_j - diag(u) W_i diag(v)||^2 / ||W_j||^2  (alternating least squares)."""
    u = torch.ones(Wi.shape[0], 1, device=Wi.device)
    v = torch.ones(1, Wi.shape[1], device=Wi.device)
    for _ in range(iters):
        A = Wi * v
        u = (A * Wj).sum(1, keepdim=True) / A.pow(2).sum(1, keepdim=True).clamp_min(1e-30)
        B = u * Wi
        v = (B * Wj).sum(0, keepdim=True) / B.pow(2).sum(0, keepdim=True).clamp_min(1e-30)
    return rel(u * Wi * v, Wj)


@torch.no_grad()
def best_lowrank_map(Wi: torch.Tensor, Wj: torch.Tensor, ranks) -> dict:
    """W_j ~= W_i @ M with rank(M)=r. Least squares then truncate M's SVD."""
    G = Wi.T @ Wi
    G += 1e-4 * torch.eye(G.shape[0], device=G.device) * G.diagonal().mean()
    M = torch.linalg.solve(G, Wi.T @ Wj)          # (in, in)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    out = {}
    for r in ranks:
        Mr = (U[:, :r] * S[:r]) @ Vh[:r]
        out[f"lowrank_map_r{r}"] = rel(Wi @ Mr, Wj)
        # bits to store the map, per weight of W_j
        out[f"lowrank_map_r{r}_bpw"] = r * (Wi.shape[1] + Wj.shape[1]) * 16.0 / Wj.numel()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--projs", default="q_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--pairs", default="adjacent,far",
                    help="adjacent: (i,i+1); far: (i, i+8)")
    ap.add_argument("--tiles", default="1x8,8x8")
    ap.add_argument("--ks", default="4096")
    ap.add_argument("--ranks", default="8,32,128")
    ap.add_argument("--max-tiles", type=int, default=120_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/phase1b.jsonl")
    args = ap.parse_args()

    tiles = [tuple(int(v) for v in t.lower().split("x")) for t in args.tiles.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    refs = list_linear_layers(model)
    by = {(r.layer_idx, r.proj): r for r in refs}
    n_layers = max(r.layer_idx for r in refs) + 1

    pairs = []
    if "adjacent" in args.pairs:
        pairs += [(i, i + 1) for i in (1, 5, 11, 17, 21)]
    if "far" in args.pairs:
        pairs += [(i, i + 8) for i in (1, 5, 11)]
    pairs = [(a, b) for a, b in pairs if b < n_layers]

    fh = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    for proj in args.projs.split(","):
        for (li, lj) in pairs:
            ri, rj = by.get((li, proj)), by.get((lj, proj))
            if ri is None or rj is None:
                continue
            Wi = get_module(model, ri.name).weight.detach().to(args.device, torch.float32)
            Wj = get_module(model, rj.name).weight.detach().to(args.device, torch.float32)
            rec = dict(model=args.model, proj=proj, li=li, lj=lj, shape=tuple(Wi.shape),
                       cosine=float((Wi * Wj).sum() /
                                    (Wi.norm() * Wj.norm()).clamp_min(1e-30)),
                       identity=rel(Wi, Wj),
                       affine=best_affine(Wi, Wj),
                       rowcol_scale=best_rowcol_scale(Wi, Wj))
            rec.update(best_lowrank_map(Wi, Wj, ranks))
            # --- codebook transfer ---
            for (th, tw) in tiles:
                Ti = collect_tiles([Wi], th, tw, "real", args.seed, args.max_tiles)
                Tj = collect_tiles([Wj], th, tw, "real", args.seed, args.max_tiles)
                Tj_sh = collect_tiles([Wj], th, tw, "shuffle", args.seed, args.max_tiles)
                for K in ks:
                    if K > min(Ti.shape[0], Tj.shape[0]) // 4:
                        continue
                    Ci = kmeans(Ti, K, iters=15, seed=args.seed)
                    half = Tj.shape[0] // 2
                    Cj = kmeans(Tj[:half], K, iters=15, seed=args.seed)
                    Tj_te = Tj[half:]
                    rec[f"cbtrans_{th}x{tw}_K{K}_cross"] = residual_energy(Tj_te, Ci)
                    rec[f"cbtrans_{th}x{tw}_K{K}_self"] = residual_energy(Tj_te, Cj)
                    rec[f"cbtrans_{th}x{tw}_K{K}_shufcross"] = residual_energy(
                        Tj_sh[half:], Ci)
                del Ti, Tj, Tj_sh
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            del Wi, Wj
            if args.device == "cuda":
                torch.cuda.empty_cache()
        print(f"[done] {proj}  ({time.time()-t0:.0f}s)")
    fh.close()
    print(f"wrote {args.out}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
