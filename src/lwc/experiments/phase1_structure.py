"""Phase 1A -- intra-matrix / pooled structure diagnostics.

Question: do LLM weight tiles carry exploitable dependence structure beyond what a matched
i.i.d. null model has?

For each projection type we pool tiles from several layers (the realistic shared-codebook
setting), fit a k-means codebook and a PCA basis on a training split, and measure held-out
residual energy. The identical pipeline runs on null models:

    real          the actual weights
    shuffle       entries permuted within each matrix -> identical marginal, no dependence
    gauss_rowcol  i.i.d. Gaussian matched to per-row and per-column RMS -> keeps the
                  outlier-channel scale structure, destroys everything else

    net_gain(real) - net_gain(shuffle)      = exploitable dependence
    net_gain(shuffle) - net_gain(gauss)     = marginal shape gain (classic VQ gain)

Usage:
    python -m lwc.experiments.phase1_structure --model Qwen/Qwen2.5-0.5B \
        --layers 2,11,20 --projs up_proj,down_proj,q_proj --out results/raw/phase1a.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model
from ..structure import collect_tiles, probe_pca_tiles, probe_vq_tiles, nn_distance_stats

DEFAULT_TILES = "1x8,1x16,4x4,8x8,16x16,32x32"
DEFAULT_VARIANTS = "real,shuffle,gauss_rowcol"


def parse_tiles(s: str):
    out = []
    for tok in s.split(","):
        a, b = tok.lower().split("x")
        out.append((int(a), int(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="2,11,20")
    ap.add_argument("--projs", default=",".join(PROJ_TYPES))
    ap.add_argument("--tiles", default=DEFAULT_TILES)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--ks", default="256,1024,4096")
    ap.add_argument("--ranks", default="1,2,4")
    ap.add_argument("--max-tiles", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/phase1a.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    tiles = parse_tiles(args.tiles)
    variants = args.variants.split(",")
    ks = [int(x) for x in args.ks.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    refs = list_linear_layers(model)
    by = {(r.layer_idx, r.proj): r for r in refs}

    fh = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    n_probe = 0
    for proj in projs:
        mats, served = [], 0
        for li in layers:
            r = by.get((li, proj))
            if r is None:
                continue
            W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
            mats.append(W)
            served += W.numel()
        if not mats:
            print(f"[skip] {proj}: not found")
            continue
        shape = tuple(mats[0].shape)
        for (th, tw) in tiles:
            d = th * tw
            for variant in variants:
                T = collect_tiles(mats, th, tw, variant=variant, seed=args.seed,
                                  max_tiles=args.max_tiles)
                base = dict(model=args.model, proj=proj, layers=layers, shape=shape,
                            variant=variant, th=th, tw=tw)
                for K in ks:
                    if K > T.shape[0] // 4:
                        continue
                    rec = probe_vq_tiles(T, K, seed=args.seed, iters=args.iters,
                                         n_weights_served=served, **base)
                    fh.write(json.dumps(rec) + "\n")
                    n_probe += 1
                    rec2 = probe_vq_tiles(T, K, seed=args.seed, iters=args.iters,
                                          normalized=True, n_weights_served=served, **base)
                    fh.write(json.dumps(rec2) + "\n")
                    n_probe += 1
                for r_ in ranks:
                    if r_ >= d:
                        continue
                    rec = probe_pca_tiles(T, r_, seed=args.seed, **base)
                    fh.write(json.dumps(rec) + "\n")
                    n_probe += 1
                del T
            torch.cuda.empty_cache() if args.device == "cuda" else None
        # nearest-neighbour sanity, cheapest tile size only
        for variant in variants:
            rec = nn_distance_stats(mats[0], 8, 8, variant=variant, seed=args.seed)
            rec.update(dict(model=args.model, proj=proj, layers=layers))
            fh.write(json.dumps(rec) + "\n")
        del mats
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(f"[done] {proj}  ({n_probe} probes, {time.time()-t0:.0f}s)")
    fh.close()
    print(f"wrote {args.out}  probes={n_probe}  elapsed={time.time()-t0:.0f}s")
    if args.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
