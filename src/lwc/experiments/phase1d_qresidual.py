"""Phase 1D -- is the LOW-BIT QUANTIZATION RESIDUAL structured?

This is the hypothesis behind Variant D:  W ~= Q_lowbit(W) + D(c).
It is attractive because scalar quantization is cheap and already handles the high-entropy
part; if the *residual* were structured, a procedural term could clean it up for few bits.

We quantize with group-wise INT-b, take R = W - Q(W), and run the exact same probes
(k-means / PCA, held-out) on R and on a shuffled version of R. If R is white noise --
which is what round-to-nearest theory predicts -- real and shuffle will coincide and
Variant D has no room to work.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..codecs.scalar import quantize_groupwise
from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model
from ..structure import collect_tiles, probe_pca_tiles, probe_vq_tiles


def parse_tiles(s: str):
    return [tuple(int(v) for v in tok.lower().split("x")) for tok in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--tiles", default="1x8,1x16,8x8,16x16")
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--ks", default="256,4096")
    ap.add_argument("--ranks", default="1,4")
    ap.add_argument("--normalize-per-group", action="store_true",
                    help="divide the residual by its group quantization step before probing")
    ap.add_argument("--max-tiles", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/phase1d.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    tiles = parse_tiles(args.tiles)
    bitlist = [int(x) for x in args.bits.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    by = {(r.layer_idx, r.proj): r for r in list_linear_layers(model)}

    fh = open(args.out, "w", encoding="utf-8")
    t0, n = time.time(), 0
    for proj in projs:
        for b in bitlist:
            resids, served, rel_energy = [], 0, []
            for li in layers:
                r = by.get((li, proj))
                if r is None:
                    continue
                W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
                Q = quantize_groupwise(W, b, args.group)
                R = W - Q
                rel_energy.append(float(R.pow(2).sum() / W.pow(2).sum()))
                if args.normalize_per_group:
                    # scale each group's residual by its own quantization step so that any
                    # remaining structure is *shape* structure, not scale structure
                    of, inf = W.shape
                    g = min(args.group, inf)
                    pad = (-inf) % g
                    Rp = torch.nn.functional.pad(R, (0, pad)) if pad else R
                    Gr = Rp.reshape(of, -1, g)
                    Gr = Gr / Gr.abs().amax(-1, keepdim=True).clamp_min(1e-12)
                    R = Gr.reshape(of, -1)[:, :inf].contiguous()
                resids.append(R)
                served += W.numel()
                del W, Q
            if not resids:
                continue
            for (th, tw) in tiles:
                for variant in ("real", "shuffle"):
                    T = collect_tiles(resids, th, tw, variant=variant, seed=args.seed,
                                      max_tiles=args.max_tiles)
                    base = dict(model=args.model, proj=proj, layers=layers, qbits=b,
                                group=args.group, variant=variant, th=th, tw=tw,
                                resid_rel_energy=sum(rel_energy) / len(rel_energy),
                                normalized_per_group=args.normalize_per_group)
                    for K in ks:
                        if K > T.shape[0] // 4:
                            continue
                        fh.write(json.dumps(probe_vq_tiles(
                            T, K, seed=args.seed, iters=args.iters,
                            n_weights_served=served, **base)) + "\n")
                        n += 1
                    for rr in ranks:
                        if rr >= th * tw:
                            continue
                        fh.write(json.dumps(probe_pca_tiles(
                            T, rr, seed=args.seed, **base)) + "\n")
                        n += 1
                    del T
            del resids
            if args.device == "cuda":
                torch.cuda.empty_cache()
            print(f"[done] {proj} int{b}  ({n} probes, {time.time()-t0:.0f}s)")
    fh.close()
    print(f"wrote {args.out}  probes={n}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
