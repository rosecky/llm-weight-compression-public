"""Phase 1C -- does a function-preserving permutation make weights more compressible?

Two permutations in a decoder-only transformer are exactly function preserving:

* **residual-stream permutation** pi_h over the hidden dimension -- permutes the *columns* of
  q/k/v/gate/up and the *rows* of o/down, plus every layernorm and the embedding. Global,
  one permutation for the whole model.
* **FFN intermediate permutation** pi_f over the intermediate dimension -- permutes the *rows*
  of gate/up and the *columns* of down. Per layer, free to choose independently.

Group-wise scalar quantization groups along the *input* dimension, so only column permutations
can change its error, and they do it by making the scale within a group homogeneous. That is
the mechanism worth measuring. Tile codecs additionally care about row order.

Variants measured:
    identity      no permutation (reference)
    sort_shared   columns sorted by a scale shared by all matrices reading this input
                  (deployable: one permutation serves every consumer)
    sort_local    columns sorted by this matrix's own column RMS
                  (NOT deployable across several consumers -- upper bound only)
    random        control; should be neutral-to-worse
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..codecs.scalar import quantize_groupwise
from ..metrics import nmse
from ..modelio import get_module, list_linear_layers, load_model
from ..structure import collect_tiles, probe_vq_tiles


def col_perm(scale: torch.Tensor, kind: str, seed: int = 0) -> torch.Tensor:
    if kind == "identity":
        return torch.arange(scale.numel(), device=scale.device)
    if kind == "random":
        g = torch.Generator(device=scale.device).manual_seed(seed)
        return torch.randperm(scale.numel(), generator=g, device=scale.device)
    return torch.argsort(scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--groups", default="64,128")
    ap.add_argument("--tiles", default="8x8")
    ap.add_argument("--ks", default="4096")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/phase1c.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    bitlist = [int(x) for x in args.bits.split(",")]
    grouplist = [int(x) for x in args.groups.split(",")]
    tiles = [tuple(int(v) for v in t.lower().split("x")) for t in args.tiles.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    by = {(r.layer_idx, r.proj): r for r in list_linear_layers(model)}

    # A "shared" column scale for the hidden dimension: RMS over the columns of all matrices
    # that read the residual stream (q,k,v,gate,up), pooled across the selected layers.
    shared_hidden = None
    for li in layers:
        for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            r = by.get((li, p))
            if r is None:
                continue
            W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
            s = W.pow(2).mean(0)
            shared_hidden = s if shared_hidden is None else shared_hidden + s
            del W
    shared_hidden = shared_hidden.sqrt()

    fh = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    for proj in projs:
        for li in layers:
            r = by.get((li, proj))
            if r is None:
                continue
            W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
            local = W.pow(2).mean(0).sqrt()
            in_is_hidden = proj in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
            for kind in ("identity", "sort_shared", "sort_local", "random"):
                if kind == "sort_shared" and not in_is_hidden:
                    # down_proj / o_proj read a per-layer space; the shared scale for them is
                    # just the local one, so skip to avoid a duplicate row
                    continue
                sc = shared_hidden if kind == "sort_shared" else local
                perm = col_perm(sc, kind, seed=args.seed + li)
                Wp = W[:, perm].contiguous()
                rec = dict(model=args.model, proj=proj, layer=li, perm=kind,
                           shape=tuple(W.shape))
                for b in bitlist:
                    for g in grouplist:
                        rec[f"int{b}_g{g}_nmse"] = nmse(Wp, quantize_groupwise(Wp, b, g))
                for (th, tw) in tiles:
                    T = collect_tiles([Wp], th, tw, "real", args.seed, 200_000)
                    for K in ks:
                        if K > T.shape[0] // 4:
                            continue
                        p = probe_vq_tiles(T, K, seed=args.seed, iters=15,
                                           n_weights_served=W.numel())
                        rec[f"vq_{th}x{tw}_K{K}_rho"] = p["rho"]
                        rec[f"vq_{th}x{tw}_K{K}_net"] = p["net_gain"]
                    del T
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                del Wp
            del W
            if args.device == "cuda":
                torch.cuda.empty_cache()
        print(f"[done] {proj}  ({time.time()-t0:.0f}s)")
    fh.close()
    print(f"wrote {args.out}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
