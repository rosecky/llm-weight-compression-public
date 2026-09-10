"""Phases 3-5: do graph communities make better quantization blocks than index adjacency?

A community is turned into a *permutation* and the matrix is physically reordered, so the
quantizer that runs afterwards is the ordinary contiguous one. Nothing here needs a runtime
gather.

    W' = P_out W P_in            quantize contiguously            un-permute, measure

Two block shapes, both at exactly the same scale budget (128 weights per scale), so the
comparison is at matched bits:

    row-group   1 x 128 : the standard group. Output order cannot matter to it, by
                          construction -- a scale is shared along a row only.
    tile 16 x 8         : the smallest 2-D block that makes an output-channel partition
                          mean anything, which is what Phase 5 asks about.

Orderings compared: contiguous (B0), random balanced (B1), channel-scale sorted (B2, the
strong simple baseline and the thing act-order does), and spectral sequencing of the A- and
G-correlation graphs (B3/B4). Every cell is run in native and Hadamard coordinates, with and
without GPTQ, because kill criterion G4 is about whether anything survives the strong
pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch

from ..gptq import ScalarQuantizer, Tile2DQuantizer, actorder_bits, gptq
from ..graphstruct import affinity, spectral_order, topk_sparsify
from ..metrics import act_nmse_from_H
from ..modelio import get_module, list_linear_layers, load_model
from ..rotate import RotationPair
from ..tailmetrics import tail_report

# (in-ordering, out-ordering) pairs per block shape
ROWGROUP_ORDERS = ["contig", "random", "scale", "specA_corr", "specA_cov"]
TILE_PAIRS = [("contig", "contig"), ("specA_corr", "contig"), ("contig", "specG_corr"),
              ("specA_corr", "specG_corr"), ("scale", "scale"), ("random", "random")]


@torch.no_grad()
def build_orders(A: torch.Tensor, G: torch.Tensor, k: int, seed: int,
                 device: str) -> Dict[str, torch.Tensor]:
    """Every candidate ordering of input channels (from A) and output channels (from G)."""
    n_in, n_out = A.shape[0], G.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    o = {
        "contig": torch.arange(n_in, device=device),
        "random": torch.randperm(n_in, generator=g, device=device),
        "scale": torch.argsort(torch.diagonal(A), descending=True),
        "specA_corr": spectral_order(topk_sparsify(affinity(A, "corr"), k)),
        "specA_cov": spectral_order(topk_sparsify(affinity(A, "cov"), k)),
    }
    o_out = {
        "contig": torch.arange(n_out, device=device),
        "random": torch.randperm(n_out, generator=g, device=device),
        "scale": torch.argsort(torch.diagonal(G), descending=True),
        "specG_corr": spectral_order(topk_sparsify(affinity(G, "corr"), k)),
    }
    return o, o_out


@torch.no_grad()
def fisher_damage(D: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> float:
    """tr(D A D^T G) without ever forming an (out, out) intermediate.

    tr(D A D^T G) = sum(D * (G D A)), so every temporary is (out, in).
    """
    return float((D * (G @ D @ A)).sum())


@torch.no_grad()
def run_one(W: torch.Tensor, A: torch.Tensor, G: torch.Tensor, pin: torch.Tensor,
            pout: torch.Tensor, quant, comp: str, percdamp: float) -> torch.Tensor:
    """Reorder, quantize contiguously, put back. Returns What in the original ordering."""
    iin = torch.argsort(pin)
    iout = torch.argsort(pout)
    Wp = W[pout][:, pin]
    Ap = A[pin][:, pin]
    Q = gptq(Wp, 2.0 * Ap if comp == "gptq" else None, quant, percdamp=percdamp)
    return Q[iout][:, iin]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--graphcal", default="cache/graphcal.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--tile", default="16x8")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--comps", default="naive,gptq")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/graph_quant.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    rb, cb = (int(x) for x in args.tile.split("x"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    geo = torch.load(args.graphcal, map_location="cpu")
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    fh = open(args.out, "a", encoding="utf-8")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_run = 0

    for name, d in sorted(geo.items()):
        li = int(name.split(".layers.")[1].split(".")[0])
        proj = name.split(".")[-1]
        if li not in layers or proj not in projs:
            continue
        W0 = get_module(model, name).weight.detach().to(args.device, torch.float32)
        A0 = d["A"].to(args.device, torch.float32)
        G0 = d["G"].to(args.device, torch.float32)
        out_f, in_f = W0.shape
        for coord in args.coords.split(","):
            rp = None
            W, A, G = W0, A0, G0
            if coord == "hadamard":
                rp = RotationPair(out_f, in_f, seed=args.seed, device=args.device)
                W = rp.forward_w(W0)
                Ar = rp.right.rotate_hessian(A0)
                A = 0.5 * (Ar + Ar.T)                 # the transform is orthogonal, but
                if rp.left is not None:               # accumulation error is not symmetric
                    Gr = rp.left.rotate_hessian(G0)
                    G = 0.5 * (Gr + Gr.T)
            o_in, o_out = build_orders(A, G, args.k, args.seed, args.device)
            den_fisher = fisher_damage(W0, A0, G0)

            plans: List[Tuple[str, str, str]] = \
                [("rowgroup", o, "contig") for o in ROWGROUP_ORDERS] + \
                [("tile%dx%d" % (rb, cb), a, b) for a, b in TILE_PAIRS]

            for shape, in_ord, out_ord in plans:
                for comp in args.comps.split(","):
                    q = (ScalarQuantizer(bits=args.bits, group=args.group)
                         if shape == "rowgroup"
                         else Tile2DQuantizer(bits=args.bits, row_block=rb, col_block=cb))
                    Qp = run_one(W, A, G, o_in[in_ord], o_out[out_ord], q, comp,
                                 args.percdamp)
                    Wh = rp.inverse_w(Qp) if rp is not None else Qp
                    dW = W0 - Wh
                    fisher = fisher_damage(dW, A0, G0)
                    sb = q.storage_bits(out_f, in_f)
                    bits = sum(sb.values()) + (32.0 if coord == "hadamard" else 0.0)
                    # an explicit, non-absorbable permutation index, priced conservatively
                    pbits = (0.0 if in_ord == "contig" else actorder_bits(in_f, out_f)) + \
                            (0.0 if out_ord == "contig" else actorder_bits(out_f, in_f))
                    rec = dict(matrix=name, layer=li, proj=proj, coord=coord, comp=comp,
                               shape=shape, in_ord=in_ord, out_ord=out_ord,
                               bits=args.bits, group=args.group, k=args.k,
                               bpw=bits / (out_f * in_f),
                               perm_bpw=pbits / (out_f * in_f),
                               fisher_err=fisher / max(den_fisher, 1e-30),
                               act_nmse=act_nmse_from_H(W0, Wh, A0),
                               **tail_report(W0, Wh))
                    fh.write(json.dumps(rec) + "\n")
                    n_run += 1
                    del Qp, Wh, dW
            fh.flush()
            print("  %-38s %-8s  %d plans  %.0fs  peak %.0f MiB"
                  % (name, coord, len(plans) * 2, time.time() - t0,
                     torch.cuda.max_memory_allocated() / 2 ** 20
                     if args.device == "cuda" else 0))
            del o_in, o_out, A, G, W
            if args.device == "cuda":
                torch.cuda.empty_cache()
        del W0, A0, G0
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("%d runs, %.0fs -> %s" % (n_run, time.time() - t0, args.out))


if __name__ == "__main__":
    main()
