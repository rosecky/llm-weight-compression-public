"""The oracle gate: how much is adaptive bit allocation worth if the layout were free?

This is the early kill test for the programmatic-layout hypothesis. We give an oracle the
complete per-weight quantization-damage table and let it spend a fixed bit budget however it
likes, charging *nothing* for describing the resulting map. If that upper bound is close to
uniform quantization at the same budget, no layout -- recursive, learned or otherwise -- can
be worth building, and the branch stops here.

Granularities, in decreasing freedom:

    weight   every weight picks its own width          (absolute upper bound)
    group    every (row, 128-column block)             (the natural codec unit)
    tile     every 128x128 block                       (hardware-friendly)
    row      every output channel
    col      every input channel                       (the "channel-only oracle")
    uniform  the baseline: one width everywhere

The comparison is run inside all four cells of the previous study's factorial
(naive/GPTQ x native/Hadamard), because kill criterion L6 asks specifically whether adaptive
layout still buys anything once compensation and incoherence rotation are already in place.

Budgets are matched exactly: the oracle's code budget is `out * in * b0`, i.e. the code bits a
uniform b0-bit quantizer would spend, so the two sides differ only in *where* the bits went.
Scales are counted separately and identically on both sides.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from ..gptq import ScalarQuantizer, gptq
from ..mixed import (BIT_OPTIONS, MixedScalarQuantizer, oracle_bits_map,
                     per_weight_distortion, quant_at_bits)
from ..modelio import get_module, list_linear_layers, load_model, total_weights
from ..rotate import RotationPair
from ..structure import make_null
from ..tailmetrics import tail_report

ALLOCS = ["uniform", "weight", "group", "tile", "row", "col"]


@torch.no_grad()
def quantize_mixed_naive(W: torch.Tensor, bits_w: torch.Tensor, group: int) -> torch.Tensor:
    """Vectorised equivalent of running `MixedScalarQuantizer` through the naive arm."""
    out_f, in_f = W.shape
    ng = in_f // group
    Wg = W.reshape(out_f, ng, group)
    Bg = bits_w.reshape(out_f, ng, group).to(W.dtype)
    lo = Wg.amin(-1, keepdim=True)
    hi = Wg.amax(-1, keepdim=True)
    rng = (hi - lo).clamp_min(1e-12)
    qmax = 2.0 ** Bg - 1.0
    s = rng / qmax.clamp_min(1.0)
    q = torch.clamp(torch.round((Wg - lo) / s.clamp_min(1e-12)),
                    torch.zeros_like(qmax), qmax)
    R = q * s + lo
    R = torch.where(Bg >= 16, Wg, R)
    R = torch.where(Bg <= 0, (lo + 0.5 * rng).expand_as(Wg), R)
    return R.reshape(out_f, in_f).contiguous()


def bit_histogram(bits_w: torch.Tensor) -> Dict[str, float]:
    n = bits_w.numel()
    out = {}
    for b in BIT_OPTIONS:
        f = float((bits_w == b).sum().item()) / n
        if f > 0:
            out["frac_b%d" % b] = f
    return out


@torch.no_grad()
def run(args):
    device = args.device
    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)
    calib = torch.load(args.calib, map_location="cpu")
    refs = [r for r in all_refs if r.layer_idx in layers and r.proj in projs
            and r.name in calib]
    print("%d matrices, amortising shared state over %.1fM weights"
          % (len(refs), amortize_over / 1e6))

    rates = [int(x) for x in args.rates.split(",")]
    allocs = args.allocs.split(",")
    coords = args.coords.split(",")
    comps = args.comps.split(",")
    group = args.group

    # acc[key] = dict of accumulators over matrices
    acc: Dict[tuple, dict] = {}
    saved_maps: Dict[str, torch.Tensor] = {}
    t0 = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ri, r in enumerate(refs):
        W_cpu = get_module(model, r.name).weight.detach()
        W = make_null(W_cpu.to(device, torch.float32), args.null, seed=args.seed)
        X = calib[r.name].to(device).float()                    # (in, n_samples)
        H = X @ X.T * (2.0 / X.shape[1])
        out_f, in_f = W.shape
        den = float((W @ H * W).sum())
        if in_f % group or out_f % args.tile_rows:
            print("  skip %s (shape %s not divisible)" % (r.name, tuple(W.shape)))
            continue

        for coord in coords:
            rp = None
            Wq, Hq = W, H
            if coord == "hadamard":
                rp = RotationPair(out_f, in_f, seed=args.seed, device=device)
                Wq = rp.forward_w(W)
                Hq = rp.forward_h(H)
            colw = torch.diagonal(Hq).clamp_min(0.0).clone()
            D_w = per_weight_distortion(Wq, colw, group)

            for b0 in rates:
                target = float(out_f) * in_f * b0
                for alloc in allocs:
                    if alloc == "uniform":
                        bits_w = torch.full((out_f, in_f), b0, dtype=torch.int16,
                                            device=device)
                    else:
                        bits_w = oracle_bits_map(D_w, (out_f, in_f), alloc, group, target,
                                                 tile_rows=args.tile_rows)
                    quant = MixedScalarQuantizer(bits_w, group=group)
                    sb = quant.storage_bits(out_f, in_f)
                    for comp in comps:
                        if comp == "naive":
                            Q = quantize_mixed_naive(Wq, bits_w, group)
                        else:
                            Q = gptq(Wq, Hq, quant, percdamp=args.percdamp)
                        Wh = rp.inverse_w(Q) if rp is not None else Q
                        dW = W - Wh
                        num = float((dW @ H * dW).sum())
                        key = (b0, alloc, comp, coord)
                        a = acc.setdefault(key, dict(num=0.0, den=0.0, bits=0.0,
                                                     nw=0, tails=[], hist=[]))
                        a["num"] += num
                        a["den"] += den
                        a["bits"] += sum(sb.values()) + (32.0 if coord == "hadamard" else 0.0)
                        a["nw"] += out_f * in_f
                        a["tails"].append(tail_report(W, Wh))
                        a["hist"].append(bit_histogram(bits_w))
                        del Q, Wh, dW
                    if args.save_maps and b0 == args.save_rate and alloc in ("group", "weight"):
                        saved_maps["%s|%s|%s|%d" % (r.name, coord, alloc, b0)] = \
                            bits_w.to(torch.int8).cpu()
                    del bits_w
            del D_w, Wq, Hq
            if device == "cuda":
                torch.cuda.empty_cache()
        del W, X, H
        if device == "cuda":
            torch.cuda.empty_cache()
        print("  [%2d/%d] %-34s %s  %.0fs  peak %.0f MiB"
              % (ri + 1, len(refs), r.name, tuple(W_cpu.shape), time.time() - t0,
                 torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0.0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as fh:
        for (b0, alloc, comp, coord), a in sorted(acc.items()):
            rec = dict(base_bits=b0, alloc=alloc, comp=comp, coord=coord, group=group,
                       variant=args.null,
                       cell="%s/%s" % (comp, coord),
                       bpw=a["bits"] / a["nw"], act_nmse=a["num"] / max(a["den"], 1e-30),
                       n_matrices=len(a["tails"]), n_weights=a["nw"],
                       model=args.model, seed=args.seed)
            for k in a["tails"][0]:
                rec[k] = float(sum(d[k] for d in a["tails"]) / len(a["tails"]))
            hk = sorted({k for d in a["hist"] for k in d})
            for k in hk:
                rec[k] = float(sum(d.get(k, 0.0) for d in a["hist"]) / len(a["hist"]))
            fh.write(json.dumps(rec) + "\n")
    if saved_maps:
        os.makedirs(os.path.dirname(args.maps_out), exist_ok=True)
        torch.save(saved_maps, args.maps_out)
        print("saved %d bit maps -> %s" % (len(saved_maps), args.maps_out))
    print("%d configs, %.0fs -> %s" % (len(acc), time.time() - t0, args.out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--calib", default="cache/calib.pt")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--allocs", default=",".join(ALLOCS))
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--comps", default="naive,gptq")
    ap.add_argument("--null", default="real",
                    help="run on a matched null instead of the real weights, e.g. gauss_rowcol")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--tile-rows", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--save-maps", action="store_true")
    ap.add_argument("--save-rate", type=int, default=3)
    ap.add_argument("--maps-out", default="cache/oracle_maps.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/oracle.jsonl")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
