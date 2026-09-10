"""Phase 1-2: build the sensitivity maps, and find out which of them predicts real damage.

Three levels of truth, each validating the one above it:

    cheap map            S1..S7, computed from W, H, and a baseline quantizer's error
    exact layer damage   ||E_u X||^2 for a perturbation confined to unit u, using the FULL
                         Hessian -- so the gap to the diagonal prediction is exactly the
                         off-diagonal contribution
    end-to-end damage    perturb one unit, measure the model's NLL on held-out text

The last one is expensive, so it runs on a stratified sample of units. It is the control that
says whether the whole diagonal-second-order framing -- which the oracle allocator depends on
-- is measuring the right thing at all.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from ..calib import get_wikitext2
from ..gptq import ScalarQuantizer, gptq
from ..mixed import quant_at_bits
from ..modelio import get_module, list_linear_layers, load_model
from ..sensitivity import (agreement, build_maps, exact_unit_damage, reduce_by_unit,
                           stratified_units)

MAP_ORDER = ["S1_absw", "S2_w2", "S3_act", "S4_hdiag", "S5_damage", "S5c_damage_gptq", "S7_obs"]


@torch.no_grad()
def nll(model, ids, device) -> float:
    tot, ntok = 0.0, 0
    for i in range(ids.shape[0]):
        x = ids[i:i + 1].to(device)
        out = model(x, labels=x)
        tot += float(out.loss) * (x.shape[1] - 1)
        ntok += x.shape[1] - 1
    return tot / ntok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--calib", default="cache/calib.pt")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--units", default="group,tile,row,col,colblk")
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--intervene-bits", type=int, default=2)
    ap.add_argument("--intervene-n", type=int, default=32)
    ap.add_argument("--intervene-layer", default="model.layers.11.mlp.up_proj")
    ap.add_argument("--intervene-seqs", type=int, default=4)
    ap.add_argument("--intervene-seqlen", type=int, default=1024)
    ap.add_argument("--skip-intervene", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/sens_maps.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    units = args.units.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    dev = args.device
    model, tok = load_model(args.model, device=dev, dtype=torch.float16)
    calib = torch.load(args.calib, map_location="cpu")
    refs = [r for r in list_linear_layers(model)
            if r.layer_idx in layers and r.proj in projs and r.name in calib]
    print("%d matrices" % len(refs))

    fh = open(args.out, "a", encoding="utf-8")
    t0 = time.time()

    # ---------------------------------------------------- map vs exact damage, full population
    for r in refs:
        W = get_module(model, r.name).weight.detach().to(dev, torch.float32)
        X = calib[r.name].to(dev).float()
        H = X @ X.T * (2.0 / X.shape[1])
        out_f, in_f = W.shape
        if in_f % args.group or out_f % args.group:
            continue
        Qn = quant_at_bits(W, args.base_bits, args.group)
        Qg = gptq(W, H, ScalarQuantizer(bits=args.base_bits, group=args.group))
        maps = build_maps(W, H, Wq_naive=Qn, Wq_gptq=Qg)
        for unit in units:
            truth = exact_unit_damage(W - Qn, H, unit, args.group, args.group)
            truth_g = exact_unit_damage(W - Qg, H, unit, args.group, args.group)
            for mname in MAP_ORDER:
                if mname not in maps:
                    continue
                pred = reduce_by_unit(maps[mname], unit, args.group, args.group)
                a = agreement(pred, truth)
                ag = agreement(pred, truth_g)
                fh.write(json.dumps(dict(
                    kind="map_vs_exact", matrix=r.name, layer=r.layer_idx, proj=r.proj,
                    unit=unit, map=mname, base_bits=args.base_bits, n_units=int(pred.numel()),
                    spearman=a["spearman"], log_pearson=a["log_pearson"],
                    top1_overlap=a["top1_overlap"],
                    spearman_gptq=ag["spearman"], top1_overlap_gptq=ag["top1_overlap"])) + "\n")
            # how much does the diagonal approximation itself lose?
            diag_pred = reduce_by_unit(maps["S5_damage"], unit, args.group, args.group)
            a = agreement(diag_pred, truth)
            ratio = float((diag_pred.sum() / truth.sum().clamp_min(1e-30)))
            fh.write(json.dumps(dict(
                kind="diag_vs_full", matrix=r.name, layer=r.layer_idx, proj=r.proj, unit=unit,
                spearman=a["spearman"], log_pearson=a["log_pearson"],
                top1_overlap=a["top1_overlap"], sum_ratio=ratio,
                n_units=int(truth.numel()))) + "\n")
        fh.flush()
        print("  %-36s %s  %.0fs" % (r.name, tuple(W.shape), time.time() - t0))
        del W, X, H, Qn, Qg, maps
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------- end-to-end intervention
    if not args.skip_intervene:
        name = args.intervene_layer
        mod = get_module(model, name)
        W = mod.weight.detach().to(dev, torch.float32)
        X = calib[name].to(dev).float()
        H = X @ X.T * (2.0 / X.shape[1])
        Qi = quant_at_bits(W, args.intervene_bits, args.group)
        maps = build_maps(W, H, Wq_naive=Qi)
        E = W - Qi
        truth_layer = exact_unit_damage(E, H, "row", args.group, args.group)
        picks = stratified_units(truth_layer, n=args.intervene_n, seed=args.seed)
        ids = get_wikitext2(tok, seqlen=args.intervene_seqlen, n_seq=args.intervene_seqs,
                            split="test", seed=args.seed)
        base = nll(model, ids, dev)
        print("  intervention on %s, base NLL %.6f, %d rows" % (name, base, len(picks)))
        orig = mod.weight.data.clone()
        recs = []
        for u in picks:
            mod.weight.data[u] = Qi[u].to(mod.weight.dtype)
            d = nll(model, ids, dev) - base
            mod.weight.data[u] = orig[u]
            recs.append(dict(unit=int(u), dnll=d,
                             exact=float(truth_layer[u]),
                             **{m: float(reduce_by_unit(maps[m], "row", args.group,
                                                        args.group)[u])
                                for m in MAP_ORDER if m in maps}))
        dn = torch.tensor([x["dnll"] for x in recs], dtype=torch.float64)
        ex = torch.tensor([x["exact"] for x in recs], dtype=torch.float64)
        summ = {"kind": "intervene", "matrix": name, "base_nll": base,
                "n": len(recs), "bits": args.intervene_bits,
                "dnll_min": float(dn.min()), "dnll_max": float(dn.max())}
        summ["exact_vs_dnll"] = agreement(ex, dn.clamp_min(0))
        for m in MAP_ORDER:
            if m in maps:
                pv = torch.tensor([x[m] for x in recs], dtype=torch.float64)
                summ["%s_vs_dnll" % m] = agreement(pv, dn.clamp_min(0))
        fh.write(json.dumps(summ) + "\n")
        fh.write(json.dumps({"kind": "intervene_raw", "matrix": name, "rows": recs}) + "\n")
        print("  exact-vs-dNLL spearman %.3f" % summ["exact_vs_dnll"]["spearman"])
    fh.close()
    print("%.0fs -> %s" % (time.time() - t0, args.out))


if __name__ == "__main__":
    main()
