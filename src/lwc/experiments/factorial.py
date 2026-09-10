"""Stage 1 of the final diagnostic: the 2x2x2 factorial, measured per layer.

    representation  A = scalar (asymmetric min-max group-wise)   B = residual VQ (d=8)
    error handling  0 = naive independent                        1 = GPTQ second-order
    coordinates     N = native                                   H = randomized Hadamard

Every cell uses the *same* compensation engine and the *same* rotation machinery, so each
factor is isolated. Metrics: full bit accounting, weight NMSE, tail errors (top 1 / 0.1 /
0.01 %), sign flips, activation NMSE with per-channel quantiles, and the Hessian proxy
tr(dW H dW^T)/tr(W H W^T).

`--null gauss_rowcol` reruns the same pipeline on a matched i.i.d. Gaussian with identical
row/column scales, answering: does the real weight matrix have any rate-distortion advantage
its matched Gaussian does not?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from ..bits import BitBudget
from ..gptq import ScalarQuantizer, VQQuantizer, actorder_bits, gptq
from ..metrics import nmse
from ..modelio import get_module, list_linear_layers, load_model, total_weights
from ..rotate import RotationPair, coordinate_stats, roundtrip_error
from ..structure import collect_tiles, kmeans, make_null
from ..tailmetrics import act_error_quantiles, hessian_proxy, tail_report
from ..tiles import to_tiles


def parse_cells(s: str):
    return [c.strip() for c in s.split(",") if c.strip()]


def build_quantizer(rep: str, cfg: dict, fit_mats, device: str, seed: int):
    """Returns (quantizer, shared_bits)."""
    if rep == "scalar":
        return ScalarQuantizer(bits=cfg["bits"], group=cfg["group"],
                               atom=cfg.get("atom", 1)), 0.0
    d, K, stages = cfg["d"], cfg["K"], cfg["stages"]
    T = collect_tiles(fit_mats, 1, d, "real", seed, cfg.get("fit_tiles", 150_000))
    books, R = [], T
    for s in range(stages):
        C = kmeans(R, K, iters=20, seed=seed + s)
        books.append(C)
        Cn = (C * C).sum(1)
        idx_chunk = int(max(512, min(8192, (1 << 24) // max(K, 1))))
        newR = torch.empty_like(R)
        for i in range(0, R.shape[0], idx_chunk):
            rc = R[i:i + idx_chunk]
            a = (Cn.unsqueeze(0) - 2.0 * (rc @ C.T)).argmin(1)
            newR[i:i + idx_chunk] = rc - C[a]
        R = newR
    q = VQQuantizer(books)
    return q, q.shared_bits()


RATE_LADDER = {
    "scalar": [
        {"bits": 2, "group": 128}, {"bits": 2, "group": 64},
        {"bits": 3, "group": 128}, {"bits": 3, "group": 64},
        {"bits": 4, "group": 128},
    ],
    "vq": [
        {"d": 8, "K": 256, "stages": 2}, {"d": 8, "K": 1024, "stages": 2},
        {"d": 8, "K": 256, "stages": 3}, {"d": 8, "K": 1024, "stages": 3},
        {"d": 8, "K": 256, "stages": 4},
    ],
}


def cfg_tag(rep, cfg):
    if rep == "scalar":
        return f"b{cfg['bits']}g{cfg['group']}" + (f"a{cfg['atom']}" if cfg.get("atom", 1) > 1 else "")
    return f"d{cfg['d']}K{cfg['K']}s{cfg['stages']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--calib", default="cache/calib.pt")
    ap.add_argument("--reps", default="scalar,vq")
    ap.add_argument("--comps", default="naive,gptq")
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--null", default="", help="also run on a matched null, e.g. gauss_rowcol")
    ap.add_argument("--actorder", action="store_true")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--extra-scalar-atom8", action="store_true",
                    help="control: scalar quantizer at atom=8 to match the VQ atom width")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/factorial.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)
    refs = [r for r in all_refs if r.layer_idx in layers and r.proj in projs]
    calib = torch.load(args.calib, map_location="cpu")
    refs = [r for r in refs if r.name in calib]
    print(f"{len(refs)} matrices, amortising shared state over {amortize_over/1e6:.1f}M weights")

    ladders = dict(RATE_LADDER)
    if args.extra_scalar_atom8:
        ladders["scalar"] = ladders["scalar"] + [
            {"bits": b, "group": 128, "atom": 8} for b in (2, 3, 4)]

    variants = ["real"] + ([args.null] if args.null else [])
    fh = open(args.out, "a", encoding="utf-8")
    t_start = time.time()
    n_run = 0

    for variant in variants:
        # cache the (possibly nulled) weights and their rotations once per variant
        for rep in args.reps.split(","):
            for coord in args.coords.split(","):
                # fit the VQ codebook in the coordinate system the quantizer will see
                fit_mats = []
                for r in refs[:6]:
                    W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
                    W = make_null(W, variant, seed=args.seed)
                    if coord == "hadamard":
                        W = RotationPair(*W.shape, seed=args.seed,
                                         device=args.device).forward_w(W)
                    fit_mats.append(W)
                for cfg in ladders[rep]:
                    quant, shared = build_quantizer(rep, cfg, fit_mats, args.device, args.seed)
                    for comp in args.comps.split(","):
                        t0 = time.time()
                        if args.device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        budget = BitBudget(n_weights=0)
                        agg = {}
                        coord_before, coord_after = [], []
                        for r in refs:
                            W0 = get_module(model, r.name).weight.detach().to(
                                args.device, torch.float32)
                            W0 = make_null(W0, variant, seed=args.seed)
                            X = calib[r.name].to(args.device).float()      # (in, n_samples)
                            H = X @ X.T * (2.0 / X.shape[1])
                            rp = None
                            Wq_in, Hq = W0, H
                            if coord == "hadamard":
                                rp = RotationPair(*W0.shape, seed=args.seed,
                                                  device=args.device)
                                Wq_in = rp.forward_w(W0)
                                Hq = rp.forward_h(H)
                                if len(coord_before) < 3:
                                    coord_before.append(coordinate_stats(W0))
                                    coord_after.append(coordinate_stats(Wq_in))
                            Q = gptq(Wq_in, Hq if comp == "gptq" else None, quant,
                                     percdamp=args.percdamp, actorder=args.actorder)
                            Wh = rp.inverse_w(Q) if rp is not None else Q

                            for k, v in quant.storage_bits(*W0.shape).items():
                                budget.add_per_matrix(k, v)
                            if coord == "hadamard":
                                budget.add_per_matrix("rotation_seed", 32.0)
                            if args.actorder and comp == "gptq":
                                budget.add_per_matrix("actorder_index",
                                                      actorder_bits(W0.shape[1], W0.shape[0]))
                            budget.n_weights += W0.numel()

                            m = tail_report(W0, Wh)
                            m.update(act_error_quantiles(W0, Wh, X))
                            m["hessian_proxy"] = hessian_proxy(W0, Wh, H)
                            for k, v in m.items():
                                agg.setdefault(k, []).append(v)
                            del W0, X, H, Wq_in, Hq, Q, Wh
                            if args.device == "cuda":
                                torch.cuda.empty_cache()

                        bpw = (sum(budget.per_matrix.values()) / budget.n_weights
                               + shared / amortize_over)
                        rec = dict(
                            variant=variant, rep=rep, comp=comp, coord=coord,
                            cell=f"{rep}/{comp}/{coord}", cfg=cfg, cfg_tag=cfg_tag(rep, cfg),
                            name=f"{rep}.{comp}.{coord}.{cfg_tag(rep, cfg)}"
                                 + ("" if variant == "real" else f".{variant}"),
                            bpw=bpw, bpw_shared=shared / amortize_over,
                            n_matrices=len(refs), n_weights=budget.n_weights,
                            actorder=args.actorder,
                            encode_s=time.time() - t0,
                            peak_vram_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                           if args.device == "cuda" else 0.0),
                        )
                        # energy-weighted where it makes sense, mean otherwise
                        for k, v in agg.items():
                            rec[k] = float(sum(v) / len(v))
                            if k in ("max_err_over_max_w", "act_nmse_ch_max"):
                                rec[k + "_worst"] = float(max(v))
                        if coord_before:
                            for k in coord_before[0]:
                                rec["coord_" + k + "_before"] = float(
                                    sum(d[k] for d in coord_before) / len(coord_before))
                                rec["coord_" + k + "_after"] = float(
                                    sum(d[k] for d in coord_after) / len(coord_after))
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        n_run += 1
                        print(f"{rec['name']:44s} bpw={bpw:6.3f} nmse={rec['nmse']:.5f} "
                              f"act={rec['act_nmse']:.5f} top.1%={rec['relerr_top001']:.5f} "
                              f"{rec['encode_s']:5.1f}s {rec['peak_vram_mib']:.0f}MiB")
                del fit_mats
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    fh.close()
    print(f"\n{n_run} runs, {time.time()-t_start:.0f}s total -> {args.out}")


if __name__ == "__main__":
    main()
