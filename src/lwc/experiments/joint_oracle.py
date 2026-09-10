"""Kill gate K1: how far is greedy GPTQ from the *actual* optimum of its own objective?

This is the experiment the whole branch is gated on. It is deliberately tiny and deliberately
expensive per decision, because the only thing it has to establish is existence:

    at the identical bit width, group size, codebook and scales, is there a materially better
    integer code array than the one greedy GPTQ commits to?

Three levels of answer, on single output rows of real matrices with real calibration geometry
(the row is the honest unit: with `G = I` the layer objective decomposes exactly over output
rows, so a single row *is* a complete subproblem, not a slice of one):

1. **Coordinate descent.** Reopen every decision, repeatedly, each time to its exact
   conditional optimum. Measures the cost of GPTQ's irreversible commits (hypothesis H1).

2. **Exact block moves.** Solve blocks of 16-32 decisions to proven optimality with a
   box-constrained sphere decoder, cycling over the row. Measures whether higher-order joint
   moves see anything single-coordinate moves cannot.

3. **The subset oracle.** Freeze everything except `n` decisions and solve those *exactly*.
   With the complement frozen at GPTQ's answer this is the true local oracle gap; with it
   frozen at the coordinate-descent answer it asks whether CD's fixed point is also optimal
   against joint moves.

Plus a simulated annealer from the same start, as a search family that fails differently from
coordinate descent -- if CD were stuck in a poor basin, a long anneal would show it.

Everything reports damage relative to GPTQ's, so a number below 1.0 is a better solution at
identical storage and a number above 1.0 is worse.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch

from ..joint import (anneal_row, cd_row, cd_subset_exact, quantize_with_state, row_damage,
                     sphere_decode, subproblem_form)
from ..modelio import get_module, list_linear_layers, load_model
from ..rotate import RotationPair


def pick_modules(cal: Dict, layers: List[int], projs: List[str]) -> List[str]:
    out = []
    for n in cal:
        parts = n.split(".")
        li = int(parts[parts.index("layers") + 1])
        if li in layers and parts[-1] in projs:
            out.append(n)
    return sorted(out)


@torch.no_grad()
def run_module(name: str, W0: torch.Tensor, A0: torch.Tensor, args, fh) -> None:
    dev = args.device
    for coord in args.coords.split(","):
        if coord == "hadamard":
            rp = RotationPair(W0.shape[0], W0.shape[1], seed=args.seed, device=dev)
            W = rp.forward_w(W0)
            A = rp.forward_h(A0)
            A = 0.5 * (A + A.T)
        else:
            W, A = W0, A0
        H = 2.0 * A

        _, st_g = quantize_with_state(W, H, args.bits, args.group, args.percdamp)
        _, st_r = quantize_with_state(W, None, args.bits, args.group, args.percdamp)
        s_g, z_g = st_g.expand()
        s_r, z_r = st_r.expand()
        qmax = st_g.qmax

        g = torch.Generator(device="cpu").manual_seed(args.seed)
        rows = torch.randperm(W.shape[0], generator=g)[:args.rows].tolist()

        for ri, r in enumerate(rows):
            w = W[r].contiguous()
            sg, zg, cg = s_g[r].contiguous(), z_g[r].contiguous(), st_g.codes[r].float()
            sr, zr, cr = s_r[r].contiguous(), z_r[r].contiguous(), st_r.codes[r].float()
            d_gptq = row_damage(w, A, sg, zg, cg)
            d_rtn = row_damage(w, A, sr, zr, cr)
            if d_gptq <= 0:
                continue
            rec = dict(module=name, coord=coord, row=r, bits=args.bits, group=args.group,
                       n=int(w.numel()), d_gptq=d_gptq, rel_rtn=d_rtn / d_gptq)

            # ---- 1. coordinate descent, GPTQ scales, GPTQ init (the primary H1 arm)
            t0 = time.time()
            c_cd = cd_row(w, A, sg, zg, cg, qmax, sweeps=args.sweeps)
            d_cd = row_damage(w, A, sg, zg, c_cd)
            rec["rel_cd"] = d_cd / d_gptq
            rec["cd_flips"] = int((c_cd != cg).sum())
            rec["cd_s"] = time.time() - t0

            # same optimizer from the *naive* start, on the naive scales: does the basin matter?
            c_cd_r = cd_row(w, A, sr, zr, cr, qmax, sweeps=args.sweeps)
            rec["rel_cd_from_rtn"] = row_damage(w, A, sr, zr, c_cd_r) / d_gptq

            # multi-start with randomised sweep order, best of `restarts`
            best = d_cd
            for k in range(args.restarts):
                ck = cd_row(w, A, sg, zg, cg, qmax, sweeps=args.sweeps,
                            order_seed=1000 * (ri + 1) + k)
                best = min(best, row_damage(w, A, sg, zg, ck))
            rec["rel_cd_multistart"] = best / d_gptq

            # ---- 1b. block moves at k = 2, 4, 8 -- the CDQuant reconciliation
            # CDQuant reports Block-CD with k=2 buying +0.3% at INT3 and +1.0% at INT2 over
            # greedy CD, and the effect growing as bits shrink. Our 32-decision exact blocks
            # bought ~0.0%. The two are only comparable if the *starting point* matches, so
            # both are run: from GPTQ directly (their comparison) and from the converged
            # single-coordinate optimum (ours).
            for sub in [int(x) for x in args.pair_subs.split(",") if int(x) > 1]:
                t0 = time.time()
                c_p, ip = cd_subset_exact(w, A, sg, zg, cg, qmax, sub=sub,
                                          rounds=args.rounds, seed=args.seed,
                                          node_budget=args.block_nodes, blocks="random")
                rec["rel_k%d_from_gptq" % sub] = row_damage(w, A, sg, zg, c_p) / d_gptq
                c_q, iq = cd_subset_exact(w, A, sg, zg, c_cd, qmax, sub=sub,
                                          rounds=args.rounds, seed=args.seed,
                                          node_budget=args.block_nodes, blocks="random")
                rec["rel_k%d_from_cd" % sub] = row_damage(w, A, sg, zg, c_q) / d_gptq
                # alternating: single-coordinate sweeps interleaved with k-blocks
                c_a = c_p.clone()
                for _ in range(args.alt_rounds):
                    c_a = cd_row(w, A, sg, zg, c_a, qmax, sweeps=args.sweeps)
                    c_a, _ = cd_subset_exact(w, A, sg, zg, c_a, qmax, sub=sub, rounds=1,
                                             seed=args.seed, node_budget=args.block_nodes,
                                             blocks="random")
                rec["rel_k%d_alternating" % sub] = row_damage(w, A, sg, zg, c_a) / d_gptq
                rec["k%d_s" % sub] = time.time() - t0

            # ---- 2. exact block moves over the whole row
            if ri < args.rows_exact:
                t0 = time.time()
                c_bx, info = cd_subset_exact(w, A, sg, zg, c_cd, qmax, sub=args.sub,
                                             rounds=args.rounds, seed=args.seed,
                                             node_budget=args.block_nodes, blocks="random")
                rec["rel_block_exact"] = row_damage(w, A, sg, zg, c_bx) / d_gptq
                rec["block_exact_s"] = time.time() - t0
                rec["block_frac_proved"] = info["frac_proved"]
                rec["block_flips_vs_cd"] = int((c_bx != c_cd).sum())

                t0 = time.time()
                c_sa = anneal_row(w, A, sg, zg, c_cd, qmax, iters=args.anneal_iters,
                                  seed=args.seed + ri)
                rec["rel_anneal"] = row_damage(w, A, sg, zg, c_sa) / d_gptq
                rec["anneal_s"] = time.time() - t0

            # ---- 3. the subset oracle: exact optimum of n decisions, complement frozen
            for nsub in [int(x) for x in args.subsets.split(",")]:
                gg = torch.Generator(device="cpu").manual_seed(args.seed + 7 * ri + nsub)
                idx = torch.randperm(int(w.numel()), generator=gg)[:nsub].to(dev)
                for tag, cbase, ss, zz in (("gptq", cg, sg, zg), ("cd", c_cd, sg, zg)):
                    R, x, idx = subproblem_form(w, A, ss, zz, cbase, idx)
                    lo = np.zeros(nsub, dtype=np.int64)
                    hi = np.full(nsub, qmax, dtype=np.int64)
                    t0 = time.time()
                    bc, _, nodes, proved = sphere_decode(R, x, lo, hi, cbase[idx],
                                                         node_budget=args.nodes)
                    c2 = cbase.clone()
                    c2[idx] = torch.tensor(bc, dtype=torch.float32, device=dev)
                    d2 = row_damage(w, A, ss, zz, c2)
                    base = d_gptq if tag == "gptq" else d_cd
                    rec["oracle%d_%s_rel" % (nsub, tag)] = d2 / d_gptq
                    rec["oracle%d_%s_gain" % (nsub, tag)] = 1.0 - d2 / max(base, 1e-30)
                    rec["oracle%d_%s_flips" % (nsub, tag)] = int((c2 != cbase).sum())
                    rec["oracle%d_%s_proved" % (nsub, tag)] = bool(proved)
                    rec["oracle%d_%s_nodes" % (nsub, tag)] = int(nodes)
                    rec["oracle%d_%s_s" % (nsub, tag)] = time.time() - t0

            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("  %-34s %-8s row %4d | rtn %.3f  cd %.4f (%.1f%% flips)  "
                  "blockx %s  oracle%d/gptq %+.2f%%  oracle%d/cd %+.2f%%"
                  % (name.split("model.layers.")[-1], coord, r, rec["rel_rtn"],
                     rec["rel_cd"], 100.0 * rec["cd_flips"] / w.numel(),
                     ("%.4f" % rec["rel_block_exact"]) if "rel_block_exact" in rec else "-",
                     int(args.subsets.split(",")[-1]),
                     100.0 * rec["oracle%s_gptq_gain" % args.subsets.split(",")[-1]],
                     int(args.subsets.split(",")[-1]),
                     100.0 * rec["oracle%s_cd_gain" % args.subsets.split(",")[-1]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,up_proj,down_proj")
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--rows-exact", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--sub", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--pair-subs", default="2,4,8",
                    help="exact block sizes for the CDQuant reconciliation")
    ap.add_argument("--alt-rounds", type=int, default=2)
    ap.add_argument("--subsets", default="16,32")
    ap.add_argument("--nodes", type=int, default=300_000)
    ap.add_argument("--block-nodes", type=int, default=30_000)
    ap.add_argument("--anneal-iters", type=int, default=200_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/joint_oracle.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cal = torch.load(args.cal, map_location="cpu")
    names = pick_modules(cal, [int(x) for x in args.layers.split(",")], args.projs.split(","))
    print("%d modules: %s" % (len(names), ", ".join(n.split("layers.")[-1] for n in names)))

    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fh = open(args.out, "a", encoding="utf-8")
    t_all = time.time()
    for n in names:
        W = get_module(model, n).weight.detach().float().to(args.device)
        A = cal[n]["A"].float().to(args.device)
        run_module(n, W, A, args, fh)
        del W, A
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("done in %.0f s, peak VRAM %.0f MiB"
          % (time.time() - t_all,
             torch.cuda.max_memory_allocated() / 2 ** 20 if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
