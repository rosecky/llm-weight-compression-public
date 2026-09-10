"""Can the rows where expensive discrete search pays be identified cheaply?

The optimality-gap study showed strong row heterogeneity: at 2 bits the k=2/k=8 exact block
moves buy 1-2% on average over converged CD, but the per-row spread is large. If a handful of
rows carries most of that gap, and those rows are recognisable from quantities that are free
by the time GPTQ has run, then an adaptive algorithm follows immediately:

    GPTQ + cheap CD everywhere  ->  expensive search only where the predictor points.

This experiment produces the raw material for that question: one record per row holding

  * cheap features, all available before any search is spent:
      - GPTQ row damage, RTN/GPTQ ratio
      - boundary-code fractions (codes at 0 / at qmax)
      - functional-margin statistics (mean, 10th percentile, fraction "ambiguous"),
        margins normalised by the row's mean per-weight damage so rows are comparable
      - weight-shape statistics (kurtosis, max/rms)
      - the step-vs-curvature scale `mean_j s_j^2 A_jj`, same normalisation
      - cheap-CD outcome (rel_cd, flips) -- CD costs 0.03 s/row, so it counts as cheap
  * the label, from the strongest practical searcher established by the k-move study:
      alternating single-coordinate + exact k=2 blocks, then exact k=8 blocks, then one pass
      of exact k=24 blocks; `gap_cd = 1 - d_strong / d_cd` is what the expensive stage buys.

The analysis (scripts/analyze_predictor.py) then asks the only two questions that matter:
how concentrated is the total gap across rows, and how much of it do the top rows ranked by
each cheap feature capture. No learned classifier -- if a single feature or a two-term score
cannot find the rows, a classifier trained on nine matrices would be noise anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..joint import (cd_row, cd_subset_exact, margins, quantize_with_state, row_damage)
from ..modelio import get_module, load_model
from ..rotate import RotationPair


def row_features(w, codes, sfull, m_row, adiag, qmax, d_gptq, d_rtn):
    n = w.numel()
    per_w = max(d_gptq / n, 1e-30)
    fin = torch.isfinite(m_row)
    m = (m_row[fin] / per_w) if int(fin.sum()) else torch.zeros(1)
    wc = w - w.mean()
    var = max(float((wc * wc).mean()), 1e-30)
    kurt = float((wc ** 4).mean()) / var ** 2
    return dict(
        d_gptq=d_gptq,
        rel_rtn=d_rtn / max(d_gptq, 1e-30),
        bound0=float((codes == 0).float().mean()),
        boundq=float((codes == qmax).float().mean()),
        margin_mean=float(m.mean()),
        margin_p10=float(m.float().quantile(0.10)),
        ambig_frac=float((m < 0.1).float().mean()),
        wkurt=kurt,
        wmax_rms=float(w.abs().max() / w.pow(2).mean().sqrt().clamp_min(1e-30)),
        step_curv=float((sfull * sfull * adiag).mean()) / per_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--modules", default="1.self_attn.q_proj,1.self_attn.o_proj,1.mlp.up_proj,"
                                         "11.self_attn.q_proj,11.self_attn.o_proj,11.mlp.up_proj,"
                                         "22.self_attn.q_proj,22.self_attn.o_proj,22.mlp.up_proj")
    ap.add_argument("--cells", default="2:native:64,2:hadamard:64,3:native:32",
                    help="bits:coord:rows_per_module")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--alt-rounds", type=int, default=2)
    ap.add_argument("--node-budget", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/oracle_pred.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))

    cal = torch.load(args.cal, map_location="cpu")
    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    fh = open(args.out, "a", encoding="utf-8")

    for spec in args.cells.split(","):
        bits, coord, nrows = spec.split(":")
        bits, nrows = int(bits), int(nrows)
        for short in args.modules.split(","):
            name = "model.layers." + short
            if name not in cal:
                continue
            W0 = get_module(model, name).weight.detach().float().clone()
            A0 = 0.5 * (cal[name]["A"].float() + cal[name]["A"].float().T)
            if coord == "hadamard":
                rp = RotationPair(W0.shape[0], W0.shape[1], seed=args.seed, device="cpu")
                W = rp.forward_w(W0)
                A = rp.forward_h(A0)
                A = 0.5 * (A + A.T)
            else:
                W, A = W0, A0

            _, st_g = quantize_with_state(W, 2.0 * A, bits, args.group, args.percdamp)
            _, st_r = quantize_with_state(W, None, bits, args.group, args.percdamp)
            s_g, z_g = st_g.expand()
            s_r, z_r = st_r.expand()
            qmax = st_g.qmax
            M = margins(W, A, st_g, None)
            adiag = torch.diagonal(A)

            g = torch.Generator(device="cpu").manual_seed(args.seed + bits)
            rows = torch.randperm(W.shape[0], generator=g)[:nrows].tolist()
            t_mod = time.time()
            for ri, r in enumerate(rows):
                w = W[r].contiguous()
                sg, zg, cg = s_g[r].contiguous(), z_g[r].contiguous(), st_g.codes[r].float()
                d_gptq = row_damage(w, A, sg, zg, cg)
                if d_gptq <= 0:
                    continue
                d_rtn = row_damage(w, A, s_r[r].contiguous(), z_r[r].contiguous(),
                                   st_r.codes[r].float())
                feat = row_features(w, st_g.codes[r], sg, M[r], adiag, qmax, d_gptq, d_rtn)

                t0 = time.time()
                c_cd = cd_row(w, A, sg, zg, cg, qmax, sweeps=args.sweeps)
                d_cd = row_damage(w, A, sg, zg, c_cd)
                feat["rel_cd"] = d_cd / d_gptq
                feat["cd_flips"] = float((c_cd != cg).float().mean())
                cd_s = time.time() - t0

                # the strongest practical searcher from the k-move study
                t0 = time.time()
                c = c_cd.clone()
                best = d_cd
                for _ in range(args.alt_rounds):
                    c, _ = cd_subset_exact(w, A, sg, zg, c, qmax, sub=2, rounds=1,
                                           seed=args.seed, node_budget=args.node_budget,
                                           blocks="random")
                    c = cd_row(w, A, sg, zg, c, qmax, sweeps=args.sweeps)
                    best = min(best, row_damage(w, A, sg, zg, c))
                c, _ = cd_subset_exact(w, A, sg, zg, c, qmax, sub=8, rounds=1,
                                       seed=args.seed, node_budget=args.node_budget,
                                       blocks="random")
                best = min(best, row_damage(w, A, sg, zg, c))
                c, _ = cd_subset_exact(w, A, sg, zg, c, qmax, sub=24, rounds=1,
                                       seed=args.seed, node_budget=args.node_budget,
                                       blocks="random")
                best = min(best, row_damage(w, A, sg, zg, c))

                rec = dict(module=short, coord=coord, bits=bits, group=args.group, row=r,
                           n=int(w.numel()), d_cd=d_cd,
                           gap_cd=1.0 - best / max(d_cd, 1e-30),
                           gap_gptq=1.0 - best / max(d_gptq, 1e-30),
                           cd_s=cd_s, strong_s=time.time() - t0, **feat)
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("%-22s %-8s b%d  %d rows in %.0fs" % (short, coord, bits,
                                                        len(rows), time.time() - t_mod))
            del W, A, st_g, st_r, s_g, z_g, s_r, z_r, M
    fh.close()


if __name__ == "__main__":
    main()
