"""Rate-allocation oracle: is a fixed bit budget better spent non-uniformly? (question C)

WaterSIC-style reverse waterfilling says uniform rate is generally suboptimal under a
weighted reconstruction metric. This experiment measures the *oracle* value of non-uniform
allocation at hardware-realistic granularity only -- per output row and per matrix, from a
small menu of (bits, group) rungs -- before anyone designs a format:

    RA-U    uniform rung everywhere (the baseline every prior result used)
    RA-A    allocation from the input geometry alone: per-row damages under A
    RA-AG   the same, with rows weighted by the module-G diagonal (GuidedQuant-flavoured;
            the off-diagonal part of G does not row-separate an allocation problem, so it
            is used for EVALUATION but not for the greedy choice)

Method. Each matrix is quantized once at every menu rung (canonical GPTQ + layer-CD --
damages measured post-refinement, so the allocator sees what deployment would see). The
layer objective decomposes exactly over rows, so per-row damage per rung is free. The
allocator is a Lagrangian sweep: for lambda >= 0 each row picks argmin_r d(row, rung) +
lambda * bits(rung); lambda is bisected until the total bits match the uniform budget
exactly (mixing the two rungs adjacent in lambda where needed). Per-matrix granularity runs
the same sweep with matrices as atoms across the whole model slice.

Reported: damage at matched total bits vs RA-U, converted to equivalent bpw via the local
rate-damage curve of the same matrix; held-out validation under the independent second
moment. Kill criterion from the brief: AG gain < ~0.1 equivalent bpw kills the branch;
> ~0.25 continues it toward compressible allocation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch

from ..joint import cd_refine, damage, metric_identity, quantize_with_state
from ..modelio import get_module, load_model
from ..rotate import RotationPair

MENU = [(2, 128), (2, 64), (2, 32), (3, 128), (3, 64), (3, 32), (4, 128), (4, 64)]


def bpw_of(bits: int, group: int) -> float:
    return bits + 32.0 / group


@torch.no_grad()
def row_damages(W, A, bits, group, percdamp, sweeps, seed) -> torch.Tensor:
    _, st = quantize_with_state(W, 2.0 * A, bits, group, percdamp)
    Wh, _ = cd_refine(W, A, st, metric_identity(), sweeps=sweeps, seed=seed)
    D = W - Wh
    return ((D @ A) * D).sum(1), D                      # (out,), (out,in)


def lagrange_alloc(dmg: torch.Tensor, bits: torch.Tensor, budget: float
                   ) -> Tuple[torch.Tensor, float]:
    """dmg: (rows, rungs), bits: (rungs,) bits-per-weight. Returns per-row rung choice
    hitting `budget` (mean bpw) as closely as the discrete menu allows from below/above
    via bisection on lambda."""
    lo, hi = 0.0, float(dmg.max() / max(bits.min(), 1e-9)) + 1.0
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        pick = (dmg + lam * bits.unsqueeze(0)).argmin(1)
        used = float(bits[pick].mean())
        if used > budget:
            lo = lam
        else:
            hi = lam
    lam = hi
    pick = (dmg + lam * bits.unsqueeze(0)).argmin(1)
    return pick, float(bits[pick].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--valcal", default="cache/jointcal_val.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--budgets", default="2.5,3.25")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--cd-sweeps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/raw/rate_alloc.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = args.device

    cal = torch.load(args.cal, map_location="cpu")
    val = torch.load(args.valcal, map_location="cpu") if os.path.exists(args.valcal) else {}
    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    names = [n for n in cal if any("layers.%s." % l in n for l in args.layers.split(","))
             and n.split(".")[-1] in args.projs.split(",")]
    fh = open(args.out, "a", encoding="utf-8")
    budgets = [float(b) for b in args.budgets.split(",")]

    # per-matrix caches for the cross-matrix (per-matrix granularity) sweep
    matrix_curves: Dict[str, Dict] = {}

    for name in sorted(names):
        W0 = get_module(model, name).weight.detach().float().to(dev)
        A0 = 0.5 * (cal[name]["A"].float() + cal[name]["A"].float().T).to(dev)
        Av = val.get(name)
        Av = 0.5 * (Av["A"].float() + Av["A"].float().T).to(dev) if Av is not None else None
        out_f, in_f = W0.shape
        rp = RotationPair(out_f, in_f, seed=args.seed, device=dev)
        W = rp.forward_w(W0)
        Ar = rp.forward_h(A0)
        A = 0.5 * (Ar + Ar.T)
        den = max(damage(W0, A0), 1e-30)

        rungs = [(b, g) for (b, g) in MENU if in_f % g == 0]
        t0 = time.time()
        dmg = torch.zeros(out_f, len(rungs), device=dev)
        dmg_v = torch.zeros_like(dmg) if Av is not None else None
        for i, (b, g) in enumerate(rungs):
            d_row, D = row_damages(W, A, b, g, args.percdamp, args.cd_sweeps, args.seed)
            dmg[:, i] = d_row
            if Av is not None:
                Dn = rp.inverse_w(W - D) - W0           # = -(quantized - fp) in native
                dmg_v[:, i] = ((Dn @ Av) * Dn).sum(1)
                del Dn
            del D
        bits = torch.tensor([bpw_of(b, g) for (b, g) in rungs], device=dev)
        matrix_curves[name] = dict(dmg=dmg.sum(0), dmg_v=(dmg_v.sum(0) if Av is not None
                                                          else None),
                                   bits=bits, n=out_f * in_f, rungs=rungs, den=den)

        for budget in budgets:
            # uniform reference = the menu rung at exactly this bpw if present, else skip
            uni = [i for i, (b, g) in enumerate(rungs) if abs(bpw_of(b, g) - budget) < 1e-9]
            if not uni:
                continue
            u = uni[0]
            d_uni = float(dmg[:, u].sum())
            pick, used = lagrange_alloc(dmg, bits, budget)
            d_ra = float(dmg[torch.arange(out_f, device=dev), pick].sum())
            # equivalent bpw: interpolate the uniform rate-damage curve at d_ra
            order = bits.argsort()
            bs, ds = bits[order].tolist(), dmg.sum(0)[order].tolist()
            eq = None
            for j in range(len(bs) - 1):
                lo_d, hi_d = ds[j + 1], ds[j]
                if min(lo_d, hi_d) <= d_ra <= max(lo_d, hi_d):
                    f = (math.log(d_ra) - math.log(hi_d)) / \
                        (math.log(lo_d) - math.log(hi_d) + 1e-12)
                    eq = bs[j] + f * (bs[j + 1] - bs[j])
                    break
            rec = dict(module=name, granularity="row", budget=budget, used_bpw=used,
                       n_rungs_used=int(pick.unique().numel()),
                       d_uniform=d_uni / den, d_alloc=d_ra / den,
                       gain_frac=1.0 - d_ra / max(d_uni, 1e-30),
                       eq_bpw_gain=(budget - eq) if eq is not None else None,
                       secs=time.time() - t0)
            if Av is not None:
                rec["dv_uniform"] = float(dmg_v[:, u].sum()) / den
                rec["dv_alloc"] = float(
                    dmg_v[torch.arange(out_f, device=dev), pick].sum()) / den
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("%-34s row  @%.2f: uni %.5f -> alloc %.5f (-%4.1f%%) eq +%.3f bpw "
                  "(%d rungs)" % (name.split("layers.")[-1], budget, rec["d_uniform"],
                                  rec["d_alloc"], 100 * rec["gain_frac"],
                                  rec["eq_bpw_gain"] or float("nan"),
                                  rec["n_rungs_used"]))
        del W0, A0, W, A, dmg, dmg_v

    # ---- per-matrix granularity: one rung per matrix across the whole slice
    for budget in budgets:
        mats = [m for m in matrix_curves.values()
                if any(abs(bpw_of(b, g) - budget) < 1e-9 for (b, g) in m["rungs"])]
        if len(mats) < 2:
            continue
        n_tot = sum(m["n"] for m in mats)
        d_uni = sum(float(m["dmg"][[i for i, (b, g) in enumerate(m["rungs"])
                                    if abs(bpw_of(b, g) - budget) < 1e-9][0]]) / m["den"]
                    for m in mats)
        # lagrangian over matrices (weights = matrix sizes)
        lo, hi = 0.0, 1e9
        for _ in range(80):
            lam = 0.5 * (lo + hi)
            tot_bits = 0.0
            for m in mats:
                sc = torch.tensor([float(m["dmg"][i]) / m["den"]
                                   for i in range(len(m["rungs"]))])
                j = int((sc + lam * m["bits"].cpu() * m["n"] / n_tot).argmin())
                tot_bits += float(m["bits"][j]) * m["n"]
            if tot_bits / n_tot > budget:
                lo = lam
            else:
                hi = lam
        lam, tot_bits, d_ra = hi, 0.0, 0.0
        for m in mats:
            sc = torch.tensor([float(m["dmg"][i]) / m["den"] for i in range(len(m["rungs"]))])
            j = int((sc + lam * m["bits"].cpu() * m["n"] / n_tot).argmin())
            tot_bits += float(m["bits"][j]) * m["n"]
            d_ra += float(sc[j])
        rec = dict(granularity="matrix", budget=budget, used_bpw=tot_bits / n_tot,
                   n_matrices=len(mats), d_uniform=d_uni, d_alloc=d_ra,
                   gain_frac=1.0 - d_ra / max(d_uni, 1e-30))
        fh.write(json.dumps(rec) + "\n")
        print("ALL matrices        matrix @%.2f: uni %.5f -> alloc %.5f (-%4.1f%%) used "
              "%.3f bpw" % (budget, d_uni, d_ra, 100 * rec["gain_frac"], rec["used_bpw"]))
    fh.close()


if __name__ == "__main__":
    main()
