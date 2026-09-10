"""Factorial-effects analysis of the final diagnostic.

The eight cells are not eight competing codecs. They are a 2x2x2 ablation, so what matters is
the decomposition into main effects and interactions.

Effects are expressed in **equivalent bits per weight**, which is the natural additive scale:
for each cell we fit `log2(distortion) = a + slope * bpw` and convert a distortion ratio into
bits using that cell's own measured slope. So "+0.8 bits" means the treatment reaches the same
quality as the control while spending 0.8 fewer bits per weight.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)

REPS = ["scalar", "vq"]
COMPS = ["naive", "gptq"]
COORDS = ["native", "hadamard"]


def load(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) \
        if os.path.exists(p) else pd.DataFrame()


def fit_cell(sub, ycol):
    """log2(y) = a + slope*bpw. Returns (a, slope) with slope < 0."""
    x = sub["bpw"].to_numpy(float)
    y = np.log2(np.clip(sub[ycol].to_numpy(float), 1e-12, None))
    if len(x) < 2:
        return float(y[0]) if len(x) else float("nan"), -2.0
    slope, a = np.polyfit(x, y, 1)
    return a, slope


def predict(a, slope, bpw):
    return 2.0 ** (a + slope * bpw)


def bits_equiv(y_ctrl, y_treat, slope):
    """How many bits/weight the improvement is worth, using the cell's own slope."""
    return (math.log2(max(y_ctrl, 1e-12)) - math.log2(max(y_treat, 1e-12))) / abs(slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="results/raw/factorial.jsonl")
    ap.add_argument("--stage2", default="results/raw/factorial_ppl.jsonl")
    ap.add_argument("--ycol", default="act_nmse")
    ap.add_argument("--rates", default="2.25,3.0,3.5,4.0")
    ap.add_argument("--out", default="results/tables/factorial.md")
    args = ap.parse_args()

    df = load(args.stage1)
    df = df[df.variant == "real"]
    df = df[~df.cfg_tag.str.contains("a8")]          # the atom-width control, reported apart
    rates = [float(x) for x in args.rates.split(",")]

    # ---------------------------------------------------------------- per-cell curves
    fits, rows = {}, []
    for rep, comp, coord in itertools.product(REPS, COMPS, COORDS):
        sub = df[(df.rep == rep) & (df.comp == comp) & (df.coord == coord)]
        if not len(sub):
            continue
        a, slope = fit_cell(sub, args.ycol)
        fits[(rep, comp, coord)] = (a, slope)
        for r in rates:
            rows.append(dict(rep=rep, comp=comp, coord=coord, bpw=r,
                             pred=predict(a, slope, r), slope=slope))
    curve = pd.DataFrame(rows)

    print("=" * 120)
    print(f"PER-CELL RATE-DISTORTION FITS on {args.ycol}   log2(D) = a + slope*bpw")
    print("=" * 120)
    ft = pd.DataFrame([dict(cell=f"{k[0]}/{k[1]}/{k[2]}", intercept=v[0], slope=v[1],
                            bits_per_decade=abs(v[1]))
                       for k, v in fits.items()])
    print(ft.round(3).to_string(index=False))
    print("\nPredicted", args.ycol, "at matched rates:")
    piv = curve.pivot_table(index=["rep", "comp", "coord"], columns="bpw", values="pred")
    print(piv.round(5).to_string())

    # ---------------------------------------------------------------- factorial effects
    print("\n" + "=" * 120)
    print("MAIN EFFECTS, in equivalent bits per weight (positive = the treatment helps)")
    print("=" * 120)
    eff = []
    for r in rates:
        def P(rep, comp, coord):
            a, s = fits[(rep, comp, coord)]
            return predict(a, s, r), s

        for comp, coord in itertools.product(COMPS, COORDS):     # representation gain
            c, sc = P("scalar", comp, coord)
            v, sv = P("vq", comp, coord)
            eff.append(dict(effect="representation (VQ - scalar)", bpw=r,
                            holding=f"{comp}/{coord}", bits=bits_equiv(c, v, (sc + sv) / 2)))
        for rep, coord in itertools.product(REPS, COORDS):       # compensation gain
            n, sn = P(rep, "naive", coord)
            g, sg = P(rep, "gptq", coord)
            eff.append(dict(effect="compensation (GPTQ - naive)", bpw=r,
                            holding=f"{rep}/{coord}", bits=bits_equiv(n, g, (sn + sg) / 2)))
        for rep, comp in itertools.product(REPS, COMPS):         # rotation gain
            n, sn = P(rep, comp, "native")
            h, sh = P(rep, comp, "hadamard")
            eff.append(dict(effect="rotation (Hadamard - native)", bpw=r,
                            holding=f"{rep}/{comp}", bits=bits_equiv(n, h, (sn + sh) / 2)))
    E = pd.DataFrame(eff)
    summ = E.groupby(["effect", "bpw"]).bits.agg(["mean", "min", "max"]).reset_index()
    print(summ.round(3).to_string(index=False))
    print("\nBroken out by the level being held fixed:")
    print(E.pivot_table(index=["effect", "holding"], columns="bpw", values="bits")
          .round(3).to_string())

    # ---------------------------------------------------------------- interactions
    print("\n" + "=" * 120)
    print("TWO-WAY INTERACTIONS, in equivalent bits per weight")
    print("(effect of X when Y is ON, minus effect of X when Y is OFF)")
    print("=" * 120)
    inter = []
    for r in rates:
        def P(rep, comp, coord):
            a, s = fits[(rep, comp, coord)]
            return predict(a, s, r), s

        for coord in COORDS:      # representation x compensation
            gs = bits_equiv(*[P("scalar", c, coord)[0] for c in ("naive", "gptq")],
                            P("scalar", "gptq", coord)[1])
            gv = bits_equiv(*[P("vq", c, coord)[0] for c in ("naive", "gptq")],
                            P("vq", "gptq", coord)[1])
            inter.append(dict(interaction="representation x compensation", bpw=r,
                              at=coord, bits=gv - gs))
        for comp in COMPS:        # representation x rotation
            gs = bits_equiv(*[P("scalar", comp, c)[0] for c in ("native", "hadamard")],
                            P("scalar", comp, "hadamard")[1])
            gv = bits_equiv(*[P("vq", comp, c)[0] for c in ("native", "hadamard")],
                            P("vq", comp, "hadamard")[1])
            inter.append(dict(interaction="representation x rotation", bpw=r,
                              at=comp, bits=gv - gs))
        for rep in REPS:          # compensation x rotation
            gn = bits_equiv(*[P(rep, c, "native")[0] for c in ("naive", "gptq")],
                            P(rep, "gptq", "native")[1])
            gh = bits_equiv(*[P(rep, c, "hadamard")[0] for c in ("naive", "gptq")],
                            P(rep, "gptq", "hadamard")[1])
            inter.append(dict(interaction="compensation x rotation", bpw=r,
                              at=rep, bits=gh - gn))
    I = pd.DataFrame(inter)
    print(I.pivot_table(index=["interaction", "at"], columns="bpw", values="bits")
          .round(3).to_string())

    # ---------------------------------------------------------------- tails
    print("\n" + "=" * 120)
    print("TAIL ERROR by cell (relative error on the largest-magnitude weights)")
    print("=" * 120)
    tcols = [c for c in ("relerr_top01", "relerr_top001", "relerr_top0001",
                         "max_err_over_max_w", "signflip_top001") if c in df.columns]
    tail = df.groupby(["rep", "comp", "coord"])[tcols].mean()
    print(tail.round(5).to_string())

    # ---------------------------------------------------------------- null control
    full = load(args.stage1)
    nulls = full[full.variant != "real"]
    if len(nulls):
        print("\n" + "=" * 120)
        print("MATCHED-GAUSSIAN CONTROL: real weights vs a matched i.i.d. Gaussian")
        print("=" * 120)
        both = full[full.cfg_tag.isin(nulls.cfg_tag.unique())]
        cmp = both.pivot_table(index=["rep", "comp", "coord", "cfg_tag", "bpw"],
                               columns="variant", values=args.ycol).reset_index()
        vcol = [c for c in cmp.columns if c not in
                ("rep", "comp", "coord", "cfg_tag", "bpw", "real")]
        if vcol:
            cmp["real_over_null"] = cmp["real"] / cmp[vcol[0]]
            print(cmp.round(5).to_string(index=False))
            print(f"\nmean real/null ratio = {cmp['real_over_null'].mean():.4f} "
                  f"(1.0 = the real matrix has no advantage over matched noise)")

    # ---------------------------------------------------------------- stage 2
    pp = load(args.stage2)
    if len(pp):
        print("\n" + "=" * 120)
        print("STAGE 2 -- full-model sequential GPTQ, wikitext-2 perplexity")
        print("=" * 120)
        pp = pp.drop_duplicates("name", keep="last").sort_values(["bpw", "name"])
        print(pp[["name", "rep", "comp", "coord", "bpw", "ppl", "ppl_delta",
                  "encode_s"]].round(3).to_string(index=False))
        print("\nperplexity by cell and rate:")
        print(pp.pivot_table(index=["rep", "comp", "coord"], columns="bpw",
                             values="ppl").round(3).to_string())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Factorial diagnostic\n\n## Per-cell rate-distortion fits\n\n")
        f.write(ft.round(3).to_markdown(index=False))
        f.write(f"\n\n## Predicted {args.ycol} at matched rates\n\n")
        f.write(piv.round(5).to_markdown())
        f.write("\n\n## Main effects (equivalent bits/weight)\n\n")
        f.write(summ.round(3).to_markdown(index=False))
        f.write("\n\n## Main effects by held level\n\n")
        f.write(E.pivot_table(index=["effect", "holding"], columns="bpw",
                              values="bits").round(3).to_markdown())
        f.write("\n\n## Two-way interactions (equivalent bits/weight)\n\n")
        f.write(I.pivot_table(index=["interaction", "at"], columns="bpw",
                              values="bits").round(3).to_markdown())
        f.write("\n\n## Tail error by cell\n\n")
        f.write(tail.round(5).to_markdown())
        if len(pp):
            f.write("\n\n## Stage 2 perplexity\n\n")
            f.write(pp[["name", "bpw", "ppl", "ppl_delta"]].round(3).to_markdown(index=False))
        f.write("\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
