"""Tables for the joint-optimization study.

    joint_oracle.jsonl  how far greedy GPTQ is from the optimum, and who closes the gap
    joint_layer.jsonl   the same at full matrix scale, on three objectives at once
    joint_ppl.jsonl     end-to-end perplexity, which is the only arbiter

Everything is reported relative to GPTQ at the identical bit budget, so a number below 1.000
is a better solution that costs exactly the same to store.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

RAW = "results/raw"
OUT = "results/tables"


def load(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else None


def fmt(v, spec="%.4f"):
    return "-" if v is None else spec % v


# ====================================================================== oracle


def oracle_table(R: List[Dict]) -> str:
    if not R:
        return ""
    # The jsonl is append-only across runs, and an interrupted earlier run with weaker settings
    # left records behind. Keep only rows from the full configuration (which is the only one
    # that carries the 64-decision oracle), and keep the last write per (module, coord, row).
    R = [r for r in R if "oracle64_gptq_gain" in r]
    if not R:
        return ""
    dedup: Dict[tuple, Dict] = {}
    for r in R:
        dedup[(r["module"], r["coord"], r["row"])] = r
    R = list(dedup.values())
    subs = sorted({int(k.split("_")[0][6:]) for r in R for k in r if k.startswith("oracle")})
    out = ["## Kill gate K1 -- the oracle gap on single rows",
           "",
           "Damage relative to GPTQ at identical storage. `cd` is exact coordinate descent from",
           "GPTQ's own solution with GPTQ's own scales; `block-exact` solves blocks of decisions",
           "to proven optimality with a sphere decoder; `anneal` is simulated annealing from the",
           "same start. `oracle-n` is the *exact* optimum of n decisions with the rest frozen.",
           ""]
    hdr = "| coords | rows | RTN | GPTQ | CD | CD multi-start | CD from RTN (RTN scales) |"
    out += [hdr, "|" + "---|" * 6]
    for coord in ("native", "hadamard"):
        S = [r for r in R if r["coord"] == coord]
        if not S:
            continue
        out.append("| %s | %d | %s | 1.0000 | **%s** | %s | %s |" % (
            coord, len(S), fmt(mean([r["rel_rtn"] for r in S])),
            fmt(mean([r["rel_cd"] for r in S])),
            fmt(mean([r.get("rel_cd_multistart") for r in S])),
            fmt(mean([r.get("rel_cd_from_rtn") for r in S]))))
    out += ["", "### Stronger search, on the rows where it was run",
            "",
            "Block-exact and the annealer both start from the coordinate-descent solution, and",
            "only run on a subset of rows, so they are shown against CD *on those same rows*.",
            "",
            "| coords | rows | CD | block-exact (sphere decoder) | simulated annealing |",
            "|" + "---|" * 5]
    for coord in ("native", "hadamard"):
        S = [r for r in R if r["coord"] == coord and "rel_block_exact" in r]
        if not S:
            continue
        out.append("| %s | %d | %s | %s | %s |" % (
            coord, len(S), fmt(mean([r["rel_cd"] for r in S])),
            fmt(mean([r["rel_block_exact"] for r in S])),
            fmt(mean([r.get("rel_anneal") for r in S]))))
    out += ["", "### What the exact oracle finds that the optimizer did not", "",
            "| coords | n free | improvement over GPTQ | improvement over CD | "
            "subproblems proved optimal |", "|" + "---|" * 5]
    for coord in ("native", "hadamard"):
        for n in subs:
            S = [r for r in R if r["coord"] == coord and ("oracle%d_gptq_gain" % n) in r]
            if not S:
                continue
            pv = [r for r in S if r.get("oracle%d_gptq_proved" % n)]
            out.append("| %s | %d | %+.3f%% | %+.3f%% | %.0f%% |" % (
                coord, n, 100 * mean([r["oracle%d_gptq_gain" % n] for r in S]),
                100 * mean([r["oracle%d_cd_gain" % n] for r in S]),
                100.0 * len(pv) / len(S)))
    out += ["", "### By matrix", "",
            "| module | coords | RTN | CD | CD flips |", "|" + "---|" * 5]
    for key in sorted({(r["module"], r["coord"]) for r in R}):
        S = [r for r in R if (r["module"], r["coord"]) == key]
        out.append("| %s | %s | %s | %s | %.1f%% |" % (
            key[0].split("layers.")[-1], key[1],
            fmt(mean([r["rel_rtn"] for r in S])), fmt(mean([r["rel_cd"] for r in S])),
            100 * mean([r["cd_flips"] / r["n"] for r in S])))
    return "\n".join(out) + "\n"


# ====================================================================== layer


def layer_table(R: List[Dict]) -> str:
    if not R:
        return ""
    out = ["## H1 and H2 at full matrix scale", "",
           "Relative damage on three objectives: the layer's own `tr(dW A dW^T)`, the",
           "Fisher-weighted `tr(dW A dW^T G)` at the model's own loss, and the layer objective",
           "against a *held-out* calibration sample. Averaged over matrices.", ""]
    for coord in ("native", "hadamard"):
        S = [r for r in R if r["coord"] == coord]
        if not S:
            continue
        arms = sorted({r["arm"] for r in S}, key=lambda a: (a.split(":")[0], a))
        out += ["### %s coordinates" % coord, "",
                "| arm | layer | Fisher | held-out | weight NMSE | flipped | mean |step| | s |",
                "|" + "---|" * 8]
        for a in arms:
            T = [r for r in S if r["arm"] == a]
            out.append("| %s | %s | %s | %s | %s | %.2f%% | %s | %.0f |" % (
                a, fmt(mean([r["rel_layer"] for r in T])),
                fmt(mean([r["rel_fisher"] for r in T])),
                fmt(mean([r["rel_heldout"] for r in T])),
                fmt(mean([r["w_nmse"] for r in T])),
                100 * mean([r["frac_flipped"] for r in T]),
                fmt(mean([r["mean_abs_step"] for r in T]), "%.2f"),
                mean([r["opt_s"] for r in T])))
        out.append("")
    sp = [r for r in R if "g_rank90" in r]
    if sp:
        out += ["### Spectrum of the downstream metric G", "",
                "| module | dim | rank for 90% of trace | rank for 99% |", "|" + "---|" * 4]
        for r in sorted(sp, key=lambda r: r["module"]):
            out.append("| %s | %d | %d | %d |" % (r["module"].split("layers.")[-1],
                                                  r["g_dim"], r["g_rank90"], r["g_rank99"]))
        out.append("")
    return "\n".join(out) + "\n"


# ====================================================================== end to end


def equivalent_bits(R: List[Dict]) -> str:
    """Convert a perplexity gain into the bit budget that would have bought the same thing.

    The reference is the *baseline's own* rate-quality curve: GPTQ with no revisiting, at every
    bit width present in the data, in the same coordinates. Interpolation is piecewise linear in
    `log(d ppl)` against bpw, so a config that lands exactly on a baseline point converts to
    exactly zero equivalent bits.
    """
    out = []
    for coord in ("native", "hadamard"):
        ref = sorted([(r["bpw"], r["ppl_delta"]) for r in R
                      if r["coord"] == coord and r["comp"] == "gptq"
                      and r.get("cd", 0) == 0 and not r.get("ambig")],
                     key=lambda t: t[0])
        if len(ref) < 2:
            continue
        xs = [b for b, _ in ref]
        ys = [math.log(max(d, 1e-9)) for _, d in ref]

        def bpw_for(delta):
            y = math.log(max(delta, 1e-9))
            if y >= ys[0]:
                return xs[0] + (y - ys[0]) * (xs[1] - xs[0]) / (ys[1] - ys[0])
            for i in range(len(xs) - 1):
                if ys[i] >= y >= ys[i + 1]:
                    t = (y - ys[i]) / (ys[i + 1] - ys[i])
                    return xs[i] + t * (xs[i + 1] - xs[i])
            return xs[-1] + (y - ys[-1]) * (xs[-1] - xs[-2]) / (ys[-1] - ys[-2])

        rows = [r for r in R if r["coord"] == coord and r.get("cd", 0) > 0]
        if not rows:
            continue
        out += ["### Equivalent bit budget, %s coordinates" % coord, "",
                "Reference curve: " + ", ".join("%.2f bpw -> +%.3f ppl" % (b, d)
                                                for b, d in ref), "",
                "| config | bits | bpw | d ppl | GPTQ would need | equivalent gain |",
                "|" + "---|" * 6]
        for r in sorted(rows, key=lambda r: (r["bits"], r["name"])):
            need = bpw_for(r["ppl_delta"])
            out.append("| %s | %d | %.3f | %+.4f | %.3f bpw | **%+.3f bits/weight** |"
                       % (r["name"], r["bits"], r["bpw"], r["ppl_delta"], need,
                          need - r["bpw"]))
        out.append("")
    return "\n".join(out) + "\n" if out else ""


def ppl_table(R: List[Dict]) -> str:
    if not R:
        return ""
    out = ["## End-to-end perplexity (wikitext-2, held-out)", "",
           "`vs GPTQ` is the change against the greedy baseline at the same bits and coordinates;",
           "`vs GPTQ+CD` is against the *local optimum* of the layer objective, which is the",
           "baseline any wider-scope arm has to beat to mean anything. Negative is better, and",
           "the `bpw` column is identical down each block by construction.", "",
           "| config | coords | start | sweeps | scope | bpw | ppl | d ppl | vs GPTQ | "
           "vs GPTQ+CD | flipped | opt s |", "|" + "---|" * 12]

    def is_plain_cd(r):
        return (r.get("cd", 0) > 0 and r.get("scope", "layer") == "layer"
                and not r.get("ambig") and not r.get("adamp") and not r.get("refit")
                and r.get("order", "forward") == "forward" and r["comp"] == "gptq")

    base, base_cd = {}, {}
    for r in R:
        k = (r["coord"], r["comp"], r["bits"])
        if r.get("cd", 0) == 0 and not r.get("ambig"):
            base[k] = r["ppl_delta"]
        if is_plain_cd(r):
            base_cd[(r["coord"], r["bits"])] = r["ppl_delta"]
    for r in R:
        ref = base.get((r["coord"], r["comp"], r["bits"]))
        ref_cd = base_cd.get((r["coord"], r["bits"]))
        gain = ("%+.4f" % (r["ppl_delta"] - ref)) if ref is not None else "-"
        gain_cd = ("%+.4f" % (r["ppl_delta"] - ref_cd)) if ref_cd is not None else "-"
        fl = r.get("flips", {})
        tot = mean([v for k, v in fl.items() if not k.startswith("depth")]) if fl else None
        out.append("| %s | %s | %s | %d | %s | %.4f | %.4f | %+.4f | %s | %s | %s | %.0f |" % (
            r["name"], r["coord"], r["comp"], r.get("cd", 0),
            r.get("scope", "layer"),
            r["bpw"], r["ppl"], r["ppl_delta"], gain, gain_cd,
            ("%.2f%%" % (100 * tot)) if tot is not None else "-", r.get("opt_s", 0.0)))
    out.append("")
    dep = [r for r in R if r.get("flips") and any(k.startswith("depth")
                                                  for k in r["flips"])]
    if dep:
        out += ["### Where the decisions change (fraction of codes moved)", "",
                "| config | q | k | v | o | gate | up | down | d0 | d1 | d2 | d3 |",
                "|" + "---|" * 12]
        keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "depth0", "depth1", "depth2", "depth3"]
        for r in dep:
            row = " | ".join("%.2f%%" % (100 * r["flips"][k]) if k in r["flips"] else "-"
                             for k in keys)
            out.append("| %s | %s |" % (r["name"], row))
        out.append("")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [
        "# Joint optimization of quantization decisions -- results",
        "",
        oracle_table(load("%s/joint_oracle.jsonl" % RAW)),
        layer_table(load("%s/joint_layer.jsonl" % RAW)),
        ppl_table(load("%s/joint_ppl.jsonl" % RAW)),
        equivalent_bits(load("%s/joint_ppl.jsonl" % RAW)),
    ]
    txt = "\n".join(p for p in parts if p)
    open("%s/joint.md" % OUT, "w", encoding="utf-8").write(txt)
    print(txt if "--print" in sys.argv else "wrote %s/joint.md" % OUT)


if __name__ == "__main__":
    main()
