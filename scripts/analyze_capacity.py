"""Tables for the compensation-capacity and calibration-generalization study.

Two questions, deliberately kept apart:

    capacity        how much of the raw quantization error can local compensation remove, and
                    does that fraction collapse at low bitrate (is there a cliff?)
    generalization  does the compensation that *is* achieved transfer off the calibration
                    sample, and is the answer a function of bitrate or of samples-per-dimension

The second is the one that decides whether a robust or uncertainty-aware GPTQ is worth pursuing:
if the held-out gap is governed by tokens per input dimension, more calibration data closes it
and there is no research problem; if it persists at the community-standard budget, there is.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

RAW = "results/raw"
OUT = "results/tables"


def load(p: str) -> List[Dict]:
    path = "%s/%s" % (RAW, p)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] \
        if os.path.exists(path) else []


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fmt(v, s="%.4f"):
    return "-" if v is None else s % v


# ====================================================================== capacity


def capacity_tables(R: List[Dict]) -> str:
    if not R:
        return ""
    out = ["## Compensation capacity as a function of bitrate", "",
           "`removed` is `1 - D_post / D_raw`: the fraction of the raw round-to-nearest damage",
           "that compensation absorbed. `held-out ratio` is the same weights scored against an",
           "independent calibration sample, divided by the score on the sample they were fitted",
           "to -- so 1.0 would mean the compensation transfers perfectly.", ""]
    for coord in ("native", "hadamard"):
        S = [r for r in R if r["coord"] == coord]
        if not S:
            continue
        out += ["### %s coordinates (mean over %d matrices)"
                % (coord, len({r["module"] for r in S})), "",
                "Group 256 does not divide the 896-wide inputs, so those rows cover only the",
                "three 4864-wide `down_proj` matrices. The `matrices` column is there because",
                "without it those rows look like a group-size effect when they are a change of",
                "sample.", "",
                "| bpw | format | matrices | raw | GPTQ | GPTQ+CD | removed by GPTQ | "
                "by GPTQ+CD | held-out ratio |", "|" + "---|" * 9]
        for bpw in sorted({r["bpw"] for r in S}):
            T = [r for r in S if r["bpw"] == bpw]
            out.append("| %.3f | b%d g%d | %d | %s | %s | %s | %.1f%% | **%.1f%%** | %s |" % (
                bpw, T[0]["bits"], T[0]["group"], len({r["module"] for r in T}),
                fmt(mean([r["d_raw"] for r in T]), "%.5f"),
                fmt(mean([r["d_gptq"] for r in T]), "%.5f"),
                fmt(mean([r["d_cd"] for r in T]), "%.5f"),
                100 * mean([r["frac_gptq"] for r in T]),
                100 * mean([r["frac_cd"] for r in T]),
                fmt(mean([r.get("gap_ratio_cd") for r in T]), "%.3f")))
        out.append("")

    # is the overfit a function of bitrate, or of samples per dimension?
    out += ["### What governs the held-out gap: bitrate, or samples per dimension?", "",
            "Calibration here is 8192 tokens, so a 896-wide input sees 9.1 samples per",
            "dimension and a 4864-wide one sees 1.7. If the gap tracks the second column rather",
            "than the first, it is a finite-sample effect and more calibration data fixes it.",
            "",
            "| module | input dim | tokens/dim | held-out ratio at 2 bit | at 3 bit | at 4 bit |",
            "|" + "---|" * 6]
    for mod in sorted({r["module"] for r in R}):
        S = [r for r in R if r["module"] == mod and r["coord"] == "hadamard"]
        if not S:
            continue
        by_bits = {b: mean([r.get("gap_ratio_cd") for r in S if r["bits"] == b])
                   for b in (2, 3, 4)}
        out.append("| %s | %d | %.1f | %s | %s | %s |" % (
            mod.split("layers.")[-1], S[0]["inn"], 8192.0 / S[0]["inn"],
            fmt(by_bits[2], "%.3f"), fmt(by_bits[3], "%.3f"), fmt(by_bits[4], "%.3f")))
    out.append("")

    # the rotation's value is not uniform across matrices -- state that rather than average it
    out += ["### How much the rotation is worth, per matrix and per rate", "",
            "Ratio of post-GPTQ+CD damage, Hadamard over native. Below 1 means the rotation",
            "helps. Averaging this over matrices hides that it ranges from negligible to 4x.",
            "", "| module | 2.25 bpw | 3.25 bpw | 4.25 bpw |", "|" + "---|" * 4]
    for mod in sorted({r["module"] for r in R}):
        row = ["| " + mod.split("layers.")[-1]]
        for bpw in (2.25, 3.25, 4.25):
            n = [r["d_cd"] for r in R if r["module"] == mod and r["coord"] == "native"
                 and abs(r["bpw"] - bpw) < 1e-9]
            h = [r["d_cd"] for r in R if r["module"] == mod and r["coord"] == "hadamard"
                 and abs(r["bpw"] - bpw) < 1e-9]
            row.append(fmt((h[0] / n[0]) if (n and h and n[0] > 0) else None, "%.3f"))
        out.append(" | ".join(row) + " |")
    out.append("")

    # GPTQ is not always an improvement
    bad = [r for r in R if r["frac_gptq"] < 0]
    if bad:
        out += ["### Cases where GPTQ made it *worse* than plain rounding", "",
                "| module | coords | bpw | format | raw | GPTQ | removed |", "|" + "---|" * 7]
        for r in sorted(bad, key=lambda r: r["frac_gptq"]):
            out.append("| %s | %s | %.3f | b%d g%d | %.5f | %.5f | %+.1f%% |" % (
                r["module"].split("layers.")[-1], r["coord"], r["bpw"], r["bits"],
                r["group"], r["d_raw"], r["d_gptq"], 100 * r["frac_gptq"]))
        out.append("")
    return "\n".join(out) + "\n"


# ====================================================================== calibration size


def calib_tables(R: List[Dict]) -> str:
    if not R:
        return ""
    geo = [r for r in R if r.get("stage") == "geometry"]
    sol = [r for r in R if r.get("stage") == "solution"]
    out = ["## Calibration budget: does low-bit quantization stay sample-sensitive?", ""]

    if geo:
        out += ["### Geometry reproducibility across disjoint draws", "",
                "Mean squared cosine of the principal angles between the leading eigenspaces of",
                "two independent draws. 1.0 would mean the directions GPTQ compensates along are",
                "a property of the model rather than of the text sample.", "",
                "| module | dim | tokens | seqlen | tokens/dim | subspace-8 | -32 | -128 | "
                "diag | off-diag |", "|" + "---|" * 10]
        for r in sorted(geo, key=lambda r: (r["module"], r["seqlen"], r["tokens"])):
            out.append("| %s | %d | %d | %d | %.1f | %s | %s | %s | %s | %s |" % (
                r["module"].split("layers.")[-1], r["dim"], r["tokens"], r["seqlen"],
                r["tokens_per_dim"], fmt(r.get("pair_subspace8"), "%.3f"),
                fmt(r.get("pair_subspace32"), "%.3f"), fmt(r.get("pair_subspace128"), "%.3f"),
                fmt(r.get("pair_diag_corr"), "%.3f"), fmt(r.get("pair_offdiag_corr"), "%.3f")))
        out.append("")

    if sol:
        out += ["### Solution stability: spread of held-out damage across draws", "",
                "All draws are scored against the *same* large held-out second moment, so the",
                "spread is sampling variability of the solution, not of the yardstick.",
                "`codes differ` is the fraction of integer codes on which two draws disagree --",
                "a large value with a small spread means the solution is merely non-unique, not",
                "unstable.", "",
                "| module | dim | tokens | seqlen | bpw | held-out | rel spread | codes differ | "
                "fit/held gap |", "|" + "---|" * 9]
        for r in sorted(sol, key=lambda r: (r["module"], r["seqlen"], r["bpw"], r["tokens"])):
            out.append("| %s | %d | %d | %d | %.3f | %s | %.2f%% | %.1f%% | %s |" % (
                r["module"].split("layers.")[-1], r["dim"], r["tokens"], r["seqlen"],
                r["bpw"], fmt(r["held_mean"], "%.5f"), 100 * r["held_rel_spread"],
                100 * r["codes_differ"], fmt(r["gap_ratio"], "%.3f")))
        out.append("")

        out += ["### The decisive view: does the gap close with budget, per bitrate?", "",
                "| module | bpw | " + " | ".join("%dk" % (t // 1024) for t in
                                                 sorted({r["tokens"] for r in sol})) + " |",
                "|" + "---|" * (2 + len({r["tokens"] for r in sol}))]
        toks = sorted({r["tokens"] for r in sol})
        for mod in sorted({r["module"] for r in sol}):
            for bpw in sorted({r["bpw"] for r in sol}):
                cells = []
                for t in toks:
                    m = [r["gap_ratio"] for r in sol if r["module"] == mod
                         and abs(r["bpw"] - bpw) < 1e-9 and r["tokens"] == t]
                    cells.append(fmt(mean(m), "%.3f"))
                out.append("| %s | %.3f | %s |" % (mod.split("layers.")[-1], bpw,
                                                   " | ".join(cells)))
        out.append("")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = ["# Compensation capacity and calibration generalization", "",
             capacity_tables(load("comp_capacity.jsonl")),
             calib_tables(load("calib_size.jsonl"))]
    txt = "\n".join(p for p in parts if p)
    open("%s/capacity.md" % OUT, "w", encoding="utf-8").write(txt)
    print(txt if "--print" in sys.argv else "wrote %s/capacity.md" % OUT)


if __name__ == "__main__":
    main()
