"""Tables for the learned-transform branch.

The two questions the tables have to separate are the ones section 5 of the brief insists on:

    does the transform put weights closer to the quantization grid,
    or does it make the resulting errors easier for GPTQ and coordinate descent to cancel?

so every arm is shown with its grid distance next to its post-compensation damage and the two
gain factors, and the random-rotation controls are summarised as a distribution rather than a
point, because "any change of basis helps" has already masqueraded as a result once in this
project.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from typing import Dict, List

RAW = "results/raw"
OUT = "results/tables"


def load(path: str) -> List[Dict]:
    p = "%s/%s" % (RAW, path)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) \
        else []


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def equiv_bits(d_ref: float, d_new: float) -> float:
    """Damage ratio converted to bits, at the high-rate scalar-quantizer slope of 6 dB/bit.

    A factor of four in squared error is one bit, so `0.5 log2(d_ref / d_new)` is what the
    improvement would have cost in rate. It is the right unit for asking whether a learned
    transform pays for the parameters it needs to store.
    """
    return 0.5 * math.log2(max(d_ref, 1e-30) / max(d_new, 1e-30))


def net_table(R: List[Dict]) -> str:
    """What a learned rotation costs *at deployment*, not in the abstract.

    Storing R is the worst of three deployments, and quoting only that cost overstates the
    price (SpinQuant/QuaRot/OSTQuant rotations ship at zero storage). But the free deployments
    carry constraints of their own, and the honest table states which of them each measured
    gain is actually eligible for.
    """
    out = ["## Deployment accounting: fusible, online, or stored?", "",
           "Three ways a rotation can exist at inference time:", "",
           "1. **Fused** -- absorbed into adjacent weight matrices; zero storage, zero runtime.",
           "   Legal only where the path is linear, and for a residual-stream input only if the",
           "   *same* rotation is shared by every consumer of that stream (q,k,v,gate,up and",
           "   every writer). SpinQuant's R1/R2 live here.",
           "2. **Online** -- applied per token at runtime. Zero storage if structured (seeded",
           "   Hadamard), but a general dense R costs a `d x d` multiply per token per site.",
           "   The only legal option where the upstream path is nonlinear (`down_proj` input).",
           "3. **Stored per matrix** -- the full parameter cost lands on one matrix. This is",
           "   what the T3/T5 arms as *trained here* (one R per matrix) would actually cost.",
           "", "Every learned arm below was trained per matrix, so its gain is an **upper",
           "bound** for the fused deployment, which would force one shared R across the stream",
           "-- a constraint the training never saw. All measured modules are `q_proj` (stream",
           "consumers).", "",
           "| module | arm | damage vs Hadamard | gain (bits) | stored: cost/net | "
           "fused: cost/net | fused legality |", "|" + "---|" * 7]
    for m in sorted({r["module"] for r in R}):
        S = [r for r in R if r["module"] == m]
        had = [r for r in S if r["arm"] == "T1_hadamard"]
        if not had:
            continue
        ref = had[0]["dam_cd"]
        for r in sorted([x for x in S if x["arm"].startswith(("T3_", "T4_", "T5_"))],
                        key=lambda x: x["arm"]):
            g = equiv_bits(ref, r["dam_cd"])
            c = r["param_bits_per_weight"]
            out.append("| %s | %s | %.4f | %+.4f | %.3f / %+.4f | 0 / %+.4f | "
                       "only if shared stream-wide |" % (
                           m.split("layers.")[-1], r["arm"], r["dam_cd"] / ref, g,
                           c, g - c, g))
    out.append("")

    # the other lever that buys the same thing: spend the bits on finer groups instead
    cap = load("comp_capacity.jsonl")
    if cap:
        out += ["### Opportunity cost: the group-size lever on the same matrices", "",
                "A learned rotation is never the only way to spend effort or bits. The same",
                "quality gain can be bought by refining the quantizer itself; here is what the",
                "group-size lever pays on the identical matrices (Hadamard coordinates,",
                "post-GPTQ+CD damage from the capacity sweep):", "",
                "| module | trade | delta bpw | damage ratio | bits bought per 0.25 bpw |",
                "|" + "---|" * 5]
        for m in sorted({r["module"] for r in R}):
            C = {(r["bits"], r["group"]): r["d_cd"] for r in cap
                 if r["module"] == m and r["coord"] == "hadamard"}
            for (b1, g1), (b2, g2), db in ((( 3, 128), (3, 64), 0.25),
                                           ((3, 64), (3, 32), 0.5),
                                           ((3, 128), (4, 128), 1.0)):
                if (b1, g1) in C and (b2, g2) in C and C[(b2, g2)] > 0:
                    eb = equiv_bits(C[(b1, g1)], C[(b2, g2)])
                    out.append("| %s | b%dg%d -> b%dg%d | +%.2f | %.3f | %.3f |" % (
                        m.split("layers.")[-1], b1, g1, b2, g2, db,
                        C[(b2, g2)] / C[(b1, g1)], eb * 0.25 / db))
        out += ["",
                "Reading: a fused learned rotation is free, so any gain it delivers survives --",
                "*if* the shared-stream constraint leaves any of the per-matrix gain standing.",
                "A stored per-matrix rotation has to beat the group-size lever at equal bits,",
                "and it does not: refining g128 to g64 costs the same 0.25 bpw and removes",
                "more damage than the best learned rotation measured here."]
    return "\n".join(out) + "\n"


def arm_rows(R: List[Dict], key: str = "dam_cd") -> str:
    mods = sorted({r["module"] for r in R})
    arms = sorted({r["arm"] for r in R})
    out = ["| arm | grid dist | RTN | GPTQ | GPTQ+CD | GPTQ gain | CD gain | "
           "additive/joint after CD | extra bpw |", "|" + "---|" * 9]
    for a in arms:
        T = [r for r in R if r["arm"] == a]
        out.append("| %s | %.4f | %.5f | %.5f | **%.5f** | %.2fx | %.4fx | %.2f | %+.4f |" % (
            a, mean([r["grid_dist"] for r in T]), mean([r["dam_rtn"] for r in T]),
            mean([r["dam_gptq"] for r in T]), mean([r["dam_cd"] for r in T]),
            mean([r["gptq_gain"] for r in T]), mean([r["cd_gain"] for r in T]),
            mean([r.get("cd_additive_over_joint") for r in T]) or 0.0,
            mean([r["param_bits_per_weight"] for r in T])))
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = load("transform_oracle.jsonl")
    if not R:
        print("no transform data yet")
        return
    parts = ["# Learned quantization-aware transforms -- results", "",
             "Damage is `tr(dW A dW^T)` relative to `tr(W A W^T)`, always computed in **native**",
             "coordinates so that arms in different bases are directly comparable. Every arm goes",
             "through the identical GPTQ and the identical coordinate descent.", "",
             "## Averaged over matrices", "", arm_rows(R), ""]

    # section 11: the random-rotation control has to be a distribution, not a point
    parts += ["## The random-rotation control (section 11)", "",
              "| module | Hadamard | random rotations: best | median | p10 | p90 | spread | "
              "best learned |", "|" + "---|" * 8]
    for m in sorted({r["module"] for r in R}):
        S = [r for r in R if r["module"] == m]
        had = [r["dam_cd"] for r in S if r["arm"] == "T1_hadamard"]
        rnd = [r["dam_cd"] for r in S if r["arm"].startswith("T2_")]
        lrn = [r["dam_cd"] for r in S if r["arm"].startswith(("T3_", "T4_", "T5_"))]
        if not had or not rnd:
            continue
        parts.append("| %s | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f | %s |" % (
            m.split("layers.")[-1], had[0], min(rnd), statistics.median(rnd),
            pct(rnd, 0.1), pct(rnd, 0.9), max(rnd) - min(rnd),
            ("%.5f" % min(lrn)) if lrn else "-"))
    parts.append("")

    # section 5: what is the learned transform actually doing?
    parts += ["## Grid proximity against compensated damage (section 5)", "",
              "If a learned basis wins by putting weights closer to the levels, `grid dist` moves",
              "with `GPTQ+CD`. If it wins by making errors more compensatable, `GPTQ gain` and",
              "`CD gain` move while `grid dist` does not.", "",
              "| module | arm | grid dist | pre-comp damage | post-comp damage | GPTQ gain |",
              "|" + "---|" * 6]
    for m in sorted({r["module"] for r in R}):
        for r in [x for x in R if x["module"] == m
                  and x["arm"] in ("T0_native", "T1_hadamard", "T3_grid_had",
                                   "T4_output_had", "T5_compensation_had")]:
            parts.append("| %s | %s | %.4f | %.5f | %.5f | %.2fx |" % (
                m.split("layers.")[-1], r["arm"], r["grid_dist"], r["dam_rtn"],
                r["dam_cd"], r["gptq_gain"]))
    parts.append("")

    # section 15: does a learned basis manufacture more compensation than Hadamard does?
    parts += ["## Error-interaction decomposition (section 15)", "",
              "`additive / joint` is the sum of what each input channel's error would do alone,",
              "divided by the damage actually achieved. Larger means the pipeline cancelled more",
              "of its own error. The question is whether a learned basis systematically raises",
              "it above Hadamard on the *same* matrix.", "",
              "| module | arm | additive/joint after GPTQ | after GPTQ+CD | damage after CD |",
              "|" + "---|" * 5]
    keep = ("T0_native", "T1_hadamard", "T2_had_seed0", "T3_grid_had", "T4_output_had",
            "T5_compensation_had", "T5b_manyouter", "T5c_from_grid")
    for m in sorted({r["module"] for r in R}):
        for a in keep:
            row = [x for x in R if x["module"] == m and x["arm"] == a]
            if not row:
                continue
            r = row[0]
            parts.append("| %s | %s | %.2f | %.2f | %.5f |" % (
                m.split("layers.")[-1], a, r.get("gptq_additive_over_joint", 0.0),
                r.get("cd_additive_over_joint", 0.0), r["dam_cd"]))
    parts.append("")
    parts.append(net_table(R))

    txt = "\n".join(parts)
    open("%s/transform.md" % OUT, "w", encoding="utf-8").write(txt)
    print(txt if "--print" in sys.argv else "wrote %s/transform.md" % OUT)


if __name__ == "__main__":
    main()
