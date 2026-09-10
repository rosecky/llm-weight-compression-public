"""Tables for the GPTQ scale-source ablation (branch C).

Separates the two candidate mechanisms behind the b4/g256 GPTQ-worse-than-RTN failure:

    M1  truncated group fit   scale fitted on the first min(group, blocksize) columns only
                              (this implementation's deviation; original GPTQ fits the full
                              group from the working matrix)
    M2  scale feedback        scale fitted on compensated working weights rather than the
                              originals (present in the original algorithm too)

`full` fixes M1 and keeps M2; `frozen` fixes both. So:  block vs full = M1's cost,
full vs frozen = M2's cost, and the sign of either can flip with rate (range shrinkage acts
as implicit clipping, which *helps* at some rates).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

RAW = "results/raw/gptq_scalefeed.jsonl"
OUT = "results/tables"


def main():
    os.makedirs(OUT, exist_ok=True)
    R = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    arms = [r for r in R if "block" in r]
    orders = [r for r in R if r.get("arm", "").startswith("order:")]

    out = ["# GPTQ scale-source ablation (branch C)", "",
           "Damage relative to `tr(W A W^T)`, identical storage in every arm. `M1 cost` is",
           "block/full - 1 (what the truncated-group fit costs; negative means the accidental",
           "range shrinkage helped). `M2 cost` is full/frozen - 1 (what fitting scales on",
           "compensated instead of original weights costs in the *original* GPTQ semantics).",
           "",
           "| module | coords | rate | RTN | block | full | frozen | M1 cost | M2 cost | "
           "vs RTN (best) | clip 2nd half |", "|" + "---|" * 11]
    for r in sorted(arms, key=lambda x: (x["module"], x["coord"], -x["group"], -x["bits"])):
        m1 = r["block"] / max(r["full"], 1e-30) - 1.0
        m2 = r["full"] / max(r["frozen"], 1e-30) - 1.0
        best = min(r["block"], r["full"], r["frozen"], r["bs=g"])
        out.append("| %s | %s | b%d g%d | %.5f | %.5f | %.5f | %.5f | %+.1f%% | %+.1f%% | "
                   "%.2fx | %.4f |" % (
                       r["module"], r["coord"], r["bits"], r["group"], r["rtn"], r["block"],
                       r["full"], r["frozen"], 100 * m1, 100 * m2,
                       r["rtn"] / max(best, 1e-30), r.get("clip_half2", 0.0)))
    out.append("")

    out += ["## Aggregate: which scale source wins, by rate", "",
            "| coords | rate | matrices | block wins | full wins | frozen wins | "
            "median frozen/block |", "|" + "---|" * 7]
    key = defaultdict(list)
    for r in arms:
        key[(r["coord"], r["bits"], r["group"])].append(r)
    for (coord, bits, group), S in sorted(key.items(), key=lambda x: (x[0][0], -x[0][2],
                                                                      -x[0][1])):
        wins = {"block": 0, "full": 0, "frozen": 0}
        ratio = []
        for r in S:
            w = min(("block", "full", "frozen"), key=lambda a: r[a])
            wins[w] += 1
            ratio.append(r["frozen"] / max(r["block"], 1e-30))
        ratio.sort()
        out.append("| %s | b%d g%d | %d | %d | %d | %d | %.3f |" % (
            coord, bits, group, len(S), wins["block"], wins["full"], wins["frozen"],
            ratio[len(ratio) // 2]))
    out.append("")

    if orders:
        out += ["## Group-order dependence (b4/g256 family)", "",
                "| module | coords | forward (block) | reverse | random |", "|" + "---|" * 5]
        fwd = {(r["module"], r["coord"]): r["block"] for r in arms
               if r["bits"] == 4 and r["group"] > 128}
        seen = {}
        for r in orders:
            seen.setdefault((r["module"], r["coord"]), {})[r["arm"].split(":")[1]] = r["d"]
        for (m, c), d in sorted(seen.items()):
            out.append("| %s | %s | %.5f | %s | %s |" % (
                m, c, fwd.get((m, c), float("nan")),
                ("%.5f" % d["reverse"]) if "reverse" in d else "-",
                ("%.5f" % d["random"]) if "random" in d else "-"))
        out.append("")

    txt = "\n".join(out) + "\n"
    open("%s/scalefeed.md" % OUT, "w", encoding="utf-8").write(txt)
    print(txt if "--print" in sys.argv else "wrote %s/scalefeed.md" % OUT)


if __name__ == "__main__":
    main()
