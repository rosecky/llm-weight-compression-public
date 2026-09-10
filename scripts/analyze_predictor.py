"""Is the oracle gap concentrated, and do cheap features find it? (branch B)

Two questions only:

1. Concentration: what fraction of rows carries 50% / 80% of the summed CD-to-strong gap
   (gap weighted by each row's absolute damage improvement, not the ratio, because a large
   relative gap on a negligible row buys nothing).
2. Capture: rank rows by each cheap feature alone (and by a two-term score), take the top
   10% / 20%, and report how much of the total weighted gap they contain. The oracle ranking
   ("if you knew the answer") is the ceiling; random selection is the floor (= the fraction
   taken).

No classifier. If no feature beats random by a wide margin, the adaptive algorithm has no
cheap trigger and that is the (negative) result.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

RAW = "results/raw/oracle_pred.jsonl"
OUT = "results/tables"

FEATURES = ["margin_p10", "ambig_frac", "margin_mean", "bound0", "boundq", "rel_cd",
            "cd_flips", "rel_rtn", "wkurt", "wmax_rms", "step_curv", "d_gptq"]
# features where a SMALL value should mean a LARGE gap
ASCENDING = {"margin_p10", "margin_mean", "rel_rtn"}


def capture(rows, weights, order, frac):
    k = max(1, int(round(len(rows) * frac)))
    idx = sorted(range(len(rows)), key=order)[:k]
    tot = sum(weights)
    return sum(weights[i] for i in idx) / max(tot, 1e-30)


def main():
    os.makedirs(OUT, exist_ok=True)
    R = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    cells = defaultdict(list)
    for r in R:
        cells[(r["bits"], r["coord"])].append(r)

    out = ["# Cheap prediction of where expensive search pays (branch B)", ""]
    for (bits, coord), S in sorted(cells.items()):
        # absolute improvement bought by the strong searcher on this row
        w = [max(r["gap_cd"], 0.0) * r["d_cd"] for r in S]
        tot = sum(w)
        gaps = sorted(((wi, r) for wi, r in zip(w, S)), key=lambda x: -x[0])
        out += ["## %d-bit %s  (%d rows, %d matrices)" % (
                    bits, coord, len(S), len({r["module"] for r in S})), "",
                "mean relative gap over CD: %.2f%%   rows with gap > 1%%: %.0f%%" % (
                    100 * sum(r["gap_cd"] for r in S) / max(len(S), 1),
                    100 * sum(1 for r in S if r["gap_cd"] > 0.01) / max(len(S), 1)), ""]
        if tot <= 0:
            out += ["total weighted gap is zero -- nothing for a predictor to find here.", ""]
            continue
        cum, n50, n80 = 0.0, None, None
        for i, (wi, _) in enumerate(gaps):
            cum += wi
            if n50 is None and cum >= 0.5 * tot:
                n50 = (i + 1) / len(S)
            if n80 is None and cum >= 0.8 * tot:
                n80 = (i + 1) / len(S)
        out += ["**Concentration**: 50%% of the total gap sits in the top %.0f%% of rows, "
                "80%% in the top %.0f%%." % (100 * n50, 100 * n80), ""]

        out += ["| ranking | top 10% captures | top 20% captures |", "|---|---|---|",
                "| oracle (upper bound) | %.0f%% | %.0f%% |" % (
                    100 * capture(S, w, lambda i: -w[i], 0.10),
                    100 * capture(S, w, lambda i: -w[i], 0.20)),
                "| random (floor) | 10% | 20% |"]
        for f in FEATURES:
            sgn = 1.0 if f in ASCENDING else -1.0
            out.append("| %s | %.0f%% | %.0f%% |" % (
                f, 100 * capture(S, w, lambda i: sgn * S[i].get(f, 0.0), 0.10),
                100 * capture(S, w, lambda i: sgn * S[i].get(f, 0.0), 0.20)))
        # a deliberately simple two-term score: much ambiguity and much damage to win back
        mu_d = sum(r["d_cd"] for r in S) / len(S)
        out.append("| ambig_frac * d_cd | %.0f%% | %.0f%% |" % (
            100 * capture(S, w, lambda i: -(S[i]["ambig_frac"] * S[i]["d_cd"] / mu_d), 0.10),
            100 * capture(S, w, lambda i: -(S[i]["ambig_frac"] * S[i]["d_cd"] / mu_d), 0.20)))
        out.append("")

    txt = "\n".join(out) + "\n"
    open("%s/predictor.md" % OUT, "w", encoding="utf-8").write(txt)
    print(txt if "--print" in sys.argv else "wrote %s/predictor.md" % OUT)


if __name__ == "__main__":
    main()
