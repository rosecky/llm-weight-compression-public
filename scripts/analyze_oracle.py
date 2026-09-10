"""Read the oracle allocation sweep and answer the gate question.

The currency is the same one the rest of the project uses: **bits/weight equivalent**. For
each cell (naive|gptq x native|hadamard) we fit the uniform rate-distortion curve
log10(act_nmse) = a + s * bpw over the uniform points, then express an oracle's distortion as
the uniform rate that would have achieved it. `gain_bits` is that rate minus the oracle's own
rate: how many bits/weight a free, perfect layout is worth.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def fit_line(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    s = num / den if den else 0.0
    return my - s * mx, s


def make_inverse_rd(points):
    """points = [(bpw, log10 nmse)] of the uniform ladder, sorted by bpw.

    Returns rate(y): the uniform rate that reaches distortion y, by piecewise-linear
    interpolation (extrapolating with the end segment). Using the measured curve rather than
    a global fit makes the uniform baseline's own gain exactly zero by construction, so
    `gain_bits` is a clean read-off and not a fit residual.
    """
    pts = sorted(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    def rate(y):
        # ys is decreasing in x; find the bracketing segment, extrapolating at the ends
        for i in range(len(xs) - 1):
            hi_y, lo_y = ys[i], ys[i + 1]
            bracketed = lo_y <= y <= hi_y
            below_first = i == 0 and y > hi_y
            above_last = i == len(xs) - 2 and y < lo_y
            if bracketed or below_first or above_last:
                sl = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
                if sl == 0:
                    return xs[i]
                return xs[i] + (y - ys[i]) / sl
        return xs[-1]
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/raw/oracle.jsonl")
    ap.add_argument("--metric", default="act_nmse")
    ap.add_argument("--out-md", default="results/tables/oracle.md")
    args = ap.parse_args()

    recs = load(args.path)
    cells = sorted({r["cell"] for r in recs})
    rates = sorted({r["base_bits"] for r in recs})
    allocs = ["uniform", "col", "row", "tile", "group", "weight"]
    allocs = [a for a in allocs if any(r["alloc"] == a for r in recs)]

    by = {(r["cell"], r["base_bits"], r["alloc"]): r for r in recs}
    lines = ["# Oracle adaptive bit allocation", "",
             "`gain_bits` = bits/weight a *free* oracle layout is worth, read off the uniform",
             "rate-distortion curve of the same cell. Budgets are matched exactly: the oracle",
             "spends the same code bits a uniform quantizer would.", ""]

    for cell in cells:
        uni = [(by[(cell, b, "uniform")]["bpw"],
                math.log10(max(by[(cell, b, "uniform")][args.metric], 1e-12)))
               for b in rates if (cell, b, "uniform") in by]
        a0, s = fit_line([u[0] for u in uni], [u[1] for u in uni])
        inv = make_inverse_rd(uni)
        lines += ["", "## cell `%s`   (uniform slope %.3f decades/bit = %.2f dB/bit)"
                  % (cell, s, -10 * s), "",
                  "| base bits | alloc | bpw | %s | vs uniform | gain_bits | frac 16b | frac<=1b |"
                  % args.metric,
                  "|---|---|---|---|---|---|---|---|"]
        print("\n=== %s   slope %.3f decades/bit ===" % (cell, s))
        for b in rates:
            ru = by.get((cell, b, "uniform"))
            if ru is None:
                continue
            for al in allocs:
                r = by.get((cell, b, al))
                if r is None:
                    continue
                v = max(r[args.metric], 1e-12)
                ratio = ru[args.metric] / v
                gain = inv(math.log10(v)) - r["bpw"]
                f16 = r.get("frac_b16", 0.0)
                flow = r.get("frac_b0", 0.0) + r.get("frac_b1", 0.0)
                lines.append("| %d | %s | %.3f | %.5f | %.2fx | %+.3f | %.3f | %.3f |"
                             % (b, al, r["bpw"], v, ratio, gain, f16, flow))
                print("  b0=%d %-8s bpw=%.3f  %s=%.5f  %5.2fx  gain=%+.3f bits"
                      % (b, al, r["bpw"], args.metric, v, ratio, gain))

    # headline: how much of the arbitrary-oracle gain do the coarse units capture?
    lines += ["", "## Oracle-gain capture (fraction of the per-weight oracle's gain_bits)", "",
              "| cell | base bits | " + " | ".join(allocs[1:]) + " |",
              "|---|---|" + "---|" * len(allocs[1:])]
    for cell in cells:
        uni = [(by[(cell, b, "uniform")]["bpw"],
                math.log10(max(by[(cell, b, "uniform")][args.metric], 1e-12)))
               for b in rates if (cell, b, "uniform") in by]
        inv = make_inverse_rd(uni)
        for b in rates:
            if (cell, b, "weight") not in by:
                continue
            def g(al):
                r = by.get((cell, b, al))
                if r is None:
                    return None
                return inv(math.log10(max(r[args.metric], 1e-12))) - r["bpw"]
            gw = g("weight")
            row = ["| %s | %d " % (cell, b)]
            for al in allocs[1:]:
                ga = g(al)
                row.append("| %.0f%% " % (100.0 * ga / gw) if ga is not None and gw
                           else "| - ")
            lines.append("".join(row) + "|")

    import os
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    open(args.out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote %s" % args.out_md)


if __name__ == "__main__":
    main()
