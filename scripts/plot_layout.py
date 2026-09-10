"""Two figures for the programmatic-layout study.

diag_oracle.png   what a free layout is worth, by granularity and by cell, with the
                  matched-Gaussian null overlaid
diag_layout_net.png  the same gains charged for the measured cost of describing the map
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from analyze_oracle import make_inverse_rd

FIG = "results/figures"
UNITS = ["col", "row", "tile", "group", "weight"]
LABEL = {"col": "per-column", "row": "per-row", "tile": "per-tile 128x128",
         "group": "per-group", "weight": "per-weight"}
CELLS = ["naive/native", "gptq/native", "naive/hadamard", "gptq/hadamard"]
MAP_BITS = {"hadamard": 2.4024, "native": 2.4095}


def gains(path, cell, bits=3):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = {}
    for variant in sorted({r.get("variant", "real") for r in recs}):
        by = {(r["cell"], r["base_bits"], r["alloc"]): r
              for r in recs if r.get("variant", "real") == variant}
        if (cell, bits, "uniform") not in by:
            continue
        uni = [(by[(cell, b, "uniform")]["bpw"], math.log10(by[(cell, b, "uniform")]["act_nmse"]))
               for b in (2, 3, 4) if (cell, b, "uniform") in by]
        inv = make_inverse_rd(uni)
        out[variant] = {u: inv(math.log10(by[(cell, bits, u)]["act_nmse"]))
                        - by[(cell, bits, u)]["bpw"]
                        for u in UNITS if (cell, bits, u) in by}
    return out


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---------------------------------------------------------------- figure 1
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = range(len(UNITS))
    colors = {"naive/native": "#c44", "gptq/native": "#e90", "naive/hadamard": "#59c",
              "gptq/hadamard": "#161"}
    for cell in CELLS:
        g = gains("results/raw/oracle.jsonl", cell).get("real", {})
        ax.plot(list(x), [g.get(u, float("nan")) for u in UNITS], "o-", color=colors[cell],
                label=cell, lw=2, ms=6)
    gn = gains("results/raw/oracle_null.jsonl", "gptq/hadamard")
    if "gauss_rowcol" in gn:
        ax.plot([UNITS.index(u) for u in ("col", "group", "weight")],
                [gn["gauss_rowcol"][u] for u in ("col", "group", "weight")],
                "x--", color="k", lw=1.5, ms=9,
                label="gptq/hadamard, matched Gaussian null")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[u] for u in UNITS])
    ax.set_ylabel("bits/weight a FREE layout is worth")
    ax.set_title("Oracle adaptive bit allocation at 3 bits — the map is charged nothing")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("%s/diag_oracle.png" % FIG, dpi=150)

    # ---------------------------------------------------------------- figure 2
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    maps = [json.loads(l) for l in open("results/raw/maps.jsonl", encoding="utf-8")]
    rows = []
    for cell, alloc in (("gptq/hadamard", "group"), ("gptq/hadamard", "weight"),
                        ("gptq/native", "group"), ("gptq/native", "weight")):
        coord = cell.split("/")[1]
        g = gains("results/raw/oracle.jsonl", cell)["real"][alloc]
        sel = [m for m in maps if m["alloc"] == alloc and m["coord"] == coord]
        cost = min(sum(m["real"][k] for m in sel) / len(sel) for k in sel[0]["real"])
        rows.append(("%s\n%s" % (LABEL[alloc], coord), g, cost))
    xs = range(len(rows))
    ax.bar([i - 0.2 for i in xs], [r[1] for r in rows], 0.4, label="gain from the layout",
           color="#2a7")
    ax.bar([i + 0.2 for i in xs], [r[2] for r in rows], 0.4, label="cost of describing it",
           color="#c44")
    for i, r in enumerate(rows):
        ax.text(i, max(r[1], r[2]) + 0.06, "net %+.2f" % (r[1] - r[2]), ha="center",
                fontsize=9)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylabel("bits/weight")
    ax.set_title("A layout must pay for its own description (3 bits, measured held-out)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig("%s/diag_layout_net.png" % FIG, dpi=150)
    print("wrote %s/diag_oracle.png and %s/diag_layout_net.png" % (FIG, FIG))


if __name__ == "__main__":
    main()
