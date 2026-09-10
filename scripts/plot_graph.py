"""Two figures for the graph study.

graph_null.png  community structure of the real graph against four surrogate matrices
graph_ppl.png   end-to-end perplexity by channel ordering, with the random-permutation band
                that makes the graph result interpretable
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "results/figures"
NULLS = ["real", "spectral", "rewire", "config"]
NULL_LABEL = {"real": "real graph", "spectral": "spectral null\n(same eigenvalues)",
              "rewire": "degree-preserving\nrewiring", "config": "configuration\nmodel"}


def mean(v, k):
    return sum(x[k] for x in v) / len(v)


def main():
    os.makedirs(FIG, exist_ok=True)

    # ---------------------------------------------------------------- figure 1
    R = [json.loads(l) for l in open("results/raw/graph_structure.jsonl", encoding="utf-8")]
    agg = defaultdict(list)
    for r in R:
        if r["block"] == 128:
            agg[(r["side"], r["kind"], r["null"])].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (side, title) in zip(axes, [("A", "input graph  A = E[x x$^T$]"),
                                        ("G", "output graph  G = E[g g$^T$]")]):
        xs = range(len(NULLS))
        vals = [mean(agg[(side, "corr", n)], "block_energy") for n in NULLS]
        rnd = mean(agg[(side, "corr", "real")], "block_energy_random")
        scl = mean(agg[(side, "corr", "real")], "block_energy_scale")
        bars = ax.bar(list(xs), vals, 0.6,
                      color=["#2a7"] + ["#c44"] * (len(NULLS) - 1))
        ax.axhline(rnd, color="#666", ls=":", lw=1.5, label="random balanced partition")
        ax.axhline(scl, color="#39c", ls="--", lw=1.5, label="channel-scale sorting")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.006, "%.3f" % v, ha="center", fontsize=9)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([NULL_LABEL[n] for n in NULLS], fontsize=8.5)
        ax.set_ylabel("off-diagonal edge energy inside blocks of 128")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Functional-graph communities against matched nulls (correlation graph)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("%s/graph_null.png" % FIG, dpi=150)

    # ---------------------------------------------------------------- figure 2
    P = [json.loads(l) for l in open("results/raw/graph_endtoend.jsonl", encoding="utf-8")
         if '"ppl"' in l]
    P = [r for r in P if r.get("mode") == "ppl"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    for ax, coord in zip(axes, ("hadamard", "native")):
        sel = [r for r in P if r["coord"] == coord]
        rnds = [r["ppl_delta"] for r in sel if r["order"] == "random"]
        named = [(r["order"], r["ppl_delta"]) for r in sel if r["order"] != "random"]
        named.sort(key=lambda t: -t[1])
        labels = [n for n, _ in named]
        vals = [v for _, v in named]
        colors = {"contig": "#888", "scale": "#39c", "specA_corr": "#2a7"}
        ax.bar(range(len(vals)), vals, 0.55,
               color=[colors.get(n, "#999") for n in labels])
        if rnds:
            ax.axhspan(min(rnds), max(rnds), color="#c44", alpha=0.18,
                       label="random permutations (n=%d)" % len(rnds))
            ax.axhline(sum(rnds) / len(rnds), color="#c44", ls="--", lw=1.4)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.06, "%.3f" % v, ha="center", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["contiguous" if n == "contig" else
                            "scale-sorted" if n == "scale" else "A-corr\ncommunities"
                            for n in labels], fontsize=9)
        ax.set_ylabel("perplexity increase over fp16")
        ax.set_title("%s coordinates, sequential GPTQ, 3.25 bpw" % coord, fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("End-to-end: a graph ordering is not distinguishable from a random one",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("%s/graph_ppl.png" % FIG, dpi=150)
    print("wrote %s/graph_null.png and %s/graph_ppl.png" % (FIG, FIG))


if __name__ == "__main__":
    main()
