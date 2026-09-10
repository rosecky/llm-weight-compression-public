"""Figures for the joint-optimization study.

joint_oracle.png   how far greedy GPTQ is from the optimum of its own objective, and how much
                   of that gap each strength of optimizer closes
joint_endtoend.png the same question where it counts: perplexity at identical storage, with the
                   compensation-radius ladder and the ambiguity-restricted arms
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "results/raw"
FIG = "results/figures"


def load(p):
    p = "%s/%s" % (RAW, p)
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fig_oracle(R):
    R = [r for r in R if "oracle64_gptq_gain" in r]
    dedup = {}
    for r in R:
        dedup[(r["module"], r["coord"], r["row"])] = r
    R = list(dedup.values())
    if not R:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, coord in zip(axes, ("native", "hadamard")):
        S = [r for r in R if r["coord"] == coord]
        E = [r for r in S if "rel_block_exact" in r]
        bars = [("RTN", mean([r["rel_rtn"] for r in S]), "#c44"),
                ("GPTQ", 1.0, "#888"),
                ("+ coordinate\ndescent", mean([r["rel_cd"] for r in S]), "#2a7"),
                ("+ multi-start", mean([r["rel_cd_multistart"] for r in S]), "#2a7"),
                ("+ exact blocks\n(sphere decoder)", mean([r["rel_block_exact"] for r in E]),
                 "#175"),
                ("+ simulated\nannealing", mean([r["rel_anneal"] for r in E]), "#175")]
        vals = [b[1] for b in bars]
        ax.bar(range(len(bars)), vals, 0.6, color=[b[2] for b in bars])
        ax.axhline(1.0, color="#888", ls=":", lw=1.2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=9)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_ylabel("layer damage relative to GPTQ")
        ax.set_ylim(0, max(1.15, max(vals) * 1.12))
        ax.set_title("%s coordinates, 3 bits, group 128" % coord, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Greedy GPTQ is far from the optimum of its own objective, and a plain "
                 "coordinate descent finds essentially all of it", fontsize=11.5)
    fig.tight_layout()
    fig.savefig("%s/joint_oracle.png" % FIG, dpi=150)


def fig_endtoend(P):
    if not P:
        return
    by = {r["name"]: r for r in P}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    groups = [("native", "#c44"), ("hadamard", "#2a7")]
    labels, vals, cols = [], [], []
    for coord, c in groups:
        for comp in ("naive", "gptq"):
            a = [r for r in P if r["coord"] == coord and r["comp"] == comp
                 and r.get("cd", 0) == 0 and not r.get("ambig") and r.get("bits") == 3]
            b = [r for r in P if r["coord"] == coord and r["comp"] == comp
                 and r.get("cd", 0) > 0 and not r.get("ambig") and not r.get("adamp")
                 and r.get("scope", "layer") == "layer" and r.get("bits") == 3
                 and r.get("order", "forward") == "forward" and not r.get("refit")]
            if not a or not b:
                continue
            labels += ["%s\n%s" % (coord, comp), "%s\n%s + CD" % (coord, comp)]
            vals += [a[0]["ppl_delta"], b[0]["ppl_delta"]]
            cols += ["#bbb", c]
    ax.bar(range(len(vals)), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, "%.3f" % v, ha="center", fontsize=8.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("perplexity increase over fp16")
    ax.set_title("H1: reopening every decision, at identical storage", fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    ladder = ["layer", "module", "block1", "block2", "block4", "block8"]
    xs, ys = [], []
    for i, s in enumerate(ladder):
        m = [r for r in P if r.get("scope", "layer") == s and r["coord"] == "hadamard"
             and r.get("cd", 0) > 0 and not r.get("ambig")]
        if m:
            xs.append(i)
            ys.append(min(r["ppl_delta"] for r in m))
    if xs:
        ax.plot(xs, ys, "o-", color="#2a7", lw=2)
        base = [r for r in P if r["coord"] == "hadamard" and r["comp"] == "gptq"
                and r.get("cd", 0) == 0 and r.get("bits") == 3]
        if base:
            ax.axhline(base[0]["ppl_delta"], color="#888", ls="--", lw=1.4,
                       label="GPTQ, no revisiting")
        ax.set_xticks(list(range(len(ladder))))
        ax.set_xticklabels(["layer\n(GPTQ scope)", "module", "1 block", "2 blocks",
                            "4 blocks", "8 blocks"], fontsize=8.5)
        ax.set_ylabel("perplexity increase over fp16")
        ax.set_title("H2: how far downstream the optimizer is allowed to look", fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("End-to-end, wikitext-2, 3 bits / group 128 -- identical storage in every bar",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig("%s/joint_endtoend.png" % FIG, dpi=150)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_oracle(load("joint_oracle.jsonl"))
    fig_endtoend(load("joint_ppl.jsonl"))
    print("wrote figures to %s" % FIG)


if __name__ == "__main__":
    main()
