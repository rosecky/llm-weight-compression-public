"""Figures for the final diagnostic: bits/weight vs activation error and vs perplexity,
drawn as a 2x2x2 ablation rather than a scatter of competing codecs."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colour = representation, line style = compensation, marker fill = coordinate system
COLOR = {"scalar": "#c0392b", "vq": "#2980b9"}
STYLE = {"naive": ":", "gptq": "-"}
MARKER = {"native": "o", "hadamard": "^"}


def load(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) \
        if os.path.exists(p) else pd.DataFrame()


def draw(ax, df, ycol, logy=True):
    for (rep, comp, coord), sub in df.groupby(["rep", "comp", "coord"]):
        sub = sub.sort_values("bpw")
        ax.plot(sub["bpw"], sub[ycol], color=COLOR[rep], ls=STYLE[comp],
                marker=MARKER[coord], ms=7, lw=1.8,
                mfc=COLOR[rep] if coord == "native" else "white",
                mec=COLOR[rep], mew=1.4,
                label=f"{rep} / {comp} / {coord}")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="results/raw/factorial.jsonl")
    ap.add_argument("--stage2", default="results/raw/factorial_ppl.jsonl")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    s1 = load(args.stage1)
    if len(s1):
        s1 = s1[(s1.variant == "real") & (~s1.cfg_tag.str.contains("a8"))]
        fig, ax = plt.subplots(figsize=(9, 6))
        draw(ax, s1, "act_nmse")
        ax.set_xlabel("effective bits per weight (codes + scales + codebook + rotation seed)")
        ax.set_ylabel(r"activation NMSE  $\||(W-\hat{W})X\||^2 / \||WX\||^2$")
        ax.set_title("Final diagnostic: representation x compensation x coordinates\n"
                     "Qwen2.5-0.5B, 20 linear layers")
        ax.legend(fontsize=8, title="representation / error handling / coordinates",
                  title_fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "diag_activation.png"), dpi=160)
        print("wrote diag_activation.png")

        # tail error, the metric that MSE hides
        fig, ax = plt.subplots(figsize=(9, 6))
        draw(ax, s1, "relerr_top0001")
        ax.set_xlabel("effective bits per weight")
        ax.set_ylabel("relative error on the top 0.01 % of weights by magnitude")
        ax.set_title("Tail error: what aggregate MSE does not show")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "diag_tail.png"), dpi=160)
        print("wrote diag_tail.png")

    s2 = load(args.stage2)
    if len(s2):
        s2 = s2.drop_duplicates("name", keep="last")
        fig, ax = plt.subplots(figsize=(9, 6))
        draw(ax, s2, "ppl")
        base = float(s2.base_ppl.iloc[0])
        ax.axhline(base, color="#111111", ls="--", lw=1.2, label=f"fp16 = {base:.2f}")
        ax.set_xlabel("effective bits per weight")
        ax.set_ylabel("wikitext-2 perplexity (all 168 linear layers quantized)")
        ax.set_title("End-to-end quality, full-model sequential GPTQ\nQwen2.5-0.5B")
        ax.legend(fontsize=8, title="representation / error handling / coordinates",
                  title_fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "diag_perplexity.png"), dpi=160)
        print("wrote diag_perplexity.png")

        # main-effect bar chart at the 3-bit operating point
        fig, ax = plt.subplots(figsize=(8, 4.4))
        sub = s2[np.isclose(s2.bpw, s2.bpw.round(2))]
        piv = s2.pivot_table(index="bpw", columns=["rep", "comp", "coord"], values="ppl")
        if len(piv):
            piv.plot(kind="bar", ax=ax, logy=True, width=0.8, legend=False)
            ax.axhline(base, color="#111111", ls="--", lw=1.2)
            ax.set_ylabel("wikitext-2 perplexity (log)")
            ax.set_xlabel("effective bits per weight")
            ax.set_title("All eight cells at each rate (dashed = fp16)")
            ax.legend(fontsize=6, ncol=2, loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "diag_cells.png"), dpi=160)
            print("wrote diag_cells.png")


if __name__ == "__main__":
    main()
