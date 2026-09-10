"""Rate-distortion plots: bits/weight vs activation error, and vs perplexity."""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FAMILY = [
    ("B0_", "fp16 reference", "#111111", "*", 14),
    ("B1_", "INT-b RTN group-wise", "#c0392b", "o", 6),
    ("B1b_", "Lloyd-Max scalar (NF-class)", "#e67e22", "s", 6),
    ("B1r", "Hadamard-rotated INT", "#8e44ad", "^", 6),
    ("B2_", "low-rank", "#7f8c8d", "v", 6),
    ("B3_", "residual VQ (AQLM-class)", "#2980b9", "D", 6),
    ("B4_", "VQ + quantized residual", "#16a085", "P", 7),
    ("VA_", "A: VQ + per-tile affine", "#d35400", "<", 7),
    ("VB_", "B: VQ + per-tile rank-1", "#95a5a6", ">", 7),
    ("VC_", "C: learned shared decoder", "#27ae60", "X", 8),
    ("VD_", "D: quant + correction", "#c2185b", "h", 7),
    ("IFS_rec", "IFS: recursive composition", "#6d4c41", "p", 8),
    ("IFS_permmod", "IFS: permutation modulation", "#00838f", "*", 12),
    ("EQ_", "equal-budget VQ controls", "#3949ab", "d", 6),
]


def family_of(name):
    for pref, label, color, marker, size in FAMILY:
        if name.startswith(pref):
            return label, color, marker, size
    return "other", "#000000", ".", 5


def pareto(df, x="bpw", y="act_nmse"):
    d = df.sort_values(x)
    best, keep = float("inf"), []
    for _, r in d.iterrows():
        if r[y] < best:
            best = r[y]
            keep.append(r)
    return pd.DataFrame(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rd", default="results/raw/rd.jsonl,results/raw/rd_recursive.jsonl")
    ap.add_argument("--ppl", default="results/raw/ppl.jsonl")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    parts = []
    for path in args.rd.split(","):
        if os.path.exists(path):
            parts.append(pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8")]))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("name", keep="last")
    df = df[df.bpw < 9]

    for ycol, ylabel, fname in [
        ("act_nmse", r"activation NMSE  $\||(W-\hat{W})X\||^2/\||WX\||^2$", "rd_activation.png"),
        ("weight_nmse", r"weight NMSE  $\||W-\hat{W}\||^2/\||W\||^2$", "rd_weight.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        seen = set()
        for _, r in df.iterrows():
            label, color, marker, size = family_of(r["name"])
            ax.plot(r["bpw"], r[ycol], marker=marker, color=color, ms=size,
                    ls="none", label=label if label not in seen else None,
                    alpha=0.9, mew=0.6, mec="white")
            seen.add(label)
        pf = pareto(df, "bpw", ycol)
        ax.plot(pf["bpw"], pf[ycol], "-", color="#34495e", lw=1.2, alpha=0.6,
                zorder=0, label="Pareto front")
        ax.set_yscale("log")
        ax.set_xlabel("effective bits per original weight (codes + codebooks + decoder + "
                      "scales + residuals + metadata)")
        ax.set_ylabel(ylabel)
        ax.set_title("Rate-distortion, Qwen2.5-0.5B, 20 linear layers "
                     "(q/o/up/down, layers 1,5,11,17,22)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        # alternate label offsets so the crowded 3-4 bit region stays readable
        for i, (_, r) in enumerate(pf.iterrows()):
            dy = 7 if i % 2 == 0 else -11
            ax.annotate(r["name"], (r["bpw"], r[ycol]), fontsize=6,
                        xytext=(5, dy), textcoords="offset points", alpha=0.85,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                  alpha=0.65))
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, fname), dpi=160)
        print("wrote", os.path.join(args.outdir, fname))

    if os.path.exists(args.ppl):
        pp = pd.DataFrame([json.loads(l) for l in open(args.ppl, encoding="utf-8")]).drop_duplicates("name", keep="last")
        if len(pp):
            fig, ax = plt.subplots(figsize=(9, 6))
            seen = set()
            pp = pp[pp.bpw < 9]          # fp16 is shown as the dashed reference line
            for _, r in pp.iterrows():
                label, color, marker, size = family_of(r["name"])
                ax.plot(r["bpw"], r["ppl"], marker=marker, color=color, ms=size + 2,
                        ls="none", label=label if label not in seen else None)
                ax.annotate(r["name"], (r["bpw"], r["ppl"]), fontsize=6,
                            xytext=(5, 4 if len(seen) % 2 == 0 else -10),
                            textcoords="offset points", alpha=0.85,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                                      alpha=0.65))
                seen.add(label)
            base = float(pp.base_ppl.iloc[0])
            ax.axhline(base, color="#111111", ls="--", lw=1,
                       label=f"fp16 = {base:.3f}")
            ax.set_yscale("log")
            ax.set_xlim(1.5, 6.6)
            ax.set_xlabel("effective bits per original weight")
            ax.set_ylabel("wikitext-2 perplexity (all layers compressed)")
            ax.set_title("End-to-end quality, Qwen2.5-0.5B")
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, "rd_perplexity.png"), dpi=160)
            print("wrote", os.path.join(args.outdir, "rd_perplexity.png"))


if __name__ == "__main__":
    main()
