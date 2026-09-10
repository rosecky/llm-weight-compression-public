"""Summarise Phase 1 probes: the only number that matters is real - null."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 400)


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8")])


def phase1a(path, out_md=None):
    df = load(path)
    vq = df[df.probe == "vq"].copy()
    vq["tile"] = vq.th.astype(str) + "x" + vq.tw.astype(str)
    key = ["proj", "tile", "d", "K", "normalized"]
    p = vq.pivot_table(index=key, columns="variant", values="net_gain")
    r = vq.pivot_table(index=key, columns="variant", values="rho")
    p["d_real_minus_shuffle"] = p["real"] - p["shuffle"]
    p["d_shuffle_minus_gauss"] = p["shuffle"] - p["gauss_rowcol"]
    p["rho_real"] = r["real"]
    p["rho_shuffle"] = r["shuffle"]
    p = p.reset_index()

    print("=" * 100)
    print("PHASE 1A -- tile VQ net gain (bits/weight). Positive d_real_minus_shuffle = real")
    print("dependence structure that a prototype codec could exploit.")
    print("=" * 100)
    cols = ["proj", "tile", "K", "normalized", "rho_real", "rho_shuffle", "real", "shuffle",
            "gauss_rowcol", "d_real_minus_shuffle", "d_shuffle_minus_gauss"]
    print(p[cols].round(4).to_string(index=False))

    print("\n--- summary of the dependence signal across every VQ probe ---")
    d = p["d_real_minus_shuffle"]
    print(f"n probes        : {len(d)}")
    print(f"mean            : {d.mean():+.5f} bits/weight")
    print(f"median          : {d.median():+.5f}")
    print(f"max             : {d.max():+.5f}  (at {p.loc[d.idxmax(), ['proj','tile','K']].to_dict()})")
    print(f"min             : {d.min():+.5f}")
    print(f"fraction > 0.05 : {(d > 0.05).mean():.3f}")
    print("\n--- shape gain (shuffle - gauss_rowcol), i.e. classic non-Gaussian VQ gain ---")
    s = p["d_shuffle_minus_gauss"]
    print(f"mean {s.mean():+.5f}   median {s.median():+.5f}   max {s.max():+.5f}")

    pca = df[df.probe == "pca"].copy()
    if len(pca):
        pca["tile"] = pca.th.astype(str) + "x" + pca.tw.astype(str)
        pp = pca.pivot_table(index=["proj", "tile", "d", "r"], columns="variant",
                             values=["net_gain", "rho", "r90", "ev_top1_frac"])
        print("\n" + "=" * 100)
        print("PHASE 1A -- tile PCA. r90 = #components for 90% energy (d = flat spectrum).")
        print("=" * 100)
        flat = pca[pca.variant == "real"].groupby(["proj", "tile", "d"]).agg(
            r90=("r90", "first"), r99=("r99", "first"), top1=("ev_top1_frac", "first"))
        flat["r90_frac_of_d"] = flat.r90 / flat.index.get_level_values("d")
        flat["top1_vs_flat"] = flat.top1 * flat.index.get_level_values("d")
        print(flat.round(4).to_string())
        print("\n(top1_vs_flat = 1.0 means the leading eigenvalue is exactly the flat-spectrum "
              "value, i.e. no low-rank structure at all)")

    nn = df[df.probe == "nn"]
    if len(nn):
        print("\n--- nearest-neighbour among unit-norm 8x8 tiles (1.0 = orthogonal) ---")
        print(nn.pivot_table(index="proj", columns="variant", values="nn_rel").round(4).to_string())

    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Phase 1A -- intra/pooled tile structure\n\n")
            f.write(p[cols].round(4).to_markdown(index=False))
            f.write("\n\n## Dependence signal (real - shuffle), bits/weight\n\n")
            f.write(f"- n = {len(d)}, mean = {d.mean():+.5f}, median = {d.median():+.5f}, "
                    f"max = {d.max():+.5f}, min = {d.min():+.5f}\n")
        print(f"\nwrote {out_md}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    phase1a(a.path, a.out_md)
