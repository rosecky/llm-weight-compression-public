"""Build the headline summary table and the equal-budget comparisons."""
from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_colwidth", 44)


def load(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) \
        if os.path.exists(p) else pd.DataFrame()


def bits_equivalent(nmse_a, nmse_b):
    """How many bits/weight the gap between two distortions is worth (Gaussian high-rate)."""
    return 0.5 * math.log2(max(nmse_b, 1e-30) / max(nmse_a, 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rd", default="results/raw/rd.jsonl,results/raw/rd_recursive.jsonl")
    ap.add_argument("--ppl", default="results/raw/ppl.jsonl")
    ap.add_argument("--out", default="results/tables/summary.md")
    args = ap.parse_args()

    parts = [load(p) for p in args.rd.split(",")]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df.drop_duplicates("name", keep="last")
    pp = load(args.ppl)
    if len(pp):
        df = df.merge(pp[["name", "ppl", "ppl_delta", "base_ppl"]], on="name", how="left")
    else:
        df["ppl"] = float("nan")
        df["ppl_delta"] = float("nan")

    df = df.sort_values("bpw")
    cols = ["name", "bpw", "weight_nmse", "act_nmse", "ppl", "ppl_delta", "encode_s",
            "decode_flops_per_weight", "decode_bytes_per_weight",
            "decode_shared_state_kib", "decode_verdict"]
    tbl = df[cols].copy()
    tbl.columns = ["method", "bits/weight", "weight err (NMSE)", "activation err (NMSE)",
                   "ppl", "ppl delta", "encode s", "decode FLOPs/w", "decode B/w",
                   "shared state KiB", "practical GPU potential"]

    print("=" * 150)
    print("SUMMARY -- all codecs, sorted by effective bits per weight")
    print("=" * 150)
    print(tbl.round(5).to_string(index=False))

    # ---- Pareto front on activation error ----
    d = df.sort_values("bpw")
    best, keep = float("inf"), []
    for _, r in d.iterrows():
        if r["act_nmse"] < best:
            best = r["act_nmse"]
            keep.append(r["name"])
    print("\nPARETO FRONT (bits/weight vs activation error):")
    print(df[df.name.isin(keep)][["name", "bpw", "act_nmse", "decode_flops_per_weight",
                                  "decode_verdict"]].round(5).to_string(index=False))

    # ---- head-to-head at matched rate ----
    print("\n" + "=" * 150)
    print("HEAD-TO-HEAD AT MATCHED RATE -- how many bits/weight is each family worth against")
    print("the best scalar quantizer at the same rate? (positive = family is better)")
    print("=" * 150)
    scalar = df[df.name.str.startswith(("B1_", "B1b_"))]
    rows = []
    for _, r in df.iterrows():
        cand = scalar[(scalar.bpw <= r["bpw"] + 0.30) & (scalar.bpw >= r["bpw"] - 0.30)]
        if not len(cand) or r["name"].startswith(("B1_", "B1b_", "B0_")):
            continue
        ref = cand.loc[cand.act_nmse.idxmin()]
        rows.append(dict(method=r["name"], bpw=r["bpw"], act_nmse=r["act_nmse"],
                         best_scalar=ref["name"], scalar_bpw=ref["bpw"],
                         scalar_act_nmse=ref["act_nmse"],
                         advantage_bits=bits_equivalent(r["act_nmse"], ref["act_nmse"])
                         - (r["bpw"] - ref["bpw"])))
    h2h = pd.DataFrame(rows).sort_values("advantage_bits", ascending=False)
    print(h2h.round(4).to_string(index=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Summary table\n\nAll storage is fully accounted: codes + codebooks + "
                "decoder parameters + scales + indices + residuals + outliers + metadata, "
                "with shared state amortised over all 357.8M target weights of the model.\n\n")
        f.write(tbl.round(5).to_markdown(index=False))
        f.write("\n\n## Pareto front (bits/weight vs activation error)\n\n")
        f.write(df[df.name.isin(keep)][["name", "bpw", "act_nmse",
                                        "decode_flops_per_weight", "decode_verdict"]]
                .round(5).to_markdown(index=False))
        f.write("\n\n## Head-to-head against the best scalar quantizer at matched rate\n\n")
        f.write("`advantage_bits` > 0 means the method genuinely beats scalar quantization "
                "after paying for everything it needs.\n\n")
        f.write(h2h.round(4).to_markdown(index=False))
        f.write("\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
