"""Join the end-to-end layout results with the cost of describing the layout.

A layout is only worth anything if `codes + scales + layout description` buys more than the
same number of bits spent uniformly. This produces that comparison directly: every row is
charged for its own map, and the last column says what uniform quantization would have cost
to reach the same perplexity.
"""
from __future__ import annotations

import argparse
import json
import math
import os

# Held-out cross-entropy of the allocation map, bits/weight, from scripts/analyze_maps.py
# (best decoder-realisable model in that coordinate system). `col` and `tile` are not measured
# there because their description is a fixed-width header, priced exactly below.
MAP_BITS = {
    ("group", "hadamard"): 0.0002,
    ("group", "native"): 0.0225,
    ("weight", "hadamard"): 2.4024,
    ("weight", "native"): 2.4095,
}

RUNTIME = {
    "uniform": ("H0", "one width for the whole model"),
    "col": ("H1", "per-input-channel width; needs an offline column reorder so that equal "
                  "widths are contiguous, which is absorbable into the checkpoint"),
    "group": ("H2", "one width per (row, 128-column) group: a header per group, uniform "
                    "inside"),
    "tile": ("H2", "one width per 128x128 tile"),
    "weight": ("H4", "per-weight width: per-weight branching and irregular bit packing"),
}


def map_bits(alloc, coord, n_out, n_in, group=128, sym_bits=4):
    if alloc == "uniform":
        return 0.0
    if alloc == "col":
        return sym_bits / n_out                       # one width per input channel
    if alloc == "tile":
        return sym_bits / (group * group)
    return MAP_BITS.get((alloc, coord), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/raw/layout_ppl.jsonl")
    ap.add_argument("--out-md", default="results/tables/layout_ppl.md")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]

    # Qwen2.5-0.5B: the average matrix is 896 x 896 or 4864 x 896 / 896 x 4864; use the
    # weighted average output width so the per-column header is priced honestly.
    n_out_avg = 1741.0

    lines = ["# End-to-end: does an adaptive bit layout help after GPTQ + Hadamard?", "",
             "`layout` is the held-out cost of transmitting the allocation map (measured, not",
             "assumed). `total` is what the checkpoint really costs. `uniform equivalent` is the",
             "uniform rate that reaches the same perplexity, read off the uniform ladder.", ""]
    ladder = {}
    for r in recs:
        if r["alloc"] == "uniform":
            ladder.setdefault(r["coord"], []).append((r["bpw"], r["ppl"]))
    # published uniform ladder from E15 for the same pipeline (scalar/gptq)
    ladder.setdefault("hadamard", []).append((4.25, 12.57))
    ladder.setdefault("native", []).append((4.25, 13.21))

    def uniform_equiv(coord, ppl):
        pts = sorted(ladder[coord])
        xs = [p[0] for p in pts]
        ys = [math.log(p[1]) for p in pts]
        y = math.log(ppl)
        for i in range(len(xs) - 1):
            if ys[i + 1] <= y <= ys[i] or (i == 0 and y > ys[i]) or \
                    (i == len(xs) - 2 and y < ys[i + 1]):
                sl = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
                return xs[i] + (y - ys[i]) / sl if sl else xs[i]
        return float("nan")

    for coord in ("hadamard", "native"):
        for bits in sorted({r["bits"] for r in recs}):
            sel = [r for r in recs if r["coord"] == coord and r["bits"] == bits]
            if not sel:
                continue
            lines += ["", "## %d-bit codes, %s coordinates, sequential GPTQ" % (bits, coord), "",
                      "| alloc | codes+scales | layout | total bpw | ppl | d ppl | "
                      "uniform equivalent | net vs uniform | runtime |",
                      "|---|---|---|---|---|---|---|---|---|"]
            base = [r for r in sel if r["alloc"] == "uniform"][0]
            for r in sorted(sel, key=lambda x: ["uniform", "col", "group", "tile",
                                                "weight"].index(x["alloc"])):
                mb = map_bits(r["alloc"], coord, n_out_avg, 0)
                tot = r["bpw"] + mb
                ue = uniform_equiv(coord, r["ppl"])
                net = ue - tot
                cls, _ = RUNTIME[r["alloc"]]
                lines.append("| %s | %.3f | %.4f | %.3f | %.3f | %+.3f | %.3f | %+.3f | %s |"
                             % (r["alloc"], r["bpw"], mb, tot, r["ppl"],
                                r["ppl"] - r["base_ppl"], ue, net, cls))
                print("%-9s %-8s b%d  total=%.3f bpw  ppl=%.3f  uniform-equiv=%.3f  net=%+.3f"
                      % (r["alloc"], coord, bits, tot, r["ppl"], ue, net))
    lines += ["", "## Runtime classes", ""]
    for k, (c, why) in RUNTIME.items():
        lines.append("- **%s** -- `%s`: %s" % (c, k, why))
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    open(args.out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote %s" % args.out_md)


if __name__ == "__main__":
    main()
