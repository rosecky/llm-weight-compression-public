"""Is the oracle's bit-allocation map compressible?

The oracle gain is only usable if the map that produces it can be *described* for fewer bits
than it saves. So this script measures the description length of the map directly, as
held-out cross-entropy under a series of decoder-realisable models:

    marginal          i.i.d. code over bit widths            -- the metadata cost with no model
    row / col         the width depends only on the output / input channel
    group             the width depends only on the (row, 128-column block)
    left / up         first-order context on an already-decoded neighbour
    left+up           two-neighbour context

Everything is fitted on a random half of the positions and scored on the other half, with
Laplace smoothing, because a plug-in entropy on the same data is exactly the mistake this
project caught itself making in E0.

Each model is also scored on a **within-row shuffle** of the same map: same marginal, same
per-row distribution, no spatial arrangement. `real - shuffle` is the part of the
compressibility that comes from arrangement rather than from the histogram.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import torch

SYMS = [0, 1, 2, 3, 4, 5, 6, 8, 16]
SYM_IX = {s: i for i, s in enumerate(SYMS)}


def to_sym(B: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(B, dtype=torch.long)
    for s, i in SYM_IX.items():
        out[B == s] = i
    return out


def xent(ctx: torch.Tensor, sym: torch.Tensor, n_ctx: int, mask: torch.Tensor,
         alpha: float = 0.5) -> float:
    """Held-out cross-entropy (bits/symbol) of p(sym | ctx), fitted where mask, scored where not."""
    K = len(SYMS)
    tr = mask
    counts = torch.zeros(n_ctx, K, dtype=torch.float64, device=sym.device)
    counts.index_put_((ctx[tr], sym[tr]), torch.ones(int(tr.sum()), dtype=torch.float64,
                                                     device=sym.device), accumulate=True)
    counts += alpha
    p = counts / counts.sum(1, keepdim=True)
    te = ~tr
    lp = torch.log2(p[ctx[te], sym[te]])
    return float(-lp.mean())


def analyse_map(B: torch.Tensor, seed: int = 0) -> dict:
    out_f, in_f = B.shape
    S = to_sym(B)
    K = len(SYMS)
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(out_f, in_f, generator=g) < 0.5

    r = torch.arange(out_f).reshape(-1, 1).expand(out_f, in_f)
    c = torch.arange(in_f).reshape(1, -1).expand(out_f, in_f)
    grp = r * (in_f // 128) + c // 128
    left = torch.cat([S[:, :1], S[:, :-1]], 1)
    up = torch.cat([S[:1, :], S[:-1, :]], 0)

    f = lambda x: x.reshape(-1)
    S1, M1 = f(S), f(mask)
    models = {
        "marginal": (torch.zeros_like(S1), 1),
        "row": (f(r), out_f),
        "col": (f(c), in_f),
        "group": (f(grp), out_f * (in_f // 128)),
        "left": (f(left), K),
        "up": (f(up), K),
        "left+up": (f(left) * K + f(up), K * K),
        "left+col": (f(left) * in_f + f(c), K * in_f),
    }
    res = {}
    for name, (ctx, n) in models.items():
        res[name] = xent(ctx, S1, n, M1)
    return res


def shuffle_rows(B: torch.Tensor, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    idx = torch.argsort(torch.rand(B.shape, generator=g), dim=1)
    return torch.gather(B, 1, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="cache/oracle_maps.pt")
    ap.add_argument("--out", default="results/raw/maps.jsonl")
    ap.add_argument("--out-md", default="results/tables/maps.md")
    args = ap.parse_args()

    maps = torch.load(args.path, map_location="cpu")
    recs = []
    for key, B in sorted(maps.items()):
        name, coord, alloc, b0 = key.split("|")
        B = B.to(torch.int16)
        real = analyse_map(B)
        null = analyse_map(shuffle_rows(B))
        frac = {int(s): float((B == s).float().mean()) for s in SYMS if (B == s).any()}
        recs.append(dict(matrix=name, coord=coord, alloc=alloc, base_bits=int(b0),
                         shape=list(B.shape), real=real, null=null, frac=frac))
        print("%-36s %-9s %-7s  marg %.3f  best %s=%.3f  (null best %.3f)"
              % (name, coord, alloc, real["marginal"],
                 min(real, key=real.get), min(real.values()), min(null.values())))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    # aggregate
    lines = ["# Description length of the oracle bit-allocation map", "",
             "Held-out cross-entropy in bits/weight: what it costs to *transmit* the map.",
             "`null` is the same map shuffled within rows (same marginal, no arrangement).", ""]
    models = list(recs[0]["real"].keys())
    for alloc in sorted({r["alloc"] for r in recs}):
        for coord in sorted({r["coord"] for r in recs}):
            sel = [r for r in recs if r["alloc"] == alloc and r["coord"] == coord]
            if not sel:
                continue
            lines += ["", "## alloc `%s`, coordinates `%s` (%d matrices)"
                      % (alloc, coord, len(sel)), "",
                      "| model | real bits/weight | null bits/weight | real - null |",
                      "|---|---|---|---|"]
            for m in models:
                a = sum(r["real"][m] for r in sel) / len(sel)
                b = sum(r["null"][m] for r in sel) / len(sel)
                lines.append("| %s | %.4f | %.4f | %+.4f |" % (m, a, b, a - b))
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    open(args.out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwrote %s and %s" % (args.out, args.out_md))


if __name__ == "__main__":
    main()
