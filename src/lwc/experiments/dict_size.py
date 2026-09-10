"""Research question 3 -- how large does the shared dictionary have to be before the
marginal benefit stops?

Sweeps codebook size K at fixed tile size and one VQ stage, on real weights, with a held-out
split. Reports the distortion, the code rate, the amortised codebook rate, and the marginal
distortion improvement per doubling of K.

For an i.i.d. source, doubling K buys exactly 1/d bits of distortion reduction -- the same
1/d bits it costs in code rate. So a flat "marginal gain == marginal cost" curve is the
signature of no structure, and any dictionary size is equally (un)justified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from ..modelio import get_module, list_linear_layers, load_model, total_weights
from ..structure import collect_tiles, kmeans, residual_energy, _split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--tiles", default="1x8,2x8,8x8")
    ap.add_argument("--ks", default="16,64,256,1024,4096,16384,65536")
    ap.add_argument("--max-tiles", type=int, default=1_000_000)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/dict_size.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    tiles = [tuple(int(v) for v in t.lower().split("x")) for t in args.tiles.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)
    refs = [r for r in all_refs if r.layer_idx in layers and r.proj in projs]

    mats = []
    for r in refs:
        mats.append(get_module(model, r.name).weight.detach().to(args.device, torch.float32))
    print(f"{len(mats)} matrices; amortising over {amortize_over/1e6:.1f}M weights")

    fh = open(args.out, "w", encoding="utf-8")
    t0 = time.time()
    for (th, tw) in tiles:
        d = th * tw
        T = collect_tiles(mats, th, tw, "real", args.seed, args.max_tiles)
        Ttr, Tte = _split(T, 0.5, args.seed)
        prev = None
        for K in ks:
            if K > Ttr.shape[0] // 4:
                print(f"[skip] {th}x{tw} K={K}: only {Ttr.shape[0]} training tiles")
                continue
            chunk = max(1024, min(16384, 2 ** 26 // K))
            C = kmeans(Ttr, K, iters=args.iters, seed=args.seed, chunk=chunk)
            rho = residual_energy(Tte, C, chunk=chunk)
            code_bpw = math.log2(K) / d
            cb_bpw = K * d * 16 / amortize_over
            rec = dict(th=th, tw=tw, d=d, K=K, rho=rho,
                       code_bpw=code_bpw, codebook_bpw=cb_bpw,
                       total_bpw=code_bpw + cb_bpw,
                       codebook_kib=K * d * 2 / 1024,
                       bits_saved=0.5 * math.log2(1.0 / max(rho, 1e-12)),
                       n_tiles=int(T.shape[0]))
            if prev is not None:
                rec["marginal_bits_saved_per_doubling"] = (
                    rec["bits_saved"] - prev) / math.log2(K / prev_K)
                rec["marginal_cost_per_doubling"] = 1.0 / d
            prev, prev_K = rec["bits_saved"], K
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{th}x{tw} K={K:6d}  rho={rho:.4f}  code={code_bpw:.4f}  "
                  f"cb={cb_bpw:.4f} bpw ({rec['codebook_kib']:.0f} KiB)  "
                  f"saved={rec['bits_saved']:.4f}  "
                  f"marg={rec.get('marginal_bits_saved_per_doubling', float('nan')):.4f} "
                  f"vs cost {1.0/d:.4f}")
        del T, Ttr, Tte
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print(f"wrote {args.out}  elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
