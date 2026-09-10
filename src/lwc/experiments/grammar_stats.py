"""Is there a real GRAMMAR, or an expensive codebook in disguise?

For a fitted recursive codec we measure, on real weights:

* transform usage histogram and its entropy (is a few transforms doing all the work?)
* bigram / trigram entropy of transform sequences vs the fixed-width path rate
  (the gap is what entropy coding could recover, and a large gap means the fixed-width
  path encoding is paying for structure it does not have)
* number of distinct programs actually used vs the reachable set
* reuse factor: how many tiles share a root, a codeword, a subsequence
* cross-layer and cross-projection sharing of the path distribution
  (total-variation distance between path histograms -- near 0 means one global grammar serves
  every layer, which is good for amortisation but also means the grammar is not layer-specific)

Kill criterion R3 fires if nearly every tile needs its own program.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import torch

from ..codecs import build
from ..modelio import get_module, list_linear_layers, load_model
from ..structure import collect_tiles
from ..tiles import to_tiles


def entropy_bits(counts: torch.Tensor) -> float:
    p = counts[counts > 0].float()
    p = p / p.sum()
    return float(-(p * p.log2()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--th", type=int, default=1)
    ap.add_argument("--tw", type=int, default=8)
    ap.add_argument("--roots", type=int, default=256)
    ap.add_argument("--transforms", type=int, default=16)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--family", default="signperm")
    ap.add_argument("--fit-tiles", type=int, default=200_000)
    ap.add_argument("--probe-tiles", type=int, default=200_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/grammar.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    depths = [int(x) for x in args.depths.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    by = {(r.layer_idx, r.proj): r for r in list_linear_layers(model)}

    groups = {}
    for proj in projs:
        for li in layers:
            r = by.get((li, proj))
            if r is None:
                continue
            W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
            groups[(proj, li)] = W

    fit_mats = list(groups.values())[:8]
    fh = open(args.out, "w", encoding="utf-8")
    for depth in depths:
        codec = build("recursive", th=args.th, tw=args.tw, n_roots=args.roots,
                      n_transforms=args.transforms, depth=depth, family=args.family,
                      seed=args.seed, device=args.device)
        codec.fit([collect_tiles(fit_mats, args.th, args.tw, "real", args.seed,
                                 args.fit_tiles)])

        per_group_hist = {}
        all_tf = torch.zeros(args.transforms, device=args.device)
        all_paths = Counter()
        n_tiles_total = 0
        for key, W in groups.items():
            T = to_tiles(W, args.th, args.tw).float()
            if T.shape[0] > args.probe_tiles:
                g = torch.Generator(device=T.device).manual_seed(args.seed)
                T = T[torch.randperm(T.shape[0], generator=g, device=T.device)
                      [:args.probe_tiles]]
            Tn = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
            _, p = codec._encode_beam(Tn)
            n_tiles_total += p.shape[0]
            if depth:
                tf = p[:, 1:].reshape(-1)
                all_tf += torch.bincount(tf, minlength=args.transforms).float()
                key_ids = torch.zeros(p.shape[0], dtype=torch.long, device=p.device)
                for lvl in range(1, depth + 1):
                    key_ids = key_ids * args.transforms + p[:, lvl]
                hist = torch.bincount(key_ids, minlength=args.transforms ** depth).float()
                per_group_hist[key] = hist / hist.sum().clamp_min(1)
            full = p[:, 0] * (args.transforms ** depth)
            if depth:
                for lvl in range(1, depth + 1):
                    full = full + p[:, lvl] * (args.transforms ** (depth - lvl))
            for v, c in zip(*torch.unique(full, return_counts=True)):
                all_paths[int(v)] += int(c)
            del T, Tn, p

        counts = torch.tensor(list(all_paths.values()), dtype=torch.float,
                              device=args.device)
        n_reach = args.roots * (args.transforms ** depth)
        rec = dict(family=args.family, th=args.th, tw=args.tw, depth=depth,
                   R=args.roots, K=args.transforms,
                   n_tiles=n_tiles_total, n_reachable=n_reach,
                   n_programs_used=len(all_paths),
                   frac_reachable_used=len(all_paths) / n_reach,
                   fixed_width_bits=math.log2(args.roots) + depth * math.log2(args.transforms),
                   program_entropy_bits=entropy_bits(counts),
                   transform_entropy_bits=entropy_bits(all_tf) if depth else 0.0,
                   transform_uniform_bits=math.log2(args.transforms) if depth else 0.0,
                   transform_max_frac=float(all_tf.max() / all_tf.sum()) if depth else 0.0,
                   reuse_mean=float(counts.mean()), reuse_median=float(counts.median()),
                   reuse_p10=float(counts.quantile(0.10)),
                   frac_programs_used_once=float((counts == 1).float().mean()))
        # cross-group sharing of the path distribution
        if depth and len(per_group_hist) > 1:
            keys = list(per_group_hist)
            tvs, tvs_same_proj = [], []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    tv = float(0.5 * (per_group_hist[keys[i]] -
                                      per_group_hist[keys[j]]).abs().sum())
                    tvs.append(tv)
                    if keys[i][0] == keys[j][0]:
                        tvs_same_proj.append(tv)
            rec["path_tv_mean_all_pairs"] = sum(tvs) / len(tvs)
            if tvs_same_proj:
                rec["path_tv_mean_same_proj"] = sum(tvs_same_proj) / len(tvs_same_proj)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"depth={depth}  programs_used={rec['n_programs_used']}/{n_reach} "
              f"({rec['frac_reachable_used']:.4f})  fixed_width={rec['fixed_width_bits']:.1f}b "
              f"program_entropy={rec['program_entropy_bits']:.2f}b  "
              f"transform_entropy={rec['transform_entropy_bits']:.3f}/"
              f"{rec['transform_uniform_bits']:.1f}b  "
              f"reuse_median={rec['reuse_median']:.0f}  "
              f"tv_across_groups={rec.get('path_tv_mean_all_pairs', float('nan')):.4f}")
        del codec
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
