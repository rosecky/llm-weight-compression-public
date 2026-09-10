"""Phase 2/3 -- rate-distortion sweep over baselines and procedural codecs.

For every codec config we report, on the same set of matrices:

    bits/weight      full accounting: codes + codebooks + decoder + scales + residuals + meta
    weight NMSE      ||W - What||^2 / ||W||^2
    activation NMSE  ||(W - What) X||^2 / ||W X||^2 on cached calibration activations
    encode time      wall clock, and peak VRAM
    decode cost      FLOPs/weight, bytes/weight, shared state, practicality verdict

Shared state (codebooks, decoders) is amortised over ALL target weights in the model, not just
the evaluated subset -- that is the deployment number and it is the one that decides whether a
dictionary is affordable.

Everything is streamed matrix by matrix; peak memory does not grow with model size.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from ..bits import BitBudget
from ..calib import act_scale_from_X
from ..codecs import build
from ..metrics import act_nmse_from_X, nmse
from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model, total_weights
from ..tiles import to_tiles


def sample_tiles_for_fit(mats, th, tw, max_tiles, seed):
    from ..structure import collect_tiles
    return [collect_tiles(mats, th, tw, "real", seed, max_tiles)]


def load_configs(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--configs", default="configs/codecs.json")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--calib", default="cache/calib.pt")
    ap.add_argument("--act-scale", action="store_true",
                    help="AWQ-style: scale columns by activation RMS before compressing")
    ap.add_argument("--fit-tiles", type=int, default=200_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="substring filter on config name")
    ap.add_argument("--out", default="results/raw/rd.jsonl")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)          # every target weight in the model
    refs = [r for r in all_refs if r.layer_idx in layers and r.proj in projs]
    print(f"evaluating {len(refs)} matrices; amortising shared state over "
          f"{amortize_over/1e6:.1f}M target weights")

    calib = torch.load(args.calib, map_location="cpu") if os.path.exists(args.calib) else {}
    if not calib:
        print(f"[warn] no calibration file at {args.calib}; activation error will be NaN. "
              f"Run scripts/make_calib.py first.")

    configs = load_configs(args.configs)
    if args.only:
        configs = [c for c in configs if args.only in c["name"]]

    fh = open(args.out, "a", encoding="utf-8")
    for cfg in configs:
        name, codec_kind = cfg["name"], cfg["codec"]
        kw = {k: v for k, v in cfg.items() if k not in ("name", "codec", "fit_on")}
        t0 = time.time()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        codec = build(codec_kind, **kw)

        # ---- shared-state fitting (streamed sample, bounded memory) ----
        if getattr(codec, "needs_fit", False):
            fit_mats = []
            for r in refs[: min(len(refs), 8)]:
                W = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
                if args.act_scale and r.name in calib:
                    W = W * act_scale_from_X(calib[r.name].to(args.device)).unsqueeze(0)
                if cfg.get("fit_on") == "residual":
                    from ..codecs.scalar import quantize_groupwise
                    W = W - quantize_groupwise(W, kw.get("bits", 3), kw.get("group", 128))
                fit_mats.append(W)
            codec.fit(sample_tiles_for_fit(fit_mats, kw.get("th", 8), kw.get("tw", 8),
                                           args.fit_tiles, args.seed))
            del fit_mats
            if args.device == "cuda":
                torch.cuda.empty_cache()

        # ---- encode every evaluated matrix, streaming ----
        budget = BitBudget(n_weights=0)
        w_num = w_den = 0.0
        a_num = a_den = 0.0
        n_with_act = 0
        for r in refs:
            W0 = get_module(model, r.name).weight.detach().to(args.device, torch.float32)
            s = None
            if args.act_scale and r.name in calib:
                s = act_scale_from_X(calib[r.name].to(args.device)).unsqueeze(0)
            W = W0 * s if s is not None else W0
            res = codec.compress_matrix(W)
            Wh = res.recon / s if s is not None else res.recon
            for k, v in res.bits.items():
                budget.add_per_matrix(k, v)
            budget.n_weights += W0.numel()
            w_num += float((W0 - Wh).pow(2).sum())
            w_den += float(W0.pow(2).sum())
            if r.name in calib:
                X = calib[r.name].to(args.device)
                dW = (W0 - Wh)
                a_num += float((dW @ X.float()).pow(2).sum())
                a_den += float((W0 @ X.float()).pow(2).sum())
                n_with_act += 1
                del X
            del W0, W, Wh, res
        # shared state amortised over the whole model, then rescaled to this subset's budget
        shared_total = sum(codec.shared_bits().values())
        shared_bpw = shared_total / amortize_over
        per_matrix_bpw = sum(budget.per_matrix.values()) / budget.n_weights
        bpw = per_matrix_bpw + shared_bpw

        dc = codec.decode_cost()
        rec = dict(
            name=name, codec=codec_kind, config=kw, act_scale=args.act_scale,
            n_matrices=len(refs), n_weights=budget.n_weights,
            amortize_over=amortize_over,
            bpw=bpw, bpw_per_matrix=per_matrix_bpw, bpw_shared=shared_bpw,
            bits_breakdown={k: v / budget.n_weights for k, v in budget.per_matrix.items()},
            weight_nmse=w_num / max(w_den, 1e-30),
            act_nmse=(a_num / max(a_den, 1e-30)) if n_with_act else float("nan"),
            n_with_act=n_with_act,
            encode_s=time.time() - t0,
            peak_vram_mib=(torch.cuda.max_memory_allocated() / 2**20
                           if args.device == "cuda" else 0.0),
            decode_flops_per_weight=dc.flops_per_weight,
            decode_bytes_per_weight=dc.bytes_streamed_per_weight,
            decode_shared_state_kib=dc.shared_state_bytes / 1024,
            decode_random_lookups=dc.random_lookups_per_weight,
            decode_verdict=dc.practical_verdict(),
            decode_notes=dc.notes,
        )
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"{name:34s} bpw={bpw:6.3f}  wNMSE={rec['weight_nmse']:.5f}  "
              f"aNMSE={rec['act_nmse']:.5f}  {rec['encode_s']:5.1f}s  {dc.practical_verdict()}")
        del codec
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
