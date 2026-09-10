"""End-to-end check: does the weight/activation error survive contact with the whole model?

Applies a codec to every target linear layer (streaming, one matrix at a time), writes the
reconstruction back in place, and measures wikitext-2 perplexity. `--upto-layer` compresses
only the first N decoder layers, which shows whether error accumulates catastrophically.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

from ..calib import act_scale_from_X, get_wikitext2
from ..codecs import build
from ..modelio import get_module, list_linear_layers, load_model, total_weights
from ..structure import collect_tiles


@torch.no_grad()
def perplexity(model, tok, device, seqlen=2048, n_seq=32, seed=0):
    ids = get_wikitext2(tok, seqlen=seqlen, n_seq=n_seq, split="test", seed=seed)
    nll, ntok = 0.0, 0
    for i in range(ids.shape[0]):
        x = ids[i:i + 1].to(device)
        out = model(x, labels=x)
        nll += float(out.loss) * (x.shape[1] - 1)
        ntok += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / ntok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--configs", default="configs/codecs.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--projs", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--upto-layer", type=int, default=-1,
                    help="compress only decoder layers < N (-1 = all)")
    ap.add_argument("--calib", default="cache/calib.pt")
    ap.add_argument("--act-scale", action="store_true")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--fit-tiles", type=int, default=200_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/ppl.jsonl")
    args = ap.parse_args()

    projs = args.projs.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    all_refs = list_linear_layers(model)
    refs = [r for r in all_refs if r.proj in projs
            and (args.upto_layer < 0 or r.layer_idx < args.upto_layer)]
    amortize_over = total_weights(all_refs)
    calib = torch.load(args.calib, map_location="cpu") if os.path.exists(args.calib) else {}

    orig = {r.name: get_module(model, r.name).weight.detach().clone().cpu() for r in refs}
    base_ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
    print(f"fp16 baseline perplexity: {base_ppl:.4f}  ({len(refs)} matrices to compress)")

    configs = json.load(open(args.configs, encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in wanted or args.only in c["name"]]

    fh = open(args.out, "a", encoding="utf-8")
    for cfg in configs:
        kw = {k: v for k, v in cfg.items() if k not in ("name", "codec", "fit_on")}
        t0 = time.time()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        codec = build(cfg["codec"], **kw)

        if getattr(codec, "needs_fit", False):
            fit_mats = []
            for r in refs[: min(len(refs), 8)]:
                W = orig[r.name].to(args.device, torch.float32)
                if cfg.get("fit_on") == "residual":
                    from ..codecs.scalar import quantize_groupwise
                    W = W - quantize_groupwise(W, kw.get("bits", 3), kw.get("group", 128))
                fit_mats.append(W)
            codec.fit([collect_tiles(fit_mats, kw.get("th", 8), kw.get("tw", 8),
                                     "real", args.seed, args.fit_tiles)])
            del fit_mats
            torch.cuda.empty_cache() if args.device == "cuda" else None

        total_bits, total_w = 0.0, 0
        for r in refs:
            W0 = orig[r.name].to(args.device, torch.float32)
            s = None
            if args.act_scale and r.name in calib:
                s = act_scale_from_X(calib[r.name].to(args.device)).unsqueeze(0)
            res = codec.compress_matrix(W0 * s if s is not None else W0)
            Wh = res.recon / s if s is not None else res.recon
            get_module(model, r.name).weight.data.copy_(Wh.to(torch.float16))
            total_bits += sum(res.bits.values())
            total_w += W0.numel()
            del W0, Wh, res
        bpw = total_bits / total_w + sum(codec.shared_bits().values()) / amortize_over

        ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
        dc = codec.decode_cost()
        rec = dict(name=cfg["name"], codec=cfg["codec"], config=kw,
                   act_scale=args.act_scale, upto_layer=args.upto_layer,
                   n_matrices=len(refs), bpw=bpw, ppl=ppl, base_ppl=base_ppl,
                   ppl_delta=ppl - base_ppl, encode_s=time.time() - t0,
                   peak_vram_mib=(torch.cuda.max_memory_allocated() / 2**20
                                  if args.device == "cuda" else 0.0),
                   decode_flops_per_weight=dc.flops_per_weight,
                   decode_verdict=dc.practical_verdict())
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"{cfg['name']:34s} bpw={bpw:6.3f}  ppl={ppl:9.4f}  "
              f"(fp16 {base_ppl:.4f}, delta {ppl - base_ppl:+.4f})  {time.time()-t0:5.1f}s")

        for r in refs:                                    # restore for the next config
            get_module(model, r.name).weight.data.copy_(orig[r.name].to(args.device))
        del codec
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
