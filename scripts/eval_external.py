"""Evaluate an externally quantised (dequantised fp16) checkpoint under the frozen protocol.

Same evaluation as `lwc.experiments.valid_eval` -- full contiguous wikitext-2 test at 2048
tokens, the 512 pre-fixed C4 validation windows, MC-KL to the fp16 base model (8 samples
per position on 64 windows) -- but with no quantisation decisions: the weights are loaded as
they come. Writes rows in the valid_eval jsonl format (per-window NLLs, then a summary row)
so `analyze_valid.py`-style paired bootstraps against the frozen gptq/module rows of the
same base model work unchanged. `bpw` is taken from meta.json if present (peer's VQ
exporter) or from --bpw.

usage: python scripts/eval_external.py --base Qwen/Qwen2.5-0.5B \
           --ckpt models/x --arm vq20_plain --out results/raw/ext_qwen05_vq20.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lwc.experiments.valid_eval import (c4_windows, full_wikitext2_windows, kl_estimate,  # noqa: E402
                                        kl_reference, window_nlls)
from lwc.modelio import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B", help="fp16 reference (KL anchor)")
    ap.add_argument("--ckpt", required=True, help="directory with the dequantised checkpoint")
    ap.add_argument("--arm", required=True, help="row label, e.g. vq20_plain / vq20_refine")
    ap.add_argument("--bpw", type=float, default=0.0, help="override if no meta.json")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--c4-windows", type=int, default=512)
    ap.add_argument("--kl-windows", type=int, default=32)
    ap.add_argument("--kl-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    bpw = args.bpw
    meta_path = os.path.join(args.ckpt, "meta.json")
    meta = None
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        for k in ("bpw", "total_bpw", "bits_per_weight"):
            if k in meta:
                bpw = float(meta[k])
                break

    # fp16 reference: eval sets and the KL anchor (identical to valid_eval)
    model, tok = load_model(args.base, device=dev, dtype=torch.float16)
    Wt2 = full_wikitext2_windows(tok, args.seqlen)
    tokslug = "%s_v%d" % (tok.__class__.__name__, len(tok))
    Wc4 = c4_windows(tok, args.c4_windows, args.seqlen,
                     "cache/c4_eval_%s_%dx%d.pt" % (tokslug, args.c4_windows, args.seqlen))
    Wkl = torch.cat([Wt2[: args.kl_windows], Wc4[: args.kl_windows]])
    print("eval sets: wt2 %d windows, c4 %d windows, kl %d windows"
          % (Wt2.shape[0], Wc4.shape[0], Wkl.shape[0]))
    fh = open(args.out, "a", encoding="utf-8")
    tag = dict(model=args.base, calib_seed=0, seqlen=args.seqlen, ckpt=args.ckpt)
    t0 = time.time()
    kl_toks, kl_lps = kl_reference(model, Wkl, dev, k=args.kl_samples, seed=args.seed)
    for name, W in (("wt2", Wt2), ("c4", Wc4)):
        nll = window_nlls(model, W, dev)
        fh.write(json.dumps(dict(tag, arm="fp16", eval=name, nlls=nll,
                                 nll=sum(nll) / len(nll))) + "\n")
        print("fp16 %-3s mean NLL %.5f  %ds" % (name, sum(nll) / len(nll), time.time() - t0))
    fh.flush()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    # the external checkpoint, loaded as-is
    model, _ = load_model(args.ckpt, device=dev, dtype=torch.float16)
    rec = dict(tag, arm=args.arm, bpw=bpw, meta=meta, flips={}, obj_ratio=None,
               quant_s=0.0, opt_s=0.0, scope_s=0.0, peak_vram_mib=0.0)
    for name, W in (("wt2", Wt2), ("c4", Wc4)):
        nll = window_nlls(model, W, dev)
        fh.write(json.dumps(dict(tag, arm=args.arm, eval=name, nlls=nll,
                                 nll=sum(nll) / len(nll))) + "\n")
        rec["nll_" + name] = sum(nll) / len(nll)
    rec["kl_fp16"] = kl_estimate(model, Wkl, kl_toks, kl_lps, dev)
    fh.write(json.dumps(rec) + "\n")
    fh.close()
    print("%-12s bpw %.4f | NLL wt2 %.5f c4 %.5f | KL %.5f"
          % (args.arm, bpw, rec["nll_wt2"], rec["nll_c4"], rec["kl_fp16"]))


if __name__ == "__main__":
    main()
