"""Frozen-protocol validation runs (P1-P3): quantize with the frozen pipeline, evaluate hard.

This module adds NO quantization decisions. It reuses `quantize_sequential_joint` exactly as
frozen and only extends the EVALUATION:

  * full contiguous wikitext-2 test at 2048-token windows (not a sampled subset),
  * 512 pre-fixed C4 validation windows (built once into cache/c4_eval.pt, then immutable),
  * an independent Monte-Carlo estimate of KL(P_fp16 || P_quant): before quantization the
    fp model samples k tokens per position on 64 fixed windows and stores their log-probs;
    after quantization the same tokens are scored under the quantized model. E_{y~P}[log P(y)
    - log Q(y)] is an unbiased KL estimate and needs megabytes, not the 40 GB of full logits.
  * per-window NLL written to jsonl so a paired bootstrap over windows is possible offline,
  * exact physical bpw as reported by the frozen pipeline (codes + scales + rotation seeds).

Primary metric everywhere: mean NLL per predicted token (log-perplexity). PPL is derived.

Arms are the frozen five: fp16, gptq, layer (gptq+cd), module, block. Calibration draws
differ only through --calib-seed; nothing else may vary between runs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from ..calib import get_wikitext2
from ..modelio import get_module, list_linear_layers, load_model
from .joint_ppl import quantize_sequential_joint

ARMS = {
    "gptq":   {"name": "gptq",   "coord": "hadamard", "comp": "gptq", "bits": 3, "cd": 0},
    "layer":  {"name": "layer",  "coord": "hadamard", "comp": "gptq", "bits": 3, "cd": 6},
    "module": {"name": "module", "coord": "hadamard", "comp": "gptq", "bits": 3, "cd": 6,
               "cd_pre": 6, "scope": "module", "gdamp": 1.0},
    "block":  {"name": "block",  "coord": "hadamard", "comp": "gptq", "bits": 3, "cd": 6,
               "cd_pre": 6, "scope": "block1", "gdamp": 1.0},
    # D2 diagnostic arms (see FROZEN_PROTOCOL.md addendum D2)
    "module_pre": {"name": "module_pre", "coord": "hadamard", "comp": "gptq", "bits": 3,
                   "cd": 6, "cd_pre": 6, "scope": "module_pre", "gdamp": 1.0},
    "module_bd":  {"name": "module_bd", "coord": "hadamard", "comp": "gptq", "bits": 3,
                   "cd": 6, "cd_pre": 6, "scope": "module_bd", "gdamp": 1.0},
    # D3 diagnostic arms (see FROZEN_PROTOCOL.md addendum D3): the two halves of the module
    # arm (attention projections only / MLP gate+up only) for the four-arm interaction
    # contrast, and the full-model Fisher horizon whose estimation budget is set by
    # --n-probe / --g-tokens (recorded in every row)
    "module_attn": {"name": "module_attn", "coord": "hadamard", "comp": "gptq", "bits": 3,
                    "cd": 6, "cd_pre": 6, "scope": "module_attn", "gdamp": 1.0},
    "module_mlp":  {"name": "module_mlp", "coord": "hadamard", "comp": "gptq", "bits": 3,
                    "cd": 6, "cd_pre": 6, "scope": "module", "gdamp": 1.0,
                    "gprojs": "gate_proj,up_proj"},
    "model":       {"name": "model", "coord": "hadamard", "comp": "gptq", "bits": 3,
                    "cd": 6, "cd_pre": 6, "scope": "model", "gdamp": 1.0},
}


def full_wikitext2_windows(tok, seqlen: int) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = ids.numel() // seqlen
    return ids[: n * seqlen].reshape(n, seqlen)


def c4_windows(tok, n_windows: int, seqlen: int, cache: str, seed: int = 17
               ) -> torch.Tensor:
    """Fixed C4 evaluation windows, built once and cached; afterwards immutable."""
    if os.path.exists(cache):
        return torch.load(cache, map_location="cpu")
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    g = torch.Generator().manual_seed(seed)
    buf: List[torch.Tensor] = []
    total = 0
    need = n_windows * seqlen + 1
    for doc in ds:
        t = tok(doc["text"], return_tensors="pt").input_ids[0]
        buf.append(t)
        total += t.numel()
        if total >= need * 1.02:
            break
    ids = torch.cat(buf)[:need]
    del buf
    W = ids[: n_windows * seqlen].reshape(n_windows, seqlen)
    torch.save(W, cache)
    return W


@torch.no_grad()
def window_nlls(model, W: torch.Tensor, device: str) -> List[float]:
    out = []
    for i in range(W.shape[0]):
        x = W[i: i + 1].to(device)
        r = model(x, labels=x)
        out.append(float(r.loss))
    return out


@torch.no_grad()
def kl_reference(model, W: torch.Tensor, device: str, k: int = 8, chunk: int = 256,
                 seed: int = 0):
    """Sample k tokens/position from the CURRENT (fp) model and store their log-probs."""
    g = torch.Generator(device=device).manual_seed(seed)
    toks, lps = [], []
    for i in range(W.shape[0]):
        x = W[i: i + 1].to(device)
        logits = model(x).logits[0].float()          # (T, V)
        tw, lw = [], []
        for j in range(0, logits.shape[0], chunk):
            lp = torch.log_softmax(logits[j: j + chunk], dim=-1)
            y = torch.multinomial(lp.exp(), k, replacement=True, generator=g)
            tw.append(y)
            lw.append(lp.gather(1, y))
        toks.append(torch.cat(tw).cpu())
        lps.append(torch.cat(lw).cpu())
        del logits
    return torch.stack(toks), torch.stack(lps)       # (n, T, k) each


@torch.no_grad()
def kl_estimate(model, W: torch.Tensor, toks: torch.Tensor, lps: torch.Tensor,
                device: str, chunk: int = 256) -> float:
    """E_{y~P_fp}[log P_fp(y) - log Q(y)] over the stored samples."""
    tot, n = 0.0, 0
    for i in range(W.shape[0]):
        x = W[i: i + 1].to(device)
        logits = model(x).logits[0].float()
        for j in range(0, logits.shape[0], chunk):
            lq = torch.log_softmax(logits[j: j + chunk], dim=-1)
            y = toks[i, j: j + chunk].to(device)
            diff = lps[i, j: j + chunk].to(device) - lq.gather(1, y)
            tot += float(diff.mean(1).sum())
            n += y.shape[0]
        del logits
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--arms", default="gptq,layer,module,block")
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--projs", default="q_proj,k_proj,v_proj,o_proj,"
                                       "gate_proj,up_proj,down_proj")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--bits", type=int, default=0,
                    help="override the arms' bit width (exact bpw = bits + 32/group); "
                         "0 keeps the frozen 3 bits")
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--c4-windows", type=int, default=512)
    ap.add_argument("--kl-windows", type=int, default=32, help="per corpus")
    ap.add_argument("--kl-samples", type=int, default=8)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--n-probe", type=int, default=2)
    ap.add_argument("--g-tokens", type=int, default=2048)
    ap.add_argument("--calib-source", default="wikitext2", choices=["wikitext2", "c4"],
                    help="calibration text source (addendum D3); wikitext2 is the frozen "
                         "default and is byte-identical to every earlier run")
    ap.add_argument("--fresh-g-seq", type=int, default=0,
                    help="extra calibration windows used ONLY to re-score the wider "
                         "objective on unseen text (no effect on any decision)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/valid_eval.jsonl")
    args = ap.parse_args()
    args.n_seq = 24                                   # unused by this eval; kept for reuse
    if args.bits:
        # rate sweep: the arms are otherwise untouched, so a point on the curve is the same
        # frozen method at a different exact bpw (bits + 32/group)
        for a in ARMS.values():
            a["bits"] = args.bits
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    dev = args.device

    model, tok = load_model(args.model, device=dev, dtype=torch.float16)
    orig = {r.name: get_module(model, r.name).weight.detach().clone().cpu()
            for r in list_linear_layers(model) if r.proj in set(args.projs.split(","))}

    Wt2 = full_wikitext2_windows(tok, args.seqlen)
    # cache keyed by the TOKENIZER (not the model): shared across sizes of one family,
    # distinct across families -- a Llama eval must never read Qwen token ids
    tokslug = "%s_v%d" % (tok.__class__.__name__, len(tok))
    Wc4 = c4_windows(tok, args.c4_windows, args.seqlen,
                     "cache/c4_eval_%s_%dx%d.pt" % (tokslug, args.c4_windows, args.seqlen))
    Wkl = torch.cat([Wt2[: args.kl_windows], Wc4[: args.kl_windows]])
    print("eval sets: wt2 %d windows, c4 %d windows, kl %d windows"
          % (Wt2.shape[0], Wc4.shape[0], Wkl.shape[0]))

    fh = open(args.out, "a", encoding="utf-8")
    tag = dict(model=args.model, calib_seed=args.calib_seed, seqlen=args.seqlen,
               calib_source=args.calib_source, n_probe=args.n_probe, g_tokens=args.g_tokens)

    # fp16 reference: NLLs + the KL sampling anchor (model currently holds fp weights)
    t0 = time.time()
    kl_toks, kl_lps = kl_reference(model, Wkl, dev, k=args.kl_samples, seed=args.seed)
    for name, W in (("wt2", Wt2), ("c4", Wc4)):
        nll = window_nlls(model, W, dev)
        fh.write(json.dumps(dict(tag, arm="fp16", eval=name, nlls=nll,
                                 nll=sum(nll) / len(nll))) + "\n")
        print("fp16 %-3s mean NLL %.5f  (ppl %.4f)  %ds"
              % (name, sum(nll) / len(nll),
                 torch.tensor(sum(nll) / len(nll)).exp(), time.time() - t0))
    fh.flush()

    for arm in args.arms.split(","):
        cfg = ARMS[arm]
        t0 = time.time()
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        info = quantize_sequential_joint(model, tok, cfg, args)
        q_s = time.time() - t0
        rec = dict(tag, arm=arm, bpw=info["bpw"], flips=info["flips"],
                   obj_ratio=info["obj_ratio"], obj_fresh=info.get("obj_fresh"),
                   obj_fresh_before=info.get("obj_fresh_before"),
                   obj_fresh_after=info.get("obj_fresh_after"),
                   quant_s=q_s, opt_s=info["opt_s"],
                   scope_s=info["scope_s"],
                   peak_vram_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                  if dev == "cuda" else 0.0))
        for name, W in (("wt2", Wt2), ("c4", Wc4)):
            nll = window_nlls(model, W, dev)
            fh.write(json.dumps(dict(tag, arm=arm, eval=name, nlls=nll,
                                     nll=sum(nll) / len(nll))) + "\n")
            rec["nll_" + name] = sum(nll) / len(nll)
        rec["kl_fp16"] = kl_estimate(model, Wkl, kl_toks, kl_lps, dev)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("%-6s bpw %.4f | NLL wt2 %.5f c4 %.5f | KL %.5f | %ds"
              % (arm, rec["bpw"], rec["nll_wt2"], rec["nll_c4"], rec["kl_fp16"], q_s))
        for kname, w in orig.items():
            get_module(model, kname).weight.data.copy_(w.to(dev))
        if dev == "cuda":
            torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    main()
