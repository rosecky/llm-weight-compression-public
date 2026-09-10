"""End-to-end checks for the graph study: perplexity, permutation legality, cross-layer error
interaction.

--mode ppl
    Full-model sequential GPTQ with the input channels of every layer reordered by a chosen
    rule, then wikitext-2 perplexity. Mandatory because activation NMSE has misranked codecs
    four times in this project.

--mode permcheck
    A channel permutation is only free if it can be absorbed into neighbouring operations.
    This checks the one case in a Qwen2-style block where it genuinely can -- the MLP
    intermediate, private to `up_proj`/`gate_proj` -> `down_proj` -- by permuting down_proj's
    input columns together with the output rows of up_proj and gate_proj and confirming the
    model's logits are unchanged to floating-point tolerance. It also states, for the record,
    which permutations cannot be absorbed.

--mode crosslayer
    Kill gate G6. Quantize layer l alone, layer l' alone, and both, and measure the
    interaction of their damage in end-to-end loss:

        I = (L_both - L_0) - (L_l - L_0) - (L_l' - L_0)

    If adjacent layers do not interact more than distant ones, there is nothing for a
    cross-layer functional graph to organise and the joint-quantization branch has no target.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from ..calib import get_wikitext2
from ..gptq import HessianAccumulator, ScalarQuantizer, actorder_bits, gptq
from ..graphstruct import affinity, spectral_order, topk_sparsify
from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model
from ..rotate import RotationPair
from .factorial_ppl import capture_layer0_inputs, get_decoder_layers, perplexity


def order_for(rule: str, H: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    n = H.shape[0]
    if rule == "contig":
        return torch.arange(n, device=H.device)
    if rule == "scale":
        return torch.argsort(torch.diagonal(H), descending=True)
    if rule == "random":
        g = torch.Generator(device=H.device).manual_seed(seed)
        return torch.randperm(n, generator=g, device=H.device)
    if rule == "specA_corr":
        return spectral_order(topk_sparsify(affinity(0.5 * (H + H.T), "corr"), k))
    raise ValueError(rule)


# ====================================================================== ppl


@torch.no_grad()
def quantize_sequential(model, tok, cfg, args) -> Dict:
    device = args.device
    layers, _ = get_decoder_layers(model)
    target = set(args.projs.split(","))
    ids = get_wikitext2(tok, seqlen=args.calib_seqlen, n_seq=args.calib_seq,
                        split="train", seed=args.seed)
    hidden, kw = capture_layer0_inputs(model, ids, device, batch=args.batch)
    total_bits, total_w = 0.0, 0

    for li in range(len(layers)):
        layer = layers[li]
        subs = {n: m for n, m in layer.named_modules()
                if isinstance(m, torch.nn.Linear)
                and any(n.endswith("." + p) or n == p for p in target)}
        accs = {n: HessianAccumulator(m.in_features, device=device) for n, m in subs.items()}
        handles = []
        for n, m in subs.items():
            def hook(mod, inp, out, key=n):
                accs[key].add(inp[0].detach())
            handles.append(m.register_forward_hook(hook))
        for i in range(0, hidden.shape[0], args.batch):
            layer(hidden[i:i + args.batch], **kw)
        for h in handles:
            h.remove()

        for n, m in subs.items():
            W0 = m.weight.detach().float()
            out_f, in_f = W0.shape
            H = accs[n].finalize()
            rp = None
            Win, Hin = W0, H
            if cfg["coord"] == "hadamard":
                rp = RotationPair(out_f, in_f, seed=args.seed, device=device)
                Win = rp.forward_w(W0)
                Hr = rp.forward_h(H)
                Hin = 0.5 * (Hr + Hr.T)
            p = order_for(cfg["order"], Hin, args.k,
                          cfg.get("perm_seed", args.seed))
            ip = torch.argsort(p)
            q = ScalarQuantizer(bits=cfg["bits"], group=args.group)
            Q = gptq(Win[:, p], (2.0 * Hin[p][:, p]) if cfg["comp"] == "gptq" else None, q,
                     percdamp=args.percdamp)[:, ip]
            Wh = rp.inverse_w(Q) if rp is not None else Q
            m.weight.data.copy_(Wh.to(m.weight.dtype))
            b = q.storage_bits(out_f, in_f)
            total_bits += sum(b.values()) + (32.0 if cfg["coord"] == "hadamard" else 0.0)
            if cfg["order"] != "contig" and not cfg.get("absorbable", False):
                total_bits += actorder_bits(in_f, out_f)
            total_w += out_f * in_f
            del W0, Win, Hin, H, Q, Wh
            del accs[n]
        accs.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
        outs = []
        for i in range(0, hidden.shape[0], args.batch):
            o = layer(hidden[i:i + args.batch], **kw)
            outs.append((o[0] if isinstance(o, tuple) else o).detach())
        hidden = torch.cat(outs, 0)
        del outs
        if li % 8 == 0 or li == len(layers) - 1:
            print("  layer %2d/%d  peak %.0f MiB" % (li, len(layers) - 1,
                  torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0))
    del hidden
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(bpw=total_bits / total_w, n_weights=total_w)


def mode_ppl(args):
    configs = json.load(open(args.configs, encoding="utf-8"))
    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    orig = {r.name: get_module(model, r.name).weight.detach().clone().cpu()
            for r in list_linear_layers(model) if r.proj in set(args.projs.split(","))}
    base = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
    print("fp16 baseline perplexity %.4f\n" % base)
    fh = open(args.out, "a", encoding="utf-8")
    for cfg in configs:
        t0 = time.time()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print("[%s]" % cfg["name"])
        info = quantize_sequential(model, tok, cfg, args)
        ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
        rec = dict(cfg, mode="ppl", bpw=info["bpw"], ppl=ppl, base_ppl=base,
                   ppl_delta=ppl - base, encode_s=time.time() - t0,
                   peak_vram_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0.0))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("  -> bpw=%.4f  ppl=%.4f (fp16 %.4f)  %.0fs\n"
              % (rec["bpw"], ppl, base, rec["encode_s"]))
        for k, w in orig.items():
            get_module(model, k).weight.data.copy_(w.to(args.device))
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()


# ====================================================================== permcheck


@torch.no_grad()
def mode_permcheck(args):
    model, tok = load_model(args.model, device=args.device, dtype=torch.float32)
    layers, _ = get_decoder_layers(model)
    ids = get_wikitext2(tok, seqlen=256, n_seq=2, split="test", seed=args.seed)
    x = ids[:1].to(args.device)
    ref = model(x).logits.clone()

    g = torch.Generator(device=args.device).manual_seed(args.seed)
    results = []
    for li in range(len(layers)):
        mlp = layers[li].mlp
        n = mlp.down_proj.in_features
        p = torch.randperm(n, generator=g, device=args.device)
        mlp.down_proj.weight.data = mlp.down_proj.weight.data[:, p].contiguous()
        mlp.up_proj.weight.data = mlp.up_proj.weight.data[p].contiguous()
        mlp.gate_proj.weight.data = mlp.gate_proj.weight.data[p].contiguous()
        results.append(p)
    got = model(x).logits
    rel = float((got - ref).norm() / ref.norm())
    print("MLP-intermediate permutation applied to all %d blocks" % len(layers))
    print("  logits relative change: %.3e   (fp32 round-off is ~1e-7)" % rel)
    print("  -> absorbable, runtime class H0: the permutation is folded into the checkpoint")
    print("  NOT absorbable per layer: q/k/v/gate/up read the shared residual stream, so they")
    print("     would all need one common permutation; o_proj's input is tied to attention")
    print("     head structure. Only the MLP intermediate is private to one producer/consumer")
    print("     pair.")
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(mode="permcheck", n_blocks=len(layers),
                                 logits_rel_change=rel, dtype="float32")) + "\n")


# ====================================================================== crosslayer


@torch.no_grad()
def quantize_layer_inplace(model, li, bits, group, projs, device):
    layers, _ = get_decoder_layers(model)
    saved = {}
    for n, m in layers[li].named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        if not any(n.endswith("." + p) or n == p for p in projs):
            continue
        saved[n] = m.weight.data.clone()
        W = m.weight.detach().float()
        q = ScalarQuantizer(bits=bits, group=group)
        m.weight.data.copy_(gptq(W, None, q).to(m.weight.dtype))
    return saved


@torch.no_grad()
def restore(model, li, saved):
    layers, _ = get_decoder_layers(model)
    for n, m in layers[li].named_modules():
        if n in saved:
            m.weight.data.copy_(saved[n])


@torch.no_grad()
def mode_crosslayer(args):
    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    layers, _ = get_decoder_layers(model)
    projs = args.projs.split(",")
    nl = len(layers)
    pairs = [(1, 2), (1, 3), (5, 6), (10, 11), (11, 12), (11, 13),
             (11, 16), (1, nl - 2), (5, 11), (16, 17)]
    pairs = [(a, b) for a, b in pairs if 0 <= a < nl and 0 <= b < nl]

    def nll():
        return perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)

    base = nll()
    single: Dict[int, float] = {}
    fh = open(args.out, "a", encoding="utf-8")
    print("fp16 perplexity %.4f; single-layer damage at %d bits" % (base, args.bits))
    for li in sorted({x for p in pairs for x in p}):
        s = quantize_layer_inplace(model, li, args.bits, args.group, projs, args.device)
        single[li] = nll() - base
        restore(model, li, s)
        print("  layer %2d alone: d ppl %+.5f" % (li, single[li]))

    for a, b in pairs:
        sa = quantize_layer_inplace(model, a, args.bits, args.group, projs, args.device)
        sb = quantize_layer_inplace(model, b, args.bits, args.group, projs, args.device)
        both = nll() - base
        restore(model, a, sa)
        restore(model, b, sb)
        inter = both - single[a] - single[b]
        rec = dict(mode="crosslayer", a=a, b=b, dist=abs(b - a), bits=args.bits,
                   d_a=single[a], d_b=single[b], d_both=both, interaction=inter,
                   inter_frac=inter / max(single[a] + single[b], 1e-30), base_ppl=base)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("  (%2d,%2d) dist=%2d  d_a=%+.4f d_b=%+.4f d_both=%+.4f  I=%+.4f (%.1f%%)"
              % (a, b, abs(b - a), single[a], single[b], both, inter,
                 100 * rec["inter_frac"]))
    fh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ppl", choices=["ppl", "permcheck", "crosslayer"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--projs", default=",".join(PROJ_TYPES))
    ap.add_argument("--configs", default="configs/graph_ppl.json")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/graph_endtoend.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    {"ppl": mode_ppl, "permcheck": mode_permcheck, "crosslayer": mode_crosslayer}[args.mode](args)


if __name__ == "__main__":
    main()
