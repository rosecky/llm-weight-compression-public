"""End-to-end check of adaptive bit layouts: full-model sequential GPTQ, then perplexity.

Same sequential machinery as `factorial_ppl` -- each layer is compensated against activations
that already carry every upstream layer's quantization error -- with one change: the scalar
quantizer's bit width per weight comes from an oracle allocation computed on that layer, in
whatever coordinate system the quantizer is working in.

`alloc = uniform` is the baseline. `alloc = weight` is the free-layout upper bound and is
*not* a deployable codec; it is here to show what the ceiling looks like in perplexity rather
than in activation NMSE, because this project has repeatedly seen activation NMSE misrank
codecs (E11, E15).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import torch

from ..calib import get_wikitext2
from ..gptq import HessianAccumulator, ScalarQuantizer, gptq
from ..mixed import MixedScalarQuantizer, oracle_bits_map, per_weight_distortion
from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model, total_weights
from ..rotate import RotationPair
from .factorial_ppl import capture_layer0_inputs, get_decoder_layers, perplexity
from .oracle_alloc import quantize_mixed_naive


@torch.no_grad()
def quantize_model_sequential(model, cfg, args) -> Dict:
    device = args.device
    layers, _ = get_decoder_layers(model)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)
    target_projs = set(args.projs.split(","))

    ids = get_wikitext2(model.dummy_tok, seqlen=args.calib_seqlen, n_seq=args.calib_seq,
                        split="train", seed=args.seed)
    hidden, layer_kwargs = capture_layer0_inputs(model, ids, device, batch=args.batch)

    total_bits, total_w = 0.0, 0
    hist = torch.zeros(17, dtype=torch.float64)
    for li in range(len(layers)):
        layer = layers[li]
        subs = {n: m for n, m in layer.named_modules()
                if isinstance(m, torch.nn.Linear)
                and any(n.endswith("." + p) or n == p for p in target_projs)}
        accs: Dict[str, HessianAccumulator] = {}
        handles = []
        for n, m in subs.items():
            accs[n] = HessianAccumulator(m.in_features, device=device)

            def hook(mod, inp, out, key=n):
                accs[key].add(inp[0].detach())
            handles.append(m.register_forward_hook(hook))
        for i in range(0, hidden.shape[0], args.batch):
            layer(hidden[i:i + args.batch], **layer_kwargs)
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
                Hin = rp.forward_h(H)

            if cfg["alloc"] == "uniform" or in_f % args.group or out_f % args.group:
                bits_w = torch.full((out_f, in_f), cfg["bits"], dtype=torch.int16,
                                    device=device)
            else:
                colw = torch.diagonal(Hin).clamp_min(0.0).clone()
                D_w = per_weight_distortion(Win, colw, args.group)
                bits_w = oracle_bits_map(D_w, (out_f, in_f), cfg["alloc"], args.group,
                                         float(out_f) * in_f * cfg["bits"],
                                         tile_rows=args.group)
                del D_w
            quant = MixedScalarQuantizer(bits_w, group=args.group)
            if cfg["comp"] == "gptq":
                Q = gptq(Win, Hin, quant, percdamp=args.percdamp)
            else:
                Q = quantize_mixed_naive(Win, bits_w, args.group)
            Wh = rp.inverse_w(Q) if rp is not None else Q
            m.weight.data.copy_(Wh.to(m.weight.dtype))

            b = quant.storage_bits(out_f, in_f)
            total_bits += sum(b.values()) + (32.0 if cfg["coord"] == "hadamard" else 0.0)
            total_w += out_f * in_f
            hist += torch.bincount(bits_w.reshape(-1).long(), minlength=17).double().cpu()
            del W0, Win, Q, Wh, H, Hin, bits_w
            del accs[n]
        accs.clear()
        if device == "cuda":
            torch.cuda.empty_cache()

        outs = []
        for i in range(0, hidden.shape[0], args.batch):
            o = layer(hidden[i:i + args.batch], **layer_kwargs)
            outs.append((o[0] if isinstance(o, tuple) else o).detach())
        hidden = torch.cat(outs, 0)
        del outs
        if li % 8 == 0 or li == len(layers) - 1:
            print("  layer %2d/%d, peak VRAM %.0f MiB"
                  % (li, len(layers) - 1,
                     torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0))

    del hidden
    if device == "cuda":
        torch.cuda.empty_cache()
    hist = hist / hist.sum().clamp_min(1)
    return dict(bpw=total_bits / total_w, n_weights=total_w,
                hist={str(i): float(hist[i]) for i in range(17) if hist[i] > 1e-6})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--projs", default=",".join(PROJ_TYPES))
    ap.add_argument("--configs", default="configs/layout_ppl.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/layout_ppl.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    configs = json.load(open(args.configs, encoding="utf-8"))
    if args.only:
        configs = [c for c in configs if args.only in c["name"]]

    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    model.dummy_tok = tok
    orig = {r.name: get_module(model, r.name).weight.detach().clone().cpu()
            for r in list_linear_layers(model) if r.proj in set(args.projs.split(","))}
    base_ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
    print("fp16 baseline perplexity %.4f, %d matrices\n" % (base_ppl, len(orig)))

    fh = open(args.out, "a", encoding="utf-8")
    for cfg in configs:
        t0 = time.time()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print("[%s]" % cfg["name"])
        info = quantize_model_sequential(model, cfg, args)
        ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
        rec = dict(cfg, bpw=info["bpw"], ppl=ppl, base_ppl=base_ppl,
                   ppl_delta=ppl - base_ppl, hist=info["hist"],
                   encode_s=time.time() - t0, group=args.group,
                   peak_vram_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0.0))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("  -> bpw=%.3f  ppl=%.4f (fp16 %.4f)  %.0fs  %.0f MiB\n"
              % (rec["bpw"], ppl, base_ppl, rec["encode_s"], rec["peak_vram_mib"]))
        for r_name, w in orig.items():
            get_module(model, r_name).weight.data.copy_(w.to(args.device))
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("appended to %s" % args.out)


if __name__ == "__main__":
    main()
