"""Stage 2 of the final diagnostic: full-model sequential GPTQ and end-to-end perplexity.

Stage 1 measures each layer against calibration activations captured from the *unquantized*
model. That is the right control for a per-layer ablation but it understates GPTQ, whose real
strength is sequential: each layer is compensated against activations that already carry the
error of every quantized layer before it.

So this stage does it properly:

    buffer <- embeddings(calibration ids)
    for each decoder layer:
        capture H for its linear sublayers from the CURRENT buffer
        quantize the layer in place
        buffer <- layer(buffer)          # now produced by the QUANTIZED layer
    perplexity(model)

The layer keyword arguments (attention mask, rotary position embeddings, ...) are captured
from a real forward pass rather than reconstructed, so this stays correct across transformers
versions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from ..calib import get_wikitext2
from ..gptq import HessianAccumulator, ScalarQuantizer, VQQuantizer, actorder_bits, gptq
from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model, total_weights
from ..rotate import RotationPair
from ..structure import collect_tiles, kmeans
from ..tiles import to_tiles


def get_decoder_layers(model):
    for path in ("model.layers", "model.decoder.layers", "transformer.h"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj, path
        except AttributeError:
            continue
    raise RuntimeError("could not locate the decoder layer list")


class _Catcher(torch.nn.Module):
    """Intercepts the first decoder layer to capture its inputs and keyword arguments."""

    def __init__(self, module, store):
        super().__init__()
        self.module = module
        self.store = store

    def forward(self, hidden_states, **kwargs):
        self.store["hidden"].append(hidden_states.detach())
        if "kwargs" not in self.store:
            self.store["kwargs"] = {k: v for k, v in kwargs.items()}
        raise RuntimeError("_catcher_stop")


@torch.no_grad()
def capture_layer0_inputs(model, input_ids, device, batch=1):
    layers, _ = get_decoder_layers(model)
    store = {"hidden": []}
    layers[0] = _Catcher(layers[0], store)
    for i in range(0, input_ids.shape[0], batch):
        try:
            model(input_ids[i:i + batch].to(device))
        except RuntimeError as e:
            if "_catcher_stop" not in str(e):
                raise
    layers[0] = layers[0].module
    return torch.cat(store["hidden"], 0), store.get("kwargs", {})


def build_vq_quantizer(fit_mats, d, K, stages, seed, fit_tiles=150_000):
    T = collect_tiles(fit_mats, 1, d, "real", seed, fit_tiles)
    books, R = [], T
    for s in range(stages):
        C = kmeans(R, K, iters=20, seed=seed + s)
        books.append(C)
        Cn = (C * C).sum(1)
        step = int(max(512, min(8192, (1 << 24) // max(K, 1))))
        newR = torch.empty_like(R)
        for i in range(0, R.shape[0], step):
            rc = R[i:i + step]
            a = (Cn.unsqueeze(0) - 2.0 * (rc @ C.T)).argmin(1)
            newR[i:i + step] = rc - C[a]
        R = newR
    return VQQuantizer(books)


@torch.no_grad()
def perplexity(model, tok, device, seqlen=2048, n_seq=24, seed=0):
    ids = get_wikitext2(tok, seqlen=seqlen, n_seq=n_seq, split="test", seed=seed)
    nll, ntok = 0.0, 0
    for i in range(ids.shape[0]):
        x = ids[i:i + 1].to(device)
        out = model(x, labels=x)
        nll += float(out.loss) * (x.shape[1] - 1)
        ntok += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / ntok)))


@torch.no_grad()
def quantize_model_sequential(model, tok, cfg, args) -> Dict:
    """Quantize every target linear, propagating calibration activations layer by layer."""
    device = args.device
    layers, _ = get_decoder_layers(model)
    all_refs = list_linear_layers(model)
    amortize_over = total_weights(all_refs)
    target_projs = set(args.projs.split(","))

    ids = get_wikitext2(tok, seqlen=args.calib_seqlen, n_seq=args.calib_seq,
                        split="train", seed=args.seed)
    hidden, layer_kwargs = capture_layer0_inputs(model, ids, device, batch=args.batch)
    print(f"  calibration buffer {tuple(hidden.shape)} "
          f"({hidden.numel()*hidden.element_size()/2**20:.0f} MiB)")

    # shared VQ codebook, fit once in the coordinate system the quantizer will see
    quant_shared_bits = 0.0
    vq_books = None
    if cfg["rep"] == "vq":
        fit_mats = []
        for r in all_refs:
            if r.proj in target_projs and r.layer_idx in (1, 5, 11, 17, 21):
                W = get_module(model, r.name).weight.detach().float()
                if cfg["coord"] == "hadamard":
                    W = RotationPair(*W.shape, seed=args.seed, device=device).forward_w(W)
                fit_mats.append(W)
        vq_books = build_vq_quantizer(fit_mats, cfg["d"], cfg["K"], cfg["stages"], args.seed)
        quant_shared_bits = vq_books.shared_bits()
        del fit_mats
        torch.cuda.empty_cache() if device == "cuda" else None

    total_bits, total_w = 0.0, 0
    for li in range(len(layers)):
        layer = layers[li]
        subs = {n: m for n, m in layer.named_modules()
                if isinstance(m, torch.nn.Linear)
                and any(n.endswith("." + p) or n == p for p in target_projs)}
        accs: Dict[str, HessianAccumulator] = {}
        handles = []
        if cfg["comp"] == "gptq":
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
            rp = None
            Win = W0
            H = accs[n].finalize() if n in accs else None
            if cfg["coord"] == "hadamard":
                rp = RotationPair(*W0.shape, seed=args.seed, device=device)
                Win = rp.forward_w(W0)
                if H is not None:
                    H = rp.forward_h(H)
            if cfg["rep"] == "scalar":
                q = ScalarQuantizer(bits=cfg["bits"], group=cfg["group"])
            else:
                q = VQQuantizer(vq_books.codebooks)
            Q = gptq(Win, H, q, percdamp=args.percdamp, actorder=args.actorder)
            Wh = rp.inverse_w(Q) if rp is not None else Q
            m.weight.data.copy_(Wh.to(m.weight.dtype))
            b = q.storage_bits(*W0.shape)
            total_bits += sum(b.values()) + (32.0 if cfg["coord"] == "hadamard" else 0.0)
            if args.actorder and cfg["comp"] == "gptq":
                total_bits += actorder_bits(W0.shape[1], W0.shape[0])
            total_w += W0.numel()
            del W0, Win, Q, Wh, H
            if n in accs:
                del accs[n]
        accs.clear()
        if device == "cuda":
            torch.cuda.empty_cache()

        # propagate the buffer through the now-quantized layer
        outs = []
        for i in range(0, hidden.shape[0], args.batch):
            o = layer(hidden[i:i + args.batch], **layer_kwargs)
            outs.append((o[0] if isinstance(o, tuple) else o).detach())
        hidden = torch.cat(outs, 0)
        del outs
        if li % 6 == 0 or li == len(layers) - 1:
            print(f"  layer {li:2d}/{len(layers)-1} done"
                  + (f", peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB"
                     if device == "cuda" else ""))

    del hidden
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(bpw=total_bits / total_w + quant_shared_bits / amortize_over,
                n_weights=total_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--projs", default=",".join(PROJ_TYPES))
    ap.add_argument("--configs", default="configs/factorial_ppl.json")
    ap.add_argument("--only", default=None)
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--actorder", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/factorial_ppl.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    configs = json.load(open(args.configs, encoding="utf-8"))
    if args.only:
        configs = [c for c in configs if args.only in c["name"]]

    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    orig = {r.name: get_module(model, r.name).weight.detach().clone().cpu()
            for r in list_linear_layers(model) if r.proj in set(args.projs.split(","))}
    base_ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
    print(f"fp16 baseline perplexity {base_ppl:.4f}, {len(orig)} matrices to quantize\n")

    fh = open(args.out, "a", encoding="utf-8")
    for cfg in configs:
        t0 = time.time()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print(f"[{cfg['name']}]")
        info = quantize_model_sequential(model, tok, cfg, args)
        ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
        rec = dict(cfg, bpw=info["bpw"], ppl=ppl, base_ppl=base_ppl,
                   ppl_delta=ppl - base_ppl, encode_s=time.time() - t0,
                   actorder=args.actorder,
                   calib=f"{args.calib_seq}x{args.calib_seqlen}",
                   peak_vram_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0.0))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(f"  -> bpw={rec['bpw']:.3f}  ppl={ppl:.4f}  (fp16 {base_ppl:.4f}, "
              f"delta {rec['ppl_delta']:+.4f})  {rec['encode_s']:.0f}s  "
              f"{rec['peak_vram_mib']:.0f} MiB\n")
        for r_name, w in orig.items():                      # restore for the next config
            get_module(model, r_name).weight.data.copy_(w.to(args.device))
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
