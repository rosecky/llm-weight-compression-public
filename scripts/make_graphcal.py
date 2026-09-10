"""Cache the input and output geometry of selected linear layers.

Stores, per module, `A = E[x x^T]` (in, in) and `G = E[g g^T]` (out, out),
symmetrised and by default in fp32. `G` comes
from the model's own next-token loss, so it is the empirical Fisher of the layer output --
the object that makes output-channel interactions well defined.
"""
from __future__ import annotations

import argparse
import os

import torch

from lwc.calib import get_wikitext2
from lwc.gradcov import collect_AG
from lwc.modelio import list_linear_layers, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store-dtype", default="float32",
                    choices=["float32", "float16"],
                    help="fp16 halves the file but can break the positive-"
                         "definiteness the Cholesky in gptq relies on")
    ap.add_argument("--out", default="cache/graphcal.pt")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16

    model, tok = load_model(args.model, device=args.device, dtype=dt)
    refs = [r for r in list_linear_layers(model) if r.layer_idx in layers and r.proj in projs]
    print("%d modules, dtype %s" % (len(refs), args.dtype))
    ids = get_wikitext2(tok, seqlen=args.seqlen, n_seq=args.n_seq, split="train",
                        seed=args.seed)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    store = collect_AG(model, refs, ids, device=args.device, batch=1)
    sdt = torch.float32 if args.store_dtype == "float32" else torch.float16
    out = {n: {k: (0.5 * (v + v.T)).to(sdt) for k, v in d.items()}
           for n, d in store.items()}
    torch.save(out, args.out)
    mib = sum(v.numel() * v.element_size() for d in out.values()
              for v in d.values()) / 2 ** 20
    print("wrote %s: %d modules, %.0f MiB, peak VRAM %.0f MiB"
          % (args.out, len(out), mib,
             torch.cuda.max_memory_allocated() / 2 ** 20 if args.device == "cuda" else 0))
    for n, d in list(out.items())[:4]:
        print("  %-38s A%s G%s" % (n, tuple(d["A"].shape), tuple(d["G"].shape)))


if __name__ == "__main__":
    main()
