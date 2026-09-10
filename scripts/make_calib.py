"""Cache calibration activations for the layers used in the rate-distortion sweep.

Stores, per target module, a subsampled matrix X of shape (in_features, n_samples) in fp16.
That is enough for the exact activation-space error and for the AWQ-style column scale.
"""
from __future__ import annotations

import argparse
import os

import torch

from lwc.calib import collect_layer_inputs, get_wikitext2
from lwc.modelio import list_linear_layers, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="1,5,11,17,22")
    ap.add_argument("--projs", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--max-samples", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/calib.pt")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    refs = [r for r in list_linear_layers(model) if r.layer_idx in layers and r.proj in projs]
    print(f"hooking {len(refs)} modules")

    ids = get_wikitext2(tok, seqlen=args.seqlen, n_seq=args.n_seq, split="train", seed=args.seed)
    store = collect_layer_inputs(model, refs, ids, device=args.device,
                                 max_samples=args.max_samples, seed=args.seed)
    total = sum(v.numel() * 2 for v in store.values()) / 2**20
    torch.save(store, args.out)
    print(f"wrote {args.out}: {len(store)} modules, {total:.0f} MiB")
    for k, v in list(store.items())[:4]:
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
