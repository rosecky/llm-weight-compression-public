"""Held-out calibration geometry: the same `A = E[x x^T]`, from different text.

Kill criterion K4 asks whether a joint solution is merely fitting the calibration sample. The
control is an input second moment collected from *disjoint* windows of the same corpus: an
optimizer tuned on `A_train` is scored on `A_val`, so a solution that only wins in-sample is
visible before any end-to-end run. No gradients are needed, so this is much cheaper than
`make_graphcal.py`.
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234,
                    help="different from make_graphcal's, so the windows are disjoint")
    ap.add_argument("--out", default="cache/jointcal_val.pt")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model, tok = load_model(args.model, device=args.device, dtype=torch.float32)
    refs = [r for r in list_linear_layers(model) if r.layer_idx in layers and r.proj in projs]
    ids = get_wikitext2(tok, seqlen=args.seqlen, n_seq=args.n_seq, split="train",
                        seed=args.seed)
    store = collect_AG(model, refs, ids, device=args.device, batch=1, need_grad=False)
    out = {n: {"A": (0.5 * (d["A"] + d["A"].T)).float()} for n, d in store.items()}
    torch.save(out, args.out)
    print("wrote %s: %d modules, %.0f MiB (seed %d)"
          % (args.out, len(out),
             sum(v["A"].numel() * 4 for v in out.values()) / 2 ** 20, args.seed))


if __name__ == "__main__":
    main()
