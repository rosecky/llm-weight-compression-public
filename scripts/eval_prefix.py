"""Addendum D4: evaluate checkpoints with a fixed "\\n\\n" prefix on every window.

Outside the frozen protocol. Each window = prefix + the frozen window's first
(2048 - len(prefix)) tokens; the NLL is averaged over the predicted positions after the
prefix so it is comparable window-by-window with the frozen run. Writes valid_eval-style
rows (arm, eval, nlls) so the same paired bootstrap applies.

usage: python scripts/eval_prefix.py --ckpt DIR --arm NAME --out FILE [--prefix "\\n\\n"]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lwc.experiments.valid_eval import c4_windows, full_wikitext2_windows  # noqa: E402
from lwc.modelio import load_model  # noqa: E402


@torch.no_grad()
def window_nlls_prefixed(model, W, prefix_ids, device):
    out = []
    npx = prefix_ids.numel()
    for i in range(W.shape[0]):
        x = torch.cat([prefix_ids, W[i, : W.shape[1] - npx]]).unsqueeze(0).to(device)
        logits = model(x).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
        out.append(float(nll[npx - 1:].mean()))       # positions predicted after the prefix
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prefix", default="\n\n")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--c4-windows", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device
    model, tok = load_model(args.ckpt, device=dev, dtype=torch.float16)
    Wt2 = full_wikitext2_windows(tok, args.seqlen)
    tokslug = "%s_v%d" % (tok.__class__.__name__, len(tok))
    Wc4 = c4_windows(tok, args.c4_windows, args.seqlen,
                     "cache/c4_eval_%s_%dx%d.pt" % (tokslug, args.c4_windows, args.seqlen))
    pid = tok(args.prefix, return_tensors="pt").input_ids[0]
    print("prefix tokens:", pid.tolist())
    fh = open(args.out, "a", encoding="utf-8")
    tag = dict(model=args.base, calib_seed=0, seqlen=args.seqlen, ckpt=args.ckpt,
               prefix=args.prefix)
    for name, W in (("wt2", Wt2), ("c4", Wc4)):
        nll = window_nlls_prefixed(model, W, pid, dev)
        fh.write(json.dumps(dict(tag, arm=args.arm, eval=name, nlls=nll,
                                 nll=sum(nll) / len(nll))) + "\n")
        print("%s %s prefixed mean NLL %.5f" % (args.arm, name, sum(nll) / len(nll)))
    fh.close()


if __name__ == "__main__":
    main()
