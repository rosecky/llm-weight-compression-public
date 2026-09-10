"""Sections 8 and 9: one rotation shared by everything that reads the same activation.

The per-matrix oracle answers "how much can a learned basis buy", but it answers it at a price
that makes the answer moot: `k` Householder vectors per axis per matrix is 0.29 bits/weight on a
896x896 matrix, which is far more than the gain is worth. A rotation on the *input* axis is
naturally shared, though: `q/k/v` all read the output of one layernorm and `gate/up` read
another, so one rotation per stream serves all of them, and a rotation shared across every block
-- which is what a deployed QuaRot/SpinQuant-style pipeline actually uses -- costs essentially
nothing per weight.

Sharing is a constraint, so the per-matrix result is an upper bound on what any shared rotation
can achieve in damage. What sharing changes is the denominator: the same gain, spread over many
more weights. This experiment measures how much of the per-matrix gain survives the constraint.

Scopes (`--share`):

    qkv      one input rotation for q, k, v of each layer
    gateup   one input rotation for gate and up of each layer
    stream   both of the above, learned per layer
    global   one rotation per stream shared across *all* selected layers

Only the input axis is shared -- an output-axis rotation is private to its matrix by
construction, so it is simply not used here, which also makes every arm deployable in the same
sense the Hadamard baseline is.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from ..joint import cd_refine, damage, quantize_with_state
from ..modelio import get_module, load_model
from ..rotate import RotationPair
from ..transforms import (ComposedTransform, HouseholderRotation, IdentityRotation,
                          grid_distance, interaction_decomposition, quant_error)

STREAMS = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn",
           "gate_proj": "mlp", "up_proj": "mlp"}


def group_modules(cal: Dict, layers: List[int], share: str) -> Dict[str, List[str]]:
    """Bucket the cached modules into the sets that will share one input rotation."""
    buckets: Dict[str, List[str]] = defaultdict(list)
    for n in sorted(cal):
        parts = n.split(".")
        li = int(parts[parts.index("layers") + 1])
        proj = parts[-1]
        if li not in layers or proj not in STREAMS:
            continue
        st = STREAMS[proj]
        if share == "qkv" and st != "attn":
            continue
        if share == "gateup" and st != "mlp":
            continue
        key = st if share == "global" else "%s.L%d" % (st, li)
        buckets[key].append(n)
    return dict(buckets)


@torch.no_grad()
def score_set(mats: List[Tuple[str, torch.Tensor, torch.Tensor]],
              fixed: Optional[RotationPair], right, args) -> Dict:
    """Quantize every matrix of the bucket under the shared basis and pool the damage.

    Damage is pooled as a ratio of sums, not a mean of ratios, so a small matrix cannot
    dominate the number that the parameter cost is being weighed against.
    """
    num = den = 0.0
    n_weights = 0
    grid, ggain = [], []
    for name, W0, A0 in mats:
        fx = fixed[name] if isinstance(fixed, dict) else fixed
        tf = ComposedTransform(fx, None, right)
        W, A = tf.forward_w(W0), tf.forward_h(A0)
        Wn, _ = quantize_with_state(W, None, args.bits, args.group, args.percdamp)
        Wg, st = quantize_with_state(W, 2.0 * A, args.bits, args.group, args.percdamp)
        Wc, _ = cd_refine(W, A, st, None, sweeps=args.cd_sweeps, seed=args.seed)
        d_rtn = damage(W0 - tf.inverse_w(Wn), A0)
        d_cd = damage(W0 - tf.inverse_w(Wc), A0)
        d_gptq = damage(W0 - tf.inverse_w(Wg), A0)
        num += d_cd
        den += damage(W0, A0)
        n_weights += W0.numel()
        grid.append(grid_distance(W, args.bits, args.group))
        ggain.append(d_rtn / max(d_gptq, 1e-30))
    return dict(rel_cd=num / max(den, 1e-30), n_weights=n_weights,
                grid=sum(grid) / len(grid), gptq_gain=sum(ggain) / len(ggain))


def learn_shared(mats, fixed, args, k: int, steps: int, outer: int, objective: str):
    """One input rotation, fitted against the pooled objective of the whole bucket."""
    in_f = mats[0][1].shape[1]
    dev = mats[0][1].device
    right = HouseholderRotation(in_f, k, dev, seed=args.seed + 11)
    opt = torch.optim.Adam(right.params(), lr=args.lr)
    for it in range(max(outer, 1)):
        codes = {}
        if objective == "O3":
            with torch.no_grad():
                for name, W0, A0 in mats:
                    fx = fixed[name] if isinstance(fixed, dict) else fixed
                    tf = ComposedTransform(fx, None, right)
                    W, A = tf.forward_w(W0), tf.forward_h(A0)
                    _, st = quantize_with_state(W, 2.0 * A, args.bits, args.group,
                                                args.percdamp)
                    cd_refine(W, A, st, None, sweeps=args.cd_sweeps, seed=args.seed)
                    codes[name] = st.codes.float().detach()
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for name, W0, A0 in mats:
                fx = fixed[name] if isinstance(fixed, dict) else fixed
                tf = ComposedTransform(fx, None, right)
                W, A = tf.forward_w(W0), tf.forward_h(A0)
                E, _ = quant_error(W, args.bits, args.group, codes.get(name))
                loss = loss + (((E @ A) * E).sum() if objective != "O1" else (E * E).sum())
            loss.backward()
            opt.step()
    return right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--share", default="gateup", choices=["qkv", "gateup", "stream", "global"])
    ap.add_argument("--objectives", default="O1,O3")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--cd-sweeps", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--outer", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/transform_shared.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cal = torch.load(args.cal, map_location="cpu")
    buckets = group_modules(cal, [int(x) for x in args.layers.split(",")], args.share)
    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    fh = open(args.out, "a", encoding="utf-8")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for key, names in buckets.items():
        mats = [(n, get_module(model, n).weight.detach().float().to(args.device),
                 cal[n]["A"].float().to(args.device)) for n in names]
        in_f = mats[0][1].shape[1]
        # every matrix keeps its own fixed Hadamard pair; only the learned part is shared
        fixed = {n: RotationPair(W.shape[0], W.shape[1], seed=args.seed, device=args.device)
                 for n, W, _ in mats}
        base = dict(share=args.share, bucket=key, n_mats=len(mats), inn=in_f,
                    bits=args.bits, group=args.group, k=args.k, steps=args.steps)
        print("  [%s] %d matrices, in=%d" % (key, len(mats), in_f))

        def emit(arm, right, extra=None, t0=None):
            s = score_set(mats, fixed, right, args)
            pb = (0.0 if right is None else args.k * in_f * 16.0) / s["n_weights"]
            rec = dict(base, arm=arm, **s, param_bits_per_weight=pb)
            rec.update(extra or {})
            if t0 is not None:
                rec["fit_s"] = time.time() - t0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("    %-16s rel_cd %.6f  grid %.4f  gptq x%.2f  +%.4f bpw"
                  % (arm, rec["rel_cd"], rec["grid"], rec["gptq_gain"], pb))
            return rec

        emit("hadamard", None)
        for s in range(args.rand_seeds):
            g = torch.Generator(device=args.device).manual_seed(5000 + s)
            r = HouseholderRotation(in_f, args.k, args.device, seed=5000 + s)
            with torch.no_grad():
                r.V += 0.7 * torch.randn(r.V.shape, generator=g, device=args.device)
            emit("random_k%d_s%d" % (args.k, s), r, dict(rand_seed=5000 + s))
        for obj in args.objectives.split(","):
            t0 = time.time()
            r = learn_shared(mats, fixed, args, args.k, args.steps, args.outer, obj)
            emit("learned_%s" % obj, r, dict(objective=obj), t0=t0)

        for _, W, A in mats:
            del W, A
        del mats, fixed
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("peak VRAM %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
