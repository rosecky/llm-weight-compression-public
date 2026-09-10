"""Is 0.00538 a real plateau of the hard objective, or an artefact of the fixed-code optimizer?

Two hypotheses have to be separated before any verdict on learned transforms:

    H-A  Hadamard plateau. After random orthogonal mixing the compensation geometry is already
         near-optimal and there is genuinely nowhere to go.
    H-B  Optimizer trap. Holding the quantization assignments fixed anchors the basis to
         whichever one produced them, so the alternating scheme cannot cross into a different
         discrete basin no matter how long it runs.

Everything in this file therefore scores **only the hard pipeline**, with codes recomputed from
scratch every single time:

    R  ->  fresh min-max scales  ->  GPTQ  ->  coordinate descent  ->  damage in native coords

No objective is ever evaluated with codes inherited from another basis. Modes:

`tangent`   random orthogonal perturbations `exp(eps A) H` for skew `A`, at several `eps`, each
            scored by the hard pipeline. If essentially every direction hurts and the best is
            inside the noise, Hadamard is a genuine local optimum of the hard objective.
`sweep`     a large best-of-N random-rotation sweep. The random distribution is tight, so what
            matters is its *extreme*: a learned point that merely matches best-of-200 is not a
            learned signal.
`basin`     many starting rotations, each scored before and after the fixed-code optimizer, with
            code mobility per outer iteration -- how many assignments move, how many come back,
            and whether the hard objective follows the surrogate at all.
`dfo`       derivative-free search directly on the hard objective over a low-dimensional random
            subspace of the tangent space. This is the oracle for "can *any* optimizer beat the
            plateau", with no gradient and no fixed codes anywhere.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from ..joint import cd_refine, damage, quantize_with_state
from ..modelio import get_module, load_model
from ..rotate import RotationPair
from ..transforms import (ComposedTransform, HaarRotation, HouseholderRotation,
                          IdentityRotation, StackedRotation, grid_distance,
                          interaction_decomposition, quant_error)


# ====================================================================== the hard objective


class ExpRotation(IdentityRotation):
    """`exp(eps A)` for a random skew-symmetric `A`, formed densely and exactly.

    This is the perturbation the diagnostic actually wants: a step of controlled size in a
    uniformly random direction of the *full* tangent space of the orthogonal group, not of the
    subspace some parameterisation happens to span.
    """

    def __init__(self, n: int, eps: float, device: str, seed: int):
        g = torch.Generator(device=device).manual_seed(seed)
        M = torch.randn(n, n, generator=g, device=device)
        A = (M - M.T) / math.sqrt(2.0 * n)          # unit-ish spectral scale
        self.Q = torch.matrix_exp(eps * A)
        self.n = n

    def apply(self, X):
        return X @ self.Q

    def apply_inv(self, X):
        return X @ self.Q.T

    apply_right = apply
    apply_right_inv = apply_inv

    def apply_left(self, W):
        return (W.T @ self.Q).T

    def apply_left_inv(self, W):
        return (W.T @ self.Q.T).T

    def rotate_hessian(self, H):
        return self.Q.T @ H @ self.Q


@torch.no_grad()
def hard_score(W0: torch.Tensor, A0: torch.Tensor, tf: ComposedTransform, args,
               full: bool = False) -> Dict:
    """The only objective this file trusts: fresh quantization, real GPTQ, real CD.

    Scored on `W - What` in native coordinates so every basis is measured in the same frame.
    """
    W = tf.forward_w(W0)
    A = tf.forward_h(A0)
    den = max(damage(W0, A0), 1e-30)
    Wg, st = quantize_with_state(W, 2.0 * A, args.bits, args.group, args.percdamp)
    st_cd = st.clone()
    Wc, _ = cd_refine(W, A, st_cd, None, sweeps=args.cd_sweeps, seed=args.seed)
    out = {"dam_cd": damage(W0 - tf.inverse_w(Wc), A0) / den}
    if full:
        Wn, _ = quantize_with_state(W, None, args.bits, args.group, args.percdamp)
        out["dam_rtn"] = damage(W0 - tf.inverse_w(Wn), A0) / den
        out["dam_gptq"] = damage(W0 - tf.inverse_w(Wg), A0) / den
        out["gptq_gain"] = out["dam_rtn"] / max(out["dam_gptq"], 1e-30)
        out["grid_dist"] = grid_distance(W, args.bits, args.group)
        out["cd_codes"] = st_cd.codes
    return out


# ====================================================================== mode: tangent


def mode_tangent(name, W0, A0, args, fh):
    """Does any small orthogonal step away from Hadamard improve the hard objective?"""
    dev = W0.device
    out_f, in_f = W0.shape
    rp = RotationPair(out_f, in_f, seed=args.seed, device=dev)
    base = hard_score(W0, A0, ComposedTransform(rp), args, full=True)
    d0 = base["dam_cd"]
    print("  [%s] Hadamard hard objective %.6f" % (name.split("layers.")[-1], d0))

    # repeat-evaluation noise floor: the pipeline is deterministic, so any spread here is
    # numerical, and it is the yardstick a "better direction" has to clear
    reps = [hard_score(W0, A0, ComposedTransform(
        RotationPair(out_f, in_f, seed=args.seed, device=dev)), args)["dam_cd"]
        for _ in range(3)]
    noise = max(reps) - min(reps)
    print("    determinism check: spread over 3 identical runs %.2e" % noise)

    for eps in [float(x) for x in args.eps.split(",")]:
        vals = []
        t0 = time.time()
        for i in range(args.dirs):
            L = ExpRotation(out_f, eps, dev, seed=7000 + 977 * i)
            R = ExpRotation(in_f, eps, dev, seed=9000 + 977 * i)
            s = hard_score(W0, A0, ComposedTransform(rp, L, R), args)["dam_cd"]
            vals.append(s)
            del L, R
        rel = [v / d0 - 1.0 for v in vals]
        rel.sort()
        n_better = sum(1 for r in rel if r < 0)
        rec = dict(module=name, mode="tangent", eps=eps, dirs=args.dirs, base=d0,
                   noise_spread=noise, frac_better=n_better / len(rel),
                   best=min(rel), p10=rel[len(rel) // 10], median=rel[len(rel) // 2],
                   worst=max(rel), mean=sum(rel) / len(rel), secs=time.time() - t0)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("    eps %-6g  better %3d/%-3d  best %+.3f%%  median %+.3f%%  worst %+.3f%%"
              % (eps, n_better, len(rel), 100 * rec["best"], 100 * rec["median"],
                 100 * rec["worst"]))


# ====================================================================== mode: sweep


def mode_sweep(name, W0, A0, args, fh):
    """Best-of-N over random rotations: what does the *extreme* of the random family reach?"""
    dev = W0.device
    out_f, in_f = W0.shape
    rp = RotationPair(out_f, in_f, seed=args.seed, device=dev)
    d0 = hard_score(W0, A0, ComposedTransform(rp), args)["dam_cd"]
    vals = []
    t0 = time.time()
    for i in range(args.n_random):
        kind = "had" if i % 2 == 0 else "haar"
        if kind == "had":
            tf = ComposedTransform(RotationPair(out_f, in_f, seed=20000 + i, device=dev))
        else:
            if max(out_f, in_f) > args.haar_max:
                continue
            tf = ComposedTransform(None, HaarRotation(out_f, dev, 30000 + i),
                                   HaarRotation(in_f, dev, 40000 + i))
        vals.append((hard_score(W0, A0, tf, args)["dam_cd"], kind))
        del tf
    v = sorted(x for x, _ in vals)
    rec = dict(module=name, mode="sweep", n=len(v), hadamard=d0, best=v[0],
               p01=v[max(0, len(v) // 100)], p10=v[len(v) // 10], median=v[len(v) // 2],
               p90=v[9 * len(v) // 10], worst=v[-1], secs=time.time() - t0)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print("  [%s] %d random rotations: best %.6f  median %.6f  worst %.6f | Hadamard %.6f"
          % (name.split("layers.")[-1], len(v), v[0], v[len(v) // 2], v[-1], d0))


# ====================================================================== mode: dfo


def subspace_transform(rp, V0L, V0R, NL, NR, c, k, dev):
    """`R(c)` for coefficients `c` over a fixed random subspace of the tangent directions."""
    left = HouseholderRotation(V0L.shape[1], k, dev, seed=0)
    right = HouseholderRotation(V0R.shape[1], k, dev, seed=0)
    m = c.numel() // 2
    left.V0, right.V0 = V0L, V0R
    with torch.no_grad():
        left.V = V0L + torch.einsum("i,ikn->kn", c[:m], NL)
        right.V = V0R + torch.einsum("i,ikn->kn", c[m:], NR)
    return ComposedTransform(rp, left, right)


def mode_dfo(name, W0, A0, args, fh):
    """Derivative-free hill climbing on the hard objective itself.

    No gradient, no surrogate, no inherited codes: every proposal is scored by running the whole
    quantization pipeline. If even this cannot beat Hadamard, the plateau is real; if it can, the
    fixed-code optimizer was the problem.
    """
    dev = W0.device
    out_f, in_f = W0.shape
    k, m = args.k, args.dfo_dims
    rp = RotationPair(out_f, in_f, seed=args.seed, device=dev)
    g = torch.Generator(device=dev).manual_seed(args.seed + 55)
    V0L = HouseholderRotation(out_f, k, dev, seed=args.seed + 2).V0
    V0R = HouseholderRotation(in_f, k, dev, seed=args.seed + 1).V0
    NL = torch.randn(m, k, out_f, generator=g, device=dev)
    NR = torch.randn(m, k, in_f, generator=g, device=dev)
    NL /= NL.norm(dim=(1, 2), keepdim=True)
    NR /= NR.norm(dim=(1, 2), keepdim=True)

    c = torch.zeros(2 * m, device=dev)
    best = hard_score(W0, A0, subspace_transform(rp, V0L, V0R, NL, NR, c, k, dev),
                      args)["dam_cd"]
    d0 = best
    step = args.dfo_step
    evals, accepted, traj = 1, 0, []
    t0 = time.time()
    while evals < args.dfo_budget:
        cand = c + step * torch.randn(2 * m, generator=g, device=dev)
        s = hard_score(W0, A0, subspace_transform(rp, V0L, V0R, NL, NR, cand, k, dev),
                       args)["dam_cd"]
        evals += 1
        if s < best:
            best, c, accepted = s, cand, accepted + 1
            step *= 1.15
            traj.append(dict(eval=evals, best=best))
        else:
            step *= 0.98
        if step < args.dfo_step * 1e-3:
            break
    rec = dict(module=name, mode="dfo", dims=2 * m, k=k, budget=args.dfo_budget,
               evals=evals, accepted=accepted, base=d0, best=best,
               rel_gain=1.0 - best / d0, final_step=step, traj=traj[-12:],
               secs=time.time() - t0)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print("  [%s] DFO over %d dims: %.6f -> %.6f  (%+.2f%%) after %d evals, %d accepted"
          % (name.split("layers.")[-1], 2 * m, d0, best, -100 * rec["rel_gain"], evals,
             accepted))


# ====================================================================== mode: basin


def mode_basin(name, W0, A0, args, fh):
    """Where do different starting rotations end up, and do the codes actually move?"""
    dev = W0.device
    out_f, in_f = W0.shape
    rp = RotationPair(out_f, in_f, seed=args.seed, device=dev)
    starts: List[Tuple[str, ComposedTransform]] = [("hadamard", ComposedTransform(rp))]
    for s in range(args.basin_seeds):
        starts.append(("had_seed%d" % s,
                       ComposedTransform(RotationPair(out_f, in_f, seed=60000 + s,
                                                      device=dev))))
    if max(out_f, in_f) <= args.haar_max:
        for s in range(args.basin_seeds):
            starts.append(("haar%d" % s, ComposedTransform(
                None, HaarRotation(out_f, dev, 61000 + s), HaarRotation(in_f, dev, 62000 + s))))
    starts.append(("identity", ComposedTransform(None)))
    for e in (0.05, 0.2):
        starts.append(("had_pert%g" % e, ComposedTransform(
            rp, ExpRotation(out_f, e, dev, 63000), ExpRotation(in_f, e, dev, 64000))))

    for tag, tf0 in starts:
        before = hard_score(W0, A0, tf0, args, full=True)
        codes0 = before.pop("cd_codes")
        # A fresh learnable correction stacked on whatever basis this start represents, so
        # every start is optimized by the identical machinery from its own position.
        left = StackedRotation(tf0.left, HouseholderRotation(out_f, args.k, dev,
                                                             seed=args.seed + 2))
        right = StackedRotation(tf0.right, HouseholderRotation(in_f, args.k, dev,
                                                               seed=args.seed + 1))
        tf = ComposedTransform(tf0.fixed, left, right)
        opt = torch.optim.Adam(tf.params(), lr=args.lr)
        prev = codes0
        moves = []
        for it in range(args.outer):
            with torch.no_grad():
                W, A = tf.forward_w(W0), tf.forward_h(A0)
                _, st = quantize_with_state(W, 2.0 * A, args.bits, args.group, args.percdamp)
                cd_refine(W, A, st, None, sweeps=args.cd_sweeps, seed=args.seed)
                codes = st.codes.float().detach()
                hard = damage(W0 - tf.inverse_w(st.dequant()), A0) / max(damage(W0, A0), 1e-30)
                moves.append(dict(outer=it, hard=hard,
                                  changed_vs_prev=float((st.codes != prev).float().mean()),
                                  changed_vs_init=float((st.codes != codes0).float().mean())))
                prev = st.codes.clone()
            for _ in range(args.steps):
                opt.zero_grad(set_to_none=True)
                W = tf.forward_w(W0)
                A = tf.forward_h(A0)
                E, _ = quant_error(W, args.bits, args.group, codes)
                loss = ((E @ A) * E).sum()
                loss.backward()
                opt.step()
        after = hard_score(W0, A0, tf, args, full=True)
        after.pop("cd_codes")
        rec = dict(module=name, mode="basin", start=tag,
                   before=before["dam_cd"], after=after["dam_cd"],
                   gptq_gain_before=before["gptq_gain"], gptq_gain_after=after["gptq_gain"],
                   grid_before=before["grid_dist"], grid_after=after["grid_dist"],
                   mobility=moves)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("    %-14s %.6f -> %.6f  codes moved/outer %s"
              % (tag, rec["before"], rec["after"],
                 " ".join("%.3f" % m["changed_vs_prev"] for m in moves[1:])))


# ====================================================================== driver


def pick_modules(cal, layers, projs):
    out = []
    for n in cal:
        p = n.split(".")
        if int(p[p.index("layers") + 1]) in layers and p[-1] in projs:
            out.append(n)
    return sorted(out, key=lambda n: (int(n.split("layers.")[1].split(".")[0]), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tangent",
                    choices=["tangent", "sweep", "basin", "dfo"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--layers", default="1,11")
    ap.add_argument("--projs", default="q_proj")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--cd-sweeps", type=int, default=4)
    ap.add_argument("--eps", default="0.01,0.03,0.1,0.3")
    ap.add_argument("--dirs", type=int, default=60)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--haar-max", type=int, default=1024)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--dfo-dims", type=int, default=16)
    ap.add_argument("--dfo-budget", type=int, default=400)
    ap.add_argument("--dfo-step", type=float, default=0.15)
    ap.add_argument("--basin-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/transform_plateau.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cal = torch.load(args.cal, map_location="cpu")
    names = pick_modules(cal, [int(x) for x in args.layers.split(",")], args.projs.split(","))
    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fh = open(args.out, "a", encoding="utf-8")
    fn = {"tangent": mode_tangent, "sweep": mode_sweep, "basin": mode_basin,
          "dfo": mode_dfo}[args.mode]
    for n in names:
        W = get_module(model, n).weight.detach().float().to(args.device)
        A = cal[n]["A"].float().to(args.device)
        fn(n, W, A, args, fh)
        del W, A
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("peak VRAM %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
