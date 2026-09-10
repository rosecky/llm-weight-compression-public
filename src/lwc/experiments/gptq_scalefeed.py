"""Why does GPTQ lose to plain rounding at b4/g256 on `down_proj` -- and is it fixable?

The reproduced failure: `11.mlp.down_proj`, native coordinates, 4 bits, group 256: GPTQ ends 7%
*worse* than RTN, and more damping makes it worse still. Two candidate mechanisms, separated
here by construction rather than argued about:

M1  **truncated group fit** (implementation interaction). The canonical engine calls
    `find_params` on `W1[:, j:g_end]` where `W1` is the current *block* (width 128), so with
    `group > blocksize` every group's scale is fitted on its first 128 columns only. The
    original GPTQ slices the full working matrix instead. A min-max scale fitted on half the
    group clips whatever the other half holds beyond that range -- and this codec's entire
    design contract is that the largest weight in a group is never clipped. This predicts
    exactly the observed signature: failure only at g256 (the one ladder rung with
    group > blocksize), ranges *shrinking* (fit over 128 instead of 256 columns), native-only
    (heavy-tailed rows make the two half-ranges differ; after Hadamard they match).

M2  **scale feedback from compensation** (the sequential-contamination hypothesis). Group
    scales are fitted on the *compensated working* weights, which drift away from the original
    ones as error is redistributed, so later groups get grids matched to a moving target.

The arms hold everything fixed except where the scale comes from:

    rtn      naive rounding, full-group scales                       (baseline)
    block    canonical engine, scale from first min(group, block) columns of working W
             -- verified bit-identical to `gptq()` before anything else runs
    full     scale from the FULL group of the working W (original GPTQ semantics)
             -- fixes M1, keeps M2
    bs=g     canonical semantics with blocksize raised to the group size
             -- fixes M1 differently, keeps M2
    frozen   scale precomputed from the ORIGINAL W and never updated
             -- fixes both M1 and M2

If `full` recovers RTN-or-better, M1 is the mechanism. If only `frozen` does, M2 is real. If
`frozen` beats `full` by a margin on top, both contribute. Storage is identical in every arm:
same codes, same per-group fp16 (scale, zero).

Group-order ablation (`--orders`): the failing family re-run with the groups visited in
reverse and in a random order (columns permuted at group granularity, A permuted to match,
result unpermuted). Order dependence is evidence for sequential feedback; indifference is
evidence against.

Per-group diagnostics for the failing configuration: the used scale vs the original full-group
scale, the clip rate in each half of the group, each half's share of the (diagonal-weighted)
damage, and the original half-range ratio that M1 says should predict all of it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import torch

from ..gptq import ScalarQuantizer, cholesky_inverse_upper, gptq
from ..joint import damage
from ..modelio import get_module, load_model
from ..rotate import RotationPair


def group_minmax(W: torch.Tensor, group: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-group min-max (scale, zero) per (row, group), from the given matrix as-is."""
    out_f, in_f = W.shape
    ng = (in_f + group - 1) // group
    scale = torch.zeros(out_f, ng, dtype=torch.float32)
    zero = torch.zeros(out_f, ng, dtype=torch.float32)
    for gi in range(ng):
        Wg = W[:, gi * group:(gi + 1) * group]
        lo, hi = Wg.amin(1), Wg.amax(1)
        scale[:, gi] = ((hi - lo) / qmax).clamp_min(1e-12)
        zero[:, gi] = lo
    return scale, zero


def gptq_variant(W: torch.Tensor, H: torch.Tensor, bits: int, group: int,
                 percdamp: float = 0.01, blocksize: int = 128, scale_mode: str = "block",
                 frozen: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                 ) -> Tuple[torch.Tensor, Dict]:
    """The canonical GPTQ loop with the scale source made explicit. `scale_mode='block'`
    reproduces `gptq()` exactly (asserted by the caller on the first run)."""
    out_f, in_f = W.shape
    W = W.clone().float()
    Q = torch.zeros_like(W)
    qmax = 2 ** bits - 1
    ng = (in_f + group - 1) // group
    used_scale = torch.zeros(out_f, ng)
    clip = torch.zeros(ng, 2)
    total = torch.zeros(ng, 2)

    Hinv, dead = cholesky_inverse_upper(H, percdamp)
    W[:, dead] = 0.0
    scale = zero = None

    for i1 in range(0, in_f, blocksize):
        i2 = min(i1 + blocksize, in_f)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        width = i2 - i1
        for j in range(width):
            col = i1 + j
            if col % group == 0:
                gi = col // group
                if scale_mode == "frozen":
                    scale = frozen[0][:, gi:gi + 1].clone()
                    zero = frozen[1][:, gi:gi + 1].clone()
                else:
                    if scale_mode == "block":
                        Wg = W1[:, j:min(j + group, width)]
                    else:                       # full: cross the block boundary, working W
                        if col + group > i2:
                            Wg = torch.cat([W1[:, j:width], W[:, i2:col + group]], 1)
                        else:
                            Wg = W1[:, j:j + group]
                    lo = Wg.amin(1, keepdim=True)
                    hi = Wg.amax(1, keepdim=True)
                    scale = ((hi - lo) / qmax).clamp_min(1e-12)
                    zero = lo
                used_scale[:, gi] = scale[:, 0]
            w = W1[:, j:j + 1]
            qi = torch.round((w - zero) / scale)
            gi = col // group
            half = 0 if (col % group) < group // 2 else 1
            clip[gi, half] += float(((qi < 0) | (qi > qmax)).sum())
            total[gi, half] += out_f
            q = qi.clamp(0, qmax) * scale + zero
            Q1[:, j:j + 1] = q
            err = (w[:, 0] - q[:, 0]) / Hinv1[j, j]
            E1[:, j] = err
            if j + 1 < width:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
        Q[:, i1:i2] = Q1
        if i2 < in_f:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    return Q, dict(used_scale=used_scale, clip=clip, total=total)


def rtn(W: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    qmax = 2 ** bits - 1
    scale, zero = group_minmax(W, group, qmax)
    s = scale.repeat_interleave(group, 1)[:, :W.shape[1]]
    z = zero.repeat_interleave(group, 1)[:, :W.shape[1]]
    return torch.round((W - z) / s).clamp(0, qmax) * s + z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--valcal", default="cache/jointcal_val.pt")
    ap.add_argument("--modules", default="1.mlp.down_proj,11.mlp.down_proj,22.mlp.down_proj,"
                                         "1.mlp.up_proj,11.mlp.up_proj,22.mlp.up_proj")
    ap.add_argument("--rates", default="4:256,3:256,2:256,4:128,3:128")
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--orders", default="reverse,random",
                    help="group-order ablation, applied to the b4/g256 family only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/gptq_scalefeed.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

    cal = torch.load(args.cal, map_location="cpu")
    val = torch.load(args.valcal, map_location="cpu") if os.path.exists(args.valcal) else {}
    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    fh = open(args.out, "a", encoding="utf-8")
    verified = False

    for short in args.modules.split(","):
        name = "model.layers." + short
        if name not in cal:
            print("no calibration for %s, skipped" % short)
            continue
        W0 = get_module(model, name).weight.detach().float().clone()
        A0 = 0.5 * (cal[name]["A"].float() + cal[name]["A"].float().T)
        Av = val.get(name)
        Av = 0.5 * (Av["A"].float() + Av["A"].float().T) if Av is not None else None
        out_f, in_f = W0.shape
        den = max(damage(W0, A0), 1e-30)
        den_v = max(damage(W0, Av), 1e-30) if Av is not None else 1.0

        for coord in args.coords.split(","):
            if coord == "hadamard":
                rp = RotationPair(out_f, in_f, seed=args.seed, device="cpu")
                W = rp.forward_w(W0)
                A = 0.5 * (rp.forward_h(A0) + rp.forward_h(A0).T)
                back = rp.inverse_w
            else:
                rp, W, A, back = None, W0, A0, (lambda X: X)
            H = 2.0 * A

            for spec in args.rates.split(","):
                bits, group = (int(x) for x in spec.split(":"))
                if in_f % group:
                    continue
                qmax = 2 ** bits - 1
                frozen = group_minmax(W, group, qmax)

                arms = {}
                t0 = time.time()
                arms["rtn"] = (rtn(W, bits, group), None)
                arms["block"] = gptq_variant(W, H, bits, group, args.percdamp,
                                             scale_mode="block")
                if not verified:
                    ref = gptq(W, H, ScalarQuantizer(bits=bits, group=group),
                               percdamp=args.percdamp)
                    dmax = float((arms["block"][0] - ref).abs().max())
                    print("engine check: |variant - canonical gptq| = %.3e" % dmax)
                    assert dmax == 0.0, "scale_mode='block' must reproduce gptq() exactly"
                    verified = True
                    del ref
                arms["full"] = gptq_variant(W, H, bits, group, args.percdamp,
                                            scale_mode="full")
                arms["bs=g"] = gptq_variant(W, H, bits, group, args.percdamp,
                                            blocksize=max(128, group), scale_mode="block")
                arms["frozen"] = gptq_variant(W, H, bits, group, args.percdamp,
                                              scale_mode="frozen", frozen=frozen)

                rec = dict(module=short, coord=coord, bits=bits, group=group,
                           out=out_f, inn=in_f, percdamp=args.percdamp,
                           secs=time.time() - t0)
                line = []
                for arm, (Q, aux) in arms.items():
                    D = W0 - back(Q)
                    d = damage(D, A0) / den
                    rec[arm] = d
                    if Av is not None:
                        rec[arm + "_val"] = damage(D, Av) / den_v
                    if aux is not None:
                        rec[arm + "_clip"] = float(aux["clip"].sum()
                                                   / aux["total"].sum().clamp_min(1))
                    line.append("%s %.5f%s" % (arm, d,
                                (" c%.4f" % rec[arm + "_clip"])
                                if arm + "_clip" in rec else ""))
                    del D

                # per-group forensics on the canonical arm
                Qb, aux = arms["block"]
                gi_scale = aux["used_scale"]
                ratio = (gi_scale / frozen[0].clamp_min(1e-12)).mean(0)
                h = group // 2
                ng = in_f // group
                r1 = W[:, :].reshape(out_f, ng, group)
                rng1 = (r1[:, :, :h].amax(2) - r1[:, :, :h].amin(2))
                rng2 = (r1[:, :, h:].amax(2) - r1[:, :, h:].amin(2))
                halfrange = (rng2 / rng1.clamp_min(1e-12)).median(0).values
                Dt = W - Qb
                adiag = torch.diagonal(A)
                e_col = adiag * (Dt * Dt).mean(0)
                e_half = e_col.reshape(ng, group)
                rec.update(
                    scale_ratio_mean=float(ratio.mean()),
                    scale_ratio_min=float(ratio.min()),
                    halfrange_median=float(halfrange.median()),
                    clip_half1=float(aux["clip"][:, 0].sum()
                                     / aux["total"][:, 0].sum().clamp_min(1)),
                    clip_half2=float(aux["clip"][:, 1].sum()
                                     / aux["total"][:, 1].sum().clamp_min(1)),
                    err_share_half2=float(e_half[:, h:].sum() / e_half.sum().clamp_min(1e-30)))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print("%-18s %-8s b%d g%-3d | %s | scale~%.3f halfrng~%.2f "
                      "clip %.4f/%.4f err2 %.2f | %.0fs"
                      % (short, coord, bits, group, "  ".join(line),
                         rec["scale_ratio_mean"], rec["halfrange_median"],
                         rec["clip_half1"], rec["clip_half2"],
                         rec["err_share_half2"], rec["secs"]))
                del arms, Qb, aux, Dt

                # order ablation, failing family only
                if args.orders and group > 128 and bits == 4:
                    g = torch.Generator().manual_seed(args.seed)
                    for order in args.orders.split(","):
                        gp = (torch.arange(ng - 1, -1, -1) if order == "reverse"
                              else torch.randperm(ng, generator=g))
                        idx = (gp.unsqueeze(1) * group
                               + torch.arange(group).unsqueeze(0)).flatten()
                        inv = torch.argsort(idx)
                        Qp, _ = gptq_variant(W[:, idx], H[idx][:, idx], bits, group,
                                             args.percdamp, scale_mode="block")
                        d = damage(W0 - back(Qp[:, inv]), A0) / den
                        orec = dict(module=short, coord=coord, bits=bits, group=group,
                                    arm="order:" + order, d=d)
                        fh.write(json.dumps(orec) + "\n")
                        print("    order %-8s %.5f" % (order, d))
                        del Qp
            del W, A, H
    fh.close()


if __name__ == "__main__":
    main()
