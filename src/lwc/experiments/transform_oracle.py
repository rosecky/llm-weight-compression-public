"""Kill gates TR-K1 and TR-K3: is there a rotation oracle gap above Hadamard at all?

The branch is gated on one question, asked on real matrices with real calibration geometry and
with an optimizer allowed to be far more expensive than anything deployable:

    at identical bits, group size and quantizer, and after the *same* GPTQ + coordinate-descent
    pipeline, can a learned orthogonal change of basis beat a randomized Hadamard?

Arms (section 1 of the brief), every one of them followed by identical GPTQ and identical CD:

    T0  native coordinates
    T1  randomized Hadamard -- the baseline, in exactly the setup every earlier study used
    T2  random orthogonal controls: more Hadamard seeds, and true Haar rotations where the
        dimension allows one to be formed
    T3  learned rotation on the weight-to-grid objective          (O1)
    T4  learned rotation on the local quantized output error      (O2)
    T5  learned rotation through the real compensation pipeline   (O3, alternating)

T3-T5 are initialised *at* the Hadamard, so T1 is literally their starting point and any gap is
attributable to learning rather than to a different transform family. A second set is initialised
at the identity, which asks the separate question of whether learning rediscovers Hadamard.

Section 5 is the point of the diagnostics: the interesting outcome is not a transform that puts
weights closer to the grid, it is one whose rounding errors are *more compensatable*. So every
arm reports grid distance, pre-compensation damage, post-GPTQ damage, post-CD damage, the two
gain factors those imply, and the independent-versus-joint damage decomposition.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch

from ..joint import cd_refine, damage, quantize_with_state
from ..metrics import nmse
from ..modelio import get_module, load_model
from ..rotate import RotationPair, roundtrip_error
from ..transforms import (ComposedTransform, HaarRotation, HouseholderRotation,
                          IdentityRotation, expand_groups, grid_distance,
                          group_scales, interaction_decomposition, quant_error)


def pick_modules(cal: Dict, layers: List[int], projs: List[str]) -> List[str]:
    out = []
    for n in cal:
        parts = n.split(".")
        li = int(parts[parts.index("layers") + 1])
        if li in layers and parts[-1] in projs:
            out.append(n)
    return sorted(out, key=lambda n: (int(n.split("layers.")[1].split(".")[0]), n))


@torch.no_grad()
def evaluate(W0: torch.Tensor, A0: torch.Tensor, tf: ComposedTransform, args,
             cd_sweeps: int) -> Dict:
    """Run the identical pipeline under a given basis and score it in native coordinates.

    Every number below is computed on `W - What` in *native* coordinates, so arms in different
    bases are directly comparable and no basis can flatter itself by being scored in its own
    frame.
    """
    W = tf.forward_w(W0)
    A = tf.forward_h(A0)
    H = 2.0 * A
    den = max(damage(W0, A0), 1e-30)

    Wn, st_n = quantize_with_state(W, None, args.bits, args.group, args.percdamp)
    Wg, st_g = quantize_with_state(W, H, args.bits, args.group, args.percdamp)
    st = st_g.clone()
    Wc, info = cd_refine(W, A, st, None, sweeps=cd_sweeps, seed=args.seed)

    out = {}
    for tag, Wq in (("rtn", Wn), ("gptq", Wg), ("cd", Wc)):
        Wh0 = tf.inverse_w(Wq)
        D = W0 - Wh0
        out["dam_" + tag] = damage(D, A0) / den
        out["wnmse_" + tag] = nmse(W0, Wh0)
        if tag != "rtn":
            out.update({("%s_%s" % (tag, k)): v
                        for k, v in interaction_decomposition(D, A0).items()})
    out["gptq_gain"] = out["dam_rtn"] / max(out["dam_gptq"], 1e-30)
    out["cd_gain"] = out["dam_gptq"] / max(out["dam_cd"], 1e-30)
    out["grid_dist"] = grid_distance(W, args.bits, args.group)
    out["cd_frac_moved"] = info["frac_moved"]
    out["param_bits_per_weight"] = tf.param_bits() / float(W0.numel())
    dl, dr = tf.drift()
    out["drift_left"], out["drift_right"] = dl, dr
    return out


def learn_rotation(W0: torch.Tensor, A0: torch.Tensor, args, objective: str,
                   fixed: Optional[RotationPair], k: int, steps: int, lr: float,
                   outer: int = 1, cd_sweeps: int = 4, both_sides: bool = True,
                   verbose: bool = False,
                   init_tf: Optional[ComposedTransform] = None) -> ComposedTransform:
    """Fit the learned correction under one of the three objectives.

    `O1` weight-to-grid, `O2` the local quantized output error, `O3` the same functional error
    but with the integer codes held at whatever the *real* GPTQ + CD pipeline chose -- which is
    what makes it compensation-aware rather than reconstruction-aware. `O3` alternates: run the
    real pipeline, freeze its codes, move the basis, run the real pipeline again.

    `init_tf` continues from an already-fitted transform instead of from the fixed baseline,
    which is how the "does compensation-awareness improve a grid-optimized basis" arm is built.

    A caution the alternation deserves: holding the codes fixed anchors the basis to wherever
    those codes came from, because the objective is minimised by moving the weights *back*
    towards the reconstruction points they were assigned. That is Lloyd-style stalling, and it
    is the reason this function is also run with the inner and outer budgets swapped -- many
    short re-quantizations instead of few long fits -- so that a null result can be attributed
    to the objective rather than to the schedule.
    """
    out_f, in_f = W0.shape
    dev = W0.device
    if init_tf is not None:
        tf = init_tf
        for p in tf.params():
            p.requires_grad_(True)
    else:
        right = HouseholderRotation(in_f, k, dev, seed=args.seed + 1)
        left = HouseholderRotation(out_f, k, dev, seed=args.seed + 2) if both_sides \
            else IdentityRotation()
        tf = ComposedTransform(fixed, left, right)
    params = tf.params()
    opt = torch.optim.Adam(params, lr=lr)
    hist = []

    prev_codes = None
    traj: List[Dict] = []
    for it in range(max(outer, 1)):
        codes = None
        if objective == "O3":
            with torch.no_grad():
                W = tf.forward_w(W0)
                A = tf.forward_h(A0)
                _, st = quantize_with_state(W, 2.0 * A, args.bits, args.group, args.percdamp)
                cd_refine(W, A, st, None, sweeps=cd_sweeps, seed=args.seed)
                codes = st.codes.float().detach()
                # Section 6 asks for the alternation to be audited rather than assumed: the
                # *hard* quantized objective at every outer step, and how many integer
                # assignments moved since the last one. A scheme that oscillates, or whose soft
                # surrogate improves while the hard objective does not, is visible here.
                hard = damage(W0 - tf.inverse_w(st.dequant()), A0) / max(damage(W0, A0), 1e-30)
                chg = (float((st.codes != prev_codes).float().mean())
                       if prev_codes is not None else None)
                traj.append(dict(outer=it, hard_rel=hard, codes_changed=chg))
                prev_codes = st.codes.clone()
                if verbose:
                    print("      O3 outer %d: hard %.6f  codes changed %s"
                          % (it, hard, ("%.4f" % chg) if chg is not None else "-"))
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            W = tf.forward_w(W0)
            if objective == "O1":
                E, _ = quant_error(W, args.bits, args.group)
                loss = (E * E).sum()
            else:
                A = tf.forward_h(A0)
                E, _ = quant_error(W, args.bits, args.group,
                                   codes if objective == "O3" else None)
                loss = ((E @ A) * E).sum()
            loss.backward()
            opt.step()
            if verbose and step % max(steps // 4, 1) == 0:
                print("      %s outer %d step %3d loss %.6e" % (objective, it, step,
                                                                float(loss)))
        hist.append(float(loss.detach()))
    tf.fit_history = hist
    tf.alt_trajectory = traj
    return tf


@torch.no_grad()
def hadamard_pair(W0, args):
    return RotationPair(W0.shape[0], W0.shape[1], seed=args.seed, device=W0.device)


def run_module(name: str, W0: torch.Tensor, A0: torch.Tensor, args, fh) -> None:
    dev = args.device
    out_f, in_f = W0.shape
    base = dict(module=name, bits=args.bits, group=args.group, out=out_f, inn=in_f,
                k=args.k, steps=args.steps, lr=args.lr)

    def emit(arm, tf, extra=None, t0=None):
        rec = dict(base, arm=arm, **evaluate(W0, A0, tf, args, args.cd_sweeps))
        rec.update(extra or {})
        if t0 is not None:
            rec["fit_s"] = time.time() - t0
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("    %-22s rtn %.5f  gptq %.5f  cd %.5f | gptq x%.2f cd x%.3f | grid %.4f"
              " | +%.4f bpw" % (arm, rec["dam_rtn"], rec["dam_gptq"], rec["dam_cd"],
                                rec["gptq_gain"], rec["cd_gain"], rec["grid_dist"],
                                rec["param_bits_per_weight"]))
        return rec

    print("  [%s  %dx%d]" % (name.split("layers.")[-1], out_f, in_f))
    emit("T0_native", ComposedTransform(None))

    rp = hadamard_pair(W0, args)
    rt = roundtrip_error(W0, rp)
    emit("T1_hadamard", ComposedTransform(rp), dict(roundtrip_err=rt))

    for s in range(args.rand_seeds):
        rp_s = RotationPair(out_f, in_f, seed=1000 + s, device=dev)
        emit("T2_had_seed%d" % s, ComposedTransform(rp_s), dict(rand_seed=1000 + s))
    if max(out_f, in_f) <= args.haar_max:
        for s in range(args.haar_seeds):
            L = HaarRotation(out_f, dev, seed=2000 + s)
            R = HaarRotation(in_f, dev, seed=3000 + s)
            emit("T2_haar_seed%d" % s, ComposedTransform(None, L, R), dict(rand_seed=2000 + s))

    grid_tf = None
    for obj, tag in (("O1", "T3_grid"), ("O2", "T4_output"), ("O3", "T5_compensation")):
        for init, fx in (("had", rp), ("id", None)):
            if init == "id" and not args.also_from_identity:
                continue
            t0 = time.time()
            tf = learn_rotation(W0, A0, args, obj, fx, args.k, args.steps, args.lr,
                                outer=args.outer if obj == "O3" else 1,
                                cd_sweeps=args.cd_sweeps, verbose=args.verbose)
            emit("%s_%s" % (tag, init), tf,
                 dict(objective=obj, init=init, fit_history=getattr(tf, "fit_history", []),
                      alt_trajectory=getattr(tf, "alt_trajectory", [])), t0=t0)
            if obj == "O1" and init == "had":
                grid_tf = tf

    # Two controls for the alternation itself, because a null result from O3 has to be
    # attributable to the objective and not to a schedule that simply never left its start.
    if args.o3_controls:
        t0 = time.time()
        tf = learn_rotation(W0, A0, args, "O3", rp, args.k,
                            max(args.steps // 5, 10), args.lr, outer=args.outer * 5,
                            cd_sweeps=args.cd_sweeps, verbose=args.verbose)
        emit("T5b_manyouter", tf, dict(objective="O3", init="had",
                                       alt_trajectory=getattr(tf, "alt_trajectory", [])), t0=t0)
        if grid_tf is not None:
            t0 = time.time()
            tf = learn_rotation(W0, A0, args, "O3", rp, args.k, args.steps, args.lr,
                                outer=args.outer, cd_sweeps=args.cd_sweeps,
                                verbose=args.verbose, init_tf=grid_tf)
            emit("T5c_from_grid", tf, dict(objective="O3", init="O1",
                                           alt_trajectory=getattr(tf, "alt_trajectory", [])),
                 t0=t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,up_proj")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--cd-sweeps", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="Householder reflections per axis")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--outer", type=int, default=4, help="alternating rounds for O3")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--rand-seeds", type=int, default=4)
    ap.add_argument("--haar-seeds", type=int, default=2)
    ap.add_argument("--haar-max", type=int, default=1024)
    ap.add_argument("--also-from-identity", action="store_true")
    ap.add_argument("--o3-controls", action="store_true",
                    help="run the many-outer and warm-from-O1 alternation controls")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/transform_oracle.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cal = torch.load(args.cal, map_location="cpu")
    names = pick_modules(cal, [int(x) for x in args.layers.split(",")], args.projs.split(","))
    print("%d modules" % len(names))
    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fh = open(args.out, "a", encoding="utf-8")
    t_all = time.time()
    for n in names:
        W = get_module(model, n).weight.detach().float().to(args.device)
        A = cal[n]["A"].float().to(args.device)
        run_module(n, W, A, args, fh)
        del W, A
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("done in %.0f s, peak VRAM %.0f MiB"
          % (time.time() - t_all,
             torch.cuda.max_memory_allocated() / 2 ** 20 if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
