"""Hypotheses H1 and H2 at full matrix scale, at identical storage.

H1 (greedy commit). GPTQ quantizes input columns left to right and never reopens a decision.
`cd_refine` reopens all of them, repeatedly, each time to the exact conditional optimum of the
*same* objective GPTQ is minimising, with the *same* scales GPTQ chose. Any gap is the price of
irreversibility and nothing else.

H2 (scope). The layer objective `tr(dW A dW^T)` treats output rows as independent problems.
A downstream objective `tr(dW A dW^T G)`, with `G = E[J^T J]` for the Jacobian of whatever we
actually care about, couples them. Running the identical optimizer under the identical bit
budget against a wider-horizon `G` is what "widening the optimization scope" means here, and
comparing the two answers is what isolates scope from search strength.

Arms, all at the same bits / group / codebook / scales-per-group budget:

    naive          RTN, no compensation                                          (S0)
    gptq           the canonical engine                                          (S1)
    gptq+cd        GPTQ then exact coordinate descent                            (H1)
    gptq+cd:aX     ... on the *damped* Hessian GPTQ itself minimises, so the optimizer gets the
                   same regularisation the baseline already has
    gptq+cd+refit  ... alternating with a least-squares scale refit (still the same storage)
    naive+cd       coordinate descent from the RTN basin (does the GPTQ start matter?)
    gptq+cdG:dX    coordinate descent against the Fisher `G` at damping X        (H2)
    gptq+cdG:rR    ... against a rank-R surrogate of it, which is the section 9.3 measurement
    gptq+cd@p      coordinate descent restricted to the p least confident decisions (section 15)

Every arm is scored on *both* objectives -- the layer's own and the Fisher-weighted one -- so
that an arm which wins on the objective it was given and loses on the other is visible
immediately rather than after an end-to-end run.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import torch

from ..joint import (DenseMetric, QuantState, additive_change, cd_refine, damage,
                     damp_dense, margins, metric_from_dense, metric_identity,
                     quantize_with_state, refit_scales)
from ..metrics import nmse
from ..modelio import get_module, list_linear_layers, load_model
from ..rotate import RotationPair


def pick_modules(cal: Dict, layers: List[int], projs: List[str]) -> List[str]:
    out = []
    for n in cal:
        parts = n.split(".")
        li = int(parts[parts.index("layers") + 1])
        if li in layers and parts[-1] in projs:
            out.append(n)
    return sorted(out, key=lambda n: (int(n.split("layers.")[1].split(".")[0]), n))


@torch.no_grad()
def score(W, Wh, A, Gm, Aval, den) -> Dict:
    """`den` holds the three denominators, which depend only on the matrix, not on the arm.

    Recomputing `tr(W A W^T G)` for every arm meant three dense (out, out) applications per
    row of output -- more compute than the optimizer being measured.
    """
    D = W - Wh
    return dict(
        rel_layer=damage(D, A) / den[0],
        rel_fisher=damage(D, A, Gm) / den[1],
        rel_heldout=(damage(D, Aval) / den[2]) if Aval is not None else None,
        w_nmse=nmse(W, Wh),
    )


@torch.no_grad()
def flip_stats(st: QuantState, ref: QuantState, rtn: Optional[QuantState] = None) -> Dict:
    d = (st.codes.int() - ref.codes.int())
    nz = d != 0
    out = dict(frac_flipped=0.0, mean_abs_step=0.0, frac_step1=0.0, max_step=0)
    n = int(nz.sum())
    if n:
        a = d[nz].abs().float()
        out = dict(frac_flipped=n / float(d.numel()), mean_abs_step=float(a.mean()),
                   frac_step1=float((a == 1).float().mean()), max_step=int(a.max()))
    if rtn is not None:
        out["frac_vs_rtn"] = float((st.codes != rtn.codes).float().mean())
    return out


@torch.no_grad()
def run_module(name: str, W0: torch.Tensor, A0: torch.Tensor, G0: Optional[torch.Tensor],
               Aval0: Optional[torch.Tensor], args, fh) -> None:
    dev = args.device
    for coord in args.coords.split(","):
        if coord == "hadamard":
            rp = RotationPair(W0.shape[0], W0.shape[1], seed=args.seed, device=dev)
            W = rp.forward_w(W0)
            A = 0.5 * (rp.forward_h(A0) + rp.forward_h(A0).T)
            Aval = None if Aval0 is None else 0.5 * (rp.forward_h(Aval0)
                                                     + rp.forward_h(Aval0).T)
            Gd = None if G0 is None else rp.left.rotate_hessian(G0)
        else:
            W, A, Aval, Gd = W0, A0, Aval0, G0
        H = 2.0 * A
        Gfull = metric_identity() if Gd is None else DenseMetric(Gd)

        base = dict(module=name, coord=coord, bits=args.bits, group=args.group,
                    out=int(W.shape[0]), inn=int(W.shape[1]))
        if Gd is not None and coord == "native" and args.spectrum:
            lam = torch.linalg.eigvalsh(0.5 * (Gd + Gd.T)).clamp_min(0.0).flip(0)
            cs = torch.cumsum(lam, 0) / lam.sum().clamp_min(1e-30)
            base["g_rank90"] = int((cs < 0.90).sum()) + 1
            base["g_rank99"] = int((cs < 0.99).sum()) + 1
            base["g_dim"] = int(lam.numel())
            del lam, cs

        den = (max(damage(W, A), 1e-30), max(damage(W, A, Gfull), 1e-30),
               max(damage(W, Aval), 1e-30) if Aval is not None else 1.0)

        Wn, st_n = quantize_with_state(W, None, args.bits, args.group, args.percdamp)
        Wg, st_g = quantize_with_state(W, H, args.bits, args.group, args.percdamp)

        def emit(arm, Wh, st, ref, t0, extra=None):
            rec = dict(base, arm=arm, **score(W, Wh, A, Gfull, Aval, den),
                       **flip_stats(st, ref, st_n), opt_s=time.time() - t0)
            if arm not in ("naive", "gptq"):
                rec.update(additive_change(W, A, st_g, st))
            rec.update(extra or {})
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("    %-16s layer %.5f  fisher %.5f  heldout %s  flipped %5.2f%%  %5.1fs"
                  % (arm, rec["rel_layer"], rec["rel_fisher"],
                     ("%.5f" % rec["rel_heldout"]) if rec["rel_heldout"] else "-",
                     100 * rec["frac_flipped"], rec["opt_s"]))
            return rec

        print("  [%s %s]" % (name.split("layers.")[-1], coord))
        emit("naive", Wn, st_n, st_g, time.time())
        emit("gptq", Wg, st_g, st_g, time.time())

        for od in args.orders.split(","):
            st = st_g.clone()
            t0 = time.time()
            Wh, info = cd_refine(W, A, st, None, sweeps=args.sweeps, order=od, seed=args.seed)
            emit("gptq+cd" + ("" if od == "forward" else ":" + od), Wh, st, st_g, t0,
                 dict(sweeps_run=info["sweeps_run"], obj_ratio=info["damage_end"]
                      / max(info["damage_start"], 1e-30), order=od))

        # GPTQ does not minimise `tr(dW A dW^T)`; it minimises the *damped* form, because an
        # empirical A from a few thousand tokens is badly conditioned for a 4864-wide input.
        # Coordinate descent on the undamped form is therefore free to exploit A's noise
        # directions, which is exactly what a held-out score would expose. These arms give the
        # optimizer the same regularisation the baseline already has.
        adiag_c = float(torch.diagonal(A).mean())
        eyeA = torch.eye(A.shape[0], device=dev)
        for ad in [float(x) for x in args.adamp.split(",") if float(x) > 0]:
            st = st_g.clone()
            t0 = time.time()
            Wh, info = cd_refine(W, A + (ad * adiag_c) * eyeA, st, None,
                                 sweeps=args.sweeps, seed=args.seed)
            emit("gptq+cd:a%g" % ad, Wh, st, st_g, t0, dict(adamp=ad))
        del eyeA

        st = st_n.clone()
        t0 = time.time()
        Wh, info = cd_refine(W, A, st, None, sweeps=args.sweeps, seed=args.seed)
        emit("naive+cd", Wh, st, st_g, t0, dict(obj_ratio=info["damage_end"]
                                                / max(info["damage_start"], 1e-30)))

        st = st_g.clone()
        t0 = time.time()
        Wh = None
        for it in range(args.refit_iters):
            Wh, info = cd_refine(W, A, st, None, sweeps=args.sweeps, seed=args.seed)
            refit_scales(W, A, st)
            Wh = st.dequant()
        emit("gptq+cd+refit", Wh, st, st_g, t0)

        # H2: the same optimizer, the same bits, a wider objective. The damping sweep is the
        # controlled path from the layer's own objective (large rel) to the raw downstream one
        # (rel -> 0); the rank sweep asks how low-rank the downstream metric really is.
        if Gd is not None:
            for rel in [float(x) for x in args.gdamp.split(",")]:
                st = st_g.clone()
                t0 = time.time()
                Gm = DenseMetric(damp_dense(Gd, rel))
                Wh, info = cd_refine(W, A, st, Gm, sweeps=args.sweeps, seed=args.seed)
                emit("gptq+cdG:d%g" % rel, Wh, st, st_g, t0,
                     dict(obj_ratio=info["damage_end"] / max(info["damage_start"], 1e-30),
                          backoffs=info["backoffs"], gdamp=rel, g_rank=0))
                del Gm
            for rk in [int(x) for x in args.granks.split(",") if int(x) > 0]:
                st = st_g.clone()
                t0 = time.time()
                Gm = metric_from_dense(damp_dense(Gd, args.grank_damp), rk)
                Wh, info = cd_refine(W, A, st, Gm, sweeps=args.sweeps, seed=args.seed)
                emit("gptq+cdG:r%d" % rk, Wh, st, st_g, t0,
                     dict(obj_ratio=info["damage_end"] / max(info["damage_start"], 1e-30),
                          gdamp=args.grank_damp, g_rank=rk))
                del Gm

        # --- section 15: is the whole benefit carried by the least confident decisions?
        m = margins(W, A, st_g.clone(), None).flatten()
        finite = torch.isfinite(m)
        for p in [float(x) for x in args.ambig.split(",")]:
            k = max(1, int(p * finite.sum().item()))
            thr = torch.kthvalue(m[finite].float(), k).values
            mask = (m <= thr).reshape(W.shape) & torch.isfinite(m).reshape(W.shape)
            st = st_g.clone()
            t0 = time.time()
            Wh, info = cd_refine(W, A, st, None, sweeps=args.sweeps, seed=args.seed, mask=mask)
            emit("gptq+cd@%g" % p, Wh, st, st_g, t0, dict(ambig_p=p))

        del W, A, H
        if dev == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--cal", default="cache/graphcal.pt")
    ap.add_argument("--valcal", default="cache/jointcal_val.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--orders", default="forward,backward,random")
    ap.add_argument("--refit-iters", type=int, default=3)
    ap.add_argument("--ambig", default="0.001,0.01,0.05,0.2")
    ap.add_argument("--adamp", default="0.01,0.1")
    ap.add_argument("--gdamp", default="1.0,0.1,0.01")
    ap.add_argument("--granks", default="16,64,256")
    ap.add_argument("--grank-damp", type=float, default=0.01)
    ap.add_argument("--spectrum", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/joint_layer.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cal = torch.load(args.cal, map_location="cpu")
    val = torch.load(args.valcal, map_location="cpu") if os.path.exists(args.valcal) else {}
    names = pick_modules(cal, [int(x) for x in args.layers.split(",")], args.projs.split(","))
    print("%d modules, held-out calibration: %s" % (len(names), bool(val)))

    model, _ = load_model(args.model, device="cpu", dtype=torch.float32)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fh = open(args.out, "a", encoding="utf-8")
    t_all = time.time()
    for n in names:
        W = get_module(model, n).weight.detach().float().to(args.device)
        A = cal[n]["A"].float().to(args.device)
        G = cal[n]["G"].float().to(args.device) if "G" in cal[n] else None
        Av = val[n]["A"].float().to(args.device) if n in val else None
        run_module(n, W, A, G, Av, args, fh)
        del W, A, G, Av
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("done in %.0f s, peak VRAM %.0f MiB"
          % (time.time() - t_all,
             torch.cuda.max_memory_allocated() / 2 ** 20 if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
