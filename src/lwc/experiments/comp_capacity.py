"""How much of the raw quantization damage can local compensation actually remove, per bitrate?

The cross-scope experiment has so far been run at 3 bits, where GPTQ is extremely effective. If
local compensation saturates at low rate -- if at 2 bits it can no longer absorb the error it is
handed -- then that, not 3 bits, is where a wider compensation scope has room to matter. This
experiment measures the capacity curve directly, and at the same time measures whether the
compensation that *is* achieved generalises off the calibration sample.

For each matrix, coordinate system and rate:

    D_raw    damage after plain round-to-nearest
    D_gptq   after GPTQ
    D_cd     after GPTQ + coordinate descent, i.e. the best local solution we know how to find

    compensated fraction = 1 - D_post / D_raw          how much of the raw error was absorbed
    residual             = D_cd                        what local compensation could not fix

The rate ladder uses exact, realisable formats. A group-wise INT-b checkpoint costs
`b + 32/group` bits per weight, so fractional rates come from the group size rather than from
imaginary fractional codes, and every number in the table is a format that could actually ship:

    2 bits: group 256 -> 2.125,  128 -> 2.25,  64 -> 2.5,  32 -> 3.0
    3 bits: group 256 -> 3.125,  128 -> 3.25,  64 -> 3.5
    4 bits: group 256 -> 4.125,  128 -> 4.25

Generalisation is measured on the spot: every arm is scored against both the calibration second
moment it was optimized on and an independent one collected from disjoint text, so the
calibration gap is available as a function of rate rather than as a separate study.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch

from ..joint import cd_refine, damage, quantize_with_state
from ..metrics import nmse
from ..modelio import get_module, load_model
from ..rotate import RotationPair

# (bits, group) pairs, ordered by the exact bits/weight they cost
LADDER: List[Tuple[int, int]] = [
    (2, 256), (2, 128), (2, 64), (2, 32),
    (3, 256), (3, 128), (3, 64),
    (4, 256), (4, 128),
]


def exact_bpw(bits: int, group: int) -> float:
    """Codes plus one fp16 scale and one fp16 zero per (row, group)."""
    return bits + 2.0 * 16.0 / group


def pick_modules(cal: Dict, layers: List[int], projs: List[str]) -> List[str]:
    out = []
    for n in cal:
        p = n.split(".")
        if int(p[p.index("layers") + 1]) in layers and p[-1] in projs:
            out.append(n)
    return sorted(out, key=lambda n: (int(n.split("layers.")[1].split(".")[0]), n))


@torch.no_grad()
def run_module(name: str, W0: torch.Tensor, A0: torch.Tensor, Aval: Optional[torch.Tensor],
               args, fh) -> None:
    dev = W0.device
    out_f, in_f = W0.shape
    for coord in args.coords.split(","):
        if coord == "hadamard":
            rp = RotationPair(out_f, in_f, seed=args.seed, device=dev)
            W = rp.forward_w(W0)
            A = 0.5 * (rp.forward_h(A0) + rp.forward_h(A0).T)
        else:
            rp, W, A = None, W0, A0
        den = max(damage(W0, A0), 1e-30)
        den_v = max(damage(W0, Aval), 1e-30) if Aval is not None else 1.0
        print("  [%s %s]" % (name.split("layers.")[-1], coord))

        for bits, group in LADDER:
            if in_f % group:
                continue
            t0 = time.time()
            Wn, _ = quantize_with_state(W, None, bits, group, args.percdamp)
            Wg, st = quantize_with_state(W, 2.0 * A, bits, group, args.percdamp)
            st_cd = st.clone()
            Wc, info = cd_refine(W, A, st_cd, None, sweeps=args.cd_sweeps, seed=args.seed)

            back = (lambda X: rp.inverse_w(X)) if rp is not None else (lambda X: X)
            Dn, Dg, Dc = W0 - back(Wn), W0 - back(Wg), W0 - back(Wc)
            d_raw, d_gptq, d_cd = (damage(Dn, A0) / den, damage(Dg, A0) / den,
                                   damage(Dc, A0) / den)
            rec = dict(module=name, coord=coord, bits=bits, group=group,
                       bpw=exact_bpw(bits, group), out=out_f, inn=in_f,
                       d_raw=d_raw, d_gptq=d_gptq, d_cd=d_cd,
                       frac_gptq=1.0 - d_gptq / max(d_raw, 1e-30),
                       frac_cd=1.0 - d_cd / max(d_raw, 1e-30),
                       cd_extra=1.0 - d_cd / max(d_gptq, 1e-30),
                       gptq_gain=d_raw / max(d_gptq, 1e-30),
                       w_nmse_cd=nmse(W0, back(Wc)), cd_frac_moved=info["frac_moved"],
                       secs=time.time() - t0)
            if Aval is not None:
                # the same weights, scored against calibration text they were not fitted on
                v_raw, v_gptq, v_cd = (damage(Dn, Aval) / den_v, damage(Dg, Aval) / den_v,
                                       damage(Dc, Aval) / den_v)
                rec.update(v_raw=v_raw, v_gptq=v_gptq, v_cd=v_cd,
                           gap_gptq=v_gptq - d_gptq, gap_cd=v_cd - d_cd,
                           gap_ratio_cd=v_cd / max(d_cd, 1e-30),
                           gap_ratio_raw=v_raw / max(d_raw, 1e-30))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("    %4.3f bpw (b%d g%-3d)  raw %.5f  gptq %.5f  +cd %.5f  | removed "
                  "%5.1f%% / %5.1f%%  | held-out ratio %s"
                  % (rec["bpw"], bits, group, d_raw, d_gptq, d_cd,
                     100 * rec["frac_gptq"], 100 * rec["frac_cd"],
                     ("%.3f" % rec["gap_ratio_cd"]) if Aval is not None else "-"))
            del Wn, Wg, Wc, Dn, Dg, Dc, st, st_cd
        del W, A
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
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--cd-sweeps", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/comp_capacity.jsonl")
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
    t0 = time.time()
    for n in names:
        W = get_module(model, n).weight.detach().float().to(args.device)
        A = cal[n]["A"].float().to(args.device)
        Av = val[n]["A"].float().to(args.device) if n in val else None
        run_module(n, W, A, Av, args, fh)
        del W, A, Av
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("done in %.0f s, peak VRAM %.0f MiB"
          % (time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 20
             if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
