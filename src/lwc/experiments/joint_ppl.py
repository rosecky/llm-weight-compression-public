"""End-to-end perplexity for joint quantization-decision optimization. The decisive test.

Everything else in this branch is a proxy. This project has now misranked methods by a
layer-local quadratic metric five separate times, so no arm counts as better until it is better
here: full-model sequential quantization, then wikitext-2 perplexity on held-out text.

The rules of section 6 are enforced structurally rather than by promise. Each arm produces a
`QuantState` -- integer codes plus one (scale, zero) per (row, group) -- and the reported bit
count comes from that state's own `storage_bits`. Coordinate descent changes which integers are
stored and nothing else: no residual, no outlier stream, no side information, no wider group, no
extra precision anywhere. An arm that wins here wins at exactly the baseline's bitrate.

Config keys (`configs/joint_ppl*.json`):

    coord    native | hadamard
    comp     gptq | naive                  what produces the starting point
    cd       number of coordinate-descent sweeps, 0 to disable
    order    forward | backward | random | alternate
    scope    layer | module | blockK -- the optimization horizon, and the only thing that
             changes between scope arms. Its `G` is measured inside the pipeline.
    gdamp    trust-region damping on that wider metric, relative to its mean diagonal
    refit    alternations of (coordinate descent, least-squares scale refit)
    ambig    if > 0, only this fraction of the least confident decisions may move
    adamp    Tikhonov damping on A during optimization, matching what GPTQ already applies
    cd_pre   layer-objective sweeps to run *before* switching to the wider one, so a scope arm
             starts from the GPTQ+CD local optimum rather than from raw GPTQ
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from ..attnscope import attn_G
from ..calib import get_calib, get_wikitext2  # noqa: F401  (get_wikitext2 kept for callers)
from ..gptq import HessianAccumulator, gptq
from ..joint import (DenseMetric, cd_refine, damp_dense, margins, metric_identity,
                     quantize_with_state, refit_scales)
from ..vq import VQFrozenQuantizer, build_vq, vq_refine
from ..modelio import PROJ_TYPES, get_module, list_linear_layers, load_model
from ..rotate import RotationPair
from ..scopeg import block_G
from .factorial_ppl import capture_layer0_inputs, get_decoder_layers, perplexity


def measure_G(scope, model, layers, li, hidden, kw, args, seed):
    """The wider metric for one block at the requested horizon, in one place so the decision
    path and the fresh-G verification path cannot drift apart.

    `module_bd` is `module` with the attention side moved to its exact algebra: heads do not
    mix before o_proj, so that endpoint's G is block-diagonal by construction and is measured
    as such (far better conditioned per sample). The MLP side is unchanged -- down_proj is the
    only place MLP channels mix at all, so there is no structure there to exploit.
    """
    common = dict(batch=args.batch, n_probe=args.n_probe, seed=seed, device=args.device,
                  max_tokens=args.g_tokens)
    if scope in ("attn_logits", "attn_smax", "attn_out", "attn_out_bd"):
        ep = {"attn_logits": "logits", "attn_smax": "softmax",
              "attn_out": "attn_out", "attn_out_bd": "attn_out"}[scope]
        return attn_G(layers[li], hidden, kw, ep,
                      block_diag=scope.endswith("_bd"), **common)
    if scope in ("module_bd", "module_pre"):
        # module_pre isolates the endpoint change (post-o -> pre-o) with the estimator
        # unchanged; module_bd adds the exact per-head structure on top of it.
        Gs = attn_G(layers[li], hidden, kw, "attn_out",
                    block_diag=(scope == "module_bd"), **common)
        Gs.update(block_G(layers, li, hidden, kw, "module",
                          keep_projs={"gate_proj", "up_proj"}, **common))
        return Gs
    tail = (model.model.norm, model.lm_head) if scope == "model" else None
    Gs = block_G(layers, li, hidden, kw,
                 "module" if scope == "module_attn" else scope, tail=tail, **common)
    if scope == "module_attn":
        Gs = {k: v for k, v in Gs.items()
              if k.split(".")[-1] in ("q_proj", "k_proj", "v_proj")}
    return Gs


@torch.no_grad()
def quantize_sequential_joint(model, tok, cfg, args) -> Dict:
    device = args.device
    layers, _ = get_decoder_layers(model)
    target = set(args.projs.split(","))
    cs = args.calib_seed if args.calib_seed >= 0 else args.seed
    # calibration source: "wikitext2" (every frozen run) or "c4" (addendum D3 only); the
    # default path is byte-identical to before
    src = getattr(args, "calib_source", "wikitext2") or "wikitext2"
    ids = get_calib(src, tok, args.calib_seqlen, args.calib_seq, cs)
    # Fresh-G verification windows (P1 audit): drawn separately with an offset seed so the
    # decision windows are byte-identical to every historical run; used ONLY to re-score
    # the wider objective on unseen text, never for any decision.
    n_fresh = int(getattr(args, "fresh_g_seq", 0) or 0)
    if n_fresh > 0:
        ids2 = get_calib(src, tok, args.calib_seqlen, n_fresh, cs + 1000)
        ids = torch.cat([ids, ids2], 0)
    hidden, kw = capture_layer0_inputs(model, ids, device, batch=args.batch)
    nc = args.calib_seq                        # decision windows; the rest are fresh-G only
    prefix = None
    for name, _ in model.named_modules():
        if name.endswith(".layers"):
            prefix = name
            break

    total_bits, total_w = 0.0, 0
    flips = defaultdict(lambda: [0, 0])          # key -> [flipped, total]
    obj = []
    obj_fresh = []
    fresh_abs = [0.0, 0.0]                 # absolute objective sums: [before, after]
    t_opt = 0.0
    t_scope = 0.0

    for li in range(len(layers)):
        layer = layers[li]
        subs = {n: m for n, m in layer.named_modules()
                if isinstance(m, torch.nn.Linear)
                and any(n.endswith("." + p) or n == p for p in target)}
        accs = {n: HessianAccumulator(m.in_features, device=device) for n, m in subs.items()}
        handles = []
        for n, m in subs.items():
            def hook(mod, inp, out, key=n):
                accs[key].add(inp[0].detach())
            handles.append(m.register_forward_hook(hook))
        for i in range(0, nc, args.batch):
            layer(hidden[i:i + args.batch], **kw)
        for h in handles:
            h.remove()

        # The wider objective, measured here rather than cached: `hidden` already carries the
        # error of every block quantized so far, and the blocks downstream of this one are
        # still full precision, so this is the geometry the optimizer would really face.
        Gs = {}
        scope = cfg.get("scope", "layer")
        if cfg.get("glayers"):
            lo, hi = (int(x) for x in cfg["glayers"].split("-"))
            if not (lo <= li <= hi):
                scope = "layer"
        if scope != "layer" and cfg.get("cd", 0) > 0:
            t0 = time.time()
            Gs = measure_G(scope, model, layers, li, hidden[:nc], kw, args, args.seed)
            if cfg.get("gprojs"):
                keep = set(cfg["gprojs"].split(","))
                Gs = {k: v for k, v in Gs.items() if k.split(".")[-1] in keep}
            t_scope += time.time() - t0
        GsB = {}
        if (scope not in ("layer",) and cfg.get("cd", 0) > 0 and n_fresh > 0
                and not scope.startswith("attn")):
            GsB = measure_G(scope, model, layers, li, hidden[nc:], kw, args,
                            args.seed + 101)
            # scoring-only tensors: keep them off the GPU so the verification set never
            # competes with the decision set for memory (they return one at a time)
            GsB = {k: v.cpu() for k, v in GsB.items()}
            if device == "cuda":
                torch.cuda.empty_cache()

        for n, m in subs.items():
            W0 = m.weight.detach().float()
            out_f, in_f = W0.shape
            A0 = 0.5 * accs[n].finalize()
            full = "%s.%d.%s" % (prefix, li, n)
            rp = None
            if cfg["coord"] == "hadamard":
                rp = RotationPair(out_f, in_f, seed=args.seed, device=device)
                W = rp.forward_w(W0)
                Ar = rp.forward_h(A0)
                A = 0.5 * (Ar + Ar.T)
                del Ar
            else:
                W, A = W0, A0

            H = 2.0 * A

            grp = cfg.get("group", args.group)
            if cfg.get("repr", "scalar") == "vq2d":
                # ---- vector representation: same pipeline, same G machinery, different code
                adiag = torch.diagonal(A)
                st = build_vq(W, adiag, d=cfg.get("vq_d", 2), K=cfg.get("vq_k", 32),
                              group=grp, scale_mode=cfg.get("vq_scale", "group"),
                              seed=args.seed)
                if cfg["comp"] == "gptq":
                    Wh = gptq(W, H, VQFrozenQuantizer(st, adiag), percdamp=args.percdamp)
                else:
                    Wh = st.dequant()
                st_ref = st.clone()
                if cfg.get("cd", 0) > 0:
                    Gm = metric_identity()
                    if n in Gs:
                        Gd = Gs.pop(n)
                        if rp is not None:
                            Gd = rp.left.rotate_hessian(Gd)
                        Gm = DenseMetric(damp_dense(Gd, cfg.get("gdamp", 1.0)))
                        del Gd
                    t0 = time.time()
                    if cfg.get("cd_pre", 0) > 0 and not Gm.is_identity:
                        vq_refine(W, A, st, metric_identity(), sweeps=cfg["cd_pre"],
                                  order=cfg.get("order", "forward"), seed=args.seed)
                    Wh, info = vq_refine(W, A, st, Gm, sweeps=cfg["cd"],
                                         order=cfg.get("order", "forward"), seed=args.seed)
                    t_opt += time.time() - t0
                    obj.append(info["damage_end"] / max(info["damage_start"], 1e-30))
                    del Gm
                nd, fr = st.n_differ(st_ref)
                proj = n.split(".")[-1]
                flips[proj][0] += nd
                flips[proj][1] += st.codes.numel()
                flips["depth%d" % (li * 4 // len(layers))][0] += nd
                flips["depth%d" % (li * 4 // len(layers))][1] += st.codes.numel()
                Wfin = rp.inverse_w(Wh) if rp is not None else Wh
                m.weight.data.copy_(Wfin.to(m.weight.dtype))
                b = st.storage_bits()
                total_bits += sum(b.values()) + (32.0 if cfg["coord"] == "hadamard" else 0.0)
                total_w += out_f * in_f
                del W0, W, A, A0, H, Wh, Wfin, st, st_ref
                del accs[n]
                continue

            Wq, st = quantize_with_state(W, H if cfg["comp"] == "gptq" else None,
                                         cfg["bits"], grp, args.percdamp)
            st_ref = st.clone()
            Wh = Wq

            if cfg.get("cd", 0) > 0:
                Gm = metric_identity()
                if n in Gs:
                    Gd = Gs.pop(n)
                    if rp is not None:
                        Gd = rp.left.rotate_hessian(Gd)
                    Gm = DenseMetric(damp_dense(Gd, cfg.get("gdamp", 1.0)))
                    del Gd
                mask = None
                if cfg.get("ambig", 0) > 0:
                    mm = margins(W, A, st.clone(), None).flatten()
                    fin = torch.isfinite(mm)
                    k = max(1, int(cfg["ambig"] * int(fin.sum())))
                    thr = torch.kthvalue(mm[fin].float(), k).values
                    mask = (mm <= thr).reshape(W.shape) & fin.reshape(W.shape)
                # GPTQ minimises the damped Hessian, not the raw one. Giving the optimizer the
                # same regularisation is the apples-to-apples comparison; `adamp: 0` is the
                # unregularised arm that shows what the difference is worth.
                Aopt = A
                if cfg.get("adamp", 0.0) > 0:
                    Aopt = A + (cfg["adamp"] * float(torch.diagonal(A).mean())) * \
                        torch.eye(A.shape[0], device=device)
                t0 = time.time()
                # A wider objective has to prove itself against the best *local* solution, not
                # against raw GPTQ. `cd_pre` runs the layer objective to its own optimum first,
                # so the scope arms start from exactly the GPTQ+CD baseline and can only be
                # credited with what the wider horizon adds on top of it.
                if cfg.get("cd_pre", 0) > 0 and not Gm.is_identity:
                    cd_refine(W, Aopt, st, metric_identity(), sweeps=cfg["cd_pre"],
                              order=cfg.get("order", "forward"), seed=args.seed, mask=mask)
                Wh, info = cd_refine(W, Aopt, st, Gm, sweeps=cfg["cd"],
                                     order=cfg.get("order", "forward"), seed=args.seed,
                                     mask=mask)
                for _ in range(cfg.get("refit", 0)):
                    refit_scales(W, Aopt, st)
                    Wh, info2 = cd_refine(W, Aopt, st, Gm, sweeps=cfg["cd"],
                                          order=cfg.get("order", "forward"), seed=args.seed,
                                          mask=mask)
                if Aopt is not A:
                    del Aopt
                t_opt += time.time() - t0
                obj.append(info["damage_end"] / max(info["damage_start"], 1e-30))
                if n in GsB:
                    # score the SAME weights under a G estimated from unseen text:
                    # does the wider-objective improvement survive off its own sample?
                    GB = GsB.pop(n).to(W.device)
                    if rp is not None:
                        GB = rp.left.rotate_hessian(GB)
                    GB = 0.5 * (GB + GB.T)
                    Db, Da = W - Wq, W - Wh
                    eb = float(((Db @ A) * (GB @ Db)).sum())
                    ea = float(((Da @ A) * (GB @ Da)).sum())
                    obj_fresh.append(ea / max(eb, 1e-30))
                    fresh_abs[0] += eb
                    fresh_abs[1] += ea
                    del GB, Db, Da
                del Gm, mask

            nd, fr = st.n_differ(st_ref)
            proj = n.split(".")[-1]
            flips[proj][0] += nd
            flips[proj][1] += st.codes.numel()
            flips["depth%d" % (li * 4 // len(layers))][0] += nd
            flips["depth%d" % (li * 4 // len(layers))][1] += st.codes.numel()

            Wfin = rp.inverse_w(Wh) if rp is not None else Wh
            m.weight.data.copy_(Wfin.to(m.weight.dtype))
            b = st.storage_bits()
            total_bits += sum(b.values()) + (32.0 if cfg["coord"] == "hadamard" else 0.0)
            total_w += out_f * in_f
            del W0, W, A, A0, H, Wq, Wh, Wfin, st, st_ref
            del accs[n]
        accs.clear()
        Gs.clear()
        if device == "cuda":
            torch.cuda.empty_cache()
        outs = []
        for i in range(0, hidden.shape[0], args.batch):
            o = layer(hidden[i:i + args.batch], **kw)
            outs.append((o[0] if isinstance(o, tuple) else o).detach())
        hidden = torch.cat(outs, 0)
        del outs
        if li % 8 == 0 or li == len(layers) - 1:
            print("  layer %2d/%d  peak %.0f MiB  opt %.0fs" % (
                li, len(layers) - 1,
                torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0,
                t_opt))
    del hidden
    if device == "cuda":
        torch.cuda.empty_cache()
    hit = sum(v[0] for k, v in flips.items() if not k.startswith("depth"))
    tot = sum(v[1] for k, v in flips.items() if not k.startswith("depth"))
    return dict(bpw=total_bits / total_w, n_weights=total_w, opt_s=t_opt, scope_s=t_scope,
                obj_ratio=(sum(obj) / len(obj)) if obj else None,
                obj_fresh=(sum(obj_fresh) / len(obj_fresh)) if obj_fresh else None,
                obj_fresh_before=(fresh_abs[0] if obj_fresh else None),
                obj_fresh_after=(fresh_abs[1] if obj_fresh else None),
                flips={k: v[0] / max(v[1], 1) for k, v in flips.items()},
                frac_flipped=hit / max(tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--projs", default=",".join(PROJ_TYPES))
    ap.add_argument("--configs", default="configs/joint_ppl.json")
    ap.add_argument("--only", default="", help="comma-separated config names to run")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--calib-seq", type=int, default=32)
    ap.add_argument("--calib-seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--calib-seed", type=int, default=-1,
                    help="draw calibration text from a different seed than the rotation, so "
                         "calibration sensitivity can be measured with everything else fixed")
    ap.add_argument("--n-probe", type=int, default=2)
    ap.add_argument("--g-tokens", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/joint_ppl.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = json.load(open(args.configs, encoding="utf-8"))
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in keep]
    model, tok = load_model(args.model, device=args.device, dtype=torch.float16)
    orig = {r.name: get_module(model, r.name).weight.detach().clone().cpu()
            for r in list_linear_layers(model) if r.proj in set(args.projs.split(","))}
    base = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
    print("fp16 baseline perplexity %.4f\n" % base)

    fh = open(args.out, "a", encoding="utf-8")
    for cfg in configs:
        t0 = time.time()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        print("[%s]" % cfg["name"])
        info = quantize_sequential_joint(model, tok, cfg, args)
        ppl = perplexity(model, tok, args.device, args.seqlen, args.n_seq, args.seed)
        rec = dict(cfg, bpw=info["bpw"], ppl=ppl, base_ppl=base, ppl_delta=ppl - base,
                   calib_seq=args.calib_seq, calib_seqlen=args.calib_seqlen,
                   calib_seed=(args.calib_seed if args.calib_seed >= 0 else args.seed),
                   calib_tokens=args.calib_seq * args.calib_seqlen,
                   nll_delta=float(torch.log(torch.tensor(ppl / base))),
                   obj_ratio=info["obj_ratio"], flips=info["flips"],
                   encode_s=time.time() - t0, opt_s=info["opt_s"],
                   scope_s=info["scope_s"],
                   peak_vram_mib=(torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0.0))
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print("  -> bpw=%.4f  ppl=%.4f (fp16 %.4f, delta %+.4f)  obj %s  %.0fs\n"
              % (rec["bpw"], ppl, base, rec["ppl_delta"],
                 ("%.4f" % info["obj_ratio"]) if info["obj_ratio"] else "-",
                 rec["encode_s"]))
        for k, w in orig.items():
            get_module(model, k).weight.data.copy_(w.to(args.device))
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    main()
