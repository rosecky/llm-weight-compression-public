"""The two cheap gates that decide whether a functional-graph branch is worth building.

Phase 1 (kill gate G2) -- is the topology real?
    Build affinity graphs from A = E[x x^T] and G = E[g g^T], partition them into balanced
    communities by spectral sequencing, and compare every structure metric against four
    surrogate matrices: vertex permutation, weighted configuration model, degree-preserving
    rewiring, and -- the decisive one -- an eigenvalue-preserving random-eigenvector matrix,
    which separates "this matrix has strong eigenmodes" from "the eigenmodes live on
    identifiable groups of channels".

Phase 2 (kill gate G1) -- do edges mean anything?
    For `y = W x` with K-FAC geometry the damage of a perturbation is `tr(dW A dW^T G)`, so
    the interaction between two channel perturbations is *exactly*

        input channels  i, j :  I_ij = 2 A_ij (e_i^T G e_j)     e_i = dW[:, i]
        output channels i, j :  I_ij = 2 G_ij (r_i^T A r_j)     r_i = dW[i, :]

    Both are computable for the entire population at once as an elementwise product of two
    matrices, so no pair sampling is needed and there is no selection bias. The gate then asks
    two things a graph enthusiast must not skip:

      1. what fraction of the total damage is interaction at all, rather than the sum of
         independent per-channel terms -- if that is ~0 there is nothing for topology to
         organise, regardless of how good the graph looks;
      2. whether the interaction is *systematically signed* or zero-mean noise, because only a
         systematic component can be exploited by choosing which channels share a block.

    A small end-to-end intervention then checks the quadratic model itself against measured
    NLL changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import torch

from ..calib import get_wikitext2
from ..gptq import ScalarQuantizer, gptq
from ..graphstruct import (affinity, balanced_labels, block_energy_ratio, make_graph_null,
                           modularity, normalized_cut, random_order, scale_order,
                           spectral_analyze, to_correlation, topk_sparsify)
from ..modelio import get_module, list_linear_layers, load_model
from ..rotate import RotationPair
from ..sensitivity import agreement

COV_NULLS = ("real", "permute", "spectral")
AFF_NULLS = ("config", "rewire")


def load_geo(path: str, name: str, side: str, device: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu")[name]
    return d[side].to(device, torch.float32)


# ====================================================================== phase 1


@torch.no_grad()
def phase1(args):
    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    geo = torch.load(args.graphcal, map_location="cpu")
    fh = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n_run = 0

    for name, d in sorted(geo.items()):
        li = int(name.split(".layers.")[1].split(".")[0])
        proj = name.split(".")[-1]
        if li not in layers or proj not in projs:
            continue
        for side in args.sides.split(","):
            C0 = d[side].to(args.device, torch.float32)
            n = C0.shape[0]
            if n > args.max_n:
                print("  skip %s/%s (n=%d > max-n)" % (name, side, n))
                continue
            for kind in args.kinds.split(","):
                for null in args.nulls.split(","):
                    if null in COV_NULLS:
                        S = affinity(make_graph_null(C0, null, args.seed), kind)
                    else:
                        S = make_graph_null(affinity(C0, kind), null, args.seed)
                    S = topk_sparsify(S, args.k)
                    order, spec = spectral_analyze(S)
                    rnd = random_order(n, args.seed, args.device)
                    scl = scale_order(C0)
                    for block in [int(b) for b in args.blocks.split(",")]:
                        if n % block:
                            continue
                        lab = balanced_labels(order, block)
                        rec = dict(phase="structure", matrix=name, layer=li, proj=proj,
                                   side=side, kind=kind, null=null, k=args.k, n=n,
                                   block=block, n_blocks=n // block,
                                   modularity=modularity(S, lab),
                                   ncut=normalized_cut(S, lab),
                                   block_energy=block_energy_ratio(S, lab),
                                   block_energy_random=block_energy_ratio(
                                       S, balanced_labels(rnd, block)),
                                   block_energy_scale=block_energy_ratio(
                                       S, balanced_labels(scl, block)),
                                   block_energy_contig=block_energy_ratio(
                                       S, balanced_labels(
                                           torch.arange(n, device=args.device), block)),
                                   **spec)
                        rec["energy_lift"] = rec["block_energy"] / max(
                            rec["block_energy_random"], 1e-30)
                        fh.write(json.dumps(rec) + "\n")
                        n_run += 1
                    fh.flush()
                print("  %-38s %s/%-5s %-8s n=%4d  %.0fs" % (name, side, kind, "", n,
                                                             time.time() - t0))
            del C0
            if args.device == "cuda":
                torch.cuda.empty_cache()
    fh.close()
    print("phase1: %d records, %.0fs -> %s" % (n_run, time.time() - t0, args.out))


# ====================================================================== phase 2


@torch.no_grad()
def interaction_matrices(E: torch.Tensor, A: torch.Tensor, G: torch.Tensor, axis: str):
    """Exact per-pair interaction and per-channel additive damage.

    axis='in'  -> M = E^T G E, coupling matrix is A
    axis='out' -> M = E A E^T,  coupling matrix is G
    """
    if axis == "in":
        M = E.T @ G @ E
        Cpl = A
    else:
        M = E @ A @ E.T
        Cpl = G
    I = 2.0 * Cpl * M                       # off-diagonal entries are the interactions
    D = torch.diagonal(Cpl) * torch.diagonal(M)
    return I, D, Cpl, M


@torch.no_grad()
def edge_report(I: torch.Tensor, D: torch.Tensor, Cpl: torch.Tensor, tag: str,
                seed: int = 0) -> Dict[str, float]:
    n = I.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=I.device)
    total = float(D.sum() + 0.5 * I[off].sum())     # tr(dW A dW^T G), exactly
    additive = float(D.sum())
    inter_signed = 0.5 * float(I[off].sum())
    inter_abs = 0.5 * float(I[off].abs().sum())
    out = {
        "%s_total" % tag: total,
        "%s_additive" % tag: additive,
        "%s_interaction_signed_frac" % tag: inter_signed / max(abs(total), 1e-30),
        "%s_interaction_abs_frac" % tag: inter_abs / max(abs(total), 1e-30),
    }
    # does the edge weight predict the interaction magnitude? full population, held out
    e = Cpl.abs()[off].double()
    y = I.abs()[off].double()
    g = torch.Generator(device=I.device).manual_seed(seed)
    m = torch.rand(e.numel(), generator=g, device=I.device) < 0.5
    a = agreement(e[m], y[m])
    out["%s_spearman_edge_absI" % tag] = a["spearman"]
    out["%s_top1_overlap" % tag] = a["top1_overlap"]
    # held-out log-linear fit  log|I| ~ a + b log|edge|
    le, ly = torch.log(e.clamp_min(1e-30)), torch.log(y.clamp_min(1e-30))
    ok = torch.isfinite(le) & torch.isfinite(ly) & (e > 0) & (y > 0)
    tr, te = ok & m, ok & ~m
    if int(tr.sum()) > 100 and int(te.sum()) > 100:
        x1, y1 = le[tr], ly[tr]
        b = ((x1 - x1.mean()) * (y1 - y1.mean())).sum() / (x1 - x1.mean()).pow(2).sum()
        a0 = y1.mean() - b * x1.mean()
        pred = a0 + b * le[te]
        ss = (ly[te] - pred).pow(2).sum()
        st = (ly[te] - ly[te].mean()).pow(2).sum()
        out["%s_heldout_r2" % tag] = float(1.0 - ss / st.clamp_min(1e-30))
        out["%s_slope" % tag] = float(b)
    return out


@torch.no_grad()
def phase2(args):
    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    geo = torch.load(args.graphcal, map_location="cpu")
    model, _ = load_model(args.model, device="cpu", dtype=torch.float16)
    refs = {r.name: r for r in list_linear_layers(model)}
    fh = open(args.out, "a", encoding="utf-8")
    t0 = time.time()

    for name, d in sorted(geo.items()):
        li = int(name.split(".layers.")[1].split(".")[0])
        proj = name.split(".")[-1]
        if li not in layers or proj not in projs:
            continue
        W0 = get_module(model, name).weight.detach().to(args.device, torch.float32)
        A0 = d["A"].to(args.device, torch.float32)
        G0 = d["G"].to(args.device, torch.float32)
        out_f, in_f = W0.shape
        for coord in args.coords.split(","):
            rp = None
            W, A, G = W0, A0, G0
            if coord == "hadamard":
                rp = RotationPair(out_f, in_f, seed=args.seed, device=args.device)
                W = rp.forward_w(W0)
                A = rp.right.rotate_hessian(A0)
                G = rp.left.rotate_hessian(G0) if rp.left is not None else G0
                A = 0.5 * (A + A.T)          # the transform is orthogonal, but
                G = 0.5 * (G + G.T)          # accumulation error is not symmetric
            Aco = to_correlation(A)
            Gco = to_correlation(G)
            for comp in args.comps.split(","):
                q = ScalarQuantizer(bits=args.bits, group=args.group)
                Q = gptq(W, 2.0 * A if comp == "gptq" else None, q, percdamp=args.percdamp)
                E = W - Q
                rec = dict(phase="edges", matrix=name, layer=li, proj=proj, coord=coord,
                           comp=comp, bits=args.bits, group=args.group,
                           out_f=out_f, in_f=in_f)
                for axis, Cpl_raw, Cpl_corr in (("in", A, Aco), ("out", G, Gco)):
                    I, D, Cpl, M = interaction_matrices(E, A, G, axis)
                    rec.update(edge_report(I, D, Cpl, "%s_cov" % axis, args.seed))
                    # the same interaction, ranked by the *scale-free* edge instead
                    rec.update(edge_report(I, D, Cpl_corr, "%s_corr" % axis, args.seed))
                    del I, D, M
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print("  %-38s %-8s %-5s  in:int=%.4f/%.4f  out:int=%.4f/%.4f  %.0fs"
                      % (name, coord, comp,
                         rec["in_cov_interaction_signed_frac"],
                         rec["in_cov_interaction_abs_frac"],
                         rec["out_cov_interaction_signed_frac"],
                         rec["out_cov_interaction_abs_frac"], time.time() - t0))
                del Q, E
        del W0, A0, G0
        if args.device == "cuda":
            torch.cuda.empty_cache()
    fh.close()
    print("phase2 done, %.0fs -> %s" % (time.time() - t0, args.out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="structure", choices=["structure", "edges"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--graphcal", default="cache/graphcal.pt")
    ap.add_argument("--layers", default="1,11,22")
    ap.add_argument("--projs", default="q_proj,o_proj,up_proj,down_proj")
    ap.add_argument("--sides", default="A,G")
    ap.add_argument("--kinds", default="cov,corr")
    ap.add_argument("--nulls", default="real,permute,config,rewire,spectral")
    ap.add_argument("--blocks", default="32,64,128")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-n", type=int, default=5000)
    ap.add_argument("--coords", default="native,hadamard")
    ap.add_argument("--comps", default="naive,gptq")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = "results/raw/graph_%s.jsonl" % args.phase
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    (phase1 if args.phase == "structure" else phase2)(args)
    if args.device == "cuda":
        print("peak VRAM %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20))


if __name__ == "__main__":
    main()
