"""Does low-bit quantization stay sensitive to *which* calibration text you happened to draw?

The practical question is sharp. If 8k tokens looks unstable but 32k is fine, there is no
research problem -- just use more calibration data. If 2-bit stays sensitive at 64k and beyond,
then the compensation is chasing sampling noise in a way more data does not fix, and an
uncertainty-aware or robust GPTQ becomes an interesting direction.

Design, chosen so the answer is not an artefact of the measurement:

* **One large fixed held-out second moment** is collected once from text disjoint from every
  calibration draw, and is the *only* yardstick. Scoring draw A against draw B would confound
  "the solution is unstable" with "the yardstick is unstable".
* **Several independent draws at each size.** Sensitivity is the spread of held-out damage
  across draws at a fixed size and rate -- with two draws you get a difference, not a spread.
* **Both halves of the question.** Geometry stability (do the draws even agree about the
  directions) and solution stability (do the resulting checkpoints, and their held-out quality,
  agree) are reported separately, because a geometry that is only 70% reproducible costs nothing
  if the compensation it produces transfers anyway.
* **Every rate.** The whole point is whether sensitivity grows as the bitrate falls.

Token budgets are `n_seq x seqlen`, so the default sizes are 4k, 8k, 16k, 32k, 64k and 128k.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import time
from typing import Dict, List

import torch

from ..calibpool import budget_to_nseq, draw_windows, token_stream
from ..gradcov import collect_AG
from ..joint import cd_refine, damage, quantize_with_state
from ..modelio import get_module, list_linear_layers, load_model
from ..rotate import RotationPair
from .comp_capacity import LADDER, exact_bpw


@torch.no_grad()
def top_eigvecs(A: torch.Tensor, r: int, oversample: int = 24, power: int = 3):
    """Leading `r` eigenvectors of a PSD matrix without a full decomposition.

    A dense `eigh` of the 4864-wide MLP input moment costs more than everything else in this
    experiment put together, and only the leading directions are ever compared. Randomized
    range-finding with a few power iterations reproduces them to far better precision than the
    sampling noise being measured.
    """
    n = A.shape[0]
    if r + oversample >= n:
        lam, U = torch.linalg.eigh(A)
        return U.flip(1)[:, :r]
    Om = torch.randn(n, r + oversample, device=A.device, dtype=A.dtype)
    Y = A @ Om
    for _ in range(power):
        Y, _ = torch.linalg.qr(Y)
        Y = A @ Y
    Q, _ = torch.linalg.qr(Y)
    S = Q.T @ (A @ Q)
    _, V = torch.linalg.eigh(0.5 * (S + S.T))
    return (Q @ V.flip(1))[:, :r].contiguous()


@torch.no_grad()
def subspace_overlap(A: torch.Tensor, B: torch.Tensor, ranks: List[int]) -> Dict:
    rmax = max(ranks)
    Ua = top_eigvecs(A, rmax)
    Ub = top_eigvecs(B, rmax)
    out = {}
    for r in ranks:
        if r < A.shape[0]:
            s = torch.linalg.svdvals(Ua[:, :r].T @ Ub[:, :r])
            out["subspace%d" % r] = float((s ** 2).mean())
    da, db = torch.diagonal(A), torch.diagonal(B)
    out["diag_corr"] = float(torch.corrcoef(torch.stack([da, db]))[0, 1])
    m = ~torch.eye(A.shape[0], dtype=torch.bool, device=A.device)
    out["offdiag_corr"] = float(torch.corrcoef(torch.stack([A[m], B[m]]))[0, 1])
    return out


def collect_A(model, refs, stream, n_parts, part, seqlen, n_seq, device, seed=0):
    ids = draw_windows(stream, n_parts, part, seqlen, n_seq, seed=seed)
    st = collect_AG(model, refs, ids, device=device, batch=1, need_grad=False)
    out = {n: 0.5 * (d["A"].float() + d["A"].float().T) for n, d in st.items()}
    del st
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="11")
    ap.add_argument("--projs", default="q_proj,up_proj")
    ap.add_argument("--budgets", default="32768,65536,131072,262144,524288",
                    help="calibration tokens per draw")
    ap.add_argument("--seqlens", default="512,2048",
                    help="realise each budget at each of these sequence lengths")
    ap.add_argument("--corpus", default="wikitext103")
    ap.add_argument("--pool-tokens", type=int, default=6_000_000)
    ap.add_argument("--draws", type=int, default=3, help="independent calibration draws per size")
    ap.add_argument("--eval-tokens", type=int, default=393216,
                    help="fixed held-out yardstick, from its own disjoint partition")
    ap.add_argument("--coords", default="hadamard")
    ap.add_argument("--ranks", default="8,32,128")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--cd-sweeps", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/raw/calib_size.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    projs = args.projs.split(",")
    ranks = [int(x) for x in args.ranks.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    seqlens = [int(x) for x in args.seqlens.split(",")]

    model, tok = load_model(args.model, device=args.device, dtype=torch.float32)
    refs = [r for r in list_linear_layers(model) if r.layer_idx in layers and r.proj in projs]
    W0 = {r.name: get_module(model, r.name).weight.detach().float().clone() for r in refs}

    # One pool, cut into disjoint partitions: `draws` for the calibration draws and one more
    # for the yardstick, so no two of them share a single token.
    n_parts = args.draws + 1
    need = n_parts * (max(budgets) + max(seqlens)) * 2
    stream = token_stream(tok, args.corpus, max(args.pool_tokens, need))
    print("%d modules | budgets %s | seqlens %s | %d disjoint draws + held-out | pool %d tokens"
          % (len(refs), budgets, seqlens, args.draws, stream.numel()))

    fh = open(args.out, "a", encoding="utf-8")
    eval_cache = {}

    for seqlen in seqlens:
        n_eval = budget_to_nseq(args.eval_tokens, seqlen)
        print("collecting the held-out yardstick at seqlen %d (%d windows) ..."
              % (seqlen, n_eval))
        A_eval = collect_A(model, refs, stream, n_parts, args.draws, seqlen, n_eval,
                           args.device)
        for budget in budgets:
            n_seq = budget_to_nseq(budget, seqlen)
            t0 = time.time()
            draws = [collect_A(model, refs, stream, n_parts, d, seqlen, n_seq, args.device,
                               seed=d) for d in range(args.draws)]
            for r in refs:
                name = r.name
                Ad = [draws[d][name].to(args.device) for d in range(args.draws)]
                Ae = A_eval[name].to(args.device)

                geo = [subspace_overlap(Ad[i], Ad[j], ranks)
                       for i, j in itertools.combinations(range(args.draws), 2)]
                gm = {k: statistics.mean(g[k] for g in geo) for k in geo[0]}
                rec = dict(module=name, stage="geometry", tokens=budget, seqlen=seqlen,
                           n_seq=n_seq, dim=int(Ad[0].shape[0]), draws=args.draws,
                           tokens_per_dim=budget / float(Ad[0].shape[0]),
                           **{("pair_" + k): v for k, v in gm.items()})
                fh.write(json.dumps(rec) + "\n")
                print("  %7d tok  L%-4d %-16s dim %4d  subspace %s | diag r %.3f | offdiag r %.3f"
                      % (budget, seqlen, name.split("layers.")[-1], rec["dim"],
                         " ".join("%.3f" % gm["subspace%d" % k] for k in ranks
                                  if "subspace%d" % k in gm),
                         gm["diag_corr"], gm["offdiag_corr"]))

                W = W0[name].to(args.device)
                for coord in args.coords.split(","):
                    if coord == "hadamard":
                        rp = RotationPair(W.shape[0], W.shape[1], seed=0, device=args.device)
                        Wt = rp.forward_w(W)
                        back = rp.inverse_w
                        rot = lambda M: 0.5 * (rp.forward_h(M) + rp.forward_h(M).T)  # noqa
                    else:
                        Wt, back, rot = W, (lambda X: X), (lambda M: M)
                    At = [rot(a) for a in Ad]
                    den_e = max(damage(W, Ae), 1e-30)
                    for bits, group in LADDER:
                        if W.shape[1] % group:
                            continue
                        fit, held, codes = [], [], []
                        for d in range(args.draws):
                            _, st = quantize_with_state(Wt, 2.0 * At[d], bits, group,
                                                        args.percdamp)
                            stc = st.clone()
                            Wc, _ = cd_refine(Wt, At[d], stc, None,
                                              sweeps=args.cd_sweeps, seed=0)
                            Dn = W - back(Wc)
                            fit.append(damage(Dn, Ad[d]) / max(damage(W, Ad[d]), 1e-30))
                            held.append(damage(Dn, Ae) / den_e)
                            codes.append(stc.codes)
                            del st, stc, Wc, Dn
                        dis = [float((codes[i] != codes[j]).float().mean())
                               for i, j in itertools.combinations(range(args.draws), 2)]
                        rec = dict(module=name, stage="solution", tokens=budget,
                                   seqlen=seqlen, n_seq=n_seq, coord=coord, bits=bits,
                                   group=group, bpw=exact_bpw(bits, group),
                                   draws=args.draws, dim=int(Ad[0].shape[0]),
                                   tokens_per_dim=budget / float(Ad[0].shape[0]),
                                   fit_mean=statistics.mean(fit),
                                   held_mean=statistics.mean(held),
                                   held_spread=max(held) - min(held),
                                   held_rel_spread=(max(held) - min(held))
                                   / max(statistics.mean(held), 1e-30),
                                   held_std=(statistics.pstdev(held) if len(held) > 1 else 0.0),
                                   gap_ratio=statistics.mean(held)
                                   / max(statistics.mean(fit), 1e-30),
                                   codes_differ=statistics.mean(dis) if dis else 0.0)
                        fh.write(json.dumps(rec) + "\n")
                        del codes
                    fh.flush()
                    del At
                del W, Ad, Ae
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            del draws
            print("  budget %d @ seqlen %d done in %.0fs" % (budget, seqlen, time.time() - t0))
        del A_eval
    fh.close()
    print("peak VRAM %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20
                                  if args.device == "cuda" else 0))


if __name__ == "__main__":
    main()
