"""Is the geometry GPTQ exploits a property of the model, or of the particular text sample?

GPTQ compensates along directions defined by `A = E[x x^T]`, estimated from a few thousand
calibration tokens. For a 4864-wide input that is fewer samples than dimensions, so the question
is not rhetorical: if `A` from two disjoint text samples disagrees about which directions matter,
then the compensation is fitting the sample and a held-out gap is guaranteed no matter how good
the optimizer is.

This compares the second moments themselves rather than their downstream effect, so it separates
"the geometry is unstable" from "the optimizer overfits a stable geometry":

    diagonal        rank correlation of per-channel second moments -- the part AWQ-style scaling
                    and min-max grouping actually see
    spectrum        eigenvalue profile, and the rate at which it decays
    subspace        principal angles between the leading eigenspaces at several ranks; the mean
                    squared cosine is the fraction of one subspace that lies inside the other
    energy          how much of sample B's energy lives in sample A's top-k directions, which is
                    exactly the quantity a compensation step relies on being right about
    interaction     correlation of the predicted pairwise compensation terms `A_ij`, which is
                    what decides whether two channels' errors can cancel

Everything is a comparison between two *independent* estimates of the same object, so a low
number is evidence about the estimator, not about the model.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def rank(x):
        idx = torch.argsort(x)
        r = torch.empty_like(x)
        r[idx] = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
        return r
    ra, rb = rank(a.float()), rank(b.float())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-30))


@torch.no_grad()
def compare(A: torch.Tensor, B: torch.Tensor, ranks: List[int]) -> Dict:
    n = A.shape[0]
    out = {"dim": n}
    da, db = torch.diagonal(A), torch.diagonal(B)
    out["diag_spearman"] = spearman(da, db)
    out["diag_rel_l1"] = float((da - db).abs().sum() / da.abs().sum().clamp_min(1e-30))

    la, Ua = torch.linalg.eigh(A)
    lb, Ub = torch.linalg.eigh(B)
    la, lb = la.flip(0).clamp_min(0), lb.flip(0).clamp_min(0)
    Ua, Ub = Ua.flip(1), Ub.flip(1)
    ca = torch.cumsum(la, 0) / la.sum().clamp_min(1e-30)
    out["rank90"] = int((ca < 0.90).sum()) + 1
    out["rank99"] = int((ca < 0.99).sum()) + 1
    out["eig_spearman"] = spearman(la, lb)

    for r in ranks:
        if r >= n:
            continue
        Pa, Pb = Ua[:, :r], Ub[:, :r]
        s = torch.linalg.svdvals(Pa.T @ Pb)              # cosines of the principal angles
        out["subspace%d" % r] = float((s ** 2).mean())
        # how much of B's energy sits in A's leading directions, and vice versa
        out["energy_b_in_a%d" % r] = float(((Pa.T @ B) * Pa.T).sum()
                                           / torch.diagonal(B).sum().clamp_min(1e-30))
        out["energy_a_in_a%d" % r] = float(la[:r].sum() / la.sum().clamp_min(1e-30))

    # the off-diagonal entries are what make two channels' errors cancel
    m = ~torch.eye(n, dtype=torch.bool, device=A.device)
    oa, ob = A[m], B[m]
    out["offdiag_corr"] = float(torch.corrcoef(torch.stack([oa, ob]))[0, 1])
    na = A / torch.sqrt(da.clamp_min(1e-30).unsqueeze(0) * da.clamp_min(1e-30).unsqueeze(1))
    nb = B / torch.sqrt(db.clamp_min(1e-30).unsqueeze(0) * db.clamp_min(1e-30).unsqueeze(1))
    out["corr_offdiag_corr"] = float(torch.corrcoef(torch.stack([na[m], nb[m]]))[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cals", default="cache/graphcal.pt,cache/jointcal_val.pt",
                    help="two or more caches holding A for the same modules")
    ap.add_argument("--ranks", default="8,32,128")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/raw/calib_geometry.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    paths = [p for p in args.cals.split(",") if os.path.exists(p)]
    if len(paths) < 2:
        raise SystemExit("need at least two calibration caches; have %s" % paths)
    stores = [torch.load(p, map_location="cpu") for p in paths]
    ranks = [int(x) for x in args.ranks.split(",")]
    common = sorted(set(stores[0]).intersection(*[set(s) for s in stores[1:]]))
    print("%d modules shared across %d calibration sets" % (len(common), len(stores)))

    fh = open(args.out, "a", encoding="utf-8")
    for n in common:
        for i in range(len(stores)):
            for j in range(i + 1, len(stores)):
                A = stores[i][n]["A"].float().to(args.device)
                B = stores[j][n]["A"].float().to(args.device)
                rec = dict(module=n, cal_a=paths[i], cal_b=paths[j],
                           **compare(A, B, ranks))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print("  %-28s dim %4d | diag rho %.4f | subspace8 %.3f 32 %.3f 128 %.3f "
                      "| offdiag r %.3f"
                      % (n.split("layers.")[-1], rec["dim"], rec["diag_spearman"],
                         rec.get("subspace8", float("nan")),
                         rec.get("subspace32", float("nan")),
                         rec.get("subspace128", float("nan")), rec["offdiag_corr"]))
                del A, B
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    main()
