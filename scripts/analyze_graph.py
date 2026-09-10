"""Tables for the graph-aware quantization study, plus two measurements that need the cached
geometry rather than a results file:

* **stability in k** -- does the community structure survive changing how many edges per
  vertex are kept?
* **cross-layer reuse** -- q_proj, k_proj, v_proj, gate_proj and up_proj in *every* block read
  the same residual-stream channels, so their A-graph partitions are directly comparable.
  If functional communities were a property of the model rather than of one layer's
  calibration noise, those partitions should agree. Adjusted Rand index, against the
  chance level of 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import torch

import sys
sys.path.insert(0, "src")
from lwc.graphstruct import affinity, balanced_labels, block_energy_ratio, spectral_order, topk_sparsify


def adjusted_rand(a: torch.Tensor, b: torch.Tensor) -> float:
    """ARI between two labelings of the same vertices. 0 = chance, 1 = identical."""
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    n = a.numel()
    M = torch.zeros(ka, kb, dtype=torch.float64)
    M.index_put_((a.cpu(), b.cpu()), torch.ones(n, dtype=torch.float64), accumulate=True)
    c2 = lambda x: x * (x - 1) / 2
    sij = c2(M).sum()
    sa = c2(M.sum(1)).sum()
    sb = c2(M.sum(0)).sum()
    tot = c2(torch.tensor(float(n)))
    exp = sa * sb / tot
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 0.0


def mean(v, k):
    return sum(x[k] for x in v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphcal", default="cache/graphcal.pt")
    ap.add_argument("--out-md", default="results/tables/graph.md")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    L = []

    # ---------------------------------------------------------------- G2 structure
    R = [json.loads(l) for l in open("results/raw/graph_structure.jsonl", encoding="utf-8")]
    agg = defaultdict(list)
    for r in R:
        if r["block"] == 128:
            agg[(r["side"], r["kind"], r["null"])].append(r)
    L += ["# Graph-aware quantization", "", "## G2 -- topology against null models", "",
          "Balanced communities of 128 channels from spectral sequencing; `block energy` is the",
          "fraction of off-diagonal edge energy inside blocks, `lift` is that over a random",
          "balanced partition. The **spectral** null keeps the entire eigenvalue spectrum and",
          "randomises only which channels the eigenmodes live on.", "",
          "| side | graph | null | block energy | lift | modularity | eff. support / n |",
          "|---|---|---|---|---|---|---|"]
    for k, v in sorted(agg.items()):
        L.append("| %s | %s | %s | %.4f | %.2f | %.4f | %.4f |"
                 % (k[0], k[1], k[2], mean(v, "block_energy"), mean(v, "energy_lift"),
                    mean(v, "modularity"), mean(v, "eff_support_frac")))
    L += ["", "### Real graph vs simple orderings (block energy captured, block=128)", "",
          "| side | graph | spectral | scale-sorted | contiguous | random |",
          "|---|---|---|---|---|---|"]
    agg2 = defaultdict(list)
    for r in R:
        if r["block"] == 128 and r["null"] == "real":
            agg2[(r["side"], r["kind"])].append(r)
    for k, v in sorted(agg2.items()):
        L.append("| %s | %s | %.4f | %.4f | %.4f | %.4f |"
                 % (k[0], k[1], mean(v, "block_energy"), mean(v, "block_energy_scale"),
                    mean(v, "block_energy_contig"), mean(v, "block_energy_random")))

    # ---------------------------------------------------------------- G1 edges
    E = [json.loads(l) for l in open("results/raw/graph_edges.jsonl", encoding="utf-8")]
    ea = defaultdict(list)
    for r in E:
        ea[(r["coord"], r["comp"])].append(r)
    L += ["", "## G1 -- do edges predict real error interaction?", "",
          "Exact decomposition of `tr(dW A dW^T G)` into per-channel terms and pair",
          "interactions, full population, 12 matrices.", "",
          "| coords | comp | axis | additive/total | signed I/total | \\|I\\|/total | "
          "Spearman(edge,\\|I\\|) | held-out R2 |",
          "|---|---|---|---|---|---|---|---|"]
    for (c, cp), v in sorted(ea.items()):
        for ax in ("in", "out"):
            add = sum(x["%s_cov_additive" % ax] / x["%s_cov_total" % ax] for x in v) / len(v)
            L.append("| %s | %s | %s | %.3f | %.3f | %.1f | %.3f | %.3f |"
                     % (c, cp, ax, add, mean(v, "%s_cov_interaction_signed_frac" % ax),
                        mean(v, "%s_cov_interaction_abs_frac" % ax),
                        mean(v, "%s_cov_spearman_edge_absI" % ax),
                        mean(v, "%s_cov_heldout_r2" % ax)))

    # ---------------------------------------------------------------- G3/G4 quantization
    Q = [json.loads(l) for l in open("results/raw/graph_quant.jsonl", encoding="utf-8")]
    qa = defaultdict(list)
    for r in Q:
        qa[(r["shape"], r["coord"], r["comp"], r["in_ord"], r["out_ord"])].append(r)
    L += ["", "## G3 / G4 -- layer-wise quantization at 3.25 bpw", "",
          "`fisher` is `tr(dW A dW^T G) / tr(W A W^T G)`; lower is better. `lift` is against",
          "the contiguous ordering in the same cell.", "",
          "| block | coords | comp | in order | out order | fisher | lift | perm bits/weight |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(qa.items()):
        base = qa.get((k[0], k[1], k[2], "contig", "contig"))
        b = mean(base, "fisher_err") if base else float("nan")
        L.append("| %s | %s | %s | %s | %s | %.5f | %.3f | %.4f |"
                 % (k[0], k[1], k[2], k[3], k[4], mean(v, "fisher_err"),
                    b / max(mean(v, "fisher_err"), 1e-30), mean(v, "perm_bpw")))

    # ---------------------------------------------------------------- end to end
    P = [json.loads(l) for l in open("results/raw/graph_endtoend.jsonl", encoding="utf-8")]
    ppl = [r for r in P if r.get("mode") == "ppl"]
    if ppl:
        base = ppl[0]["base_ppl"]
        L += ["", "## End-to-end perplexity, full-model sequential GPTQ", "",
              "| coords | order | bpw | ppl | d ppl | lift over contiguous |",
              "|---|---|---|---|---|---|"]
        for coord in ("hadamard", "native"):
            ref = [r for r in ppl if r["coord"] == coord and r["order"] == "contig"]
            rd = ref[0]["ppl_delta"] if ref else float("nan")
            for r in [x for x in ppl if x["coord"] == coord]:
                L.append("| %s | %s | %.4f | %.4f | %+.4f | %.3f |"
                         % (coord, r["order"], r["bpw"], r["ppl"], r["ppl_delta"],
                            rd / r["ppl_delta"]))
        L += ["", "fp16 baseline perplexity %.4f." % base]
    cross = [r for r in P if r.get("mode") == "crosslayer"]
    if cross:
        L += ["", "## G6 -- cross-layer error interaction (end-to-end, independent 3-bit)", "",
              "| layers | distance | d ppl A | d ppl B | d ppl both | interaction | % of additive |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(cross, key=lambda x: x["dist"]):
            L.append("| %d, %d | %d | %+.4f | %+.4f | %+.4f | %+.4f | %.1f%% |"
                     % (r["a"], r["b"], r["dist"], r["d_a"], r["d_b"], r["d_both"],
                        r["interaction"], 100 * r["inter_frac"]))
        adj = [r for r in cross if r["dist"] == 1]
        far = [r for r in cross if r["dist"] >= 5]
        L += ["", "adjacent (distance 1): %.1f%% of additive damage; distant (>=5): %.1f%%."
              % (100 * mean(adj, "inter_frac"), 100 * mean(far, "inter_frac"))]
    pc = [r for r in P if r.get("mode") == "permcheck"]
    if pc:
        L += ["", "## Permutation legality", "",
              "Permuting the MLP intermediate of all %d blocks (down_proj columns together with"
              % pc[0]["n_blocks"],
              "up_proj and gate_proj rows) changes the fp32 logits by **%.1e** -- round-off."
              % pc[0]["logits_rel_change"]]

    # ---------------------------------------------------------------- stability + reuse
    geo = torch.load(args.graphcal, map_location="cpu")
    dev = args.device
    L += ["", "## Community stability", "",
          "### In the sparsification level k (layer 11, A-correlation graph, block 128)", "",
          "| matrix | k=8 vs 32 (ARI) | k=32 vs 128 (ARI) | block energy k=8 / 32 / 128 |",
          "|---|---|---|---|"]
    for name in sorted(geo):
        if ".11." not in name:
            continue
        A = geo[name]["A"].to(dev, torch.float32)
        labs, be = {}, {}
        for k in (8, 32, 128):
            S = topk_sparsify(affinity(A, "corr"), k)
            lab = balanced_labels(spectral_order(S), 128)
            labs[k] = lab
            be[k] = block_energy_ratio(S, lab)
        L.append("| %s | %.3f | %.3f | %.3f / %.3f / %.3f |"
                 % (name.split(".", 2)[-1], adjusted_rand(labs[8], labs[32]),
                    adjusted_rand(labs[32], labs[128]), be[8], be[32], be[128]))
        del A

    L += ["", "### Across layers (same residual-stream channels, so directly comparable)", "",
          "| projection | layers | ARI | chance |", "|---|---|---|---|"]
    for proj in ("q_proj", "up_proj"):
        names = sorted(n for n in geo if n.endswith(proj))
        labs = {}
        for n in names:
            A = geo[n]["A"].to(dev, torch.float32)
            if A.shape[0] != 896:
                continue
            labs[n] = balanced_labels(
                spectral_order(topk_sparsify(affinity(A, "corr"), 32)), 128)
            del A
        ks = sorted(labs)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                L.append("| %s | %s vs %s | %.4f | 0.0000 |"
                         % (proj, ks[i].split(".")[2], ks[j].split(".")[2],
                            adjusted_rand(labs[ks[i]], labs[ks[j]])))

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    open(args.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-40:]))
    print("\nwrote %s" % args.out_md)


if __name__ == "__main__":
    main()
