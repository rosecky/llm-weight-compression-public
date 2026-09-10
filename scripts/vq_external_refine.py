"""Run `lwc.vq.vq_refine` on an externally quantised VQ matrix (one .npz per linear).

Written for the embedding-quantization peer's capture format. Everything is in the basis
the weight was quantised in (their rotated input basis): pass `W` and `H` already rotated;
the output-side metric `G` is basis-independent.

npz keys (all float32 unless stated):
  W        (out, in)         original weights in the quantised basis
  H        (in, in)          input Gram / GPTQ Hessian in the same basis (optionally damped)
  codebook (K, d)            centroids in amplitude-normalised units: w = scale * codebook[idx]
  codes    (out, in // d)    int (any width); centroid index per d consecutive input columns
  scale    (out, in // group) frozen amplitudes, one per row per `group` input columns
  group    scalar int        columns per amplitude block (256 for the peer format)
  G        (out, out)        OPTIONAL output-side metric E[J^T J]; omitted -> layer-local
  gdamp    scalar            OPTIONAL trust-region damping for G (default 1.0)

Writes `<out>.npz` with the refined `codes`, plus `obj_before`, `obj_after` (the objective
tr(D H D^T G) with D = W - dequant), `n_changed`, and `secs`. Never touches scale/codebook.

usage: python scripts/vq_external_refine.py in.npz out.npz [--sweeps 6] [--order forward|random] [--device cuda]
       python scripts/vq_external_refine.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lwc.joint import DenseMetric, damp_dense, metric_identity  # noqa: E402
from lwc.vq import VQState, vq_refine  # noqa: E402


def load_state(z, device) -> VQState:
    C = torch.as_tensor(np.asarray(z["codebook"]), dtype=torch.float32, device=device)
    codes = torch.as_tensor(np.asarray(z["codes"]), dtype=torch.int16, device=device)
    scale = torch.as_tensor(np.asarray(z["scale"]), dtype=torch.float32, device=device)
    group = int(np.asarray(z["group"]))
    d = int(C.shape[1])
    out_f, nvec = codes.shape
    assert scale.shape[0] == out_f, "scale rows must match codes rows"
    assert (nvec * d) % group == 0 and group % d == 0, "group must be a multiple of d"
    assert scale.shape[1] == (nvec * d) // group, "scale has one entry per group of columns"
    assert int(codes.max()) < C.shape[0], "code index outside the codebook"
    return VQState(codes, scale, C, d, group)


def objective(W, H, st, G):
    D = W - st.dequant()
    GD = D if G is None else G @ D
    return float(((D @ H) * GD).sum())


def refine_file(path_in: str, path_out: str, sweeps: int, order: str, device: str):
    z = np.load(path_in)
    W = torch.as_tensor(np.asarray(z["W"]), dtype=torch.float32, device=device)
    H = torch.as_tensor(np.asarray(z["H"]), dtype=torch.float32, device=device)
    st = load_state(z, device)
    assert st.dequant().shape == W.shape, "codes*d must equal W's input width"
    G = None
    if "G" in z.files:
        G = torch.as_tensor(np.asarray(z["G"]), dtype=torch.float32, device=device)
        G = damp_dense(G, float(np.asarray(z["gdamp"])) if "gdamp" in z.files else 1.0)
    metric = metric_identity() if G is None else DenseMetric(G)
    # per-column-block codebooks stored concatenated: restrict each column vector to its
    # block's slice so the result stays representable in the peer's container
    cb_slice = None
    if "block_of_group" in z.files:
        bog = torch.as_tensor(np.asarray(z["block_of_group"]), dtype=torch.long, device=device)
        kpb = int(np.asarray(z["k_per_block"]))
        assert bog.numel() == st.codes.shape[1], "block_of_group must have one entry per column vector"
        cb_slice = torch.stack([bog * kpb, (bog + 1) * kpb], 1)
        c = st.codes.long()
        assert bool(((c >= cb_slice[:, 0]) & (c < cb_slice[:, 1])).all()), \
            "input codes outside their block's codebook slice"
    before = objective(W, H, st, G)
    codes0 = st.codes.clone()
    t0 = time.time()
    vq_refine(W, H, st, metric, sweeps=sweeps, order=order, cb_slice=cb_slice)
    secs = time.time() - t0
    after = objective(W, H, st, G)
    if cb_slice is not None:
        c = st.codes.long()
        assert bool(((c >= cb_slice[:, 0]) & (c < cb_slice[:, 1])).all()), \
            "refined codes left their block's codebook slice"
    n_changed = int((st.codes != codes0).sum())
    np.savez(path_out, codes=st.codes.cpu().numpy(), obj_before=before, obj_after=after,
             n_changed=n_changed, frac_changed=n_changed / codes0.numel(), secs=secs,
             sweeps=sweeps)
    print("%s: obj %.6g -> %.6g (ratio %.4f), %d codes changed (%.2f%%), %.1f s"
          % (os.path.basename(path_in), before, after, after / max(before, 1e-30),
             n_changed, 100.0 * n_changed / codes0.numel(), secs))
    return before, after, n_changed


def selftest(device: str):
    """Synthetic matrix in the peer's format: d=4, K=64, group 256, one codebook."""
    torch.manual_seed(0)
    out_f, in_f, d, K, group = 512, 1024, 4, 64, 256
    W = torch.randn(out_f, in_f)
    X = torch.randn(4096, in_f) @ torch.randn(in_f, in_f) / in_f ** 0.5     # correlated inputs
    H = X.T @ X / X.shape[0]
    H = H + 0.01 * torch.diagonal(H).mean() * torch.eye(in_f)
    scale = W.reshape(out_f, in_f // group, group).pow(2).mean(2).sqrt()
    V = (W / scale.repeat_interleave(group, 1)).reshape(-1, d)
    C = V[torch.randperm(V.shape[0])[:K]].clone()                            # crude codebook
    codes = torch.cdist(V, C).argmin(1).reshape(out_f, in_f // d)
    J = torch.randn(64, out_f)
    G = J.T @ J / 64
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vq_selftest.npz")
    np.savez(tmp, W=W.numpy(), H=H.numpy(), codebook=C.numpy(), codes=codes.numpy(),
             scale=scale.numpy(), group=group, G=G.numpy(), gdamp=1.0)
    b, a, n = refine_file(tmp, tmp.replace(".npz", "_out.npz"), sweeps=2, order="forward",
                          device=device)
    assert a <= b + 1e-6 and n > 0, "refinement must not increase the objective"
    with np.load(tmp.replace(".npz", "_out.npz")) as z:
        assert z["codes"].shape == codes.shape
    for p in (tmp, tmp.replace(".npz", "_out.npz")):
        os.remove(p)
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--sweeps", type=int, default=6)
    ap.add_argument("--order", default="forward", choices=["forward", "random"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.device)
        return
    if not (a.inp and a.out):
        ap.error("need in.npz and out.npz (or --selftest)")
    refine_file(a.inp, a.out, a.sweeps, a.order, a.device)


if __name__ == "__main__":
    main()
