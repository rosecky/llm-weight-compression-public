"""Input and output geometry of a linear layer: A = E[x x^T] and G = E[g g^T].

For `y = W x` the local second-order damage of a weight perturbation is, under the K-FAC
factorisation `H ~ G (x) A`,

    D(dW) ~ tr(dW A dW^T G)

which is the identity every gate in the graph study rests on, because it makes the
*interaction* between two perturbations exact rather than hypothetical:

    dW = e_i u^T  and  e_j v^T   (two output channels)  ->  I = 2 G_ij (u^T A v)
    dW = u e_i^T  and  v e_j^T   (two input channels)   ->  I = 2 A_ij (u^T G v)

So `A` is the natural graph over input channels and `G` the natural graph over output
channels, and neither is a metaphor: if the corresponding covariance entry is zero, the two
perturbations cannot interact at second order.

`G` needs gradients. We take them from the model's own LM loss on calibration text. All
parameters are frozen and the graph is anchored by detaching the embedding output and
requiring grad on it, so backward builds activation gradients only -- no parameter gradient
buffers, which is what makes this fit on a 6 GB laptop GPU.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .modelio import LayerRef, get_module


class CovAccumulator:
    """Bounded-memory accumulation of sum_t v_t v_t^T in fp32."""

    def __init__(self, n: int, device: str = "cuda"):
        self.C = torch.zeros(n, n, device=device, dtype=torch.float32)
        self.n = 0

    @torch.no_grad()
    def add(self, V: torch.Tensor, chunk: int = 4096) -> None:
        V = V.reshape(-1, V.shape[-1])
        for i in range(0, V.shape[0], chunk):
            v = V[i:i + chunk].float()
            self.C += v.T @ v
            self.n += v.shape[0]

    def finalize(self) -> torch.Tensor:
        return self.C / max(self.n, 1)


def collect_AG(model, refs: List[LayerRef], input_ids: torch.Tensor, device: str = "cuda",
               batch: int = 1, need_grad: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
    """Returns {module_name: {"A": (in,in), "G": (out,out)}} as fp32 CPU tensors.

    `G` is the empirical Fisher of the layer output under the model's own next-token loss.
    """
    names = [r.name for r in refs]
    accA = {n: CovAccumulator(get_module(model, n).in_features, device) for n in names}
    accG = {n: CovAccumulator(get_module(model, n).out_features, device) for n in names} \
        if need_grad else {}

    handles = []
    for n in names:
        m = get_module(model, n)

        # A tensor hook on the module *output* rather than `register_full_backward_hook`.
        # The module hook silently delivered an all-zero gradient for q_proj (whose output is
        # reshaped and consumed by the rotary/attention path), which would have made its
        # output graph identically empty. A hook on the tensor itself is unambiguous: it
        # receives dL/dout, summed over every use of that tensor.
        def fwd(mod, inp, out, key=n):
            accA[key].add(inp[0].detach())
            if need_grad and out.requires_grad:
                out.register_hook(lambda g, k=key: accG[k].add(g.detach()))
        handles.append(m.register_forward_hook(fwd))

    anchor = None
    if need_grad:
        emb = model.get_input_embeddings()

        def anchor_hook(mod, inp, out):
            return out.detach().requires_grad_(True)
        anchor = emb.register_forward_hook(anchor_hook)
        for p in model.parameters():
            p.requires_grad_(False)

    try:
        for i in range(0, input_ids.shape[0], batch):
            x = input_ids[i:i + batch].to(device)
            if need_grad:
                out = model(x, labels=x)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    model(x)
    finally:
        for h in handles:
            h.remove()
        if anchor is not None:
            anchor.remove()

    store = {}
    for n in names:
        d = {"A": accA[n].finalize().cpu()}
        if need_grad:
            d["G"] = accG[n].finalize().cpu()
        store[n] = d
        del accA[n]
        if need_grad:
            del accG[n]
        if device == "cuda":
            torch.cuda.empty_cache()
    return store
