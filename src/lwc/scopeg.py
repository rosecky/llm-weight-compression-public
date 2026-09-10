"""The output metric of a *widened optimization scope*, measured rather than assumed.

A linear layer's own error is `tr(dW A dW^T)` -- the identity output metric, which is exactly
what GPTQ minimises and exactly why GPTQ can treat every output row as its own independent
problem. If instead we care about the error at some point further downstream, `z = f(y)`, then
to first order `dz = J dy` and the objective becomes

    tr(dW A dW^T G)     with   G = E[J^T J]

so *the entire content of "optimize over a wider scope" is a different G*, and the compensation
radius of section 8 is a ladder of horizons for the endpoint `z`:

    identity   z = y                      the layer's own output          (GPTQ's scope)
    module     z = attention / MLP output  Q,K,V,O jointly; gate,up,down jointly   (S3, S4)
    block K    z = output of block m+K-1   one, two, four transformer blocks (S5, S6, S7)
    nll        the model's own loss        the whole remaining network

`E[J^T J]` would need one backward pass per output channel to form exactly. It does not have to
be formed exactly: for Rademacher `v`, `E[J^T v v^T J] = E[J^T J]`, so one backward pass of
`L = sum_t v_t . z_t` contributes one rank-1 term *per token*. With thousands of calibration
tokens the estimate has ample rank after a handful of passes, and the variance is controlled by
`n_probe`.

Only the top eigenmodes are kept, because a dense `(out, out)` is 95 MiB for this model's MLP
matrices and 4.5 GiB across the network -- and because the rank that turns out to be *needed*
is itself one of the questions the brief asks (section 9.3).
"""
from __future__ import annotations

from typing import Dict

import torch


from .modelio import PROJ_TYPES

ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


class _GradSink:
    """Accumulates `sum_t g_t g_t^T` for one module, in fp32, over a capped token sample.

    The outer product costs `out^2` per token, which for this model's 4864-wide MLP matrices is
    24 GFLOP per 1000 tokens -- more than the forward and backward passes that produce the
    gradients. Since only the leading eigenmodes are ever used, a uniform random subsample of a
    couple of thousand tokens estimates them to far better precision than the probe noise
    already allows, so the cap costs nothing real and turns hours into minutes.
    """

    def __init__(self, n: int, device: str, frac: float = 1.0, seed: int = 0):
        self.C = torch.zeros(n, n, device=device, dtype=torch.float32)
        self.n = 0
        self.frac = frac
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def add(self, Gm: torch.Tensor, chunk: int = 2048) -> None:
        V = Gm.reshape(-1, Gm.shape[-1])
        if self.frac < 1.0:
            k = max(1, int(round(V.shape[0] * self.frac)))
            sel = torch.randperm(V.shape[0], generator=self.g)[:k].to(V.device)
            V = V[sel]
        for i in range(0, V.shape[0], chunk):
            v = V[i:i + chunk].float()
            self.C += v.T @ v
            self.n += v.shape[0]


def linear_children(layer) -> Dict[str, torch.nn.Linear]:
    out = {}
    for n, m in layer.named_modules():
        if isinstance(m, torch.nn.Linear):
            tail = n.split(".")[-1]
            if tail in PROJ_TYPES:
                out[n] = m
    return out


@torch.no_grad()
def _noop():
    pass


def block_G(layers, m: int, hidden: torch.Tensor, kw: Dict, horizon: str,
            batch: int = 2, n_probe: int = 2, seed: int = 0, device: str = "cuda",
            max_tokens: int = 2048, guard_mib: float = 512.0,
            tail=None, tail_tokens: int = 128,
            keep_projs=None) -> Dict[str, torch.Tensor]:
    """Dense `G = E[J^T J]` for the linear layers of one block, at the given horizon.

    Used *inside* the sequential quantization pipeline, where `hidden` already carries the
    error of every block quantized so far and the downstream blocks are still full precision.
    That is the honest reference geometry: it is what the wider objective would actually be
    optimising against.

    Dense rather than truncated on purpose. A rank-64 surrogate of this metric was measured to
    be worse than useless as an objective -- minimising it made the true downstream error 32x
    larger than not optimising at all -- because the optimizer simply walks into the
    surrogate's null space. Nothing is stored to disk, so the (out, out) matrices exist only
    for the duration of one block.

    Modules whose own output *is* the horizon endpoint return no entry: their `G` is the
    identity by construction, and estimating it from probes would only add noise.
    """
    # The pipeline's captured kwargs carry a DynamicCache. Under `no_grad` its growth is
    # harmless (verified: repeat calls give bit-identical outputs), but under `enable_grad`
    # the K/V it stores keep the first probe's graph alive, and the second probe's backward
    # then walks a freed graph. The probes therefore run cache-free.
    kw = {k: v for k, v in kw.items() if k not in ("past_key_value", "past_key_values")}
    if "use_cache" in kw:
        kw["use_cache"] = False
    L = len(layers)
    subs = linear_children(layers[m])
    if horizon == "module":
        targets = [("self_attn", [n for n in subs if n.split(".")[-1] in
                                  ("q_proj", "k_proj", "v_proj")]),
                   ("mlp", [n for n in subs if n.split(".")[-1] in
                            ("gate_proj", "up_proj")])]
    elif horizon.startswith("block"):
        span = max(1, min(int(horizon[5:]), L - m))
        targets = [(None, list(subs))]
    elif horizon == "model":
        # S3: the full-model KL geometry. The probe loss is -log p(y) with y sampled from
        # the model's own distribution, so E[g g^T] at each captured output is the exact
        # Fisher of the remaining network -- the YAQA-style objective, realised as the same
        # dense per-matrix G the narrower horizons use (their H_O is a Kronecker
        # approximation of exactly this quantity). `tail` must be (final_norm, lm_head).
        # The tail is evaluated on a token subsample: the Fisher estimate stays unbiased,
        # and the vocab-sized logits (the memory hazard) shrink by seqlen/tail_tokens.
        if tail is None:
            raise ValueError("horizon 'model' needs tail=(final_norm, lm_head)")
        span = L - m
        batch = 1
        targets = [("__model__", list(subs))]
    else:
        raise ValueError(horizon)

    if keep_projs is not None:
        # skip probe work for matrices whose G comes from elsewhere (e.g. the attention side
        # measured with its exact per-head structure)
        targets = [(ep, [k for k in ks if k.split(".")[-1] in keep_projs])
                   for ep, ks in targets]
    keys = sorted({k for _, ks in targets for k in ks})
    for k in keys:
        n = subs[k].out_features
        mib = n * n * 4 / 2 ** 20
        if mib > guard_mib:
            raise MemoryError("G for %s would need %.0f MiB (limit %.0f)"
                              % (k, mib, guard_mib))
    total_tok = n_probe * hidden.shape[0] * hidden.shape[1]
    frac = min(1.0, max_tokens / max(total_tok, 1))
    sinks = {k: _GradSink(subs[k].out_features, device, frac, seed + m) for k in keys}
    handles = []
    capture = {"on": set()}
    for k in keys:
        def fwd(module, inp, out, key=k):
            if key in capture["on"] and out.requires_grad:
                out.register_hook(lambda gr, kk=key: sinks[kk].add(gr.detach()))
        handles.append(subs[k].register_forward_hook(fwd))

    g = torch.Generator(device=device).manual_seed(seed + 97 * m)
    # The quantization pipeline runs under `no_grad`; the probes are the one place that needs
    # a graph, and only for activations -- every parameter stays frozen.
    try:
      with torch.enable_grad():
          for endpoint, ks in targets:
              if not ks:
                  continue
              for _ in range(n_probe):
                  for i in range(0, hidden.shape[0], batch):
                      capture["on"] = set(ks)
                      x = hidden[i:i + batch].detach().clone().requires_grad_(True)
                      if endpoint == "__model__":
                          h = x
                          for k2 in range(span):
                              o = layers[m + k2](h, **kw)
                              h = o[0] if isinstance(o, tuple) else o
                          sel = torch.randperm(h.shape[1], generator=g, device=device
                                               )[:tail_tokens]
                          logits = tail[1](tail[0](h[:, sel, :])).float()
                          V = logits.shape[-1]
                          with torch.no_grad():
                              p = torch.softmax(logits.reshape(-1, V), dim=-1)
                              y = torch.multinomial(p, 1, generator=g).squeeze(1)
                              del p
                          loss = torch.nn.functional.cross_entropy(
                              logits.reshape(-1, V), y, reduction="sum")
                          loss.backward()
                          capture["on"] = set()
                          del h, logits, loss, y, x
                          continue
                      if endpoint is not None:
                          holder = {}
                          sub = getattr(layers[m], endpoint)
                          hh = sub.register_forward_hook(
                              lambda mod, inp, out, h=holder: h.__setitem__(
                                  "z", out[0] if isinstance(out, tuple) else out))
                          try:
                              layers[m](x, **kw)
                          finally:
                              hh.remove()
                          z = holder["z"]
                      else:
                          h = x
                          for k2 in range(span):
                              o = layers[m + k2](h, **kw)
                              h = o[0] if isinstance(o, tuple) else o
                          z = h
                      v = (torch.randint(0, 2, z.shape, generator=g, device=device,
                                         dtype=z.dtype) * 2 - 1)
                      (z * v).sum().backward()
                      capture["on"] = set()
                      del z, x, v
    finally:
        for h_ in handles:
            h_.remove()
        capture["on"] = set()
    out = {k: sinks[k].C / max(sinks[k].n, 1) for k in keys}
    sinks.clear()
    return out


