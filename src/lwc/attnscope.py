"""Objective geometries at intermediate points INSIDE attention.

The module-scope result says optimizing q/k/v against the attention-module output transfers
end-to-end where layer-local objectives do not. This module builds the controlled ladder that
locates the mechanism:

    layer          Delta(y) reconstruction                      (existing baseline)
    logits         Z = Q K^T / sqrt(d), per head, causal        A1
    softmax        Fisher of the attention distribution         A2  (BoA-like local metric)
    attn_out       Y = softmax(Z) V, per head, pre-o_proj       A5-pre

Each endpoint is realised the same way as every wider scope in this project: probes give
`G = E[J^T J]` at the *outputs of q_proj / k_proj (/ v_proj)*, and the identical coordinate-
descent solver consumes it. Two exactness notes, stated rather than glossed:

  * For Q with K held at its current (already-quantized) value the map to Z is LINEAR, so the
    logit-endpoint geometry is exact, not a Taylor step; likewise for K given Q. The Q<->K
    cross term (second order in the joint perturbation) enters through sequential refinement,
    exactly as GPTQ handles cross-column terms.
  * The softmax endpoint uses Fisher sampling (key j ~ p per query row, loss = -log p_j), so
    `E[g g^T]` is the softmax-Jacobian metric `diag(p) - p p^T` pushed back to the q/k
    outputs -- the same mechanism the full-model horizon uses at the lm_head.

The Q/K/V -> Z -> Y map is rebuilt here from the layer's own weights (current, possibly
partially quantized state -- the honest geometry) using the installed transformers' rotary
and GQA helpers, and verified against the module's own forward before first use.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from .scopeg import _GradSink

ENDPOINTS = ("logits", "softmax", "attn_out")


def _heads(attn):
    Hq = attn.q_proj.out_features // attn.head_dim if hasattr(attn, "head_dim") else None
    dh = attn.head_dim if hasattr(attn, "head_dim") else \
        attn.q_proj.out_features // attn.config.num_attention_heads
    Hq = attn.q_proj.out_features // dh
    Hk = attn.k_proj.out_features // dh
    return Hq, Hk, dh


class _BlockGradSink:
    """Accumulates `sum_t g g^T` as per-head blocks instead of one dense matrix.

    At the pre-o_proj endpoint head h's output `Y_h = P_h V_h` depends only on that head's
    q/k/v, so `G = E[J^T J]` is EXACTLY block-diagonal with `head_dim`-wide blocks -- this is
    algebra, not an approximation. The reason it matters is estimation, not memory: the same
    probe sample gives `n / head_dim` samples per dimension instead of `n / out_features`
    (2048 probes: 32 vs 2.3 at 0.5B, 16 vs 1.3 at 1.5B), so the metric the optimizer sees is
    an order of magnitude better conditioned.
    """

    def __init__(self, n_heads: int, dh: int, device: str, frac: float = 1.0, seed: int = 0):
        self.C = torch.zeros(n_heads, dh, dh, device=device, dtype=torch.float32)
        self.n = 0
        self.frac = frac
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.n_heads, self.dh = n_heads, dh

    def add(self, Gm: torch.Tensor, chunk: int = 2048) -> None:
        V = Gm.reshape(-1, self.n_heads * self.dh)
        if self.frac < 1.0:
            k = max(1, int(round(V.shape[0] * self.frac)))
            sel = torch.randperm(V.shape[0], generator=self.g)[:k].to(V.device)
            V = V[sel]
        for i in range(0, V.shape[0], chunk):
            v = V[i:i + chunk].float().reshape(-1, self.n_heads, self.dh)
            self.C += torch.einsum("thi,thj->hij", v, v)
            self.n += v.shape[0]

    def dense(self) -> torch.Tensor:
        """Materialise the block-diagonal matrix so the solver stays unchanged."""
        n = self.n_heads * self.dh
        out = torch.zeros(n, n, device=self.C.device, dtype=self.C.dtype)
        for h in range(self.n_heads):
            s = h * self.dh
            out[s:s + self.dh, s:s + self.dh] = self.C[h]
        return out / max(self.n, 1)


@torch.no_grad()
def _noop():
    pass


def attn_G(layer, hidden: torch.Tensor, kw: Dict, endpoint: str, batch: int = 2,
           n_probe: int = 2, seed: int = 0, device: str = "cuda",
           max_tokens: int = 2048, block_diag: bool = False) -> Dict[str, torch.Tensor]:
    """G for q_proj/k_proj (and v_proj at the attn_out endpoint) at the chosen intermediate
    attention endpoint. Keys are module-local names ('self_attn.q_proj', ...) to match
    `block_G`'s convention.

    `block_diag` exploits the exact per-head structure of the attn_out endpoint (see
    `_BlockGradSink`); it is only valid there, because the logit and softmax endpoints are
    per-head too but o_proj-free, while any endpoint past o_proj genuinely mixes heads."""
    assert endpoint in ENDPOINTS
    assert not block_diag or endpoint == "attn_out", \
        "block-diagonal G is exact only at the pre-o_proj endpoint"
    attn = layer.self_attn
    Hq, Hk, dh = _heads(attn)
    rep = Hq // Hk
    g = torch.Generator(device=device).manual_seed(seed)

    names = ["self_attn.q_proj", "self_attn.k_proj"] + \
        (["self_attn.v_proj"] if endpoint == "attn_out" else [])
    mods = {n: dict(layer.named_modules())[n] for n in names}
    total_tok = n_probe * hidden.shape[0] * hidden.shape[1]
    frac = min(1.0, max_tokens / max(total_tok, 1))
    if block_diag:
        sinks = {n: _BlockGradSink(mods[n].out_features // dh, dh, device, frac, seed)
                 for n in names}
    else:
        sinks = {n: _GradSink(mods[n].out_features, device, frac, seed) for n in names}

    cos, sin = kw["position_embeddings"]
    with torch.enable_grad():
        for _ in range(n_probe):
            for i in range(0, hidden.shape[0], batch):
                hb = hidden[i:i + batch].detach()
                with torch.no_grad():
                    x = layer.input_layernorm(hb)
                b, t, _ = x.shape
                yq = F.linear(x, attn.q_proj.weight, attn.q_proj.bias
                              ).detach().requires_grad_(True)
                yk = F.linear(x, attn.k_proj.weight, attn.k_proj.bias
                              ).detach().requires_grad_(True)
                q = yq.view(b, t, Hq, dh).transpose(1, 2)
                k = yk.view(b, t, Hk, dh).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos[:1], sin[:1])
                k = repeat_kv(k, rep)
                Z = (q @ k.transpose(-1, -2)) / math.sqrt(dh)         # (b,Hq,t,t)
                mask = torch.ones(t, t, dtype=torch.bool, device=device).tril_()

                if endpoint == "logits":
                    v = (torch.randint(0, 2, Z.shape, generator=g, device=device,
                                       dtype=Z.dtype) * 2 - 1) * mask
                    loss = (Z * v).sum()
                elif endpoint == "softmax":
                    Zm = Z.masked_fill(~mask, float("-inf"))
                    logp = torch.log_softmax(Zm, dim=-1)
                    with torch.no_grad():
                        p = logp.exp().reshape(-1, t).clamp_min(0)
                        j = torch.multinomial(p, 1, generator=g).squeeze(1)
                        del p
                    loss = -(logp.reshape(-1, t).gather(1, j.unsqueeze(1))).sum()
                else:                                                  # attn_out
                    yv = F.linear(x, attn.v_proj.weight, attn.v_proj.bias
                                  ).detach().requires_grad_(True)
                    vv = repeat_kv(yv.view(b, t, Hk, dh).transpose(1, 2), rep)
                    Zm = Z.masked_fill(~mask, float("-inf"))
                    P = torch.softmax(Zm, dim=-1)
                    Y = P @ vv                                         # (b,Hq,t,dh)
                    v = (torch.randint(0, 2, Y.shape, generator=g, device=device,
                                       dtype=Y.dtype) * 2 - 1)
                    loss = (Y * v).sum()
                loss.backward()
                sinks["self_attn.q_proj"].add(yq.grad.detach())
                sinks["self_attn.k_proj"].add(yk.grad.detach())
                if endpoint == "attn_out":
                    sinks["self_attn.v_proj"].add(yv.grad.detach())
                    del yv
                del yq, yk, q, k, Z, loss, x, hb
    if block_diag:
        return {n: sinks[n].dense() for n in names}
    return {n: sinks[n].C / max(sinks[n].n, 1) for n in names}


@torch.no_grad()
def verify_attn_map(layer, hidden: torch.Tensor, kw: Dict) -> float:
    """Max |difference| between this module's Q/K/V->Y->o_proj reconstruction and the
    layer's own attention output. Run once before trusting any endpoint."""
    attn = layer.self_attn
    Hq, Hk, dh = _heads(attn)
    rep = Hq // Hk
    x = layer.input_layernorm(hidden)
    b, t, _ = x.shape
    cos, sin = kw["position_embeddings"]
    q = F.linear(x, attn.q_proj.weight, attn.q_proj.bias).view(b, t, Hq, dh).transpose(1, 2)
    k = F.linear(x, attn.k_proj.weight, attn.k_proj.bias).view(b, t, Hk, dh).transpose(1, 2)
    v = F.linear(x, attn.v_proj.weight, attn.v_proj.bias).view(b, t, Hk, dh).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos[:1], sin[:1])
    k, v = repeat_kv(k, rep), repeat_kv(v, rep)
    Z = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    Z = Z.masked_fill(~torch.ones(t, t, dtype=torch.bool, device=x.device).tril_(),
                      float("-inf"))
    Y = torch.softmax(Z, dim=-1) @ v
    out = F.linear(Y.transpose(1, 2).reshape(b, t, Hq * dh),
                   attn.o_proj.weight, attn.o_proj.bias)

    holder = {}
    h = attn.register_forward_hook(
        lambda m, i, o, s=holder: s.__setitem__("y", o[0] if isinstance(o, tuple) else o))
    try:
        kw2 = {kk: vv for kk, vv in kw.items()
               if kk not in ("past_key_value", "past_key_values")}
        if "use_cache" in kw2:
            kw2["use_cache"] = False
        layer(hidden, **kw2)
    finally:
        h.remove()
    return float((out - holder["y"]).abs().max())
