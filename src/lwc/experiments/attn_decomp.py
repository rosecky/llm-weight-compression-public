"""Mechanistic decomposition of the module-scope gain inside attention (layer-level study).

Four objective geometries quantize the SAME two matrices (q_proj, k_proj; v/o stay fp so the
comparison is exactly controlled), at the same rate, rotation, calibration text and solver:

    layer     G = I                       (A0: what layer-CD optimizes)
    logits    G at Z = QK^T/sqrt(d)       (A1: exact for each matrix given the other)
    softmax   Fisher of attn distribution (A2: BoA-like local softmax metric)
    attn_out  G at Y = softmax(Z) V       (A5-pre: adds the V-weighting -- the H-value term)

Q<->K cooperation is sequential and honest: refine q against G measured with the current
quantized k, re-measure, refine k, and repeat (`--rounds`). All arms then get scored on
held-out text at every rung of the ladder: layer damage, logit MSE, softmax KL, attention-
output MSE, and top-1 flip / margin statistics -- so "best on its own objective" and "best
downstream" are visibly different columns, which is the entire point.

Diagnostics (section 3/9 of the brief): per (head, query-row) properties of the fp state --
top-logit, top1-top2 margin, entropy, max prob, Jacobian trace 1 - sum p^2, and the p-weighted
value dispersion s_V = sum_j p_j ||V_j - V_bar||^2 -- are correlated (Spearman) against the
realized per-row softmax KL and attention-output damage of the layer-quantized state (H-high /
H-margin / H-entropy / H-value). The section-9 test asks: at matched KL (per-decile), does
s_V explain the remaining spread in output damage? If yes, preserving the distribution is the
wrong target when the competing values agree -- the mechanism by which the functional
endpoint can beat softmax preservation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict

import torch
import torch.nn.functional as F

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from ..attnscope import attn_G, verify_attn_map, _heads
from ..calib import get_wikitext2
from ..joint import DenseMetric, cd_refine, damp_dense, metric_identity, quantize_with_state
from ..modelio import load_model
from ..rotate import RotationPair
from .factorial_ppl import capture_layer0_inputs, get_decoder_layers


@torch.no_grad()
def attn_state(layer, hidden, kw, Wq=None, Wk=None):
    """(Z, P, Y, V) of the layer's attention with optionally substituted q/k weights."""
    attn = layer.self_attn
    Hq, Hk, dh = _heads(attn)
    rep = Hq // Hk
    x = layer.input_layernorm(hidden)
    b, t, _ = x.shape
    cos, sin = kw["position_embeddings"]
    q = F.linear(x, Wq if Wq is not None else attn.q_proj.weight, attn.q_proj.bias)
    k = F.linear(x, Wk if Wk is not None else attn.k_proj.weight, attn.k_proj.bias)
    v = F.linear(x, attn.v_proj.weight, attn.v_proj.bias)
    q = q.view(b, t, Hq, dh).transpose(1, 2)
    k = k.view(b, t, Hk, dh).transpose(1, 2)
    v = repeat_kv(v.view(b, t, Hk, dh).transpose(1, 2), rep)
    q, k = apply_rotary_pos_emb(q, k, cos[:1], sin[:1])
    k = repeat_kv(k, rep)
    Z = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
    mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril_()
    Zm = Z.masked_fill(~mask, float("-inf"))
    P = torch.softmax(Zm, dim=-1)
    return Z, P, P @ v, v, mask


@torch.no_grad()
def ladder_metrics(layer, hidden, kw, Wq, Wk) -> Dict:
    Z0, P0, Y0, _, mask = attn_state(layer, hidden, kw)
    Z1, P1, Y1, _, _ = attn_state(layer, hidden, kw, Wq=Wq, Wk=Wk)
    dZ = ((Z1 - Z0).pow(2) * mask).sum() / mask.sum() / Z0.shape[0] / Z0.shape[1]
    kl = (P0.clamp_min(1e-12) * (P0.clamp_min(1e-12).log()
                                 - P1.clamp_min(1e-12).log())).sum(-1).mean()
    dY = (Y1 - Y0).pow(2).mean()
    top0 = Z0.masked_fill(~mask, float("-inf")).argmax(-1)
    top1 = Z1.masked_fill(~mask, float("-inf")).argmax(-1)
    # margins only where at least two positions are visible
    flips = (top0 != top1).float().mean()
    return dict(logit_mse=float(dZ), softmax_kl=float(kl), attnout_mse=float(dY),
                top1_flips=float(flips))


@torch.no_grad()
def row_diagnostics(layer, hidden, kw, Wq, Wk, max_rows: int = 20000) -> Dict:
    """Per-(head, query) fp-state properties vs realized damage of the quantized state."""
    Z0, P0, Y0, V, mask = attn_state(layer, hidden, kw)
    _, P1, Y1, _, _ = attn_state(layer, hidden, kw, Wq=Wq, Wk=Wk)
    b, H, t, _ = Z0.shape
    Zm = Z0.masked_fill(~mask, float("-inf"))
    z_sorted = Zm.sort(-1, descending=True).values
    zmax = z_sorted[..., 0]
    margin = z_sorted[..., 0] - z_sorted[..., 1].clamp_min(-1e30)
    ent = -(P0.clamp_min(1e-12) * P0.clamp_min(1e-12).log()).sum(-1)
    pmax = P0.max(-1).values
    jtr = 1.0 - (P0 * P0).sum(-1)
    Vbar = P0 @ V                                             # (b,H,t,dh) p-weighted mean
    # s_V = sum_j p_j ||V_j - Vbar||^2 = sum_j p_j ||V_j||^2 - ||Vbar||^2
    sV = (P0 @ V.pow(2).sum(-1, keepdim=True)).squeeze(-1) - Vbar.pow(2).sum(-1)
    kl_row = (P0.clamp_min(1e-12) * (P0.clamp_min(1e-12).log()
                                     - P1.clamp_min(1e-12).log())).sum(-1)
    dY_row = (Y1 - Y0).pow(2).sum(-1)

    keep = torch.arange(t, device=Z0.device) >= 2             # rows with >=3 positions
    def flat(x):
        return x[:, :, keep].reshape(-1).float()
    props = dict(zmax=flat(zmax), margin=flat(margin), entropy=flat(ent),
                 pmax=flat(pmax), jtrace=flat(jtr), v_disp=flat(sV))
    kl_f, dy_f = flat(kl_row), flat(dY_row)
    n = kl_f.numel()
    if n > max_rows:
        idx = torch.randperm(n)[:max_rows]
        kl_f, dy_f = kl_f[idx], dy_f[idx]
        props = {k: v[idx] for k, v in props.items()}

    def spearman(a, b):
        ra = a.argsort().argsort().float()
        rb = b.argsort().argsort().float()
        return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])

    out = {}
    for k, v in props.items():
        out["rho_%s_kl" % k] = spearman(v, kl_f)
        out["rho_%s_dY" % k] = spearman(v, dy_f)
    # section 9: within KL deciles, does value dispersion explain output damage?
    dec = torch.quantile(kl_f, torch.linspace(0, 1, 11))
    rs = []
    for i in range(10):
        m = (kl_f >= dec[i]) & (kl_f <= dec[i + 1])
        if int(m.sum()) > 50:
            rs.append(spearman(props["v_disp"][m], dy_f[m]))
    out["rho_vdisp_dY_at_fixed_kl"] = float(sum(rs) / len(rs)) if rs else float("nan")
    out["n_rows"] = int(kl_f.numel())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="5,11,17")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--calib-seq", type=int, default=16)
    ap.add_argument("--val-seq", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--sweeps", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2, help="q<->k alternations with G refresh")
    ap.add_argument("--n-probe", type=int, default=2)
    ap.add_argument("--g-tokens", type=int, default=2048)
    ap.add_argument("--gdamp", type=float, default=1.0)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/attn_decomp.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = args.device

    model, tok = load_model(args.model, device=dev, dtype=torch.float32)
    layers, _ = get_decoder_layers(model)
    ids = get_wikitext2(tok, seqlen=args.seqlen, n_seq=args.calib_seq + args.val_seq,
                        split="train", seed=args.seed)
    hidden, kw = capture_layer0_inputs(model, ids, dev, batch=2)
    fh = open(args.out, "a", encoding="utf-8")
    targets = [int(x) for x in args.layers.split(",")]

    print("attention-map fidelity at layer %d: %.2e"
          % (targets[0], verify_attn_map(layers[targets[0]], hidden[:2], kw)))

    for li in range(max(targets) + 1):
        layer = layers[li]
        if li in targets:
            hc, hv = hidden[:args.calib_seq], hidden[args.calib_seq:]
            attn = layer.self_attn
            with torch.no_grad():
                xc = layer.input_layernorm(hc).reshape(-1, hc.shape[-1])
                A0 = (xc.T @ xc) / xc.shape[0]
            W0 = {p: getattr(attn, p).weight.detach().float().clone()
                  for p in ("q_proj", "k_proj")}

            for arm in ("layer", "logits", "softmax", "attn_out"):
                t0 = time.time()
                rp = {p: RotationPair(W0[p].shape[0], W0[p].shape[1], seed=args.seed,
                                      device=dev) for p in W0}
                Wt = {p: rp[p].forward_w(W0[p]) for p in W0}
                Ar = {p: 0.5 * (rp[p].forward_h(A0) + rp[p].forward_h(A0).T) for p in W0}
                st, Wh = {}, {}
                for p in ("q_proj", "k_proj"):
                    Wh[p], st[p] = quantize_with_state(Wt[p], 2.0 * Ar[p], args.bits,
                                                       args.group, args.percdamp)
                    cd_refine(Wt[p], Ar[p], st[p], metric_identity(),
                              sweeps=args.sweeps, seed=args.seed)
                # arm-specific refinement with G refreshed against the partner's current state
                if arm != "layer":
                    for _ in range(args.rounds):
                        for p in ("q_proj", "k_proj"):
                            bak = {q: getattr(attn, q + "").weight.data.clone()
                                   for q in ("q_proj", "k_proj")}
                            for q in ("q_proj", "k_proj"):
                                s_full, z_full = st[q].expand()
                                dq = rp[q].inverse_w(
                                    st[q].codes.float() * s_full + z_full)
                                getattr(attn, q).weight.data.copy_(dq)
                            Gs = attn_G(layer, hc, kw,
                                        {"logits": "logits", "softmax": "softmax",
                                         "attn_out": "attn_out"}[arm],
                                        batch=2, n_probe=args.n_probe, seed=args.seed,
                                        device=dev, max_tokens=args.g_tokens)
                            Gd = rp[p].left.rotate_hessian(Gs["self_attn." + p])
                            Gm = DenseMetric(damp_dense(Gd, args.gdamp))
                            cd_refine(Wt[p], Ar[p], st[p], Gm,
                                      sweeps=args.sweeps, seed=args.seed)
                            for q in ("q_proj", "k_proj"):
                                getattr(attn, q).weight.data.copy_(bak[q])
                            del Gs, Gd, Gm
                Wq = {}
                for p in ("q_proj", "k_proj"):
                    s_full, z_full = st[p].expand()
                    Wq[p] = rp[p].inverse_w(st[p].codes.float() * s_full + z_full)

                rec = dict(layer=li, arm=arm, bits=args.bits, group=args.group,
                           secs=time.time() - t0)
                for tag, hh in (("cal", hc), ("val", hv)):
                    m = ladder_metrics(layer, hh, kw, Wq["q_proj"], Wq["k_proj"])
                    rec.update({tag + "_" + k: v for k, v in m.items()})
                for p in ("q_proj", "k_proj"):
                    D = W0[p] - Wq[p]
                    rec["dmg_%s" % p] = float(((D @ A0) * D).sum())
                if arm == "layer":
                    rec.update(row_diagnostics(layer, hv, kw, Wq["q_proj"], Wq["k_proj"]))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print("L%-3d %-9s | val: logitMSE %.4e  KL %.4e  dY %.4e  flips %.4f | %ds"
                      % (li, arm, rec["val_logit_mse"], rec["val_softmax_kl"],
                         rec["val_attnout_mse"], rec["val_top1_flips"], rec["secs"]))
        with torch.no_grad():
            outs = []
            for i in range(0, hidden.shape[0], 2):
                o = layer(hidden[i:i + 2], **{k: v for k, v in kw.items()
                                              if k not in ("past_key_value",
                                                           "past_key_values")})
                outs.append(o[0] if isinstance(o, tuple) else o)
            hidden = torch.cat(outs, 0)
    fh.close()


if __name__ == "__main__":
    main()
