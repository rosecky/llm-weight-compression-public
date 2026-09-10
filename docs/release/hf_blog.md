---
title: "Which objective should a post-training quantizer optimize? A pre-registered study, with the negatives left in"
thumbnail: /blog/assets/llm-weight-compression/thumbnail.png
authors:
  - user: honza-rosecky
---

# Which objective should a post-training quantizer optimize? A pre-registered study, with the negatives left in

*Code, per-window evidence and the full experiment log: [github.com/rosecky/llm-weight-compression-public](https://github.com/rosecky/llm-weight-compression-public). Evidence as a dataset: [honza-rosecky/llm-weight-compression-evidence](https://huggingface.co/datasets/honza-rosecky/llm-weight-compression-evidence).*

GPTQ and its descendants round each weight matrix to minimise the error at the output of
*that matrix*. Everyone knows that is a proxy. The question this study asks is narrow and,
we think, under-measured: **if you keep the solver, the bit budget and the calibration
data fixed and only change what the objective looks at — the layer, the attention logits,
the attention output, the whole attention/MLP module, the transformer block, or the
model's own loss — which target actually makes the quantized model better end-to-end?**

We answer it on small decoders (0.5–1.5B), under a protocol frozen before the runs, with
every calibration draw reported separately, and we say up front what turned out to be
prior art and what did not survive.

## The setup in one paragraph

Hadamard-rotated scalar INT3 with group 128 (exactly 3.25 bits per weight including
scales and rotation seeds), GPTQ as the rounding rule. Then a **post-pass**: exact
coordinate descent over the integer codes, block by block *inside* the sequential
quantisation, under the objective `tr(ΔW·A·ΔWᵀ·G)` where `A` is the usual input Gram and
`G = E[JᵀJ]` is the output-side metric of the chosen horizon, measured by Rademacher probes
on the calibration activations as they flow through the already-quantised prefix. The
post-pass changes integer values only: same file size, same layout, same kernel. It costs
20–30× the GPTQ time offline and nothing at inference. Evaluation is mean NLL per token on
the *full* wikitext-2 test (146 windows of 2048 tokens) and 512 fixed C4 validation
windows, plus a Monte-Carlo KL to the fp16 model; per-window NLLs are kept so every
comparison is a paired bootstrap within a calibration draw.

## What the module objective buys

| model | calibration draws | GPTQ → GPTQ + module post-pass (wt2 NLL) | quantization gap closed | perplexity |
|---|---|---|---|---|
| Qwen2.5-0.5B | 3 | 2.921 → 2.891 | **+8.7%**, 3/3 draws | −2.7% |
| Qwen3-0.6B | 2 | 3.543 → 3.505 | **+8.2%**, 2/2 | −3.7% |
| Llama-3.2-1B | 3 | 2.769 → 2.749 | **+4.6%**, 3/3 | −1.9% |
| Qwen2.5-1.5B | 2 | 2.549 → 2.555 | **−2.5%**, 0/2 | +0.6% |

"Gap closed" is the fraction of the GPTQ-vs-fp16 NLL gap recovered. All positive rows
have paired-bootstrap intervals strictly below zero in every draw, and the effect is 2.5–4.5×
the across-draw spread of the paired difference. The last row is a real, replicated
failure, and **nothing we tested predicts it**: not model size, family, matrix width,
samples per output dimension, RMSNorm extremity, depth, head dimension, depth/width
ratio or baseline damage. Qwen3-0.6B shares both remaining architectural features with
the failure and is our second-best result.

Converted to what a buyer cares about — bits saved at equal quality along the GPTQ
rate–distortion curve — the best case is **+0.084 bits per weight** at 3.25 bpw, +0.021
at 3.50, +0.023 at 4.25 (extrapolated), and the post-pass is harmful at 2.50 bpw where
the network is already degenerate. We had set 0.1–0.2 bits as the product threshold before
measuring. We are below it at every usable rate. This is a scientific result, not a
product.

## The ladder, and what on it is prior art

The interesting curve is the objective *horizon* at matched everything (one model, one
rate, one solver, one probe budget):

| objective horizon | gap closed |
|---|---|
| layer-local + coordinate descent | +0.5% |
| QKᵀ logits (BoA/BaKron-style, q/k only) | worst attention rung |
| attention output | better |
| **module output (attention + MLP)** | **+10.4%** (exploratory), +8.7% (frozen) |
| block output | +9.6% |
| full-model Fisher, same cheap budget | +0.2% |
| full-model Fisher, 4× calibration tokens in the metric | +5.5% |

Before publishing we were told, correctly, that the non-monotone shape and its
bias–variance explanation are **BRECQ (2021)** and **MREM (2022)**, and that **BaKron
(2026)** derives the MLP-output Hessian we use for gate/up in closed form and compares
local vs global objectives at one solver on Llama-3/Qwen3 up to 8B — where the ordering
flips from model to model and between Kronecker factorizations. Our prior-art note walks
through it projection by projection. What remains ours is smaller and, we hope, useful:

- **The rungs inside attention.** Logits → softmax → attention output → module, at one
  solver and budget. The logit objective that BoA and BaKron use for q/k is the *worst*
  attention rung; the attention-output rung is exactly block-diagonal per head.
- **Text-limited, not probe-limited.** Holding the calibration set fixed and varying only
  the metric estimator: 4× the Rademacher probes changes nothing (+0.004 [−0.001, +0.009]
  NLL), 4× the calibration tokens seen by the metric moves the full-model objective from
  0.3% to 5.5% of the gap, and its KL to fp16 moves with the NLL — so the global
  objective's poor rank at cheap budgets is data coverage, not estimator noise and not an
  objective/metric mismatch. The same holds for the attention endpoint (+0.6% → +6.6%)
  and, surprisingly, *not* for the whole module, which gets slightly worse at the higher
  budget.
- **Superadditivity, with an interval.** Refining the attention half alone closes +0.6%
  of the gap (CI covers zero), the MLP half alone +2.4%, both together +8.7%; the
  interaction is −0.020 [−0.026, −0.014] NLL.
- **Hard negatives with stated scope.** Layer-local coordinate descent is mildly *harmful*
  end-to-end on three models (−0.5 / −1.8 / −2.9% of the gap). The post-GPTQ rotation
  landscape is a flat ±1% plateau under the local objective (tangent, basin, derivative-free
  search and 200 random rotations all reach the same floor). Per-row rate allocation is
  worth ≤0.11 bits. A diagonal output metric changes exactly zero codes. A rank-64
  surrogate of the metric makes the true error 32× worse.

## A methodological point we would like others to steal

A fresh-metric audit — re-scoring the refined weights under an independent probe draw of
the *same* budget — passed on the model where the method fails. Two independent estimates
agree on their dominant directions and are wrong about the tail in the same way. An
out-of-sample check at the same estimator resolution validates *sampling*, not
*resolution*. If your objective is an estimate, check it at a higher resolution, not just
on fresh data.

Relatedly: a paired bootstrap over evaluation windows does not contain calibration-draw
variance. On our decoder the GPTQ level moves ~0.01 NLL between draws while the paired
module−GPTQ difference stays at −0.030 / −0.031 / −0.021, so the *difference* is far more
stable than the *level* — but you only know that after running the draws. Report every
draw.

## What happened at 2 bits

We also ran the post-pass on a 2.12-bpw vector-quantised Qwen2.5-0.5B built by a
per-block VQ quantiser from the embedding-quantization project (d=4, K=256, input-side
rotation; quantiser not released). The representation is the story there — the wt2 gap is
0.69 NLL versus 6.47 for scalar GPTQ at 2.50 bpw — and that is prior-art territory
(GPTVQ/VPTQ class). The post-pass missed our pre-registered criterion on wt2 (−3.8% of
the gap, n.s.) while being positive on C4 (+7.1%) and KL (−9.5%), and the reason is a
tail: better on 127 of 146 wt2 windows, catastrophically worse (+1.7 nats) on ten.

Every one of those ten windows starts with a digit-like token. The damage is uniform from
position 3 onward, so the cause sits in the first tokens: **the refined 2-bit model fails
to form an attention sink on some first tokens**, the objective cannot see it (its
calibration windows start mid-text, so those rows are outside the metric's support), and
a fixed `"\n\n"` prefix on every window removes it entirely — after which the post-pass
wins on both sets (−0.030 [−0.034, −0.025] wt2, −0.030 [−0.034, −0.026] C4) and the plain
checkpoint improves too. If you deploy a 2-bit model of this kind, a BOS or fixed prefix at
inference is not optional; inputs beginning with a number or a symbol will find the
failure for you.

## What we would tell a reader in a hurry

1. Refining GPTQ's codes against the *module* output is worth 5–9% of the quantization
   gap at ~3 bits on three of four small decoders, for free at inference, and we cannot
   predict the fourth. Below ~2.5 bits scalar the quadratic objective decouples from the
   metric.
2. The shape of the objective ladder is prior art; the measurements inside attention, the
   probes-vs-tokens separation, and the validation record are what this repository adds.
3. Everything above is reproducible from committed per-window data with
   `python scripts/analyze_valid.py --print`, `analyze_rd.py` and `analyze_d3.py`;
   anything not in `evidence/` should be treated as unsupported.

The experiments were run with Claude (Anthropic) acting as an autonomous research agent
under the author's direction; every pre-registration, every result and every correction —
several of them, recorded as such — is in `EXPERIMENT_LOG.md`.
