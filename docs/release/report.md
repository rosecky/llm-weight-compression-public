# Module-scope refinement of GPTQ codes: a pre-registered study on small decoders

Jan Rosecký (independent), September 2026. Code, evidence and the full experiment log:
[github.com/rosecky/llm-weight-compression-public](https://github.com/rosecky/llm-weight-compression-public).
Companion to the [Thinletter](https://thinletter.io) embedding-quantization project.

## 1. Setting and claims

Post-training quantizers of the GPTQ family round each weight matrix to minimise the error
at the output of that matrix. We ask what changes when the integer codes GPTQ produces are
refined, block by block inside the same sequential pass, against the error at the output
of the enclosing attention or MLP *module* instead — with the bit budget, the storage
format, the solver family and the calibration data held fixed. The refinement changes
integer values only, so the quantized file, its layout and its inference kernel are
identical to GPTQ's; it costs 20–30× the GPTQ time offline and nothing at inference.

Claims, in decreasing strength:

1. On three of four small decoders (Qwen2.5-0.5B, Qwen3-0.6B, Llama-3.2-1B) the
   module-scope post-pass recovers 5–9% of the quantization loss at 3.25 bits per weight,
   consistently across independent calibration draws. On the fourth (Qwen2.5-1.5B) it is
   mildly harmful, replicated, and unexplained.
2. Inside the attention block the objective that BoA/BaKron use for q/k (the QKᵀ logit
   error) is the worst rung; the attention-output rung is better and is exactly
   block-diagonal per head.
3. The full-model objective's poor rank at cheap estimation budgets is caused by
   calibration-text coverage in the metric, not by probe noise, and not by an
   objective/metric mismatch.
4. Layer-local coordinate descent — refining against the matrix's own objective — is
   mildly harmful end-to-end on all three positive models.
5. At ~2 bits with a per-block vector-quantized representation, the post-pass improves
   typical text and breaks a tail of inputs that begin with a digit-like token: a
   first-token attention-sink failure that the objective cannot see and a fixed prefix
   removes.

The non-monotone "objective horizon" curve (layer < module > full model) and its
bias–variance explanation are prior art (BRECQ 2021, MREM 2022); BaKron (2026) derives the
MLP-output Hessian we use and compares local against global objectives at one solver up
to 8B. The repository's `docs/PRIOR_ART_LADDER.md` places every claim above against those
papers projection by projection. Converted to bits at equal quality, the gain is at most
~0.08 bits per weight, below the 0.1–0.2 product threshold we set before measuring.

## 2. Method (what is run)

**Representation.** Randomized Hadamard rotation pair per matrix (seed 0, both sides),
asymmetric min-max INT3 with group 128: exactly 3.25 bits per weight including scales and
a 32-bit rotation seed per matrix. Rate–distortion points at 2.50, 3.50 and 4.25 bpw use
the same pipeline with the bit width and group changed.

**Rounding.** GPTQ with block size 128, percdamp 0.01, full-group scale fit.

**Post-pass.** For each transformer block in sequence: capture the block's inputs from the
already-quantized prefix; measure, for each projection, `G = E[JᵀJ]` of the module output
with respect to that projection's output by Rademacher probes on the calibration
activations (2 probes over 2048 tokens; damping `G + mean(diag G)·I`); round with GPTQ;
run 6 sweeps of exact coordinate descent under the layer objective and then 6 sweeps under
`tr(ΔW·A·ΔWᵀ·G)`, each move verified against the exact joint change and backed off to a
single guaranteed-improving row when the joint move is not a decrease; propagate the
refined block's outputs to the next block. Attention endpoints for q/k/v are the attention
module output past o_proj; MLP endpoints for gate/up are the MLP output past down_proj;
o_proj and down_proj keep the layer objective.

**Calibration.** wikitext-2 train, 32 windows × 512 tokens; three independent draws (seeds
0/1/2); rotation and probe seeds fixed.

**Evaluation.** Mean NLL per token on the full contiguous wikitext-2 test (146 windows of
2048 tokens) and on 512 pre-fixed C4 validation windows; Monte-Carlo KL to the fp16 model
(8 samples per position, 64 windows). Per-window NLLs are stored; every comparison is a
paired bootstrap (10 000 resamples) over windows within a calibration draw. The protocol
was frozen and committed before any validation run (`docs/FROZEN_PROTOCOL.md`), and its
four diagnostic addenda were pre-registered with interpretation tables before running.

## 3. Results

### 3.1 The four-model table

| model | draws | GPTQ NLL → + post-pass | gap closed (wt2) | perplexity | across-draw consistency |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 3 | 2.921 → 2.891 | **+8.7%** | −2.7% | 3/3, effect 2.5× spread |
| Qwen3-0.6B | 2 | 3.543 → 3.505 | **+8.2%** | −3.7% | 2/2, 4.5× |
| Llama-3.2-1B | 3 | 2.769 → 2.749 | **+4.6%** | −1.9% | 3/3, 2.7× |
| Qwen2.5-1.5B | 2 | 2.549 → 2.555 | −2.5% | +0.6% | 0/2 |

Paired intervals for module − GPTQ are strictly negative in every positive draw (e.g.
Qwen2.5-0.5B draw 0: −0.030 [−0.035, −0.026]). The out-of-domain (C4) column is consistent
with the in-domain one on Qwen2.5-0.5B (+7.2%) and not reportable on the other two positive
models, where the across-draw spread of the paired C4 difference exceeds its mean. The
1.5B failure is carried by C4 (six times its spread) and only directional on wikitext-2.

Nothing we tested predicts the failure: size, family, matrix width, calibration samples
per output dimension, RMSNorm scale extremity, depth, head dimension, depth-to-width ratio,
baseline damage, single-draw noise. Qwen3-0.6B shares both remaining architectural
features (28 layers, head dimension 128) with the failure and is our second-best result.

### 3.2 Rate–distortion, Qwen2.5-0.5B

| bpw | GPTQ gap (NLL) | with post-pass | gap closed | module − GPTQ | equivalent bits saved |
|---|---|---|---|---|---|
| 2.50 | 6.471 | 6.564 | −1.4% | +0.094 [+0.019, +0.165] | harmful |
| 3.25 | 0.351 | 0.321 | +8.7% | −0.030 [−0.035, −0.026] | **+0.084** |
| 3.50 | 0.261 | 0.256 | +2.1% | −0.005 [−0.009, −0.002] | +0.021 |
| 4.25 | 0.067 | 0.061 | +8.9% | −0.006 [−0.008, −0.004] | +0.023 (extrapolated) |

"% gap closed" is not monotone in rate; the 3.25 bpw point was locked before validation,
so the headline is not a selected maximum, but it does not generalise across rates. At
2.50 bpw the scalar network is degenerate and the quadratic objective no longer describes
the end metric.

### 3.3 The objective ladder (one model, one rate, one budget)

| horizon | gap closed |
|---|---|
| layer-local + CD | +0.5% |
| QKᵀ logits (q/k) | worst attention rung |
| attention output | better |
| module output | **+10.4%** (exploratory protocol) |
| block output | +9.6% |
| full-model Fisher, same budget | +0.2% |

Superadditivity, under the frozen evaluation (draw 0): attention half alone +0.6% of the
gap (−0.002 [−0.007, +0.002]), MLP half alone +2.4% (−0.008 [−0.013, −0.003]), whole
module +8.7%; interaction −0.020 [−0.026, −0.014] NLL. On C4 the attention half alone is
harmful (+1.6%).

### 3.4 Estimator variance versus calibration text

Full-model Fisher arm, calibration set fixed, only the metric estimator varied:

| probes | tokens seen by the metric | gap closed | model − GPTQ | KL to fp16 |
|---|---|---|---|---|
| 2 | 2 048 | +0.3% | −0.001 [−0.005, +0.003] | 0.359 |
| 8 | 2 048 | −1.1% | +0.004 [−0.001, +0.009] | 0.365 |
| 2 | 8 192 | **+5.5%** | **−0.019 [−0.024, −0.015]** | 0.349 |

Four times the probes buys nothing; four times the calibration tokens moves the arm from
0.3% to 5.5% of the gap, with the KL moving alongside the NLL. The pre-registered reading
was "token coverage is the operative error", i.e. the global objective is text-limited,
which is MREM's finding in kind; the clean separation is what this study adds. The
attention half of the module arm behaves the same way (+0.6% → +6.6% at 8 192 tokens); the
whole module arm does not — it is slightly *worse* at 8 192 tokens (7.3% vs 8.7% on wt2,
paired +0.005 [+0.001, +0.009]) — so the frozen budget stands and the interaction between
the halves is budget-dependent in a way we have not resolved.

### 3.5 Matched-domain calibration

Calibrating on C4 train instead of wikitext-2 train (same seed) moves both arms alike on
wikitext-2 (+0.099 / +0.094 NLL) and on the C4 evaluation improves GPTQ by 0.006 and the
post-pass by 0.011; the post-pass's advantage grows by 0.005 [0.002, 0.009] (C4 gap closed
7.2% → 8.9%). Matched domain helps the post-pass slightly more than GPTQ; the effect of
domain itself is small on this model.

### 3.6 Two bits, vector-quantized

A per-block VQ quantiser from the embedding-quantization project (d=4, K=256, ~2.1 bpw,
input-side rotation; not released) produced two Qwen2.5-0.5B checkpoints from one
quantisation: plain, and plain with our post-pass inside its loop. Under the frozen
evaluation: plain wt2 gap 0.692 (scalar GPTQ at 2.50 bpw: 6.471); post-pass − plain wt2
+0.027 [−0.038, +0.098] (n.s.), C4 −0.067 [−0.088, −0.045], KL −9.5%. The pre-registered
criterion (≥3% of the gap on wt2) was not met. Per window the post-pass is better on 127 of
146 wt2 windows and 467 of 512 C4 windows, and worse by more than one nat on ten wt2
windows, every one of which begins with a digit-like token. The damage is uniform from
position 3 onward; dropping, prepending to, or replacing the first token removes it. With a
fixed "\n\n" prefix on every window: post-pass − plain −0.030 [−0.034, −0.025] on wt2 and
−0.030 [−0.034, −0.026] on C4, no window worse by more than one nat, and the plain
checkpoint improves too (−0.06 / −0.05). The refined 2-bit model fails to form an
attention sink on some first tokens; the objective cannot see it because its calibration
windows start mid-text and those rows lie outside the metric's support.

### 3.7 Negatives, with their scope

- Layer-local coordinate descent end-to-end: −0.5 / −1.8 / −2.9% of the gap on the three
  positive models. Converged CD is within 0.015% of a certified box-constrained optimum on
  the row subproblem, so this is the objective, not the search.
- Post-GPTQ rotation landscape under the local objective: a flat ±1% plateau per matrix;
  tangent analysis, basin test from ten starts, derivative-free search and 200 random
  rotations reach the same floor. End-to-end-trained rotations are not covered.
- Per-row rate allocation (WaterSIC-style oracle): ≤0.11 equivalent bits, zero at 3.25 bpw.
- A diagonal output metric with shared input Gram changes exactly zero codes (KronQ
  Prop. 1); GuidedQuant's per-sample weighting is a different object and is not covered.
- A rank-64 surrogate of the metric makes the true error 32× worse; diagonal-plus-low-rank
  forms untested.
- The true post-hoc form — refining a finished checkpoint without re-fitting downstream
  blocks — was never validated by us; in a neighbouring encoder-quantisation project the
  post-pass gave no effect resolvable above calibration-draw variance at 1.8 bpw.

## 4. Practical guidance

- If you already run a GPTQ-class pipeline at ~3 bits on a small decoder and can afford
  20–30× the quantisation time once, the module post-pass is a free 2–4% of perplexity on
  most models and a small loss on some; evaluate both checkpoints on held-out text and
  keep the better.
- Do not expect it below ~2.5 bits scalar. On vector-quantized codes at ~2 bits it helps
  typical text and needs a fixed prefix at inference to avoid the first-token failure;
  the representation, not the refinement, is what makes 2 bits usable.
- If your objective is an estimate, audit it at a *higher* resolution, not only on fresh
  data: a fresh-metric check at the same budget passed on the model where the method
  fails.
- Report every calibration draw. A window-paired interval does not contain draw variance.

## 5. Limitations

Nothing above 1.5B is tested (a 3B run was lost with a rented instance). One calibration
corpus for the headline. One draw for the rate–distortion, budget and 2-bit results. No
predictor of the 1.5B failure. Factorization error in the metric (BaKron's K-FAC versus
Shampoo swings) is untested by us. The 2-bit checkpoints depend on a quantiser that is not
part of this release; the dequantised weights and all evaluations are.

## 6. Reproducibility

Everything in `evidence/` regenerates the tables without a GPU:
`python scripts/analyze_valid.py --print` (four-model table and paired bootstraps),
`scripts/analyze_rd.py` (rate–distortion), `scripts/analyze_d3.py` (halves, calibration
domain, estimation budgets). One validation draw of one model is
`python -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --calib-seed 0
--fresh-g-seq 16`, about two hours on a 6 GB laptop GPU. The pipeline is bit-deterministic:
fp16 and GPTQ numbers reproduced exactly across machines and allocator settings. Every
pre-registration, result and correction is in `EXPERIMENT_LOG.md` (E1–E56).

The experiments were run with Claude (Anthropic) as an autonomous research agent under the
author's direction.
