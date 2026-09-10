# Reviewer package

*Written to be attacked. Every claim is paired with the evidence file that backs it and the
reason it might be wrong. Where the evidence is weaker than the claim, that is marked
inline rather than left for you to find. 2026-09-07.*

## The claim under review

After GPTQ produces integer codes, an exact coordinate-descent pass revisits those codes
against the error at the output of the enclosing **attention/MLP module** rather than of the
single linear layer — objective `tr(ΔW·A·ΔWᵀ·G)`, `G = E[JᵀJ]` measured by probes inside the
sequential pipeline. Only integer values change: identical bits, layout and kernel.

## What was pre-registered, and when

`docs/FROZEN_PROTOCOL.md`, committed at `b57c9ac` **before** any validation run. Method,
hyperparameters, evaluation sets, metric and pass criteria all fixed there; no tuning was
permitted afterwards and none was done. Two later diagnostics (D1, D2) were pre-registered
with their interpretation tables before running, in the same file. The git history is the
audit trail: every result commit postdates the protocol commit.

## Results, with the evidence behind each

Primary metric is mean NLL per token. Comparison is a **paired bootstrap over evaluation
windows within each calibration draw** (10k resamples). Calibration is wikitext-2 *train*;
evaluation is the full contiguous wikitext-2 *test* plus 512 pre-fixed C4 *validation*
windows — disjoint by construction, verified not assumed.

| model | draws | in-domain (wt2) gap closed | ΔNLL (mean over draws) | perplexity change | evidence |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 3 | **+8.7%**, 3/3 draws, all CIs strictly negative | −0.0274 | −2.7% | `evidence/validation/valid_qwen05_s{0,1,2}.*` |
| Qwen3-0.6B | 2 | **+8.2%**, 2/2 | −0.0380 | −3.7% | `evidence/validation/valid_qwen3_06b_s{0,1}.*` |
| Llama-3.2-1B | 3 | **+4.6%**, 3/3 | −0.0196 | −1.9% | `evidence/validation/p3_llama1b*.*` |
| Qwen2.5-1.5B | 2 | **−2.5%**, 0/2 | +0.0058 | +0.6% | `evidence/validation/valid_qwen15_s{0,1b}.*` |

"Gap closed" is the fraction of the GPTQ-vs-fp16 NLL gap recovered; the perplexity column
is the same effect in absolute terms and is the number to quote to anyone who does not
care about the gap. The post-pass costs 20–30× the GPTQ time (50 min on 0.5B, ~2.7 h on
1–1.5B, ~9 h projected on 3B, RTX 5090 / 24 GB) and nothing at inference.

Per-window NLLs are in the jsonl, so every interval above can be recomputed without
re-running anything: `python scripts/analyze_valid.py --print`.

## The five places I would attack first

**1. The out-of-domain column does not survive its own variance test.** Comparing the effect
to the across-draw spread of the *paired* difference (not of the scores):

    model            eval  effect    spread   ratio
    Qwen3-0.6B       c4    +0.0077   0.0407   0.2x   <- says nothing
    Llama-3.2-1B     c4    -0.0360   0.0908   0.4x   <- says nothing
    Qwen2.5-0.5B     wt2   -0.0274   0.0108   2.5x
    Qwen3-0.6B       wt2   -0.0380   0.0084   4.5x
    Llama-3.2-1B     wt2   -0.0196   0.0071   2.7x
    Qwen2.5-1.5B     wt2   +0.0058   0.0049   1.2x   <- weak

So: the in-domain result is solid on three models; the C4 numbers on two of four carry
uncertainty larger than their effect and are reported as such; and **the failure case is
carried by the out-of-domain metric and only directional in-domain**. The ratio is a
consistency check on 2–3 draws, not a statistical test.

**2. There is no predictor of when it fails, and we tested hard for one.** Ruled out: size,
family, matrix width, samples per output dimension, RMSNorm scale extremity, depth,
head_dim, depth/width ratio, baseline damage, and single-draw noise. Qwen3-0.6B shares
*both* remaining architectural features (28 layers, head_dim 128) with the failure and is
our second-best result. Evidence: `evidence/diagnostics/probe_*.jsonl`, `noise_probe.jsonl`,
`d2_*`, and E38–E46 in `EXPERIMENT_LOG.md`.

**3. As a product it does not clear its own bar.** Converted to bits saved at equal quality
(Qwen2.5-0.5B, the best model): **+0.084 equivalent bits at 3.25 bpw**, +0.021 at 3.50,
+0.023 at 4.25 (extrapolated), harmful at 2.50. The pre-registered product threshold was 0.1–0.2. The
"% of gap closed" figure is also **non-monotone in rate** (8.7 / 2.1 / 8.9), so our headline
number is the best of three and no single-rate quote is stable.

**4. Breadth above 1.5B is zero, and one evidence file was briefly wrong.** The rented
instance was shut down with a Llama-3.2-3B run — the only test above 1.5B — at layer 8/27 of
its module arm after 2.2 h; only its fp16 and GPTQ references exist
(`evidence/validation/p6_llama3b_s0.*`, wt2 2.36377 vs 2.05586). A brief power-on recovered
the per-window data for all four rate points (`evidence/validation/rd_q05_*`), so every row
of the rate–distortion table is now bootstrapped (`python scripts/analyze_rd.py`). While
doing so I found that the committed copy of the 2.50 bpw file had been pulled mid-run and
held only the fp16 records; it was replaced. Every other file on the instance matched its
committed copy by md5. Full account in `EXPERIMENT_LOG.md` E47.

**5. Single calibration corpus for the headline, one matched-domain measurement.** D3c
(0.5B, draw 0) calibrated both arms on C4 train: on wt2 both lose ≈0.1 NLL alike
(difference-of-differences −0.006 [−0.013, +0.001]); on the C4 evaluation GPTQ gains
0.006 and the post-pass 0.011, so the post-pass's advantage grows by 0.005 [0.002, 0.009]
(gap closed 7.2% → 8.9%). So matched-domain calibration favours the post-pass slightly,
and domain itself matters little on this model; one model, one draw. The earlier
"systematically pessimistic" wording was withdrawn before this was measured and stays
withdrawn: the sign was right, the implied size was not.

**6. The non-monotone objective ladder is prior art, and we said otherwise.** BRECQ (2021)
showed the layer/block/stage/net curve with net-wise worse than layer-wise and gave the
bias–variance explanation; MREM (2022) showed the layer-vs-module ordering depends on
calibration size; BaKron (Aug 2026) derives the MLP-output Hessian we use for gate/up and
compares local vs global at one solver on Llama-3/Qwen3 to 8B, with the ordering flipping
by model and by Kronecker factorization. What remains ours is listed in
`docs/PRIOR_ART_LADDER.md` §4: the intra-attention rungs, budget scaling at a fixed
calibration set (which did *not* yet separate probe count from text), integer-only
refinement inside the sequential pass (not post-hoc on a finished checkpoint — see 7), and
the validation record. Our full-model arm
optimises the Fisher of KL-to-fp16, not token NLL. D3b (E52) settled what governs its
rank at cheap budget: 4× probes +0.004 n.s., 4× calibration tokens in the metric −0.019
[−0.024, −0.015] (0.3% → 5.5% of the gap) with KL moving alongside NLL. So it is
text-limited, not noise-limited, and there is no objective/metric mismatch; factorization
error (BaKron's K-FAC/Shampoo swings) remains untested by us. The same limitation reaches
the module metric itself: D3b-ext / ext2 (E53/E54, 2026-09-09, one draw, diagnostics): the attention half alone, given 4× the calibration tokens in its metric, goes from +0.6% to +6.6% of the wt2 gap (−0.023 [−0.028, −0.019]) and from harmful to +4.8% on C4 — but the whole module arm at the same 8k budget is *worse* than at the frozen 2k (7.3% vs 8.7% wt2, 4.4% vs 7.2% C4, paired CIs excluding zero). So the text-limitation is specific to the attention endpoint; the frozen budget stands, no re-freeze, and the validated +8.7% is not a lower bound (an earlier draft said so; withdrawn). The MLP half at 8k is the one diagnostic not yet run.

**7. The first run outside our pipeline is a negative.** In a neighbouring encoder-quantisation project the post-pass gave no effect resolvable above calibration-draw variance at 1.8 bpw (E50–E51;
per-arm numbers belong to that project and are reported only in aggregate here, by
agreement). Two things it taught us are kept in full: a query-paired confidence interval
does not contain calibration-draw variance, and their first attempt refined a frozen
capture without re-fitting downstream blocks, which our validated procedure never does —
our wording had implied it did, and that is our error to own (see 6 and E50–E51).

**8. The decoder P4 test at 2 bits: criterion missed, and the reason is a tail.** The
same project's per-block VQ quantiser on Qwen2.5-0.5B at 2.12 bpw, our sequential
post-pass inside its loop, our frozen evaluation (E55): refine − plain on wt2 +0.027
[−0.038, +0.098] (−3.8% of the gap, n.s.; pre-registered "≥ ~3%" not met), on C4 −0.067
[−0.088, −0.045] (+7.1%), KL −9.5%. Per window the post-pass is better on 127 of 146 wt2
and 467 of 512 C4 windows (medians −0.07 / −0.09 nats) and catastrophically worse on three
wt2 windows (+1.7 nats each), which the mean criterion correctly refuses to forgive. The
tail was then diagnosed and pre-registered (D4, E56): all ten worst windows begin with a
digit-like fragment, the damage is uniform from position 3, and dropping or replacing the
first token removes it — the refined 2-bit model fails to form an attention sink on some
first tokens, which the objective cannot see (its calibration windows start mid-text, so
those rows are outside the metric's support). With a fixed "\n\n" prefix on every window
the tail vanishes (0 windows worse by >1 nat) and the post-pass wins on both sets: wt2
−0.030 [−0.034, −0.025], C4 −0.030 [−0.034, −0.026], ≈3–5% of the gap; prefixing also
helps the plain checkpoint (−0.06 / −0.05), so the fragility is 2-bit VQ's and the
refinement aggravates it. The frozen verdict stands as the frozen number; the finding to
carry is the failure mode and its fix, one draw. The representation is the win at 2 bits
(wt2 gap 0.69 vs 6.47 for scalar GPTQ at 2.50 bpw) and that is prior art (GPTVQ/VPTQ
class).

## What I think survives review

- **The intra-attention rungs of the objective ladder.** Layer → QKᵀ logits →
  softmax-Fisher → attention output → module → block → full-model Fisher, at matched solver,
  rate, calibration and probe budget. The curve is non-monotone with the optimum at the
  module endpoint (as BRECQ/BaKron would predict); the logit objective that BoA/BaKron use
  for q/k is the worst attention rung; the full-model objective is *worse than layer-local*
  at matched cheap estimation while improving its own objective the most; 4× the
  calibration tokens in its metric move it to 5.5% of the gap, 4× the probes move nothing
  (D3b, with intervals), still 60% of the module effect on the same calibration set.
  Evidence: `evidence/diagnostics/joint_ppl.jsonl`, `attn_decomp.jsonl`,
  `evidence/validation/d3b_*`, E32/E34/E52.
- **Layer-local coordinate descent is mildly harmful end-to-end** (−0.5 / −1.8 / −2.9% on
  three models) — contradicts the natural assumption and is cheap to verify.
- **Superadditivity, now with an interval** (D3a, frozen evaluation, draw 0): attention
  half +0.6% of the gap (CI covers 0), MLP half +2.4% [+1.0, +3.8], whole +8.7%;
  interaction −0.020 [−0.026, −0.014] NLL on wt2, −0.031 [−0.035, −0.028] on C4. Half the
  layers gives ~20% of the full effect (exploratory). E49, E32d–f.
- **Hard negatives with scoped certificates**: the post-GPTQ rotation landscape is a flat
  ±1.1% plateau (per matrix, fixed-code local objective; tangent analysis, basin test from
  10 starts, derivative-free optimisation and a 200-rotation sweep all reach the same
  floor — it does not cover end-to-end-trained rotations); converged CD is within 0.015% of
  a certified box-constrained optimum on the row subproblem; per-row rate allocation is
  worth ≤0.11 equivalent bits; a diagonal output metric with shared input Gram changes
  exactly 0 codes (KronQ Prop. 1 — GuidedQuant's per-sample weighting is a different object
  and is not covered); a rank-64 surrogate makes true error 32× worse (diagonal-plus-low-rank
  untested).
- **Methodological**: an out-of-sample check at the *same estimator resolution* validates
  sampling, not resolution — ours passed on the model where the method fails. And removing
  provably-noisy metric components (18% of the pre-o_proj metric's energy is in entries that
  are exactly zero by construction) made results *worse* on both models.

## Reproduction

```
pip install -e .                       # torch, transformers, datasets
python -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --calib-seed 0 \
       --fresh-g-seq 16 --out results/raw/valid_qwen05_s0.jsonl
python scripts/analyze_valid.py --print
```

The pipeline is bit-deterministic: fp16 and GPTQ numbers reproduced exactly across process
restarts, machines and memory-allocator settings, which is why single runs per cell are
defensible. Anything not reproducible from `evidence/` should be treated as unsupported.
