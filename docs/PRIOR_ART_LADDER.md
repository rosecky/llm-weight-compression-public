# The objective ladder against its nearest prior work

*2026-09-07. Written after an external review pointed at BRECQ, MREM, YAQA and BaKron as
direct prior art for the "non-monotone objective horizon" story. All four were read from the
primary PDFs for this note; page and table references are to those PDFs. This supersedes
the positioning in `PITCH.md` §1 and `BRIEFING.md` §4 wherever they disagree.*

## 1. What each paper already shows

| paper | what it varies | what it finds | how it explains it |
|---|---|---|---|
| **BRECQ** (Li et al., ICLR 2021, §3.2, §4.1 Table 1) | reconstruction granularity layer / block / stage / net, AdaRound-style rounding on CNNs, 1024 images | block wins; net-wise is far *worse than layer-wise* (ResNet-18 54.15 vs 65.19; MobileNetV2 40.76 vs 52.13) | "optimizing the whole network over 1024 calibration samples leads to over-fitting easily"; layer-wise "acts like a regularizer"; there "should be a better bias-variance trade-off choice … at an intermediate granularity" |
| **MREM** (Bai et al., NeurIPS 2022, §5.4.3–5.4.4, Fig. 5) | module-wise (groups of Transformer layers) vs layer-wise reconstruction of BERT; number of modules; calibration size 32–8192 | fewer, larger modules slightly better; **layer-wise REM is ahead below 128 samples, module-wise overtakes above** | "the simple training objective in REM can hardly hold more training instances"; module-wise "admits higher flexibility" |
| **YAQA** (Tseng et al., 2025, §3.1, §4.1, Tables 1–2) | layer-local LDLQ vs full-model KL Hessian (Kronecker sketches), same rounding rule | full-model KL objective beats LDLQ and GuidedQuant by ≈30% KL; **but** DiscQuant (descent on the full KL) loses to LDLQ: "descent-based methods do not always outperform local adaptive rounding" | the local Hessian is "a poor proxy" for the end-to-end error; sketches need 20M–134M tokens |
| **BaKron** (Aug 2026, arXiv 2608.06291, §4.1, App. A.1, App. D Tables 4–10) | with one solver: GPTQ (layer-local) vs MLP-local vs fully-local (MLP + BoA-style attention) vs backpropagated full-model Fisher; two Kronecker factorizations; Llama-3 / Qwen3 1B–8B, 2.81 bpw, 256×2048 Pile tokens, single run, no intervals | see §3 below: the ordering **flips from model to model and from factorization to factorization** | not analysed; "Backprop-Shampoo usually attains a lower perplexity than GPTQ, although not on every model. The local variants stay closer to GPTQ." |

So the two things we had presented as ours — that the horizon curve is non-monotone, and
that the explanation is a dependency-versus-estimation trade-off — were stated by BRECQ
in 2021 and refined by MREM in 2022. The sentences "nobody sweeps the ladder", "nobody has
stated" and "that framing is ours" were wrong and have been removed.

## 2. The mathematics, module by module

BaKron Table 4 lists the Hessian each variant uses per projection. Ours, for comparison,
with `A = E[xxᵀ]` the input Gram and `G` the output-side metric measured by Rademacher /
Fisher probes inside the sequential pipeline:

| projection | GPTQ | BaKron-MlpLocal | BaKron-FullyLocal | BaKron-Backprop | **ours, `module` scope** |
|---|---|---|---|---|---|
| q_proj | A ⊗ I | A ⊗ I | A ⊗ KKᵀ (attention-matrix Frobenius, per head) | E[xxᵀ ⊗ hhᵀ] | A ⊗ G, G = Gauss–Newton of the **attention-module output** w.r.t. q's output (through softmax, V, o_proj) |
| k_proj | A ⊗ I | A ⊗ I | A ⊗ QQᵀ | same | same, w.r.t. k's output |
| v_proj | A ⊗ I | A ⊗ I | A ⊗ OᵀO | same | A ⊗ G through attention weights and o_proj (≈ OᵀO ⊙ attention-mass structure) |
| o_proj | A ⊗ I | A ⊗ I | A ⊗ I | same | A ⊗ I (module output *is* its output) |
| up_proj | A ⊗ I | E[xxᵀ ⊗ (ggᵀ ⊙ DᵀD)] | same | same | A ⊗ E[ggᵀ ⊙ DᵀD] measured by probes — the K-FAC-style factorization of exactly the BaKron quantity |
| gate_proj | A ⊗ I | E[xxᵀ ⊗ (ffᵀ ⊙ DᵀD)] | same | same | A ⊗ E[ffᵀ ⊙ DᵀD], same remark |
| down_proj | A ⊗ I | A ⊗ I | A ⊗ I | same | A ⊗ I |

Consequences:

- **For gate/up our module objective is BaKron-MlpLocal's Hessian under the K-FAC
  independence assumption**, with `G` estimated stochastically (probes) rather than
  formed in closed form. We did not know this when we wrote the pitch; BaKron App. A.1 is
  the closed form of what our probes measure.
- **For q/k/v our objective is different from FullyLocal.** BoA/BaKron use a Frobenius loss
  on the attention *logit* matrix for q/k; we measured that rung explicitly
  (`attn_logits`) and found it worse than the attention-*output* rung, which is what our
  `module` scope uses. Our intra-attention ladder (logits → softmax → attention output →
  module) has no counterpart in any of the four papers.
- **Our full-model horizon is BaKron-Backprop's Hessian** (Fisher with `y ~ p_model`,
  K-FAC-style factorization), at 1/32 of their calibration text and with a probe estimator
  instead of a full backward pass. This is the arm our ladder ranks below layer-local at
  matched cheap budget.
- **Where the metric enters differs.** BaKron rounds *with* the two-sided Hessian inside a
  generalised GPTQ/LDLQ sweep. We round with plain GPTQ (A ⊗ I) and apply `G` afterwards
  in an exact coordinate-descent pass over the integer codes — but *inside the sequential
  quantisation*: block l is rounded, refined, and the calibration activations are
  re-propagated through the refined block before block l+1 is rounded, so every downstream
  block's GPTQ compensates for the refined upstream. Only integer codes change, so the
  output is format-identical to the GPTQ checkpoint. What we have **not** validated is
  the true post-hoc form — refining a finished checkpoint without re-fitting the blocks
  downstream; our planned P4 test of that never ran, and the one outside attempt (E50, an
  encoder VQ point below 2 bpw, refined on a frozen capture) was negative, plausibly for
  exactly this reason. That distinction was missing from our earlier wording.
- **Solver.** Both are the same family (LDLQ/GPTQ-type sequential rounding); our post-pass
  adds an exact per-coordinate minimiser with verify-and-backoff, which BaKron does not
  have. BaKron's contribution is the O(mn(m+n)) two-sided solver and the recursive
  Hessian accumulation, neither of which we have or need at ≤1.5B.

## 3. BaKron's own ladder, read as data

wikitext-2 perplexity from BaKron Tables 5–10 (2.81 bpw, per-row scales, no groups,
single run each). "gap closed" is our computation, (GPTQ − variant)/(GPTQ − base).

| model | base | GPTQ | MlpLocal K-FAC / Shampoo | FullyLocal K-FAC / Shampoo | Backprop K-FAC / Shampoo | best rung |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 11.98 | 25.63 | 22.32 / 22.32 (24%) | **21.54** / 26.59 (30% / −7%) | 22.04 / 22.29 (26% / 24%) | fully-local |
| Llama-3.2-3B | 9.53 | 13.70 | 13.60 / 13.55 (2–4%) | 13.41 / **13.28** (7–10%) | 14.55 / 15.47 (**−20% / −42%**) | fully-local; global worse than layer |
| Llama-3-8B | 7.44 | 53.47 | 19.20 / 42.03 | 17.19 / 45.31 | **11.90** / 12.24 (90%) | global |
| Qwen3-1.7B | 12.42 | 35.97 | 36.75 / 37.59 (−3 / −7%) | 39.16 / 38.09 (−14 / −9%) | 60.97 / **20.26** (−106% / +67%) | global-Shampoo; every local rung harmful |
| Qwen3-4B | 10.37 | 14.72 | 14.42 / 14.52 (7 / 5%) | 15.05 / 15.17 (−8 / −10%) | 15.49 / **13.29** (−18% / +33%) | global-Shampoo |
| Qwen3-8B | 11.12 | 14.58 | 21.62 / 13.91 (−203% / +19%) | 14.48 / 15.22 (3 / −18%) | 17.10 / **11.94** (−73% / +76%) | global-Shampoo |

Three readings, all of which bear on our claims:

1. **The horizon ordering is not a property of the horizon.** On Llama-3.2-3B the
   backpropagated full-model Fisher with 524k calibration tokens is worse than layer-local
   GPTQ by 20–42% of the gap; on the 8B models it closes 76–90%. On Qwen3-1.7B every local
   rung is harmful and the global rung is either the best or the worst arm in the table
   depending only on the factorization. This is the same "no predictor" picture we found
   on Qwen2.5-1.5B, at larger scale and in someone else's hands.
2. **Factorization error is a confound of the same order as estimation noise.** K-FAC vs
   Shampoo moves the global arm from 60.97 to 20.26 on Qwen3-1.7B and from 21.62 to 13.91
   for the *MLP-local* arm on Qwen3-8B, with identical data and solver. Our 8× probe-budget
   result (0.2% → 5.4% of the gap) shows estimation noise matters; it does not show it is
   the whole story, because our metric is also a K-FAC-style product and we never varied
   that. The reviewer's objection stands.
3. **Their MLP-local effect sizes on Llama-3.2-3B (2–4%) and Qwen3-4B (5–7%) are in the same
   band as ours** (4.6–8.7% of the NLL gap on 0.5–1B models), despite 32× the calibration
   text and rounding-time rather than post-hoc use of the metric. That is mildly reassuring
   about our estimator and mildly discouraging about the ceiling.

## 4. What is left that is ours

Checked against the four papers above and the earlier prior-art map in
`JOINT_OPTIMIZATION.md` §4:

1. **The intra-attention rungs** (QKᵀ logits → softmax-Jacobian → attention output →
   module output) isolated at one solver, one rate, one probe budget, with the finding that
   the BoA-style logit objective is the *worst* attention rung and the attention-output rung
   is exactly block-diagonal per head. BaKron uses the logit objective for q/k without
   comparing it to anything.
2. **Probe count separated from calibration text, at one solver and rate.** MREM varies
   the number of calibration samples; BaKron and YAQA fix one budget. D3b (E52) holds the
   calibration set fixed and varies only the metric estimator: 4× probes +0.004 [−0.001,
   +0.009] (nothing), 4× tokens −0.019 [−0.024, −0.015] (0.3% → 5.5% of the gap), KL to
   fp16 moving with NLL. The full-model rung is *text*-limited, which is MREM's result in
   kind; ours is the clean measurement and the negative on probes. One model, one rate.
3. **Integer-only refinement after rounding, inside the sequential pass**, format
   preserved: the metric enters after rounding, through an exact coordinate-descent pass
   with a certified optimality gap on the row subproblem, and downstream blocks are then
   rounded against the refined upstream. BaKron/YAQA/BRECQ/MREM put the metric into the
   rounding itself. The "existing checkpoint" form (no downstream re-fit) is untested by
   us and negative in the one outside run (E50).
4. **The validation record**: multiple calibration draws, paired bootstrap within draw,
   every draw reported, one documented failure, and the negative result that layer-local
   coordinate descent is mildly harmful end-to-end on three models. BaKron's tables are
   single runs without intervals and contain swings (Qwen3-8B MLP-local 21.62 vs 13.91) that
   are larger than any effect they claim and are not discussed.
5. **The superadditivity of attention-half and MLP-half refinement** (−1.8%, +1.7%, +10.4%
   together). BaKron's MlpLocal → FullyLocal step is the closest datum (Llama-1B 22.32 →
   21.54) and is roughly additive there; it does not contain the attention-only arm.

Not ours, now with the correct attribution: the non-monotone horizon curve (BRECQ), its
bias–variance explanation (BRECQ), its dependence on calibration budget (MREM), the
module-output Hessian for gated MLPs (BaKron App. A.1), the full-model Fisher objective
(YAQA, BaKron), local-vs-global at one solver (BaKron App. D).

## 5. What this changes in the plan

- The scientific claim narrows to: *at fixed calibration text, which horizon is worth its
  estimation cost?* D3b answered the variance part: probe variance is not the limiting
  error, calibration-text coverage in the metric is (0.3% → 5.5% at 4× tokens), and the
  KL moves with the NLL. What is left is factorization error (BaKron's K-FAC/Shampoo
  swings) and the whole-calibration-set point for the full-model arm; both are single
  runs on the same protocol. E53/E54 add that the attention endpoint is text-limited too (attention
  half +0.6% → +6.6% at 4× tokens) while the whole module arm is *worse* at 4× tokens;
  so the ladder's shape is budget-dependent rung by rung, and R\* = module is a statement
  at the frozen budget, not a budget-free one.
- BaKron is the natural second solver for the "does the objective conclusion survive a
  solver change" test the review asked for, since its MLP-local Hessian is ours in closed
  form.
- Above 1.5B, BaKron already covers Llama-3 and Qwen3 to 8B with the same MLP-local
  objective, so a 7–8B run of ours would replicate rather than extend. Deferred.
