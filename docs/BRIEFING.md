# Functional-scope refinement for LLM weight quantization — research briefing

*Self-contained summary for external consultation. English is used for terminology
compatibility with the literature; all numbers are our own measurements unless cited.
Status: 2026-09-05, after a frozen-protocol validation phase on three models.*

## 1. Setup

- **Models:** Qwen2.5-0.5B, Llama-3.2-1B, Qwen2.5-1.5B (validation phase); the exploratory
  phase used Qwen2.5-0.5B only (24 blocks, hidden 896, GQA 14/2, MLP 4864).
- **Task:** post-training weight-only quantization (PTQ), no retraining, no QAT.
- **What weight-only PTQ actually buys, stated honestly up front.** On a GPU it buys bytes,
  not milliseconds: measured on a neighbouring project's client (Core Ultra 7 155H, 8
  threads, vs RTX 5090), the fp16 GPU path is several times faster than their best quantized CPU client. So with a GPU in the picture this is a download-size and VRAM problem, and latency
  only becomes the motivation on CPU or WASM targets, where quantization buys both. (Source:
  the neighbouring embedding-quantisation project, 2026-09-05.) Our contribution is a quality gain at
  fixed size, so it inherits that motivation exactly.
- **Evaluation (validation phase):** mean NLL per token as the primary metric, on the full
  contiguous wikitext-2 test and 512 pre-fixed C4 windows, plus a Monte-Carlo estimate of
  KL to fp16; per-window NLLs kept so the comparison is a paired bootstrap *within* each
  calibration draw. Three independent calibration draws per model. The exploratory phase
  used a lighter 24-window perplexity protocol (fp16 11.8134) — its numbers are not directly
  comparable to the validation table.
- **Hard rules we enforce structurally:** every comparison at *bit-identical storage*
  (exact bpw accounting including scales and codebooks; refinement may only change which
  integers are stored — no residuals, no extra metadata); every method judged **only** on
  end-to-end perplexity (local proxies misranked methods five separate times in this
  program); the whole pipeline is bit-deterministic, so single runs are trustworthy.

## 2. The approach in one paragraph

Standard PTQ (GPTQ family) optimizes each linear layer against its own output
reconstruction. We add a cheap **post-pass**: after GPTQ produces integer codes, an exact
coordinate-descent solver revisits the discrete decisions against a **wider functional
objective** — the error at the output of the enclosing attention/MLP *module*, expressed as
`tr(ΔW·A·ΔWᵀ·G)` where `A = E[xxᵀ]` is the usual input second moment and `G = E[JᵀJ]` is a
dense output metric measured by Rademacher/Fisher probes **inside the sequential
quantization pipeline** (upstream layers already quantized, downstream still fp — the
geometry the decision actually faces). Same storage, same quantizer, offline-only cost.

Key implementation choices (each one earned by a measured failure of the alternative):

1. **Dense G or no G** — a rank-64 surrogate has an exploitable null space (optimizer made
   true downstream error 32× worse). Dense per-matrix G + trust-region damping only.
2. **G never enters GPTQ itself** — it cancels algebraically from column-wise OBS updates
   (KronQ Prop. 1; we verified: diagonal G changes exactly 0 codes). It enters only the
   coordinate descent, where off-diagonal structure can act.
3. **Start from the layer optimum** (`cd_pre`): the wider scope is credited only with what
   it adds beyond converged layer-local CD.
4. **Exact moves** — the objective is quadratic, so each code flip / VQ codeword swap has a
   closed-form cost; a maintained matrix `M = G·D·A` makes sweeps cheap; multi-row moves
   under off-diagonal G go through verify-and-backoff with a guaranteed single-row fallback.
5. **Endpoints with G = I get no probes** (don't estimate what is known).
6. **Frozen amplitudes from original weights** (VQ variant) — scales fitted on compensated
   working weights inflate at 2 bits (a feedback loop costing up to 45%).

## 3. Headline results

### 3.0 The validated bottom line (read this first)

Everything in §3.1–§3.4 below was measured in an exploratory phase on **one model** with a
24-window evaluation. A subsequent validation phase froze the method and protocol at a commit
(no tuning permitted afterwards), hardened the evaluation (full contiguous wikitext-2 test,
512 pre-fixed C4 windows, Monte-Carlo KL to fp16, per-window NLLs, paired bootstrap within
each calibration draw), and re-ran it on three models. That is the result that should be
believed:

| model | fp16 | GPTQ | + module post-pass | wt2 gap closed | C4 | KL |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 13.07 | 18.57 | **18.01** | **+8.7%** | +7.2% | +8.0% |
| Llama-3.2-1B | 9.75 | 15.89 | **15.54** | **+4.6%** | +3.4% | +2.4% |
| Qwen2.5-1.5B | 9.26 | 12.85 | 12.96 | **−2.5%** | −4.1% | −2.1% |

*(wikitext-2 perplexity, 3.25 bpw exactly, bit-identical storage; "gap closed" = fraction of
the NLL distance between GPTQ and fp16 that the post-pass removes.)*

Replication across calibration draws: Qwen2.5-0.5B **3/3** on both corpora, every paired
bootstrap interval strictly negative. Llama-3.2-1B **3/3 on wikitext-2** (+3.1…+4.6%), 2/3 on
C4 with one draw neutral. Qwen2.5-1.5B fails on 1 draw and the failure survives every fix
attempted.

**Honest summary: the effect is real and replicates across two model families and two
tokenizers with no tuning — and we cannot predict when it fails.** The 1.5B failure is not
explained by size (0.5B and 1B work), family, matrix width (Llama's MLP is 8192 vs 1.5B's
8960), or metric conditioning (0.25 vs 0.23 samples per output dimension). Two untested
candidates remain: depth (16 and 24 layers work, 28 fails) and head_dim (64 and 64 work, 128
fails). With n=3 those are hypotheses, not findings.

Cost: ~2.7 h of optimizer time per model on an RTX 5090, 8.5 GiB peak, 2.2% of integer codes
moved, offline only.

**Precisely what "same storage" means, since it is easy to overstate.** The post-pass changes
only integer code values: relative to its GPTQ baseline it costs nothing at all — same bits,
same file layout, same kernel. The *pipeline* both arms share does use a two-sided randomized
Hadamard rotation per matrix, and that is not free: the stored codes live in the rotated
basis (accounted as codes + scales + 32 bits of seed per matrix, included in the 3.25 bpw),
and deployment re-associates rather than folding anything —
`y = Rₒᵀ(W'_q(Rᵢᵀx))`, so each matrix undoes its own rotation locally and no residual-stream
sharing or norm folding is required. The price is two structured transforms per linear per
token; with a seeded Hadamard these are O(d log d). A peer project has now measured this
properly on AVX2 — a hand-written kernel at the vector-add throughput limit — and gets
**~10% latency overhead**, not the ~1% an operation count suggests (their correction; the
operation count is not the time). The overhead is linear in the number of rotation rounds.
So: the post-pass is kernel-free, the rotation underneath it costs on the order of 10% of
inference latency unless fused, and both arms of every comparison carry it equally.

### 3.0b Exploratory-phase results (one model, lighter protocol)

| method | actual bpw | wikitext-2 ppl |
|---|---|---|
| scalar GPTQ+CD (QuantEase/CDQuant class) | 2.25 → 2.50 | 13,914 → 2,054 (model broken) |
| external GPTVQ (Qualcomm code, own protocol) | ~2.3 | 61.5 |
| **ours: VQ-2D + module-scope post-pass** | **2.514** | **18.06** |
| external GPTVQ | ~3.05 | 16.45 |
| **ours: scalar + module-scope post-pass** | **3.25** | **16.28** (GPTQ alone: 16.80) |

*(fp16 = 11.81 on our protocol; GPTVQ rows use its own full-test protocol — unified re-eval
pending, but the low-bit difference is categorical.)*

### 3.1 The compensation-radius curve is sharply non-monotone (R\* = module)

One solver, one rate (3.25 bpw, Hadamard-rotated scalar INT3), one probe budget; only the
objective horizon varies. *Prior art (added 2026-09-07): BRECQ 2021 showed this shape and
its bias–variance explanation on CNNs; BaKron 2026 runs layer / MLP-local / fully-local /
global at one solver on Llama-3 and Qwen3 to 8B with the same MLP-output Hessian we use,
and its ordering flips by model and factorization. See `PRIOR_ART_LADDER.md`.*

| objective horizon | ppl | % of quantization gap closed |
|---|---|---|
| GPTQ (baseline) | 16.7968 | — |
| layer-local + CD | 16.7700 | 0.5% |
| **module output** | **16.2785** | **10.4%** |
| block output | 16.3174 | 9.6% |
| full-model Fisher (same budget) | 16.7883 | 0.2% |
| full-model Fisher (8× budget) | 16.5275 | 5.4% |

The full-model ("theoretically correct", YAQA-style) objective is *worse than layer-local*
at matched cheap estimation and reaches only half the module effect at 8× the cost — while
improving its own objective the most. **The radius curve is really a horizon ×
estimation-noise frontier**: distant curvature is mostly sampling noise at cheap budgets,
and the discrete optimizer exploits noise. The module endpoint is the sweet spot: most of
the "end-loss-aware" value at ~0.1% of the estimation cost of full-model sketching methods
(YAQA uses ~134M-token sketches; our probes use 2k tokens per matrix).

### 3.2 The gain is a global cooperative phenomenon (superadditive, superlinear in coverage)

*Validated 2026-09-08 under the frozen evaluation (D3a, 146 wt2 windows, paired bootstrap): attention half +0.6% of the gap (CI covers 0), MLP half +2.4% [+1.0, +3.8], whole +8.7%; interaction −0.020 [−0.026, −0.014] NLL, strictly negative. On C4 the attention half is harmful (+1.6%), the MLP half is 0, the whole +7.2%. Superadditive, with an interval; the exploratory halves were noisier than they looked.*

Decomposition at 3.25 bpw (each arm refines only the named parts with module-G, everything
else layer-local):

| refined under wider G | % gap closed |
|---|---|
| q,k (QKᵀ-logit endpoint) | 2.5% |
| q,k (softmax-Fisher endpoint) | **−2.1% (harmful)** |
| q,k,v (attention output, pre-o_proj) | 3.8% |
| q,k,v (post-o_proj) | −1.8% |
| gate,up (MLP output) | 1.7% |
| layers 0–11 only (everything) | 1.8% |
| layers 12–23 only (everything) | 2.3% |
| **everything, everywhere** | **10.4%** |

No isolated sub-mechanism carries the effect: the parts sum to ≈0 within a block and to
~40% of the whole across depth halves. Partial coverage of any kind forfeits most of the
benefit. (This also explains why prior block-local or attention-only objectives don't see
the full effect.) The softmax-Fisher failure is the estimation-noise law again: one sampled
key per query row is too noisy, and it compounds over 24 layers — notably, BoA's published
relaxation *discards* the softmax Jacobian for memory reasons, and our measurement says
that relaxation is the right call.

Sensitivity theory (Spearman over ~20k attention rows): top-logit magnitude predicts
nothing (ρ≈0); competition does (top1−top2 margin −0.67, Jacobian trace +0.69), and
**p-weighted value dispersion is the strongest predictor** (+0.70/+0.64), still predictive
at fixed softmax-KL (+0.13…0.19) — i.e., attention-distribution changes matter mainly where
the competing V vectors disagree functionally (previously only a KV-cache-pruning
heuristic, VATP; first measured here as a weight-PTQ objective property).

### 3.3 The scope effect is representation-independent (and representation sets the floor)

Matched-rate factorial at ~2.5 bpw:

| @ ~2.5 bpw | GPTQ | +layer refine | +module refine |
|---|---|---|---|
| scalar INT2/g64 (2.500 bpw) | 4,197 | 3,288 | 2,054 (broken) |
| **VQ-2D** (d=2, K=32, 2.514 bpw) | 19.16 | 18.76 | **18.06** |

- The INT2 collapse is a *representation* limit (~200× at matched rate), not a compensation
  limit; a deliberately minimal GPTVQ-like 2-dim VQ removes it.
- The module-scope increment is **+9.5 pp of the gap on VQ vs +9.9 pp on scalar** — the
  scope principle transfers across representations essentially unchanged (additive, not a
  substitute, not amplified).
- External reference (Qualcomm GPTVQ code, patched for Qwen2, its own eval protocol):
  90.9 ppl @ ~2.2 bpw, 61.5 @ ~2.3, 16.45 @ ~3.05. Our 18.06 @ 2.514 is categorically
  better at the low end even allowing protocol slack (unified re-eval pending).
- Best usable result under 3 actual bpw on this model, no retraining: **18.06 @ 2.514 bpw**.

### 3.4 Supporting results that shaped the method

- **Row-level discrete optimality:** converged coordinate descent sits within 0.015% of a
  certified exact CVP oracle (sphere decoder) on 64-decision subproblems; k=2 block moves
  add +1–2% only at 2-bit native (consistent with CDQuant). The local problem is solved;
  the room is cross-output/cross-scope. An adaptive "expensive search where predicted"
  algorithm has a working trigger (row damage) but a prize of only 0.25–0.5% — not viable.
- **Rotations:** Hadamard's value is not grid alignment (grid distance stays at chance); it
  makes *compensation* 3.2× more effective. The post-GPTQ rotation landscape is a flat ±1%
  plateau (200 random rotations, DFO, basin tests agree); substantially better bases exist
  (−14% via a surrogate-trained rotation) but are unreachable by unguided hard-objective
  search — and fusible rotations must be shared stream-wide to be storage-free.
- **Calibration:** the held-out fit gap follows 1/(tokens per input dimension) exactly,
  with **no bitrate dependence** (2-bit is not more calibration-sensitive than 4-bit);
  ~20% of integer codes differ across independent calibration draws with <1% quality
  consequence (solutions are non-unique, not unstable). Long calibration windows are less
  sample-efficient per token (~0.55× at 4× length — within-window correlation).
- **A reproduced "GPTQ worse than RTN" anomaly** traced to a group-fit implementation
  interaction (scale fitted on a truncated group), plus a real scale-feedback effect at
  2 bits (frozen original-weight scales win 24/24 cells, −10…−24%) whose remedy coincides
  with published pre-sweep grid methods — kept as a diagnostic, not a claim.
- **Rate allocation** (WaterSIC-style oracle at per-row/per-matrix granularity): ≤0.11
  equivalent bpw, low-bit only, zero at 3.25 bpw — killed by its pre-registered criterion.

## 4. Positioning: what is ours and what is not

**Not ours (do not credit us with):** GPTQ≡Babai (ICLR'26 ×2); exact per-coordinate CD
(QuantEase); GPTQ-init+CD (ReQuant/SchurQuant); k=2 moves (CDQuant); attention-output >
layer-local pairwise (BoA, APTQ); full-model two-sided Hessians (YAQA, BaKron); end-loss
weighting (GuidedQuant); module-wise objectives per se (CBQ/ACBQ); Hessian-aware VQ
(GPTVQ/VPTQ); fused rotations (QuaRot/SpinQuant); shape–gain VQ (PCDVQ); G-cancellation
(KronQ); weighted rate allocation theory (WaterSIC).

**Also not ours, corrected 2026-09-07 after external review:** the non-monotone horizon
curve and its bias–variance explanation (BRECQ 2021); its dependence on calibration budget
(MREM 2022); the gated-MLP output Hessian (BaKron 2026, App. A.1); local-vs-global at one
solver on LLMs (BaKron App. D). Full comparison in `PRIOR_ART_LADDER.md`.

**Ours (checked against 2023–2026 literature by targeted searches):**
1. The **estimation-budget dependence** of the horizon ordering at a fixed calibration set,
   with R\* = module at matched cheap budget, and the measurement that the full-model
   metric is calibration-text-limited, not probe-limited (D3b: 4× tokens 0.3% → 5.5% of
   the gap, 4× probes nothing; D3b-ext/ext2: the attention half of the module arm goes from
   +0.6% to +6.6% with 4× tokens, while the whole module arm gets *worse* at the same
   budget, so the frozen budget stands) — one model, one rate.
2. **Superadditivity / coverage-superlinearity** of functional-scope refinement.
3. The **intra-attention objective ladder** (logit / softmax-Jacobian / exact-KL /
   attention-output rungs isolated), including the measured cost of BoA's discarded
   Jacobian and the **value-aware sensitivity theory** for weight PTQ.
4. **Representation-independence** of the scope increment (scalar vs VQ factorial).
5. A certified **box-constrained CVP optimality gap** for GPTQ/CD under clipping (where the
   published Babai bound does not apply).

## 4b. What the validation phase overturned

Recorded because it is the most useful part of the record for anyone building on this:

- **Layer-scope refinement is mildly harmful**, not neutral: −0.5% / −1.8% / −2.9% of the gap
  on the three models. Anyone running "GPTQ + coordinate descent on the layer objective" and
  measuring end-to-end should expect a small loss, not a small gain.
- **An out-of-sample check at the same estimator resolution proves less than it looks.** Our
  fresh-G test (same probe budget, disjoint text) passed on the model where the method fails.
  Two independent estimates agree on their dominant directions and are wrong about the tail in
  the same way. Resolution and sampling are different failure modes.
- **Provably-noisy components of the metric are not safe to remove.** At the pre-o_proj
  endpoint 18% of the dense metric's energy sits in entries that are exactly zero by the
  algebra of attention. Zeroing them made results *worse* on both models — the off-block mass
  acts as an implicit regulariser, and the endpoint change it was bundled with broke the
  consistency between the attention and MLP halves of the block, which matters more.
- **Objective-level diagnostics did not predict end-to-end outcomes, again.** Scoring the same
  weight change under an 8×-better-estimated metric shows the "improvement" on the widest
  matrices to be entirely illusory — on the model where the method *works* as well as on the
  one where it fails. This is the sixth time in this project that a local proxy misranked.

## 5. Honest limits

- **Effect sizes are baseline-dependent, and ours are measured against a strong one.** The
  neighbouring project traced this continuously on its own grids: the same calibration
  manipulation is worth several times more as the base quantizer gets weaker, i.e. large
  published effects are substantially an artefact of a weak quantizer. Our comparisons are
  against Hadamard+GPTQ at 3.25 bpw, not against RTN or an unrotated baseline, which is the
  conservative choice — the same post-pass measured against RTN would look several times
  larger and mean less. Conversely it means our numbers should not be compared to
  effect sizes quoted against weak baselines.
- **No predictor of when the method helps.** Two of three models positive, one negative and
  unexplained; that is the single biggest weakness and it is not a presentation problem.
- Three models, all ≤1.5B; nothing at 7B+, where the dense output metric would also hit a
  memory wall (604 MiB per MLP matrix at 12288 wide) and would need a structured form.
- **One calibration corpus (wikitext-2 train) for all decisions; C4 is evaluation only — so
  every C4 number here is systematically pessimistic.** A neighbouring retrieval project found in-domain calibration text worth more than
  much larger volumes of generic text on its task (their report; single calibration draw).
  Our own matched-domain measurement (D3c) later showed a small effect for both arms on
  the decoder, so this line is kept as history, not as a claim.
- Only 3.25 bpw is validated. The 2.5 bpw and VQ results are exploratory, single-model.
- QTIP/trellis untested as the strong-representation column; YAQA compared as *geometry at
  matched estimation budget*, not as their shipped system; external GPTVQ numbers are on its
  own evaluation protocol, not unified with ours.
- VQ-2D is deliberately minimal, not a competitive codebook.
- The o_proj-endpoint sign flip is now partly understood (endpoint consistency between block
  halves) but not quantitatively explained.

## 6. Questions we want to pressure-test with others

0. **The one that matters most:** what distinguishes Qwen2.5-1.5B, where the method loses
   2.5% of the gap, from Qwen2.5-0.5B (+8.7%) and Llama-3.2-1B (+4.6%)? Size, family, width
   and metric conditioning are all ruled out by the pair (Llama-1B works, Qwen-1.5B does not,
   and they are nearly identical in MLP width and samples-per-dimension). Depth and head_dim
   are the untested candidates. What is the cheapest experiment that would settle it?
1. Is there a cleaner explanation for the superadditivity than "consistent refinement keeps
   errors in mutually-cancelling directions everywhere"? What experiment would falsify it?
2. Why does moving the attention endpoint past o_proj flip the sign of the q/k/v
   refinement? (Conditioning of G? o_proj mixing making the metric long-tailed?)
3. Is the horizon × estimation-noise frontier formalizable (e.g., bias–variance of plug-in
   quadratic metrics under discrete argmin, predicting R\* from probe budget)?
4. Does the module optimum move with model scale (0.5B → 7B), depth, or GQA ratio?
5. At what probe budget would the full-model geometry overtake the module — and is that
   crossover cheaper than YAQA's sketching?
6. Is the value-aware sensitivity result (V-dispersion at fixed KL) strong enough to build
   an objective on, or only a diagnostic?
