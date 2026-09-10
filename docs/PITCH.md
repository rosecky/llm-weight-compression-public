# Where we stand, what we can claim, and what we'd need to claim more

*2026-09-05, revised 2026-09-07 after external review. Numbers are from the frozen-protocol
validation phase unless marked exploratory. Written to be argued with: every claim is graded
by the evidence behind it. Prior-art positioning follows `PRIOR_ART_LADDER.md`.*

---

## 1. The class of problem we are genuinely strong at

**Deciding *what objective* a post-training quantizer should optimize, and proving the
answer end-to-end at bit-identical storage.**

Not "a better quantizer" — a better *target* for whatever quantizer you already have. Our
whole apparatus is built to answer "does this objective actually make the model better, or
does it just make its own number go down", and that turns out to be the question the field
gets wrong most often.

Evidence that we are good at this specific thing:

- We built a controlled ladder that isolates every objective between the layer and the
  whole model at matched solver, rate, calibration and probe budget: layer → QKᵀ logits →
  softmax-Fisher → attention output → module output → block output → full-model Fisher.
  Three-rung versions of this exist (BRECQ 2021 on CNNs; BaKron 2026 on Llama-3/Qwen3 with
  the same MLP-output Hessian we use); the rungs *inside* attention are ours.
- On that ladder **the curve is sharply non-monotone with the optimum at the module
  endpoint**, and the full-model objective is *worse than layer-local* at matched cheap
  estimation, reaching half the module effect at 8× the cost. Non-monotonicity and its
  bias–variance explanation are BRECQ's; BaKron's own tables show the same flip at 3B with
  32× our calibration text. What we add is the controlled measurement of *which*
  estimation error governs it (D3b): at one solver and rate, 4× the probe count changes
  nothing (+0.004 n.s.) and 4× the calibration tokens in the metric moves the full-model
  arm from 0.3% to 5.5% of the gap, with its KL to fp16 moving alongside. The global
  objective is text-limited, not noise-limited — MREM's finding, measured cleanly; one
  model, one rate. The same limitation reaches the attention endpoint inside the module (the
  attention half alone goes from +0.6% to +6.6% of the gap with 4× the metric tokens,
  E53) but *not* the whole module arm, which is worse at that budget (E54, paired CIs
  excluding zero); the frozen budget stands and the validated +8.7% is not a lower bound.
- We have caught our own proxies lying six separate times, each documented with the
  experiment that caught it. That discipline is the actual asset: it is why our surviving
  claims are worth something.

## 2. What we can claim strongly today

| claim | evidence | strength |
|---|---|---|
| A cheap offline post-pass closes **5–9% of the in-domain quantization gap at bit-identical storage** | 3 of 4 models, 2 families, 2 tokenizers, 2–3 calibration draws each, paired bootstrap within draw, and the effect is 2.5–4.5× the across-draw spread of that paired difference | **strong, in-domain only** |
| The same gain **out of domain is draw-dependent** and not reportable on 2 of 4 models | on Qwen3-0.6B and Llama-3.2-1B the across-draw spread of the paired C4 difference exceeds the mean effect | **a limit, stated as one** |
| **Layer-local coordinate descent is mildly harmful** end-to-end (−0.5 / −1.8 / −2.9%) | 3 models, same protocol | **strong**, and useful: it contradicts the natural assumption |
| **R\* = module** on our ladder: intermediate objectives beat both layer-local and full-model at matched calibration set | 5-point ladder + D3b (probes vs tokens separated), one model, one rate; consistent with BRECQ (2021), MREM (2022) and BaKron's 3B tables | **strong as a measurement**, single model; mechanism: the full-model metric is calibration-text-limited (D3b), not probe-noise-limited; factorization error untested |
| The gain is **superadditive and superlinear in coverage** — each half alone ≤2.4% of the gap, together 8.7%; interaction −0.020 [−0.026, −0.014] NLL (D3a, frozen eval, 146 windows) | full decomposition (7 arms) + half-depth split | **strong**, single model |
| A set of **hard negatives that save others months**: rotation landscape is a flat ±1.1% plateau; converged CD is within 0.015% of a certified box-constrained optimum; per-row rate allocation is worth ≤0.11 equivalent bits; diagonal output metrics are exactly inert; low-rank surrogates are catastrophic (32×) | each measured directly, several with certificates | **strong** |
| **Methodological**: an out-of-sample check at the same estimator resolution validates sampling, not resolution — it passed on the model where the method fails | direct measurement | **strong**, and transferable to anyone doing calibration work |

## 2b. The number that settles the product question, and it is not favourable

Converting the effect into the only unit a buyer cares about — bits saved at equal quality —
on the model where the method performs best (Qwen2.5-0.5B, frozen protocol, wikitext-2):

| bpw | GPTQ gap (NLL) | with post-pass | gap closed | module − GPTQ, paired bootstrap | **equivalent bits saved** |
|---|---|---|---|---|---|
| 2.50 | 6.4705 | 6.5643 | −1.4% | +0.094 [+0.019, +0.165] | — (model destroyed, post-pass harmful) |
| 3.25 | 0.3510 | 0.3206 | +8.7% | −0.030 [−0.035, −0.026] | **+0.084** |
| 3.50 | 0.2610 | 0.2556 | +2.1% | −0.005 [−0.009, −0.002] | +0.021 |
| 4.25 | 0.0674 | 0.0614 | +8.9% | −0.006 [−0.008, −0.004] | +0.023 (extrapolated: no higher rate measured) |

Every usable point is significant on its own (10k-resample paired bootstrap over the 146
wikitext-2 test windows, `python scripts/analyze_rd.py`); the C4 intervals are strictly
negative at the same three rates. Equivalent bits are read off the piecewise-linear GPTQ
curve; the 4.25 value uses the slope of the segment below it because nothing above 4.25 was
run.

**At best ~0.08 equivalent bits, against a pre-registered product threshold of 0.1–0.2.** We
are below it at every usable rate. The conversion is unfavourable because the
rate–distortion curve is steep (a quarter of a bit buys 0.09 NLL) while the post-pass buys
0.03. That explains why *today's* effect is worth few bits; it is not a ceiling on the
effect. We have one model, four sparse rate points, one representation, and the 4.25 value
is extrapolated. Whether a regime exists where the same principle is worth several times
more (vector-quantized codes below 3 bits, where the exploratory numbers were far larger)
is unmeasured, not ruled out.

Two consequences. The "% of gap closed" figure is **not monotone in rate** (8.7 / 2.1 /
8.9); the 3.25 bpw rate was locked before validation, so the headline is not a selected
maximum, but it does not generalise across rates. And the honest positioning is
*scientific result, not product* at these rates and this representation: a well-measured
statement about which objective a quantizer should optimize, not a technique that saves a
customer money at 3.25–4.25 bpw scalar.

## 3. What we cannot claim, and should not be caught claiming

- **We cannot predict when the method fails.** One model of four (Qwen2.5-1.5B) is harmed —
  clearly on the out-of-domain metric (6× its across-draw spread) and only directionally
  in-domain (1.2×, n=2), which is the mirror image of the positive results — and *nothing*
  separates it from the three that work: not size, family, matrix width, samples-per-dimension, RMSNorm
  extremity, depth, head_dim, depth/width ratio, or baseline damage. Qwen3-0.6B shares both
  architectural features with the failure and is our second-best result. BaKron's tables
  show the same model-dependent flips for the same objective family at 1.7–8B. A predictor
  is not a precondition for deployment — run the post-pass, evaluate both checkpoints on a
  held-out representative set, keep the better — but it is what would make the cost
  worthwhile, and the cost is not small: the post-pass takes 20–30× the GPTQ time (50 min
  on 0.5B, ~2.7 h on 1–1.5B, ~9 h projected on 3B on a 24 GB card), with zero inference
  overhead.
- **Nothing above 1.5B is tested.** A Llama-3.2-3B run reached its GPTQ baseline (wt2 NLL
  2.364 vs fp16 2.056) and its module arm was lost at layer 8/27 when the instance was shut
  down; on a 24 GB card it would need ~9 h, which is the price of the dense module metric. At 7–8B the dense output
  metric hits a memory wall — 604 MiB per MLP matrix — and would need a structured form
  (Kronecker or per-head blocks), which is a different method needing its own validation.
- **One calibration corpus for the headline**, wikitext-2. Measured once (D3c, 0.5B, draw
  0): calibrating on C4 train instead moves both arms alike on wt2 (+0.099 / +0.094 NLL)
  and on the C4 evaluation improves GPTQ by 0.006 and the post-pass by 0.011, so the
  post-pass's advantage grows by 0.005 [0.002, 0.009] and the C4 gap closed goes 7.2% →
  8.9%. Matched domain helps the post-pass slightly more than GPTQ; the effect of domain
  itself is small on this model.
- **No head-to-head against QTIP, YAQA or GuidedQuant as systems.** We compared *geometries*
  at matched estimation budget, which is a fair scientific comparison and an unfair
  engineering one.
- The **VQ and sub-3-bit results are exploratory**: single model, lighter protocol. The one
  external application on an encoder — in a neighbouring encoder-quantisation project the post-pass gave no effect resolvable above calibration-draw variance at 1.8 bpw (E50–E51, aggregate by agreement with
  that project) — is below the rate where our own quadratic model broke, but it is outside
  evidence and it does not point our way. The decoder version of that test (P4, E55) — the same project's VQ quantiser
  on Qwen2.5-0.5B at 2.12 bpw, our post-pass inside its loop — misses the pre-registered
  criterion on wt2 (−3.8% of the gap, n.s.) while being positive on C4 (+7.1%) and KL
  (−9.5%): better on 127/146 and 467/512 windows, catastrophic on three wt2 windows
  (+1.7 nats). The tail is a first-token effect (D4, E56): the ten worst windows all
  start with a digit-like fragment, the refined 2-bit model fails to form an attention
  sink there, and with a fixed "\n\n" prefix on every window the tail vanishes and the
  post-pass wins on both sets (−0.030 [−0.034, −0.025] wt2, −0.030 [−0.034, −0.026] C4,
  ≈3–5% of the gap). The frozen verdict stands; the sink fragility is the 2-bit failure
  mode to report, with its fix (BOS/prefix at inference). The representation, not the
  post-pass, is what makes 2 bits usable (wt2 gap 0.69 vs 6.47 for scalar at 2.50 bpw).
- Effect sizes are baseline-dependent. Ours are measured against Hadamard+GPTQ, a strong
  baseline; the same post-pass against RTN would look several times larger and mean less.
- The across-draw "effect ÷ spread" ratios are a consistency check on 2–3 draws, not a
  test. The C4 results on Qwen3-0.6B and Llama-3.2-1B are part of the record with their
  uncertainty; they neither support nor refute a positive out-of-domain claim.
- The hard negatives are certificates for the settings they were run in: the rotation
  plateau is per matrix under the fixed-code local objective (tangent, basin, DFO and
  200-rotation sweep all reach the same floor); the rank-64 failure does not cover
  diagonal-plus-low-rank forms; the diagonal-metric inertness covers shared-input
  row weighting (KronQ Prop. 1), not GuidedQuant's per-sample weighting, which gives each
  row its own input Gram and does move codes.

## 4. Where we could be best, and what it would take

**(a) The reference answer on *which* estimator error governs the objective ladder.**
*Closest to reach, and narrower than we said before BaKron.* The three-rung ladder and the
non-monotone finding are prior art; the intra-attention rungs, the integer-only
refinement inside the sequential pass and the validation record are ours. D3b answered the first half of the open question: the horizon ordering at cheap budget
is set by how much calibration text the wider metric sees (4× tokens: 0.3% → 5.5% of the
gap; 4× probes: nothing), not by probe variance, and the KL moves with the NLL so there is
no objective/metric mismatch. What remains is factorization error (BaKron's K-FAC/Shampoo
swings) and whether the full-model arm overtakes the module arm on the whole calibration
set; both are one run each on the same protocol. The whole module arm at 8k metric
tokens was run (D3b-ext2, E54) and is worse than at 2k, so the headline does not rise and
no re-validation is triggered; the attention-half gain at 8k (E53) and the whole-module
loss at 8k are the same draw, and the MLP half at 8k is the one diagnostic that would say
which half absorbs the other. Days of compute, no new engineering.

**(b) A deployable "improve an already-quantized model" post-pass.** *Two blockers.* The
first is procedural and our own wording hid it: the validated post-pass runs *inside* the
sequential quantisation, so blocks downstream of a refined block are re-fitted to its new
output. Applying it to a finished checkpoint means either re-running that sequential pass
or re-fitting downstream blocks after each refinement; neither was tested by us, and the
one outside attempt without a downstream re-fit (E50, an encoder VQ point below 2 bpw) was negative.
The second: the method needs a predictor — or at minimum a cheap pre-flight check that says "this model
will benefit" — because a technique that silently harms one model in four cannot ship. The
honest path is more models, not more theory: ~10 checkpoints across families and sizes at
one rate would either reveal the boundary or establish the hit rate as a number
("helps 7 of 10, harms 1, neutral 2"), which is itself sellable if stated plainly.

**(c) The estimation-budget frontier as a general principle.** *Least proven, and now
narrower.* D3b showed the budget that matters is calibration text in the metric, which is
MREM's 2022 finding; our contribution is the clean separation of tokens from probes at one
solver and rate, and it has one model behind it. Two more models and a second solver (BaKron is the natural one: its
MLP-local Hessian is ours in closed form) would make it a contribution independent of our
particular post-pass.

**What we should not chase:** a better codebook, a better rotation, a better per-row
discrete search, or a mixed-precision allocation scheme. We measured all four in the
settings that mattered to us and found nothing; the certificates are scoped as stated in
§3, not general.

## 5. The honest one-paragraph version

We are strongest at measuring *what a quantizer should optimize for* without being fooled
by the objective's own number, and we have the apparatus and the record to show it. Our
concrete result is a post-pass costing 20–30× the GPTQ time and nothing at inference that
recovers 5–9% of what quantization costs you on the calibration domain (1.9–3.7% lower
perplexity), at identical file size and with no kernel change, replicated across two model
families — with one unexplained failure in four models, and with out-of-domain transfer
that depends on which calibration sample you drew. The non-monotone objective ladder we
built on is prior art (BRECQ, BaKron); the rungs inside attention, the integer-only
refinement inside the sequential pass and the validation discipline are ours. The gap between "interesting finding" and
"shippable technique" is two things: knowing which models it helps, and finding a regime
where the gain is worth more than ~0.08 bits. At 3.25–4.25 bpw scalar it is not. The
unmeasured candidate is a vector-quantized code below 3 bits, where our exploratory numbers
were far larger and our validation never went.
