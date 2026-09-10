# Substantive findings

*Status: 2026-09-01. Qwen2.5-0.5B, wikitext-2/103 calibration, group-wise asymmetric min-max
INT-b (exact bpw = bits + 32/group), identical storage format in every comparison. fp16
baseline perplexity 11.8134 (2048-token windows, held-out test). No paper-level verdict is
issued here; four gating results are still running (see §8).*

Findings are ordered by branch, strongest-evidence first within each. Every claim below is
measured in this repository; prior-art attributions are explicit.

---

## 1. The local discrete problem is essentially solved — and that is a finding

**GPTQ+CD sits within measurement noise of the exact discrete optimum at the row level.**
The exact subset oracle (sphere decoder with certified optimality, free coordinates ordered by
conditional precision) improves converged coordinate descent by **+0.000/0.011/0.015%** on
16/32/64-decision subproblems, where the same oracle improves raw GPTQ by +1.0/1.7/2.7%.
Block-exact moves over whole rows, k∈{2,4,8} exact block moves, simulated annealing and
multi-start all land in the same place: CD from the GPTQ start is the optimum for practical
purposes.

| arm (3-bit, damage rel. to GPTQ) | native | Hadamard |
|---|---|---|
| RTN | 5.192 | 6.933 |
| GPTQ | 1.000 | 1.000 |
| GPTQ + CD | **0.822** | **0.931** |
| CD multi-start | 0.807 | 0.918 |
| CD from RTN basin | 0.986 | 1.324 |

The last row matters: local search **cannot replace** second-order compensation — starting CD
from the RTN basin is worse than GPTQ alone under Hadamard. At 2 bits CD reaches 0.70 of GPTQ
(native); the gap GPTQ leaves grows as bits shrink, and CD closes almost all of it.

**Higher-order moves, reconciled with CDQuant.** Exact k=2 blocks alternated with
single-coordinate sweeps buy **+1.88% at 2-bit native** over converged CD (CDQuant reports
+1.0% at INT2 — consistent), and ≈0 at 3-bit and under Hadamard. Block moves *from GPTQ*
(without CD convergence first) are worse than plain CD in all four (bits × coord) cells.

**Attribution.** GPTQ ≡ Babai's nearest-plane was proved twice at ICLR 2026 (arXiv 2507.18553,
2508.01077) — not ours. CD with exact per-coordinate minimiser is QuantEase; GPTQ-init+CD is
benchmarked by ReQuant/SchurQuant. What is ours: the **certified box-constrained CVP oracle**
as an optimality measurement (Chen et al.'s Babai bound holds only without clipping; OJBKQ
poses the box-constrained problem but does not measure the gap), and the empirical result that
the gap above CD is negligible.

## 2. Local gains do not transfer end-to-end — the decisive negative result so far

Full-model sequential quantization at 3.25 bpw, wikitext-2 perplexity:

| pipeline | layer objective | end-to-end ppl | ppl gap closed |
|---|---|---|---|
| Hadamard GPTQ | — | 16.7968 | — |
| Hadamard GPTQ+CD | −5.65% | 16.7700 | **−0.54%** |
| native GPTQ | — | 23.2516 | — |
| native GPTQ+CD | −13.07% | 23.2590 | **+0.06%** |

A −5.65% layer-objective gain becomes a −0.54% perplexity gain (factor ~10 shortfall); in
native coordinates a −13% local gain transfers as *nothing at all*. This is the fifth
independent instance in this project of a local quadratic proxy misranking methods, and the
reason every surviving branch is now judged **only** on held-out end-to-end perplexity.

Two diagnostics constrain the explanation:

- **Ambiguity-restricted CD is a regulariser.** Restricting moves to the 5% least confident
  decisions moves 1.78% of codes (vs 9.50% for full CD) and gives *better* held-out layer
  damage in native (0.02488 vs 0.02782). Full CD's moves are cooperative (additive/actual
  = −1.50 under Hadamard): they cancel jointly on the calibration geometry, and that joint
  cancellation is what fails to generalise.
- **Calibration overfit is not bitrate-dependent** (§6), so the transfer failure is a property
  of the *moves*, not of the rate.

## 3. Wider-scope compensation: algebraically alive, empirically undecided

The entire content of "optimize over a wider scope" is a different output metric `G` in
`tr(ΔW A ΔWᵀ G)`; interactions are exactly pairwise (quadratic objective — a theorem, not a
measurement). The branch survives because **G-cancellation provably does not apply here**:
KronQ shows `G` cancels from column-wise OBS updates, and we verified the boundary — GPTQ
never sees G; a *diagonal* G reproduces the identity solution to exactly **0 differing codes**;
a full off-diagonal G moves **5.42%** of codes, lowers G-weighted damage to 0.958 and raises
layer damage 3.1%. Only off-diagonal structure acts, and it does.

Two hard-won implementation facts: a rank-64 surrogate of G is *worse than useless* as an
objective (optimizing it makes true downstream error 32× worse — the optimizer walks into the
surrogate's null space; only dense G with trust-region damping works), and the honest G must be
measured inside the sequential pipeline (upstream already quantized, downstream still fp).

**First end-to-end scope result — the first positive transfer in the program.** Widening the
optimization horizon from the layer's own output to the *module* output (Q,K,V jointly toward
the attention output; gate,up jointly toward the MLP output), at bit-identical storage and
starting from the GPTQ+CD layer optimum (`cd_pre`):

| arm (3.25 bpw, Hadamard) | ppl | quantization gap (ppl − fp16) | gap closed vs GPTQ |
|---|---|---|---|
| GPTQ | 16.7968 | 4.9834 | — |
| GPTQ+CD, layer scope | 16.7700 | 4.9566 | 0.54% |
| GPTQ+CD, **module scope** | **16.2785** | **4.4650** | **10.4%** |
| GPTQ+CD, block scope | 16.3174 | 4.5040 | 9.6% |
| GPTQ+CD, full-model Fisher | 16.7883 | 4.9748 | 0.17% |

*(Prior-art note, 2026-09-07: the non-monotone shape and its bias–variance reading are
BRECQ's (2021); BaKron (2026) shows the same local/global flips on Llama-3/Qwen3 at one
solver. See `PRIOR_ART_LADDER.md`. The paragraph below stands as our measurement. Update
2026-09-08, D3b/E52: the "noise-limited" reading below is corrected to text-limited — at
the frozen protocol, 4× probes +0.004 n.s., 4× calibration tokens in the metric −0.019
[−0.024, −0.015], gap closed 0.3% → 5.5%, KL moving with NLL. D3b-ext / ext2 (E53/E54, 2026-09-09, one draw, diagnostics): the attention half alone, given 4× the calibration tokens in its metric, goes from +0.6% to +6.6% of the wt2 gap (−0.023 [−0.028, −0.019]) and from harmful to +4.8% on C4 — but the whole module arm at the same 8k budget is *worse* than at the frozen 2k (7.3% vs 8.7% wt2, 4.4% vs 7.2% C4, paired CIs excluding zero). So the text-limitation is specific to the attention endpoint; the frozen budget stands, no re-freeze, and the validated +8.7% is not a lower bound (an earlier draft said so; withdrawn). The MLP half at 8k is the one diagnostic not yet run.)*

**The radius curve is sharply non-monotone: R\* = module.** The full-model geometry — dense
per-matrix Fisher of the remaining network, measured in-pipeline with the same solver and
the same probe budget as every narrower horizon — transfers *worse than layer scope* while
improving its own objective the most of any arm (0.855 vs module's 0.893). At fixed cheap
estimation, widening the horizon past the module trades signal for estimation noise, and the
discrete optimizer walks into the noise (the rank-64-surrogate failure family, in milder
form). Honest positioning against YAQA: their sketches use ~134M tokens where our sinks cap
at 2048 — this result does not say full-model objectives lose; it says the radius curve is
really a **horizon-versus-estimation-noise frontier**, and at matched cheap estimation the
module endpoint is the sweet spot. The probe-budget scaling test settled the mechanism: at
8× the estimation budget the full-model arm improves from 0.17% to **5.40%** of the gap —
noise-limited, not fundamentally unusable — yet still reaches only half the module effect at
eight times the cost. The radius curve is a two-axis frontier (horizon × estimation budget):
cheap estimation → module dominates outright; the full-model direction pays only with
orders-of-magnitude more measurement, the regime YAQA actually operates in.

The wider scope buys ~19× what layer-scope CD bought, where every *local* refinement so far
failed to transfer (§2). The move accounting is internally consistent: extra codes move
*only* on matrices whose G is non-identity (q/k/v/gate/up: 2.4–3.9% vs 1.1–1.6% under
layer-CD), while o_proj/down_proj — whose own output is the horizon endpoint — flip exactly
as under layer-CD (1.05%/1.50%). The wider objective itself improved 10.7% (obj ratio 0.893)
over its layer-optimal start. Cost: ~9,500 s optimizer time for the full model (~15× plain
CD), offline only.

**The scope gain requires a functioning network — the 2.25 bpw inversion.** At 2.25 bpw
(b2/g128, Hadamard) the scalar pipeline collapses (GPTQ ppl 121,567; layer-CD 13,914) and
module scope *inverts*: 20,654 — worse than layer scope while improving its own wider
objective (0.868). `G` is a linearisation measured through the pipeline's actual hidden
states; once those are degenerate, the metric stops describing the real loss and optimizing
it is actively harmful. Scalar b2/g128 is representation-limited on this model (the phase-2
VQ study exists to separate that from the scope principle), and the module-scope mechanism is
certified only above the usability cliff.

**The radius curve flattens at the module.** The block-1 endpoint lands at 16.3174 — equal to
the module arm within +0.04 ppl, not better, despite covering all seven matrices (o_proj and
down_proj get non-identity G there). The exploitable cross-output structure evidently lives
*inside* the module — the softmax mixing of Q,K,V and the GLU product of gate,up — and the
residual stream past the module endpoint adds nothing at this horizon. Two candidate readings
(radius saturation vs probe-noise growth with horizon depth) are separated by the 2-block arm,
now running: further degradation = noise; parity = saturation. Either way the practical
recipe so far is the *cheap* end of the ladder: module-G, one extra probe pass, ~10% of the
gap.

*Caveats: one model, one rate, Hadamard coordinates; 2-block and native controls still
running. A sweep-count confound (cd_pre+cd = 12 sweeps vs baseline 6) is unlikely — layer CD
converges and breaks early — and the horizon shape (module ≈ block1 ≫ layer at equal sweeps)
is not explicable by sweep count at all.*

## 4. The GPTQ-worse-than-RTN failure: mechanism found, fix is prior art

The reproduced anomaly (`down_proj`, b4/g256 native: GPTQ 7% *worse* than RTN, worse with more
damping) has two separable mechanisms, isolated by arms that differ only in where the group
scale comes from (identical storage everywhere; the `block` arm reproduces the canonical
engine bit-exactly, |Δ| = 0.0):

- **M1 — truncated group fit (implementation trap).** With group > blocksize the engine fits
  each group's min-max scale on its first 128 columns only. Signature confirmed: all clipping
  lands in second group-halves (0.0000/0.0112), scales shrink (ratio ~0.91 — the earlier
  "range inflation" hypothesis pointed at the right quantity with the wrong sign), only the
  g256 rung is hit, native only (heavy-tailed rows make half-ranges differ; Hadamard equalises
  them). Fixing the fit removes the failure entirely: 0.01527 → **0.00537** (RTN 0.01429).
- **M2 — scale feedback from compensation (real, rate-dependent, general).** Scales fitted on
  the compensated *working* weights are inflated ~15–19% at 2 bits (compensation mass grows as
  the grid coarsens — a feedback loop). Freezing scales computed from the *original* weights
  wins **24/24 cells at 2 bits** across g64/g128/g256 and both coordinate systems:

| median frozen/working damage | g64 | g128 | g256 |
|---|---|---|---|
| 2-bit | 0.902 | 0.884 | 0.762–0.827 |
| 3-bit | 0.99 | 0.985 | mixed |
| 4-bit | — | ~1.00 | M1-dominated |

  The win transfers to held-out data (e.g. b2/g128 native: held-out 0.294 → 0.208) and applies
  to the *original* GPTQ semantics, not just our engine.

**Attribution:** the fix is published — Two-Stage Grid Optimization (arXiv 2602.02126)
initialises group scales before the GPTQ sweep and refits post-hoc (4–6 pp at INT2, consistent
with our −11…−24%); GPTQ's `--static-groups` flag has always precomputed grids (motivated by
act-order, not quality). Ours is the forensic decomposition and the anomaly's resolution — a
diagnostic, not a headline.

**Engine repaired, tables corrected — with one ironic footnote.** The canonical engine now
fits full groups (verified bit-exact against original semantics at g256 and against its old
self at g≤128), and the capacity table's "GPTQ worse than RTN" section is empty: b4/g256
native goes from −7.1% to +62.5% removed. But at b2/g256 the *correct* semantics is ~35%
worse than the bug was — the truncated fit had been acting as a crude beneficial clip-search,
confirming from a second direction that plain min-max scale estimation is the weak link at
2 bits and pre-computed/optimised scales dominate either variant.

## 5. Rotations: what Hadamard actually does, and the landscape around it

**Hadamard is not a grid transform.** It leaves grid distance at chance (0.246 vs 0.25) and
improves RTN error only 1.27×. Its entire value is that **compensation works 3.2× better**
after it (GPTQ gain ×1.94 → ×6.18). The specific rotation is irrelevant — 4 Hadamard seeds and
3 Haar rotations land within ~1% (a tight plateau).

**Hadamard is not a strict local optimum of the hard post-GPTQ objective.** Random tangent
perturbations with fresh hard requantization: at ε=0.01, 9/50 directions improve (best
−0.347%, noise floor 0); at ε=0.03, 15/50 (best −0.284%). But fixed-code alternating descent
(T5) is an optimizer trap — it moves 0.6% and stays at Hadamard; started from identity it ends
*worse* than Hadamard. The learned grid-surrogate rotation (T3) reaches a genuinely different
basis at **−14–15% below Hadamard** on all 3 matrices tested, far outside the random band —
proof a better basis *exists*, on an objective (O1) that is not the hard pipeline objective.
Resolution of the two hypotheses: the plateau is locally flat (H-A holds locally), and
fixed-code optimizers cannot travel between basins (H-B holds globally).

**Basin test (complete).** Ten starts — Hadamard, 3 Hadamard seeds, 3 Haar, identity, and
perturbed-Hadamard at ε=0.05/0.2 — under alternating optimization with *fresh hard
requantization* each outer step: every rotation-family start converges into a **±0.5% band**
(0.00608–0.00614) despite high code mobility (24–39% of codes move per outer step — this is
not the fixed-code trap); identity converges to a separate basin 2× worse (0.0123). The
plateau is a wide flat basin that alternating optimization cannot leave, in either direction.
**Hard-objective search (complete — the branch's last gating item).** Derivative-free search
(400 evaluations of the full requantize→GPTQ→CD pipeline, 16 accepted steps) finds −0.65%;
best-of-200 random rotations finds the *same* value (0.006103 — a common floor); the
200-rotation plateau spans ±1.1% with Hadamard slightly better than the median. Verdict:
unguided search on the hard objective lands in the weak band (<1–2% over Hadamard) every way
it was tried — tangent, basin, DFO, random sweep. Hadamard is not a local optimum, but the
flat ±1% plateau is a property of the landscape, not of any optimizer. The only demonstrated
escape is **surrogate-guided** (T3's grid loss, −14% on the hard metric). The well-posed
remaining question is no longer "does a better basis exist" (it does) but whether T3-class
gains survive (a) the shared-stream constraint of a fused deployment and (b) end-to-end
perplexity — a new experiment, not a continuation of this search.

**Deployment accounting** (corrected): three classes — *fused* (SpinQuant-style, zero storage,
legal only on linear paths and only as one rotation shared stream-wide), *online* (the only
option at `down_proj` input; free if structured, d²/token if dense), *stored per matrix* (what
our per-matrix T3/T5 would really cost: 0.286 bpw). At 3 bits on `q_proj`: fused T3 gain is
+0.11 equivalent bits at zero cost (an **upper bound** — the shared-stream constraint was
never trained against); the stored deployment is dominated by simply refining group size
(+0.157 equivalent bits per 0.25 bpw). Learned rotations only make sense fused.

## 5b. Inside attention: where the module gain comes from

**End-to-end ladder** (3.25 bpw; each arm refines only the named matrices with the wider G,
everything else at layer-CD; identical solver/rate/calibration):

| rung | refined | ppl | gap closed vs GPTQ |
|---|---|---|---|
| layer-CD (all) | all | 16.7700 | 0.54% |
| QKᵀ logits | q,k | 16.6713 | 2.52% |
| softmax-Fisher | q,k | 16.9164 | **−2.13%** (harmful) |
| attention output (pre-o) | q,k,v | 16.6066 | 3.82% |
| module endpoint (post-o) | q,k,v | 16.8886 | −1.84% |
| module endpoint (MLP out) | gate,up | 16.7111 | +1.72% |
| **module (both halves)** | q,k,v + gate,up | **16.2785** | **+10.4%** |

*(Validated 2026-09-08 under the frozen evaluation (D3a, 146 wt2 windows, paired bootstrap): attention half +0.6% of the gap (CI covers 0), MLP half +2.4% [+1.0, +3.8], whole +8.7%; interaction −0.020 [−0.026, −0.014] NLL, strictly negative. On C4 the attention half is harmful (+1.6%), the MLP half is 0, the whole +7.2%. Superadditive, with an interval; the exploratory halves were noisier than they looked. E49.)*

**The module gain is strongly superadditive.** The two halves refined separately sum to
≈ 0% (−1.84 + 1.72), refined together they close 10.4% — no isolated sub-mechanism (QK
logits, softmax, V-aggregation, or the MLP alone) carries it. The gain is an emergent
property of refining both halves of every block against their functional outputs, with
and the half-depth test sharpens it: module-G in layers 0–11 only closes 1.83%, in 12–23
only 2.25% — each half captures ~20% of the full effect at 50% coverage, so the gain is
**superlinear in coverage** and superadditive across both partitions tested (across depth:
joint = 2.5× the sum; within block: the sum is ≈ 0). Simple early-layer compounding is
refuted (the halves are nearly symmetric). The standing summary: the module-scope benefit
requires *consistent joint refinement everywhere* — partial coverage of any kind forfeits
most of it, and attention-side-only coverage is actively harmful. Also notable: moving the
attention endpoint past o_proj flips the q/k/v refinement from +3.8% to −1.8% at identical
refined sets — the one unexplained detail queued for a targeted follow-up.

The emerging law, now seen three times (full-model Fisher, softmax-Fisher, rank-64
surrogate): **transfer tracks the estimation quality of the geometry**. Exactly-measurable
objectives (logits: a linear map given the partner; attention output: dense Rademacher
probes on a compact endpoint) transfer; sampled-Fisher metrics at cheap budgets are noise
the optimizer exploits, and 24 layers compound it — softmax-G ends worse than plain GPTQ
even though the same geometry looked fine in the single-layer study.

A controlled ladder — layer → QKᵀ logits → softmax-Fisher → attention output — quantizing
only q/k at identical rate/rotation/solver, scored on held-out text (3 layers, monotone, no
inversions): **the dominant step is layer → logits** (−40–50% on every downstream metric —
plain Q/K cooperation on QKᵀ, incidentally the exact term BoA's shipped relaxation keeps
after discarding the softmax Jacobian); the softmax metric adds a few percent; the
attention-output metric adds another ~7–13% on the functional error. Sensitivity theory
(Spearman over ~20k attention rows/layer): top-logit magnitude predicts nothing (ρ≈0 —
"protect the big logits" is falsified); competition does — top1-top2 margin −0.61/−0.67,
Jacobian trace +0.69, and **p-weighted value dispersion is the strongest predictor overall**
(+0.70/+0.64). The §9 control: at *fixed* softmax-KL, value dispersion still predicts
attention-output damage (ρ +0.13…+0.19 in every layer) — the first measurement of the
"score × value" argument (previously only a KV-cache-pruning heuristic, VATP) as a
weight-PTQ objective property. Preserving the attention distribution is the wrong target
precisely where the competing value vectors disagree — but at 3 bits this refinement is
second-order next to Q/K logit cooperation.

## 5c. Scope × representation: the scope effect is representation-independent (R2)

VQ-2D factorial at 2.514 bpw (d=2, K=32 codewords, row-amortised amplitudes frozen from the
original weights, per-matrix diag-Hessian-weighted codebooks; full storage accounting incl.
codebook):

| VQ-2D arm | ppl | VQ gap closed | scope increment |
|---|---|---|---|
| GPTQ assignment | 19.1554 | — | |
| + layer refine | 18.7567 | 5.4% | |
| + **module refine** | **18.0595** | **14.9%** | **+9.5 pp** |

Two results in one table. First, the representation sets the usability floor: scalar
collapses at 2.25 bpw (ppl 121,567) while VQ-2D is a working model at 2.51 (19.2 before any
refinement) — the INT2 cliff was representation-limited, confirmed independently by the
external GPTVQ reference (own protocol: 90.9 at ~2.2 bpw, 61.5 at ~2.3, 16.45 at ~3.05 —
at ~3 bpw it lands in the same band as our scalar+module 16.28 @ 3.25). Second, the
module-scope increment over layer scope is **+9.5 pp of the gap on VQ vs +9.9 pp on
scalar** — nearly identical across representations (scenario R2): the functional-scope
effect sits additively on top of the representation gain, neither absorbed by the stronger
code (R1) nor amplified by its extra degrees of freedom (R3). A secondary observation: on
the richer discrete space even *layer*-scope refinement transfers (5.4% vs scalar's 0.54%).

**The completed matched-rate factorial (~2.5 bpw):**

| @ ~2.5 bpw | GPTQ | +layer | +module |
|---|---|---|---|
| scalar b2/g64 (2.500) | 4,197 | 3,288 | 2,054 |
| VQ-2D (2.514) | 19.16 | 18.76 | **18.06** |

The scalar column adds two facts: the 2.25-bpw module *inversion* is gone at 2.5 — module
scope again helps (−38%) even in a badly broken model, locating the G-usability boundary
between 2.25 and 2.5 bpw; and the representation gap at matched rate is ~200×.

**The phase's five questions, answered:**
1. *Scalar GPTQ representation-limited at INT2?* Yes — categorically (121,567 vs 19.2 at
   matched ~2.25–2.5 bpw; two independent VQ implementations agree).
2. *Does the module-scope benefit survive VQ?* Yes — +9.5 pp vs scalar's +9.9 pp.
3. *Substitute, independent, or synergy?* Independent additive gains (S1/R2).
4. *Does the scope benefit grow with VQ dimension?* Untested (4D deferred) — R2's "no
   amplification at d=2" removes the urgency.
5. *Best usable PPL under 3 actual bpw on this 0.5B model, no retraining:* **18.06 at
   2.514 bpw** (VQ-2D + module scope), vs external GPTVQ 90.9 / 61.5 / 16.45 at ~2.2 / 2.3 /
   3.05 bpw (protocol differences pending unified re-eval).

## 5d. Rate allocation: killed by its own criterion

The WaterSIC-style oracle (per-row Lagrangian allocation from an 8-rung (bits, group) menu
at matched total bits, damages measured post-GPTQ+CD): at 2.5 bpw the per-row prize is
+0.13–0.17 equivalent bpw on attention matrices, +0.11 on up_proj, +0.03 on the widest
matrix (down_proj), and per-matrix allocation buys −6.3% damage; at 3.25 bpw the allocator
degenerates to uniform and the gain is exactly zero. Mean ≈ 0.11 eq-bpw, low-bit-only —
below the pre-registered continue threshold (0.25), and an AG variant differs only by
diagonal-G row weights, which cannot plausibly bridge the gap. **Branch closed**; the
diagnostic stays: a real but small per-row allocation prize exists only in the deep low-bit
regime.

## 6. Calibration sensitivity is finite-sample noise, not a low-bit phenomenon

*(Both decisive cells are in: 262k meets every closure condition the brief set (~50–100
samples/dim: gap → 1, draw variance < 1%, eigenspaces stabilising) and 524k confirms the
trend. Only the seqlen-2048 control pass is still running.)*

- **No compensation cliff.** Across 2.25→4.25 bpw, compensation removes 64.8→73.2% of raw
  damage (native) and 74.5→79.0% (Hadamard) — a gentle slope, no collapse at 2 bits.
- **The held-out ratio is flat in bitrate** (native ~2.05–2.11, Hadamard ~1.78–1.84 at the
  8k-token budget) on 12 matrices in both coordinate systems. The §12 hypothesis — that
  aggressive low-bit compensation overfits calibration — is falsified at this scale.
- **The gap is a pure function of tokens per dimension** and follows 1/N exactly (excess gap
  halves per doubling): `down_proj` 1.29 → 1.15 → 1.08 → **1.041** at 6.7/13.5/26.9/53.9 t/d;
  `q_proj` 1.07 → 1.04 → 1.02 → **1.014**. At *matched* t/d the two matrices coincide (1.076
  vs 1.069) across a 5.4× dimension difference. At GPTQ's community-standard 262k-token
  calibration the `down_proj` gap is 4.1% — **identical at all nine bitrates to three decimal
  places** (1.041–1.042 from 2.125 to 4.25 bpw), with draw-to-draw spread ≤ 0.28%. At 524k it
  falls to 1.024–1.025 (`q_proj` 1.010), still flat in bitrate, codes still 16% different.
- **Solutions are non-unique, not unstable.** Across disjoint calibration draws ~20% of integer
  codes differ while held-out quality differs by <1% — at *every* bitrate including 2-bit,
  with no trend. Parameter instability without performance instability.
- **Geometry reproducibility improves with budget** (leading-subspace overlap `down_proj`
  0.785 → 0.862 rank-128 from 32k → 128k) and lags the diagonal (r ≈ 1.000) and off-diagonal
  (r 0.94–0.99) — but the lag has no measurable quality cost.
- **Token budget ≠ token budget: sequence length matters.** The same 32k tokens drawn as
  2048-token windows instead of 512-token windows give a *worse* gap (q_proj 1.13 vs 1.07,
  `down_proj` 1.37 vs 1.29) and ~2× the draw-to-draw spread, with matched-seqlen yardsticks.
  Tokens within a window are correlated: 16 long windows carry less information than 64 short
  ones (effective sample ≈ 0.55× at 4× the length — between iid and fully-window-correlated).
  Practical corollary: calibration budgets quoted in tokens are not comparable across sequence
  lengths, and short-window calibration is more sample-efficient for the second moment.
  *(Full 2048 pass still running; numbers from the 32k cell.)*

Consequence: uncertainty-aware / robust GPTQ has no observed target here; the likely closure
is "use enough tokens per dimension". Also: there is no bitrate at which the cross-scope
experiment would see more headroom — the §13 relocation is moot.

## 7. Adaptive expensive search: the prize is concentrated but tiny

Per-row analysis, 1440 rows over 9 matrices (complete):

| cell | mean gap CD→strong | 50% of gap mass in | 80% in | best cheap trigger (top-10% budget) |
|---|---|---|---|---|
| 2-bit native | 0.25% | top **1%** of rows | top 4% | `d_gptq` 48% (oracle 97%) |
| 2-bit Hadamard | 0.49% | top 3% | top 7% | `ambig_frac·d_cd` 43%, `d_gptq` 33% (oracle 91%) |
| 3-bit native | 0.31% | top 2% | top 5% | `rel_rtn` 34%, `d_gptq` 30% (oracle 94%) |

The gap is **extremely concentrated** — the "10% of rows carry 80% of the gap" hypothesis is
exceeded (it is 4–7%) — and a usable cheap trigger exists (row damage, plus ambiguity×damage
under Hadamard, capturing 3–5× the random floor). But the total prize is ≤0.5% of the local
objective in every cell, so even the oracle ranking buys nothing that could survive to
end-to-end perplexity. Margin features alone, the intended smart trigger, are near-useless in
native coordinates (0–1%) and mediocre under Hadamard (22%). An adaptive "GPTQ+CD everywhere,
exact search where predicted" algorithm therefore has a working trigger and nothing worth
triggering on. The scientific content is the certification that CD is row-wise near-optimal
(§1), which is what pushes all remaining hope to cross-output/cross-layer structure (§3).

## 8. Closed branches and the one finding they left behind

Tile/fractal/IFS structure, output-channel graph communities, programmatic bit-allocation maps
and permutation layouts are all closed (verdict GA-B: the topology is real, but everything
practically relevant about it is first-order anisotropy, which rotation removes more cheaply).
The finding worth keeping (**GA-F**): compensation is not a refinement of independent
quantization but a different regime — additive per-channel damage is **7.7–10.8×** the damage
actually achieved, i.e. GPTQ destroys an order of magnitude of error through interactions it
creates itself. Every grouping study reached this same conclusion from a different direction.

## 8b. Phase decision table (per the reframing brief §13)

| question | closest existing work | our experiment | verdict |
|---|---|---|---|
| Does the intermediate module objective beat layer-local **and** full-model? | BoA/APTQ (attention-vs-layer pairwise), YAQA (full-model, huge sketches) | 5-point radius curve + probe-budget scaling, one solver | **Yes at matched estimation budget**: module 10.4% vs layer 0.5% vs full-model 0.2% (1×) / 5.4% (8×). The curve is a horizon × estimation-budget frontier; full-model pays only in the YAQA-scale regime. |
| Is the module optimum stable across bitrate? | — | 2.25 / 2.5 / 3.25 bpw arms | Stable in the usable regime (+9.9 pp @3.25 scalar, +9.5 pp @2.51 VQ, −38% damage even on the broken 2.5 scalar); inverts only below the usability cliff (2.25). |
| Does it survive a modern VQ representation? | GPTVQ (representation), GuidedQuant (end-loss, row-decoupled) | VQ-2D factorial | **Yes — R2**, near-identical increment. (QTIP/trellis untested — the honest caveat.) |
| Selection vs rate allocation? | WaterSIC | RA-A oracle, 8-rung menu | Nearly everything is codeword/rounding selection; allocation ≤ 0.11 eq-bpw, low-bit only, **branch killed** by its own criterion. |
| Best PPL < 3 actual bpw, no retraining? | no published <1.5B VQ numbers | — | **18.06 @ 2.514 bpw** (VQ-2D + module scope); external GPTVQ ~61–91 @ ~2.2–2.3 (own protocol). |
| Smallest claim prior art does not cover? | BoA discards J_σ unmeasured; VATP is KV-pruning only; CBQ/ACBQ per-module objectives exist | — | (a) the horizon × estimation-noise frontier with R\*=module at matched budget; (b) superadditivity / coverage-superlinearity of functional-scope refinement; (c) the intra-attention objective ladder incl. the measured cost of BoA's relaxation and the value-aware sensitivity theory; (d) representation-independence of the scope increment. |

## 9. What actually remains open

All four original verdict gates closed (calibration sweep §6; scope end-to-end §3/5b/5c;
rotation search §5; engine fix §4). The residual open items, none gating:

1. **QTIP/trellis representation** for the scope test's strongest-code column (pure-torch
   quantization path verified feasible; ~1–2 days of porting).
2. **Unified evaluation protocol** for external comparisons (GPTVQ numbers are on its own
   full-test protocol; ours on 24 windows).
3. The **o_proj-endpoint anomaly**: q/k/v refinement flips from +3.8% (pre-o) to −1.8%
   (post-o) at identical refined sets — unexplained, one targeted experiment.
4. Second model (e.g. Qwen3-0.6B) for external validity of the module-scope numbers.
5. GPTVQ dim-4 config (assertion at g65536 on the small widths) if a 4D reference is wanted.

## Appendix: prior-art map (what is *not* ours)

| result | source |
|---|---|
| GPTQ ≡ Babai nearest-plane (back-to-front) | Chen et al., Birnick — both ICLR 2026 |
| Exact cyclic CD on fixed grids | QuantEase |
| GPTQ-init + CD benchmarked | ReQuant, SchurQuant |
| k=2 block moves at INT2 (+1%) | CDQuant |
| G cancels from column-wise OBS | KronQ (Prop. 1) |
| Pre-sweep group scales + post-hoc refit | Two-Stage Grid Optimization (2602.02126); `--static-groups` |
| Fused zero-cost rotations | QuaRot / SpinQuant / OSTQuant / OptRot |
| Babai bound void under clipping (noted, unmeasured) | Chen et al., YAQA, OJBKQ |

---

# Validation phase (2026-09-03 → 09-05)

Everything above was exploratory: one model, a 24-window perplexity protocol, and decisions
made while looking at results. The method and protocol were then frozen at a commit, the
evaluation hardened, and the whole thing re-run on three models with no tuning permitted.
Where the two disagree, this section wins.

**Frozen setup.** Five arms (fp16 / GPTQ / layer / module / block) at exactly 3.25 bpw,
bit-identical storage. Primary metric mean NLL per token. Full contiguous wikitext-2 test
(146 windows), 512 pre-fixed C4 windows, Monte-Carlo KL to fp16, per-window NLLs for a
paired bootstrap *within* each calibration draw, three independent draws per model.

| model | fp16 | GPTQ | + module | wt2 gap closed | C4 | KL | draws (wt2) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 13.07 | 18.57 | **18.01** | **+8.7%** | +7.2% | +8.0% | 3/3 ✅ |
| Llama-3.2-1B | 9.75 | 15.89 | **15.54** | **+4.6%** | +3.4% | +2.4% | 3/3 ✅ |
| Qwen2.5-1.5B | 9.26 | 12.85 | 12.96 | −2.5% | −4.1% | −2.1% | 0/1 ❌ |

Verdicts: Qwen2.5-0.5B **clear pass** (3/3 draws, both corpora, every bootstrap interval
strictly negative). Llama-3.2-1B **conditional pass** (3/3 on wikitext-2, 2/3 on C4 with one
neutral draw; note the GPTQ baseline itself moves 0.064 NLL across draws on C4, ~2.5× our
effect, so single C4 draws are not readable). Qwen2.5-1.5B **fail**, and unexplained.

**The failure resists every explanation offered.** Not size (0.5B and 1B work). Not family
(Llama works). Not width (Llama's MLP 8192 vs 1.5B's 8960). Not metric conditioning (0.25 vs
0.23 samples per output dimension). Attempted fixes that did not rescue it: exact per-head
block-diagonal metric at the pre-o_proj endpoint, the same with a dense estimator, and an 8×
probe budget. Untested candidates: depth (16, 24 work; 28 fails) and head_dim (64, 64 work;
128 fails) — hypotheses only, at n=3.

**What the validation overturned in the sections above.**

1. *Layer-scope CD is mildly harmful*, not neutral: −0.5% / −1.8% / −2.9% across the three
   models on the hardened protocol.
2. *The fresh-G check was weaker evidence than claimed.* It uses the same probe budget on
   disjoint text, so it validates sampling, not resolution — and it passed on the model where
   the method fails.
3. *Removing provably-noisy metric components hurt.* At the pre-o endpoint 18% of the dense
   metric's energy is in entries that are exactly zero by construction; zeroing them was
   worse on both models. The off-block mass acts as an implicit regulariser, and endpoint
   consistency between the attention and MLP halves matters more than metric purity.
4. *Objective-level diagnostics misranked once more.* Rescoring the same weight change under
   an 8×-better metric shows the improvement on the widest matrices to be entirely illusory —
   on the model where the method **works** as much as on the one where it fails. Sixth
   instance of a local proxy failing to predict end-to-end outcome.

**Standing position.** A cheap offline post-pass that closes 3–9% of the quantization gap at
bit-identical storage, replicated across two model families and two tokenizers without
tuning, with one unexplained failure and no predictor of when it applies. The missing
predictor — not the effect size — is the main open problem.
