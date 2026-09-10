# Programmatic quantization layout

**Question.** The weight matrix itself is locally structureless (E0–E13). But the map of *how
expensive it is to damage each weight* need not be. Is that map structured enough to be
replaced by a small program or hierarchical partition, and does doing so beat standard
GPTQ + rotation quantization?

**Answer, in one sentence.**

> No: the map of the optimal bit budget is **not** compressible enough to be worth
> replacing with a program — its usable part costs **2.40 bits/weight** to transmit while
> buying **0.78 bits/weight** of distortion, and the part that *is* cheap to describe
> (per-channel, per-group, per-tile) is worth **+0.00 bits/weight** once GPTQ and an
> incoherence rotation are already in place.

**Verdict: PL-D** — GPTQ and rotation already solve the problem well enough that a smart
layout adds nothing — with one qualification from PL-C: an *arbitrary per-weight* oracle is
worth more than a bit per weight end to end, but its map is indistinguishable from noise and
its description destroys the gain three times over.

---

## 0. What this tested, and what it deliberately did not

Tested: the hypothesis that `S = sensitivity(W, X, H, quantization error)` has lower
complexity than `W`, and that a compact description of `S` yields a better low-bit
quantization layout at small metadata and runtime cost.

The order was the one the brief prescribed, with the oracle as an early gate: build several
sensitivity maps, validate them against measured damage, run the free-layout oracle, run the
channel-only oracle, run the matched-null controls, price the description of the map, and
only then consider recursive partitioning. Section 5 explains why the recursive branch was
not built, and states the measured bound that makes it unnecessary rather than merely
unpromising.

Everything runs on one consumer GPU. Peak VRAM: 1101 MiB for the layer-wise work, 4573 MiB
for the full-model perplexity runs (dominated by the 151936-entry vocabulary logits, not by
the method). Every large tensor is size-checked before allocation via `guard_alloc`.

Model: Qwen2.5-0.5B, wikitext-2, seed 0. Layer-wise studies use layers 1/11/22 x
{q_proj, o_proj, up_proj, down_proj} (12 matrices, 21.5M weights); perplexity uses all 168
linear layers with sequential GPTQ.

---

## 1. Sensitivity maps: which ones predict real damage?

Seven maps (`src/lwc/sensitivity.py`), validated against two levels of ground truth.

**Ground truth 1 — exact layer damage.** For every unit of a partition, the exact
`||E_u X||^2` that perturbing *only* that unit produces, computed with the **full** Hessian.
The diagonal proxy `S5 = H_jj (W-Q(W))^2` that the oracle allocator minimises captures
**97.8%** of it in total, and ranks units almost perfectly at coarse granularity:

| granularity | Spearman(S5, exact) | log-Pearson | top-1% overlap | total captured |
|---|---|---|---|---|
| column | 1.000 | 1.000 | 1.00 | 100.0% |
| tile 128x128 | 0.990 | 0.996 | 0.83 | 97.8% |
| group (row x 128) | 0.929 | 0.939 | 0.72 | 97.8% |
| row | 0.732 | 0.802 | 0.52 | 97.8% |

So the off-diagonal Hessian terms are worth 2.2% of the damage. The allocator's objective is
sound; nothing downstream is limited by that approximation.

**Which cheap maps predict the damage** (12 matrices, INT3 g128, tie-aware Spearman):

| map | per column | per group | per row | per tile |
|---|---|---|---|---|
| S1 `\|W\|` | **-0.245** | 0.501 | 0.523 | 0.190 |
| S2 `W^2` | **-0.203** | 0.569 | 0.565 | 0.278 |
| S3 `W^2 E[x^2]` | 0.948 | 0.571 | 0.510 | 0.781 |
| S4 `H_jj` | 0.989 | 0.354 | 0.000 | 0.820 |
| S5 `H_jj E^2` | 1.000 | 0.929 | 0.732 | 0.990 |
| S7 `W^2/[H^-1]_jj` | 0.616 | 0.562 | 0.559 | 0.476 |

Two things worth stating plainly:

1. **Magnitude is an *anti*-predictor of which input channel is expensive to damage**
   (-0.245). Which column matters is decided by the activation scale (`S4` alone: 0.989), and
   large-magnitude columns tend to be the ones whose activations are small. This is the
   quantitative reason magnitude-based mixed precision does not work on the input axis.
2. The axes are complementary and mutually blind. `H_jj` is *exactly* uninformative about
   rows (0.000 — a row's summed `H_jj` is the same for every row, which is why the tie-aware
   ranking matters here; naive argsort ranking invents a spurious ±0.35 correlation out of
   the tie order). On rows, magnitude is the best cheap map.

**Compensation changes the answer.** A map fitted to naive quantization damage predicts
*GPTQ* damage much worse: S5 drops from 1.000 to 0.704 (column) and 0.929 to 0.617 (group).
Any allocator used with GPTQ must be told about GPTQ. Ours is: the allocation is recomputed
in whatever coordinate system and compensation regime it is used in.

**Ground truth 2 — end-to-end intervention.** 96 output channels of
`layers.11.mlp.up_proj`, stratified across the damage range, each quantized alone to 1 bit
while the rest of the model stays fp16; NLL measured on 8 x 1024 held-out tokens.
ΔNLL spans -3.4e-4 to +1.6e-2. Spearman SE at n=96 is ±0.104.

| predictor | Spearman | log-Pearson | identifies the worst row |
|---|---|---|---|
| exact layer damage | +0.351 | +0.537 | yes |
| S5 `H_jj E^2` | +0.325 | +0.523 | yes |
| S2 `W^2` | +0.380 | +0.478 | yes |
| S7 OBS | +0.348 | +0.304 | no |
| S3 activation-weighted | +0.210 | +0.067 | no |
| S1 `\|W\|` | +0.302 | -0.047 | no |

This is the honest limit of the whole framing: **even the exact layer-level damage predicts
end-to-end loss change only moderately** (0.351 ± 0.104), and the differences among the top
predictors are inside one standard error. What every good map does get right is the extreme —
all of `exact`, `S2`, `S5`, `S4` place the single worst row (ΔNLL 1.6e-2, 20x the next) in
their top 1%. The tail is predictable; the bulk ordering is close to noise. That is the same
pattern this project found for weight values, now for sensitivities.

---

## 2. Oracle bound: how much is a *free* layout worth?

The oracle gets the complete per-weight damage table and spends exactly the code bits a
uniform quantizer would (`out * in * b0`), choosing from {0,1,2,3,4,5,6,8,16} bits per unit
by a Lagrangian (BFOS) sweep, with **no charge whatsoever** for describing the result.
`gain_bits` is read off the measured uniform rate–distortion curve of the same cell, so
uniform's own gain is exactly zero by construction.

**gptq + hadamard — the strongest existing baseline:**

| base bits | uniform | per-column | per-row | per-tile | per-group | **per-weight** |
|---|---|---|---|---|---|---|
| 2 | 0.000 | -0.054 | -0.004 | +0.000 | -0.027 | **+0.787** |
| 3 | 0.000 | -0.002 | +0.000 | +0.000 | +0.000 | **+0.777** |
| 4 | 0.000 | -0.006 | +0.000 | +0.000 | +0.000 | **+0.771** |

**gptq + native coordinates:**

| base bits | uniform | per-column | per-row | per-tile | per-group | **per-weight** |
|---|---|---|---|---|---|---|
| 2 | 0.000 | +0.368 | -0.099 | +0.087 | +0.167 | **+1.158** |
| 3 | 0.000 | +0.216 | -0.022 | -0.000 | +0.027 | **+1.181** |
| 4 | 0.000 | +0.195 | +0.028 | -0.002 | +0.038 | **+1.145** |

Fraction of the per-weight oracle's gain that each realisable granularity captures:

| cell | 2 bits | 3 bits | 4 bits |
|---|---|---|---|
| gptq/hadamard, best coarse unit | 0% | **0%** | 0% |
| gptq/native, per-column | 32% | 18% | 17% |
| naive/native, per-column | 23% | 18% | 18% |
| naive/hadamard, best coarse unit | 2% | **0%** | 0% |

**The gate result.** The oracle bound is not small — but it is entirely concentrated in the
one granularity that cannot be described cheaply. Under GPTQ + Hadamard, every
hardware-friendly unit (channel, row, group, tile) is worth **zero to three hundredths of a
bit**. That is L1 passing and L3/L6 already firing at the same time.

**The bit histogram rules out the outlier reading.** At 3 bits the per-weight oracle puts
**0.00%** of weights in the 16-bit class. Its gain comes from a broad spread —
13% at 0 bits, 2% at 1, 16% at 2, 27% at 3, 26% at 4, 14% at 5, 1% at 6 — i.e. classic
reverse water-filling on the *realised* rounding residual, not from protecting outliers.
That observation is what motivated the control in the next section.

---

## 3. Is the map structured, or is it noise?

### 3.1 Matched-null control on the oracle itself

The same pipeline was rerun on `gauss_rowcol`: i.i.d. Gaussian weights with identical
per-row and per-column RMS, quantized against the **real** activation statistics. 8 matrices,
paired, 96 configurations.

| cell | granularity | gain real | gain null | real/null |
|---|---|---|---|---|
| gptq/hadamard | per-weight | +0.811 | +0.810 | **1.00** |
| naive/hadamard | per-weight | +1.582 | +1.576 | **1.00** |
| gptq/native | per-weight | +1.309 | +1.262 | 1.04 |
| naive/native | per-weight | +2.112 | +2.061 | 1.02 |
| gptq/native | per-column | +0.280 | +0.260 | 1.08 |
| naive/native | per-column | +0.416 | +0.404 | 1.03 |

**The entire oracle gain is reproduced by matched noise.** Real weights are worth 0–4% more
than a Gaussian with the same row and column scales. Even the per-column gain in native
coordinates — the one that looked like genuine outlier-channel structure — is 92–97%
reproduced, because `gauss_rowcol` preserves exactly the column scales that gain uses.

The mechanism is now clear and it is a familiar trap in a new costume. Under min-max group
quantization, each weight's rounding residual is close to uniform on `[-Δ/2, Δ/2]`. The
oracle sees the *realisation*: it hands few bits to the weights that happened to land near a
reconstruction level. That is information about the rounding noise, not about the weights.
It is the same class of error as the in-sample k-means result this project caught in E0.

### 3.2 Description length of the map

If the map cannot be predicted, it must be transmitted. Held-out cross-entropy (fitted on a
random half of the positions, scored on the other half, Laplace-smoothed), against the same
map shuffled within rows:

**per-weight map, Hadamard coordinates** (12 matrices, 3-bit budget):

| model | real bits/weight | within-row shuffle | real - shuffle |
|---|---|---|---|
| i.i.d. marginal | 2.4266 | 2.4264 | +0.0001 |
| conditioned on row | 2.4351 | 2.4350 | +0.0001 |
| conditioned on column | **2.4024** | 2.4365 | -0.0342 |
| conditioned on group | 2.5035 | 2.5068 | -0.0034 |
| left neighbour | 2.4265 | 2.4265 | +0.0000 |
| left + up neighbour | 2.4239 | 2.4268 | -0.0030 |

The map costs **2.40 bits/weight** to transmit and buys **0.78**. Net **-1.62 bits/weight**.
And the best model beats the marginal by 0.024 bits while beating a *shuffle of itself* by
0.034 bits: there is essentially no arrangement to exploit. L2 and L4 both fire.

**per-group map** — the one granularity where structure does exist:

| coordinates | marginal | best model | (which) | shuffle | oracle gain at 3 bits |
|---|---|---|---|---|---|
| native | 1.1138 | **0.0225** | left neighbour | 0.9064 | +0.023 |
| hadamard | 0.0111 | 0.0002 | left neighbour | 0.0103 | +0.000 |

In native coordinates the group-level allocation map is **genuinely and strongly structured**
— a 50x compression over its own marginal, 40x better than its shuffle. It is essentially
run-length structure along rows, driven by column scale. But it is worth +0.023 bits/weight,
so compressing it well is compressing almost nothing. Under Hadamard the map is 99.9%
constant: there is no longer anything to describe *or* to gain.

---

## 4. Simple grouping baselines

P0 contiguous groups is the uniform baseline. P1 row-based, P2 column-based, P3 sorting by a
scalar score and P4 2-D tiling after a row/column permutation are all dominated by, or
identical to, the corresponding oracle at that granularity — the oracle is the best possible
member of each of those families, since it optimises the same partition's allocation exactly.
So the oracle rows in section 2 *are* the upper bound for every simple baseline, and they
read:

* under GPTQ + Hadamard: per-column, per-row, per-group, per-tile all worth **0.00 ± 0.03
  bits/weight** at 3 and 4 bits.
* under GPTQ, native coordinates: per-column worth **+0.216 bits/weight**, and that is the
  only simple baseline that does anything. Per-group +0.027, per-tile -0.000, per-row -0.022.

Sorting cannot rescue the coarse granularities either: sorting changes which weights share a
unit, but the oracle already allocates optimally *given* a partition, and the per-weight
oracle bounds every possible partition. The gap between per-column (+0.216) and per-weight
(+1.181) is not a gap that reordering can close, because the per-weight map has no
arrangement structure to sort by (section 3.2).

---

## 5. Recursive / IFS partition

**Not built, and the reason is a measurement rather than a hunch.**

A recursive spatial split, a Morton/butterfly index transform or a shared split grammar are
all *models of arrangement*: they earn their keep by making the partition map cheaper to
describe than its own marginal. The measured headroom for that on the only map with a
worthwhile gain is:

```
description length of the per-weight map          2.4266 bits/weight (marginal)
best decoder-realisable model found               2.4024 bits/weight
same map shuffled within rows, best model         2.4365 bits/weight
arrangement information available                 0.034 bits/weight
gain the map buys                                 0.777 bits/weight
```

A two-neighbour context model is a strictly more general predictor of local arrangement than
a quadtree at the same scale, and it recovers 0.003 bits/weight. Any hierarchical partition
of this map is therefore bounded above by a few hundredths of a bit against a 2.4 bit
description cost. Kill criterion **R-equivalent L2** applies: after row/column effects are
removed the map is not more predictable than its matched null, so the
recursive/programmatic branch stops.

The one map that *is* highly structured — the per-group map in native coordinates — was
measured for exactly the properties the IFS branch cares about, and the answer is that its
structure is **first-order and flat, not hierarchical**:

* description length falls from 1.114 to **0.0225** bits/weight with a *depth-1* context (the
  neighbouring group in the same row). Adding the vertical neighbour changes it to 0.0221.
  There is no depth to recurse into.
* it compresses 40x better than its own shuffle, so the structure is real, but
* the whole object is worth +0.023 bits/weight, and under a rotation it flattens to a
  constant (0.011 bits/weight marginal, 99.9% one symbol).

Runtime classification of the layouts that were evaluated: `uniform` H0, `col` H1 (needs an
offline column reorder so equal widths are contiguous — absorbable into the checkpoint),
`group` and `tile` H2 (a header per block), `weight` H4 (per-weight branching and irregular
bit packing). Nothing in H3–H4 came close to earning its keep, so no H3/H4 design was pursued.

---

## 6. End-to-end results

Full-model sequential GPTQ over all 168 linear layers, wikitext-2 perplexity on 24 x 2048
tokens, fp16 baseline **11.813**. `layout` is the *measured* held-out cost of transmitting
the allocation map; `uniform equiv` is the uniform rate that reaches the same perplexity.

### 3-bit codes, Hadamard coordinates, sequential GPTQ

| alloc | codes+scales | layout | total bpw | ppl | Δppl | uniform equiv | **net** | runtime |
|---|---|---|---|---|---|---|---|---|
| uniform | 3.250 | 0.0000 | 3.250 | 16.859 | +5.045 | 3.250 | +0.000 | H0 |
| per-column | 3.250 | 0.0023 | 3.252 | 17.324 | +5.510 | 3.247 | -0.006 | H1 |
| per-group | 3.250 | 0.0002 | 3.250 | 16.744 | +4.931 | 3.273 | **+0.023** | H2 |
| per-tile | 3.250 | 0.0002 | 3.250 | 16.859 | +5.045 | 3.250 | -0.000 | H2 |
| per-weight | 3.250 | 2.4024 | 5.652 | 12.470 | +0.657 | 4.277 | **-1.375** | H4 |

### 3-bit codes, native coordinates, sequential GPTQ

| alloc | codes+scales | layout | total bpw | ppl | Δppl | uniform equiv | **net** | runtime |
|---|---|---|---|---|---|---|---|---|
| uniform | 3.250 | 0.0000 | 3.250 | 23.568 | +11.754 | 3.250 | +0.000 | H0 |
| per-column | 3.250 | 0.0023 | 3.252 | 18.310 | +6.497 | 3.686 | **+0.434** | H1 |
| per-group | 3.250 | 0.0225 | 3.272 | 24.160 | +12.347 | 3.207 | -0.065 | H2 |
| per-weight | 3.250 | 2.4095 | 5.659 | 12.302 | +0.489 | 4.373 | **-1.286** | H4 |

### 2-bit codes, Hadamard coordinates

| alloc | total bpw | ppl | net |
|---|---|---|---|
| uniform | 2.250 | 55322.5 | +0.000 |
| per-column | 2.252 | 18028.6 | +0.137 |
| per-group | 2.250 | 24644.2 | +0.100 |
| per-weight | 4.652 | **16.227** | -1.272 |

The 2-bit rows other than `per-weight` are not interpretable as quality — everything there is
broken relative to fp16 11.81 — but they are reported because the brief asked for the rate
band and because the per-weight row is striking: **a free layout at 2.25 bpw reaches
perplexity 16.23, better than uniform at 3.25 bpw (16.86)**.

Three things this table says that the activation-error table did not:

1. **The free per-weight oracle is worth much more end to end than in activation NMSE.** At
   3.25 bpw it improves activation NMSE 3.4x but reduces Δppl by **7.7x** (5.045 → 0.657),
   and beats uniform INT4 at 4.25 bpw (12.57). In perplexity terms it is worth over 1 bit per
   weight, not 0.78. Charged for its map it is still 1.29–1.38 bits/weight **worse** than
   doing nothing.
2. **Activation NMSE misranks the coarse layouts, again.** The per-group oracle in native
   coordinates improves activation NMSE by 1.04x and makes perplexity *worse* (24.16 vs
   23.57). This is the fourth time in this project that a mean-squared metric has ordered
   codecs incorrectly. Only the per-weight and per-column rows survive the check.
3. **Per-column allocation in native coordinates is a real, cheap, deployable +0.43
   bits/weight** — and it is beaten outright by simply rotating instead. Uniform + Hadamard
   (16.86) is better than the best native adaptive layout (18.31), and stacking them gives
   -0.006.

---

## 6b. Local codebook shapes (Phase 8)

Groups already carry their own `lo` and `hi`. The remaining freedom is the *shape* of the
levels inside that range: learn `T` shared shapes, let each group pick one for `log2(T)/128`
bits/weight. `T = uniform` is the plain min-max baseline; `T = 1` is a global companded
(NF-family) quantizer; `T > 1` is locally adaptive code semantics.
8 matrices, GPTQ, activation NMSE:

| coords | bits | uniform | T=1 | T=4 | T=16 | T=16 vs T=1 | matched-null T=16 vs T=1 |
|---|---|---|---|---|---|---|---|
| hadamard | 3 | 0.010197 | 0.007112 | 0.006732 | 0.006519 | 1.091x | **1.093x** |
| hadamard | 2 | 0.068074 | 0.029782 | 0.028180 | 0.027568 | 1.080x | 1.078x |
| native | 3 | 0.018507 | 0.012162 | 0.010963 | 0.010316 | 1.179x | 1.104x |
| native | 2 | 0.181245 | 0.053527 | 0.050302 | 0.048815 | 1.097x | 1.096x |

Two conclusions:

* Almost all of the benefit is the step from uniform levels to *one* learned shape (1.43x at
  3 bits under Hadamard) — the classical companding gain of a Gaussian-like marginal, which
  costs nothing to store and is not a layout effect at all.
* **Local adaptivity of the shape adds 1.09x for 0.031 bits/weight**, i.e. about +0.05 bits
  of distortion for 0.031 bits of metadata — net +0.02 bits/weight — and the matched Gaussian
  gets *the same* 1.09x. Group heterogeneity here is sampling noise, not structure.

One caution, consistent with E11: learned shapes improve aggregate error while making the
**tail worse** (relative error on the top 0.01% of weights, 3 bits native: 0.0052 uniform →
0.0196 for T=1, 3.8x worse). No perplexity claim is made for this family on the strength of
activation NMSE alone, because this project has already been burned by exactly that.

---

## 7. Ablation: does any of it survive GPTQ + rotation?

This is kill criterion L6, and it is the cleanest result in the study.

| what | native, naive | native, GPTQ | hadamard, GPTQ |
|---|---|---|---|
| per-column oracle | +0.366 | +0.216 | **-0.002** |
| per-group oracle | +0.138 | +0.027 | **+0.000** |
| per-tile oracle | +0.001 | -0.000 | **+0.000** |
| per-weight oracle | +2.028 | +1.181 | +0.777 |
| per-weight, after its map | -0.38 | -1.23 | **-1.63** |

Adaptive layout and incoherence rotation are **substitutes, not complements**. Both attack
the same thing: the anisotropy of the coordinate system. The rotation removes it globally for
32 bits of seed per matrix; adaptive allocation removes part of it locally for real metadata.
Once rotated, there is nothing left to allocate — and the rotation gets there first and
cheaper. The per-weight oracle survives rotation (it is exploiting rounding-noise realisation,
which rotation does not remove) but it was never usable.

---

## 8. Verdict

**PL-D — no significant adaptive benefit.** GPTQ and incoherence rotation already resolve the
problem to the point where a compactly describable layout adds `+0.00` bits/weight at 3 and
4 bits, and `+0.02` at 2 bits where quality is broken anyway.

Against the seven conditions the brief set for calling this direction interesting:

| # | condition | result |
|---|---|---|
| 1 | oracle allocation clearly beats uniform | **yes**, +0.78 to +2.03 bits/weight |
| 2 | much of it is predictable from a small structured map | **no**, 0.03 of 0.78 bits |
| 3 | simple row/column grouping does not explain it | yes, but they explain ~0 after rotation |
| 4 | recursive layout approaches the oracle | **not attempted**; bounded at ~0.03 bits |
| 5 | description overhead is small | **no**, 2.40 vs 0.78 bits |
| 6 | layout is physically reorderable or cheaply addressed | only for the granularities worth nothing |
| 7 | benefit survives GPTQ + rotation | **no** |

Conditions 2, 5 and 7 fail decisively. Kill criteria **L2** (map not more predictable than
matched null), **L4** (metadata exceeds gain by 3.1x) and **L6** (nothing survives
GPTQ + rotation) are all met.

There is a genuine PL-C component that should not be swept away: an arbitrary per-weight
allocation is worth **more than a bit per weight in perplexity** and nearly recovers fp16
quality at 3.25 bpw. The theoretical room is real. It is simply not addressable, because the
information that would exploit it is the realisation of the quantization noise itself, which
by construction cannot be predicted before decoding the weight it belongs to.

**Recommendation: close this branch.** No recursive partitioning, no programmatic addressing,
no tile grammar. The two findings worth carrying forward are diagnostic rather than
constructive: (i) magnitude is an anti-predictor of input-channel sensitivity while activation
scale predicts it almost perfectly, and (ii) adaptive layout and rotation are substitutes, so
a pipeline that already rotates should not spend metadata on mixed precision.

---

## Reproduction

```bash
export PYTHONPATH=src HF_HOME=$PWD/cache/hf

# Phase 1-2: sensitivity maps, exact damage, end-to-end intervention   (47 s / 187 s)
python -m lwc.experiments.sens_maps --out results/raw/sens_maps.jsonl
python -m lwc.experiments.sens_maps --layers 11 --projs up_proj --intervene-bits 1 \
    --intervene-n 96 --intervene-seqs 8 --out results/raw/sens_intervene.jsonl

# Phase 3-4: the oracle gate and the channel-only oracle              (621 s, 1101 MiB)
python -m lwc.experiments.oracle_alloc --save-maps
python scripts/analyze_oracle.py

# Matched-null control, paired                                        (597 s)
python -m lwc.experiments.oracle_alloc --layers 1,11 --allocs uniform,col,group,weight \
    --null real         --out results/raw/oracle_null.jsonl
python -m lwc.experiments.oracle_alloc --layers 1,11 --allocs uniform,col,group,weight \
    --null gauss_rowcol --out results/raw/oracle_null.jsonl

# Description length of the allocation maps                           (100 s, CPU)
python scripts/analyze_maps.py

# Phase 7: end-to-end perplexity                                      (48 min, 4573 MiB)
python -m lwc.experiments.layout_ppl
python scripts/analyze_layout.py

# Phase 8: local codebook shapes                                      (364 s, 870 MiB)
python -m lwc.experiments.local_codebooks --layers 1,11 --ts 1,2,4,8,16 --comps gptq
```

Raw results in `results/raw/{oracle,oracle_null,maps,sens_maps,sens_intervene,layout_ppl,local_cb}.jsonl`;
tables in `results/tables/{oracle,maps,layout_ppl}.md`; figures
`results/figures/diag_oracle.png` (what a free layout is worth, by granularity, with the
matched-Gaussian null overlaid) and `results/figures/diag_layout_net.png` (the same gains
charged for the measured cost of describing the map).
