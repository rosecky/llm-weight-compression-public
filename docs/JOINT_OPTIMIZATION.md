# Joint optimization of quantization decisions

> Standard GPTQ may be losing quality because it lets quantization errors compensate only in
> too local a space. If we jointly optimize the quantization decisions over a larger part of the
> computational graph, a materially better solution may exist at identical bitrate.

This is the fifth study in this repository and the first that does not look for structure in the
weights at all. The four before it asked whether there is something in `W`, in the sensitivity
map, or in the coupling graph that a cleverer *representation* could exploit; all four said no.
This one keeps the representation exactly as it is and asks a question about the *search*:

**At identical bits, group size, codebook and scales, is the particular array of integers that
greedy GPTQ commits to anywhere near the best one?**

---

## 1. What "identical storage" means here, and why it is not a promise

Section 6 of the brief is the whole experiment, so it is enforced structurally rather than by
assertion. Every arm produces a `QuantState` (`src/lwc/joint.py`) and nothing else:

```python
codes  (out, in)  int16   one integer level per weight, in [0, 2^bits - 1]
scale  (out, n_groups)    one scale per (output row, input group)
zero   (out, n_groups)    one offset per (output row, input group)
```

and the reported bit count is that object's own `storage_bits()`. There is no residual stream,
no sparse outlier list, no side information, no wider group, no higher-precision island, and no
extra metadata of any kind. The optimizer is allowed to be arbitrarily expensive offline; the
only thing it is given that GPTQ is not is **freedom in choosing the integers**. Two consequences
worth stating plainly:

* the `bpw` column is *literally identical* across every arm at a given (bits, group), so any
  difference in perplexity is a difference in the quality of the discrete solution;
* an arm that wins is not a new codec. It is the same checkpoint format with different numbers
  in it, which is the strongest possible form of the claim and also the least interesting one to
  fake.

Two arms deliberately relax one thing each, and are reported separately: `+refit` re-solves the
stored `(scale, zero)` by weighted least squares with the codes fixed (the same *count* of
scales, so the storage is unchanged, which section 6 explicitly permits), and `naive+cd` starts
from RTN's scales rather than GPTQ's.

## 2. The objective, and what "wider scope" actually changes

For `y = W x` with input second moment `A = E[x x^T]`, the layer's own output error is

```
D(dW) = tr(dW A dW^T)
```

This decomposes **exactly** over output rows. That is not a modelling choice, it is why GPTQ can
treat every row as its own independent problem, and it means scope level S2 of the brief --
"joint optimization over all output channels of one matrix" -- is a null by construction under
the layer objective. It only becomes meaningful when the objective couples rows.

It couples rows as soon as we care about the error somewhere downstream. If `z = f(y)` is the
thing we actually care about, then to first order `dz = J dy`, and

```
E||dz||^2 = tr(dW A dW^T G)      with   G = E[J^T J]
```

So **the entire content of "widen the optimization scope" is a different `G`**, and the
compensation radius of section 8 is a ladder of endpoints for `z`:

| scope | endpoint `z` | brief |
|---|---|---|
| `layer` | the layer's own output, `G = I` | S1, GPTQ's own scope |
| `module` | the attention or MLP module output | S3, S4 |
| `block1` | the transformer block's residual output | S5 |
| `block2`, `block4`, `block8` | one, three, seven blocks further downstream | S6, S7 |

`E[J^T J]` would need one backward pass per output channel to form exactly. It does not need to
be formed exactly: for Rademacher `v`, `E[J^T v v^T J] = E[J^T J]`, so a single backward pass of
`L = sum_t v_t . z_t` contributes one rank-1 term *per token*. `scopeg.block_G` does this
**inside** the sequential quantization pipeline, where the activation buffer already carries the
error of every block quantized so far and the downstream blocks are still full precision -- which
is the geometry a wider objective would really be facing, not a cached approximation of it.

### Interactions are exactly pairwise, and that is a theorem, not a measurement

Section 9 asks for `I(a, b) = D(a+b) - D(a) - D(b)` over quantization changes, for the
distribution of higher-order interactions, and for whether a pairwise model explains joint moves.
For a quadratic objective these have closed-form answers. The interaction of two single-code
moves `d_a` at `(r_a, j_a)` and `d_b` at `(r_b, j_b)` is exactly

```
I(a, b) = 2 d_a d_b G_{r_a r_b} A_{j_a j_b}
```

and there are **no third- or higher-order terms at all**, because a quadratic's Taylor expansion
in the changes stops at two. So the interesting question is not whether interactions exist -- they
provably do, densely -- but whether an optimizer that can only move one coordinate at a time is
thereby handicapped. That is measured directly (section 4 below), not assumed.

`joint.additive_change` reports the consequence in one number per arm: the damage change actually
achieved, against the sum of what each changed decision would have achieved alone from the same
starting point.

## 3. The optimizers

Three strengths, in `src/lwc/joint.py`:

**`cd_refine` -- exact cyclic coordinate descent (J1, J2).** Each coordinate is set to its *exact*
conditional minimiser over the integer grid, vectorised over all output rows, with `M = G dW A`
maintained incrementally. With `G = I` the rows within a column are exactly independent, so a
whole column can be set to its joint conditional minimum in one step and the objective cannot
increase. With a non-diagonal `G` they are not, so the exact change

```
dD = 2 d^T M[:, j] + A_jj (d^T G d)
```

is evaluated for the proposed column move and the accepted row set is halved until it is a
decrease, with a final fallback to the single best row -- which cannot fail, because a one-row
move has no cross term. **Every optimizer here is monotone in the objective it is given**, which
makes "it got worse end to end" unambiguously a statement about the objective and not about the
search.

**`cd_subset_exact` -- exact block moves (J3).** Blocks of a few dozen decisions solved to proven
optimality by a box-constrained sphere decoder, cycling over the row. This is what would find a
benefit that single-coordinate moves cannot see.

**`sphere_decode` -- the oracle (J6).** Depth-first Schnorr-Euchner enumeration with the
incumbent as the initial radius. When it terminates inside its node budget the answer is *the*
optimum of that subproblem, not a good solution, which is what makes the oracle gap a measurement
rather than an estimate. Verified against brute force on small instances (exact match).
Free coordinates are ordered by increasing conditional precision before the Cholesky, so the
decoder commits to the best-determined decisions first; without that ordering it essentially
never proved optimality inside any sane budget, with it most subproblems close in a few thousand
nodes.

A simulated annealer (`anneal_row`) is included as a search family that fails differently from
coordinate descent: if CD were stopping in a poor basin, a long anneal from the same start would
escape it.

## 4. Where this sits in the literature

Written after the fact, because a prior-art survey run mid-study corrected two things this
report would otherwise have claimed.

**The framing is not ours.** That GPTQ, run back-to-front, *is* Babai's nearest-plane algorithm
on the lattice whose basis is the Cholesky factor of the permuted Hessian was proved
independently twice, both at ICLR 2026 (Chen, Shabanzadeh, Crnčević, Hoefler & Alistarh,
arXiv 2507.18553; Birnick, arXiv 2508.01077). Note the direction: the equivalence needs the
back-to-front sweep, so "GPTQ processes columns left to right" is the convention but not the
form in which the theorem holds.

**The optimizer is not ours either.** `cd_refine` is QuantEase (Behdin et al., arXiv 2309.01885)
— cyclic coordinate descent with the exact per-coordinate minimiser on a fixed grid, whose
Lemma 2 proves convergence to a coordinate-wise minimum, which also means our CD terminates at
a CW-minimum by construction rather than by luck. QuantEase names GPTQ-initialization and never
benchmarks it; ReQuant (arXiv 2608.07019) and SchurQuant (arXiv 2608.15567), both from August
2026, have since filled that cell. DiscQuant (COLT 2025, arXiv 2501.06417) solves the same
fixed-grid rounding problem by discrepancy theory.

**What is left, and what this report is actually about.** No published work runs an exact
box-constrained closest-vector search as an *optimality oracle* for LLM weight quantization.
That matters more than it sounds: Chen et al.'s Babai bound is proved **only in the absence of
clipping**, and YAQA (arXiv 2505.22988) independently notes that LDLQ/OPTQ optimality dies once
clamping restricts the range. OJBKQ (arXiv 2602.08376) writes the box-constrained integer least
squares problem explicitly and states its solution is sub-optimal without measuring by how much.
Measuring it is the contribution here.

## 5. Results

### 5.1 The oracle gap, and who closes it

Single output rows of real matrices, 3 bits, group 128, damage relative to GPTQ's, 96 rows per
coordinate system. The pipeline is exactly deterministic — three identical runs give a spread of
0.00e+00 — so every difference below is real rather than noise.

| coords | RTN | GPTQ | + coordinate descent | + multi-start | CD from the RTN basin |
|---|---|---|---|---|---|
| native | 5.192 | 1.000 | **0.822** | 0.807 | 0.986 |
| hadamard | 6.933 | 1.000 | **0.931** | 0.918 | 1.324 |

So greedy commitment costs 18% of the objective in native coordinates and 7% under an
incoherence rotation. Two things this table settles:

* **Compensation is not replaceable by local search.** Coordinate descent started from the RTN
  solution, on RTN's own scales, lands at 1.324x GPTQ under Hadamard — *worse* than GPTQ.
  Whatever GPTQ's sequential error feedback does, reopening decisions afterwards does not
  reproduce it.
* **The gap is essentially all reachable.** Freezing everything but `n` decisions and solving
  those *exactly* improves on GPTQ by 1.0%/1.7%/2.7% for n = 16/32/64, and on the
  coordinate-descent solution by 0.000%/0.011%/0.015%. CD's fixed point is the exact optimum of
  every subproblem the decoder could prove.

### 5.2 Higher-order moves: a correction

The 3-bit numbers above supported "joint moves over several decisions buy nothing", and that
generalisation was wrong. CDQuant reports Block-CD at k=2 buying +0.3% at INT3 and +1.0% at
INT2 over greedy CD, with the effect growing as bits shrink; ADMM-Q loses 7.26 -> 7.97 PPL when
its pair-swap step is removed. Re-running the gate at **2 bits**, with the block sizes and the
protocol CDQuant actually uses:

| bits, coords | RTN | CD | k=2 from GPTQ | k=8 on top of CD | alternating CD and k=2 |
|---|---|---|---|---|---|
| 2, native | 6.814 | 0.7049 | 0.7170 | 0.7011 | **0.6972** |
| 2, hadamard | 18.228 | 0.9436 | 0.9407 | 0.9364 | 0.9377 |

Reading it carefully:

* Block moves are **not a substitute** for single-coordinate descent — k=2 applied to GPTQ's
  solution gives 0.7170, worse than plain CD's 0.7049.
* Block moves **on top of** a converged CD solution buy little: +0.04% at k=2, +0.54% at k=8.
* **Alternating** single-coordinate sweeps with k=2 blocks buys +1.09% in native coordinates,
  against CDQuant's published +1.0% at INT2. We agree with them.

The original claim failed for two reasons at once: it was measured at 3 bits, and it used a
different protocol (exact 32-decision blocks from a converged solution) than the one the
published number refers to. Also worth recording: greedy commitment costs far more at 2 bits
than at 3 — CD reaches 0.70 of GPTQ's damage in native coordinates, against 0.82 at 3 bits.

### 5.3 End-to-end, which is the only arbiter

Full-model sequential quantization, wikitext-2 test perplexity, fp16 baseline 11.8134. The
`bpw` column is identical by construction: coordinate descent changes which integers are
stored and nothing else.

| arm | bpw | ppl | d ppl | layer objective |
|---|---|---|---|---|
| Hadamard + GPTQ | 3.2500 | 16.7968 | +4.9833 | 1.000 |
| Hadamard + GPTQ + CD | 3.2500 | **16.7700** | **+4.9566** | 0.9435 |
| native + GPTQ | 3.2500 | 23.2516 | +11.4381 | 1.000 |

**The gain is real, survives the rotation, and is an order of magnitude smaller than the local
metric promised.** Coordinate descent cut the layer objective by 5.65% and the perplexity gap by
0.54% — a factor of ten. This is the sixth time in this repository that a layer-local quadratic
metric has misrepresented an end-to-end result, and the first time it did so in *magnitude*
rather than in ranking: the ordering was right, the size was not.

### 5.4 Diagnostics

At matrix scale (3 bits, mean over matrices, `+cd` against GPTQ):

| arm | layer | Fisher | held-out layer | codes moved |
|---|---|---|---|---|
| gptq | 0.01391 | 0.01193 | 0.01752 | — |
| gptq+cd | 0.01319 | 0.01128 | 0.01708 | 1.38% |
| gptq+cd, damped A | 0.01315 | 0.01121 | **0.01687** | 1.37% |
| gptq+cd, 5% most ambiguous only | 0.01343 | 0.01147 | 0.01718 | **0.56%** |

Two results here were not expected:

* **Restricting the optimizer to the least confident decisions is a regulariser.** In native
  coordinates the ambiguity-restricted arm reaches a *better* held-out score (0.02488) than
  unrestricted coordinate descent (0.02782), while moving 1.78% of codes instead of 9.50%. Part
  of what full CD buys on the calibration sample it gives back off it.
* **The moves are cooperative, not independently good.** The ratio of additive to actual damage
  change is −1.50 under Hadamard: applied one at a time these moves would make things *worse*,
  and only help together. The ambiguity-restricted moves have a ratio of +1.17, i.e. they are
  individually beneficial. Since the objective is quadratic its interactions are exactly
  pairwise and there are no higher-order terms at all, so this is a complete description of the
  structure rather than a first-order summary of it.

### 5.5 Open, running

The scope ladder (module / block / 2 blocks / 4 blocks) is running end-to-end. One control is
already settled: KronQ (COLM 2026, arXiv 2607.07964) proves that `G` cancels algebraically from
the column-wise OBS update, so feeding a downstream metric into GPTQ itself would produce
bit-identical weights and a flat radius curve for algebraic rather than scientific reasons. That
is not what happens here, and it was verified rather than assumed: GPTQ never receives `G`; a
*diagonal* `G` in the coordinate-descent objective reproduces the identity solution to **exactly
zero differing codes**; and a full off-diagonal `G` moves 5.4% of codes and lowers the
G-weighted damage to 0.958 while raising the layer damage by 3.1%. Only off-diagonal structure
in `G` can act, and it does.

## 6. Verdict

*Not yet. Pending the scope ladder end-to-end, the 262k-524k calibration sweep, and the
derivative-free rotation search.*

*(below)*
