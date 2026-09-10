# Experiment log

Reverse-chronological within phases. Every entry: hypothesis / config / result / interpretation /
decision. Seeds fixed (default 0). Hardware: RTX 4050 Laptop 6 GB, Core Ultra 7 155H, 31.5 GB RAM.

---

## E0 — Metric validation on a synthetic null (2026-08-31)

**Hypothesis.** The scalar we will use for all of Phase 1,

```
net_gain = 0.5*log2(1/rho)  -  log2(K)/d          [bits per weight]
```

must be **<= 0 for an i.i.d. Gaussian source** (rate-distortion theorem: no code beats D(R)=s^2 2^-2R).
If the metric shows positive gain on i.i.d. noise, it is measuring overfitting, not structure.

**Config.** Synthetic `W = randn(4864, 896)` on CUDA, tiles 8x8 and 16x16, k-means K in {256,1024,4096},
20 Lloyd iterations, seed 0.

**Result — first attempt (codebook fit and evaluated on the same tiles):**

| d | K | rho | net_gain |
|---|---|---|---|
| 64 | 4096 | 0.721 | **+0.048** |
| 256 | 4096 | 0.694 | **+0.217** |

Positive gain on pure i.i.d. noise. The metric was broken.

**Diagnosis.** With d=256 a 4864x896 matrix yields only 17 024 tiles; K=4096 centroids means ~4 points
per cluster. In-sample residual energy is memorisation.

**Fix.** 50/50 train/held-out tile split; codebook fit on train, `rho` measured on held-out.
Also added explicit amortized codebook storage `codebook_bpw = K*d*16 / n_weights_served`.

**Result — after fix:**

| d | K | rho_train | rho_heldout | net_gain | codebook bpw |
|---|---|---|---|---|---|
| 64 | 256 | 0.851 | 0.883 | **-0.035** | 0.06 |
| 64 | 1024 | 0.786 | 0.860 | **-0.047** | 0.24 |
| 64 | 4096 | 0.666 | 0.848 | **-0.068** | 0.96 |
| 256 | 256 | 0.928 | 0.981 | **-0.017** | 0.24 |
| 256 | 1024 | 0.826 | 0.983 | **-0.027** | 0.96 |
| 256 | 4096 | 0.460 | **0.994** | **-0.043** | 3.85 |

**Interpretation.** Metric now behaves as theory requires. Two things this already tells us before
touching a real model:

1. The train/held-out gap at d=256, K=4096 is 0.46 -> 0.99. Any tile-VQ paper-style number measured
   in-sample at these tile sizes is meaningless.
2. Codebook storage is not a rounding error. A K=4096, d=256 codebook costs **3.85 bits/weight** if it
   only serves one matrix. A tile codec must share its codebook across a large number of matrices or
   it is dead on arrival.

**Decision.** Adopt held-out `net_gain` as the Phase 1 primary statistic. Pool tiles across many
matrices when fitting, both to fix sample size and to reflect the realistic shared-codebook setting.
Run every probe against nulls `shuffle` (same marginal, no dependence) and `gauss_rowcol`
(no marginal shape either) so `real - shuffle` isolates genuine dependence structure.

---

## E1 — Positive control: can the probe detect structure that IS there? (2026-08-31)

**Hypothesis.** Before trusting a negative result, the instrument must be shown to be sensitive.

**Config.** Synthetic 4864x896 matrices whose 8x8 tiles are built from a known generator, run
through the identical probe (k-means K=256 held-out; PCA r=8 held-out).

| source | VQ rho | VQ net_gain | PCA rho (r=8) | PCA net_gain | r90 |
|---|---|---|---|---|---|
| tiles from a 256-entry codebook, no noise | 0.023 | **+2.584** | 0.764 | -0.806 | 49 |
| same + noise at 30% amplitude | 0.199 | **+1.039** | 0.773 | -0.814 | 50 |
| same + noise at 100% amplitude | 0.540 | **+0.320** | 0.818 | -0.855 | 55 |
| tiles in an 8-dim subspace of R^64 | 0.328 | +0.680 | 0.000 | **+10.524** | 8 |
| i.i.d. Gaussian | 0.883 | -0.035 | 0.876 | -0.904 | 58 |

**Interpretation.** The probe detects codebook structure at +2.58 bits/weight, and still finds
+0.32 bits/weight when the structure is buried under noise of equal amplitude. It detects a
low-rank subspace at +10.5 bits/weight. Sensitivity is not the limiting factor.

**Decision.** Any near-zero reading on real weights is a real negative, not a blind instrument.

---

## E2 — Phase 1A: intra-matrix / pooled tile structure (2026-08-31)

**Hypothesis.** LLM weight tiles contain dependence structure that a shared prototype
dictionary could exploit, beyond a matched i.i.d. null.

**Config.** Qwen2.5-0.5B, layers {1,5,11,17,22}, all 7 projections, tiles
{1x8, 1x16, 4x4, 8x8, 16x16, 32x32}, K in {256, 1024, 4096}, raw and per-tile-normalized,
PCA r in {1,2,4}, 15 Lloyd iterations, 200k tiles pooled per (proj, tile, variant),
50/50 train/held-out split, seed 0. 1155 probes, 7 min wall clock, peak VRAM 1.1 GiB.

**Result — held-out residual energy rho, pooled over all 7 projections:**

| tile | K | rho gauss_rowcol | rho real | rho shuffle | real/shuffle | real/gauss |
|---|---|---|---|---|---|---|
| 1x8 | 256 | 0.3564 | 0.3578 | 0.3348 | 1.069 | **1.004** |
| 1x8 | 4096 | 0.1951 | 0.1958 | 0.1830 | 1.070 | **1.004** |
| 1x16 | 4096 | 0.4542 | 0.4524 | 0.4323 | 1.047 | **0.996** |
| 4x4 | 4096 | 0.4468 | 0.4442 | 0.4325 | 1.027 | **0.994** |
| 8x8 | 4096 | 0.8406 | 0.8346 | 0.8292 | 1.007 | **0.993** |
| 16x16 | 4096 | 0.9710 | 0.9704 | 0.9703 | 1.000 | **0.999** |
| 32x32 | 4096 | 0.9992 | 0.9987 | 1.0008 | 0.998 | **1.000** |

**Dependence signal across all 216 VQ probes** (net_gain(real) - net_gain(shuffle)):

```
mean   -0.0148 bits/weight
median -0.0019
max    +0.0230
min    -0.1136
fraction of probes above +0.05 bits/weight:  0.000
```

**Marginal shape gain** (net_gain(shuffle) - net_gain(gauss_rowcol)): mean +0.017, max +0.117.

**PCA:** for 8x8 tiles the leading eigenvalue holds 1.06x the flat-spectrum share and 57 of 64
components are needed for 90% of the energy. For 16x16, 227 of 256. The tile spectrum is flat.

**Nearest neighbour** among 65k unit-norm 8x8 tiles: mean squared NN distance 0.99-1.08 for real,
and the same for both nulls. The best match to a random tile has cosine ~0.5 -- tiles are
essentially mutually orthogonal.

**Interpretation.** Three separate readings agree:

1. `real/gauss ~ 1.00` at every tile size and every K. LLM weight tiles are **exactly as
   compressible as a matched i.i.d. Gaussian**. There is no tile-level dependence to exploit.
2. `real/shuffle > 1` for small tiles -- shuffling makes the weights *more* compressible.
   The only real non-i.i.d. property is **per-row/per-column scale heterogeneity**, and it
   *hurts* a global codebook. That structure is already fully exploited by group-wise scalar
   quantization with per-group scales; it is not new capacity.
3. The shape gain (non-Gaussian marginal) is +0.017 bits/weight on average, peaking at +0.12.
   This is the classic VQ gain that AQLM/QuIP# already collect, and it is small.

This is **Kill criterion A** for the intra-matrix version of `W_t ~= D(c_t) + R_t`: a shared
tile dictionary cannot beat scalar quantization, because the source has no tile structure.

**Caveat handled:** the 32x32 PCA r90 for k_proj/v_proj (179/1024) is a sample-size artifact --
those matrices are 128x896, giving only ~560 pooled tiles for a 1024-dim covariance. The
held-out `net_gain` for those cells is unaffected and still ~0.

**Decision.** Stop pursuing intra-matrix tile dictionaries as the primary direction. Move the
remaining structural budget to the three places structure could still hide: cross-layer (1B),
function-preserving permutation (1C), and the low-bit quantization residual (1D). Build the
Phase 2/3 rate-distortion harness in parallel so the negative result is quantified against real
baselines rather than asserted.

---

## E3 — Phase 1D: is the low-bit quantization residual structured? (2026-08-31)

**Hypothesis (Variant D).** `W ~= Q_lowbit(W) + D(c)`. Scalar quantization handles the
high-entropy bulk; a cheap procedural term mops up *structured* residual error.

**Config.** Qwen2.5-0.5B, layers {1,5,11,17,22}, projections {q, o, up, down}, group-128 RTN at
2/3/4 bits, residual `R = W - Q(W)`, tiles {1x8, 1x16, 8x8, 16x16}, K in {256, 4096}, held-out
split, `real` vs `shuffle` null. Also a per-group-normalized variant so any surviving signal
would be shape, not scale. 372 probes x2 runs, 105 s each.

**Residual energy after quantization** (`||W-Q||^2 / ||W||^2`, mean over projections):
INT2 g128 -> 0.288, INT3 -> 0.052, INT4 -> 0.0113.

**Result — dependence signal in the residual, net_gain(real) - net_gain(shuffle):**

| qbits | tile | K | rho real | rho shuffle | delta |
|---|---|---|---|---|---|
| 2 | 1x8 | 4096 | 0.1634 | 0.1649 | +0.0064 |
| 2 | 8x8 | 4096 | 0.8359 | 0.8381 | +0.0019 |
| 3 | 1x16 | 256 | 0.5947 | 0.5750 | -0.0242 |
| 3 | 8x8 | 4096 | 0.8415 | 0.8418 | +0.0003 |
| 4 | 1x8 | 4096 | 0.1683 | 0.1686 | +0.0011 |
| 4 | 16x16 | 4096 | 0.9696 | 0.9689 | -0.0006 |

Full range across all 372 probes: **-0.025 to +0.008 bits/weight**. Per-group normalization
changes nothing.

**Interpretation.** The round-to-nearest residual is white noise, exactly as quantization theory
predicts: RTN error is uniform on `[-s/2, s/2]` and independent across coordinates once the
per-group scale is removed. There is no structured error for a procedural term to capture.
**Variant D's premise is false.** Bits spent on a correction term would be strictly better spent
on more quantization levels.

**Decision.** Keep Variant D in the rate-distortion sweep (to show it losing quantitatively
against plain quantization at equal bpw), but stop treating it as the promising direction.

---

## E4 — Phase 1B: cross-layer redundancy (2026-08-31)

**Hypothesis.** The same projection in different layers shares structure a shared basis,
prototype set, or learned inter-layer map could exploit.

**Config.** Qwen2.5-0.5B, projections {q, o, gate, up, down}, adjacent pairs
(1,2)(5,6)(11,12)(17,18)(21,22) and far pairs (1,9)(5,13)(11,19). Predictors of increasing
generosity; all numbers are `||W_j - pred||^2 / ||W_j||^2`, so 1.0 = no better than zero.

**Result (representative, up_proj / q_proj):**

| predictor | q_proj 11->12 | up_proj 11->12 | cost if used |
|---|---|---|---|
| cosine(W_i, W_j) | -0.0035 | -0.0006 | -- |
| identity (W_j ~= W_i) | 1.928 | 1.940 | 0 bits |
| best affine a*W_i + b | 1.0000 | 1.0000 | ~0 bits |
| best row+col rescale of W_i | 0.9909 | 0.9976 | ~0.04 bpw |
| low-rank map rank 8 | 0.9516 | 0.9948 | 0.36 bpw |
| low-rank map rank 32 | 0.8446 | 0.9788 | 1.43 bpw |
| low-rank map rank 128 | 0.5466 | 0.9271 | 5.72 bpw |

**Codebook transfer** (fit on layer i, applied to held-out tiles of layer j), 1x8, K=4096:

| proj | cross (i->j) | self (j->j) | cross onto shuffled j |
|---|---|---|---|
| q_proj 11->12 | 0.2052 | 0.2037 | 0.1919 |
| up_proj 11->12 | 0.1832 | 0.1841 | 0.1794 |
| gate_proj 1->2 | 0.1822 | 0.1835 | 0.1790 |

**Interpretation.**

1. Cosine between the same projection in two layers is **0.000 +/- 0.01**. Layers are mutually
   orthogonal. `identity` at ~1.93 is exactly `||W_i||^2 + ||W_j||^2` -- the signature of two
   independent matrices.
2. The best affine map explains **0.00%**. Row+column rescaling explains ~1%.
3. A rank-128 inter-layer map (out of 896) explains 45% of q_proj energy but costs **5.72
   bits/weight** -- INT4 at 4.25 bpw already leaves only 1.1% error. Non-competitive by an
   order of magnitude.
4. Codebook transfer cross == self to within noise. A codebook is not layer-specific -- but only
   because there is nothing layer-specific to learn; both are just fitting the same Gaussian.

**Useful side finding:** because codebooks transfer freely, a dictionary *can* be amortised over
the entire model at no accuracy cost. That fixes the codebook-storage problem from E0. It does
not create any rate gain.

**Decision.** Cross-layer redundancy is not a source of compression. Kill.

---

## E5 — Phase 1C: function-preserving permutation / alignment (2026-08-31)

**Hypothesis.** Channel permutations are function-preserving; a good one could group similar
channels and make weights more compressible.

**Config.** Column permutations of q/o/up/down over layers {1,5,11,17,22}. `sort_shared` uses a
scale shared by every consumer of the residual stream (deployable: one permutation serves all).
`sort_local` sorts by the matrix's own column RMS (an upper bound -- not simultaneously
realisable for several consumers). `random` is the control. Metrics: group-wise RTN NMSE at
2/3/4 bits, groups 64 and 128; plus 8x8 tile VQ.

**Result — NMSE relative to the identity permutation (below 1.0 = permutation helps):**

| proj | perm | INT2 g64 | INT3 g64 | INT4 g64 | INT2 g128 | VQ rho |
|---|---|---|---|---|---|---|
| q_proj | sort_shared | 0.978 | 0.982 | 0.982 | 0.979 | -- |
| q_proj | sort_local | 0.932 | 0.935 | 0.933 | 0.926 | -- |
| q_proj | random | 1.000 | 0.999 | 1.000 | 0.998 | -- |
| up_proj | sort_shared | 0.988 | 0.987 | 0.988 | 0.988 | 1.003 |
| up_proj | sort_local | 0.971 | 0.973 | 0.973 | 0.966 | 1.014 |
| down_proj | sort_local | 0.967 | 0.967 | 0.967 | 0.957 | 1.007 |
| o_proj | sort_local | 1.034 | 1.034 | 1.034 | 0.979 | -- |
| o_proj | random | 1.123 | 1.123 | 1.123 | 1.089 | -- |

**Interpretation.**

1. The **deployable** permutation (`sort_shared`) buys 1-2% NMSE = `0.5*log2(1/0.98)` =
   **0.015 bits/weight**. Real but negligible.
2. The non-deployable per-matrix upper bound buys 3-7% = **0.05 bits/weight**. Still negligible,
   and it cannot be realised for several consumers of the same input at once.
3. Permutation makes tile VQ **worse** (rho ratio 1.003-1.014), because sorting by scale
   concentrates scale heterogeneity into whole tiles.
4. `o_proj` is the one interesting case: a random permutation *hurts* it by 12%, and sorting by
   norm also hurts at g=64. Its natural channel order (head-block structure) is already better
   than anything a scale sort produces. Attention head locality is real -- but it is already
   present in the checkpoint, not something a permutation search can add.

**Decision.** Alignment is not the missing ingredient. Answer to research question 1 is **no**.

---

## E6 — Phase 2/3: rate-distortion sweep, all baselines and Variants A-D (2026-08-31)

**Config.** Qwen2.5-0.5B, 20 matrices (q/o/up/down x layers 1,5,11,17,22), 51.6M weights.
Shared state amortised over all 357.8M target weights of the model. Activation error on 2048
cached wikitext-2 token activations per module. Seed 0.

**Fairness correction made mid-sweep.** The initial INT2 baseline used min-max RTN levels,
which for Gaussian-ish data wastes most levels on the tails: 0.250 NMSE at 2 bits versus 0.116
for a Lloyd-Max quantizer at the same rate. Comparing VQ against that would have overstated the
VQ advantage by a factor of two. A Lloyd-Max / NF-class codec (`codecs/companded.py`) was added
and is used as the scalar reference from here on.

**Result — selected points (activation NMSE):**

| method | bits/weight | weight NMSE | act NMSE | decode FLOPs/w | practical |
|---|---|---|---|---|---|
| INT4 RTN g128 | 4.25 | 0.0113 | 0.0122 | 2 | good |
| INT3 RTN g128 | 3.25 | 0.0517 | 0.0558 | 2 | good |
| INT2 RTN g64 | 2.50 | 0.2210 | 0.2529 | 2 | good |
| Lloyd-Max 3b g128 | 3.125 | 0.0401 | 0.0545 | 1 | good |
| Lloyd-Max 2b g64 | 2.25 | 0.1251 | 0.1514 | 1 | good |
| Hadamard + INT3 g64 | 3.50 | 0.0369 | 0.0397 | 2 | good |
| **residual VQ d=8 K=4096 s=2** | **3.003** | **0.0382** | **0.0406** | **1** | **good** |
| residual VQ d=8 K=256 s=3 | 3.000 | 0.0489 | 0.0521 | 2 | medium |
| whole-matrix low rank r=64 | 1.50 | 0.7719 | 0.6156 | 128 | impractical |
| per-tile low rank 16x16 r=1 | 1.00 | 0.7618 | 0.7226 | 2 | good |
| A: VQ + per-tile affine 8x8 s16 | 3.44 | 0.0393 | 0.0434 | 17 | poor |
| A: VQ + per-tile scale d=8 | 3.000 | 0.0830 | 0.0875 | 2 | good |
| B: VQ + per-tile rank-1 8x8 | 2.80 | 0.5303 | 0.4897 | 5 | poor |
| C: learned linear k=16 | 1.125 | 0.7512 | 0.7554 | 33 | poor |
| C: learned separable a8b8 | 4.125 | 0.0213 | 0.0222 | 33 | poor |
| C: learned MLP k=16 h=128 | 1.126 | 0.7332 | 0.7414 | 321 | poor |
| D: INT2 g64 + low rank r=64 | 4.00 | 0.1818 | 0.1955 | 130 | impractical |
| D: INT3 g128 + 0.5% sparse | 3.44 | 0.0492 | 0.0534 | 3 | good |
| D: INT2 g64 + 1% sparse | 2.88 | 0.2047 | 0.2349 | 3 | good |

**Interpretation.**

1. **Residual VQ wins, but by much less than this run suggested.** At 3.0 bpw it reaches act
   NMSE 0.0406 against the Lloyd-Max scalar quantizer's 0.0545 at 3.125 bpw, which would be
   0.34 bits/weight. **That number is wrong and is corrected in E11**: the Lloyd-Max codec used
   per-group RMS scaling, which clips weight outliers and is a bad baseline. With absmax
   scaling the same scalar codec reaches **0.0405 at 3.125 bpw**, so the real VQ advantage at
   3 bits/weight is **+0.12 bits/weight**, and at 4 bits/weight scalar quantization is
   **0.43 bits/weight better** than VQ. See E11.
2. **Every per-tile transform is a net loss.** Variant A's per-tile scale at d=8 costs 1 bpw and
   more than doubles the error at fixed rate (0.0875 vs 0.0406). Variant B's rank-1 term costs
   2 bpw and lands at 0.49. Side information only pays if it removes structure, and there is
   none to remove.
3. **Variant C is explained exactly by the flat tile spectrum.** A linear decoder with code_dim
   16 of d=64 gives NMSE 0.7512; the flat-spectrum prediction is `1 - 16/64 = 0.75`. At
   code_dim 36, measured 0.4451 vs predicted 0.4375. The decoder is not undertrained -- it is
   at the ceiling the spectrum allows. The MLP buys 2.4% over the linear map for **10x the
   FLOPs** (321 vs 33 per weight), which answers research question 4 directly: a learned
   procedural decoder gives nothing an equally sized basis does not.
4. **Variant D loses in every configuration.** Its premise died in E3. INT2 + rank-64 costs
   4.0 bpw for act NMSE 0.196, while plain INT3 at 3.25 bpw reaches 0.056. The correction term
   buys less than the same bits spent on more quantization levels.

**Kill criteria triggered:** A (no better rate-distortion than VQ/low-rank baselines) for all of
Variants A, B, C, D; C (decoder FLOPs) for Variant C's MLP and for every low-rank correction;
E is not reached because weight-space and activation-space agree closely here.

**Decision.** The original hypothesis is answered negatively and quantitatively. Proceed to the
recursive/IFS branch as a separate candidate, evaluated against this same Pareto front.

---

## E7 — Kill criterion R2: does composition add expressiveness? (2026-08-31)

**Hypothesis.** A depth-d path over a dictionary of K transforms costs `d*log2(K)` bits. That is
only honest if the `K^d` paths reach `K^d` distinct operators, and it is only *useful* if the
composition cannot be replaced by a single member of the same family.

**Config.** d=64 tiles (8x8), K=16 transforms, depths 1-3, 5 families. Closure residual =
best single-family fit to a random depth-2 composition. Distinct operators counted by hashing
each composed operator's action on random probes.

| family | closure residual (depth 2) | closed? | distinct ops / K^d at depth 3 |
|---|---|---|---|
| affine `aB+b` | 0.000000 | **yes** | 0.974 |
| diag `D_r B D_c` | 0.000000 | **yes** | **0.200** |
| lowrank_add `B + uv^T` | 0.000000 | **yes** | **0.200** |
| blockrot `A B C`, shifted Givens | 0.155800 | no | 0.819 |
| signperm | group-closed, generators reach beyond the dictionary | -- | **1.000** |

**Interpretation.**

* `affine`, `diag` and `lowrank_add` are **exactly closed under composition** -- `T_b(T_a(B))`
  is another one-step member. Depth adds zero expressiveness. **R2 fires.**
* `diag` and `lowrank_add` are additionally **commutative**, so the `K^d` paths collapse onto
  multisets: at depth 3, K=16, only `C(18,3) = 816` distinct operators exist and we measured
  821. The code pays 12 bits per tile to convey 9.7 bits. **20% of the path rate is wasted.**
* `blockrot` and `signperm` are the only families where composition genuinely enlarges the
  reachable operator set. `signperm` is the ideal case on paper: every path reaches a distinct
  operator (fraction 1.000), yet because signed permutations form a group the decoder can fold
  an entire path into **one shuffle regardless of depth**.

**Decision.** Drop `affine`, `diag`, `lowrank_add` as recursive families (keep one of each in
the sweep as a documented negative). Carry `signperm` and `blockrot` forward.

---

## E8 — IFS branch: finite-depth recursive composition on real weights (2026-08-31)

**Hypothesis (Branch A).** `W_t ~= T_{k_d}(...T_{k_1}(B_r))` -- a short program over a small
shared transformation dictionary describes tiles better than a direct codebook, because the
reachable set `R * K^depth` is exponentially larger than the stored parameters `R*d + K*p`.

**Config.** Qwen2.5-0.5B, 20 matrices, d=8 tiles (1x8), R=256 roots, K=16 transforms,
depth 0-4, per-tile 8-bit scale, beam search width 16, 2 Lloyd refinement rounds on the roots.
Encoder cost `O(n_tiles * (R + depth*beam*K) * d)` -- linear in tiles, no all-pairs (R8 clear).

**Result — signperm family:**

| depth | bits/weight | weight NMSE | act NMSE | encode s |
|---|---|---|---|---|
| 0 | 2.000 | 0.2496 | 0.2633 | 13 |
| 1 | 2.500 | 0.2563 | 0.2667 | 18 |
| 2 | 3.000 | 0.2573 | 0.2697 | 30 |
| 3 | 3.500 | 0.2549 | 0.2647 | 47 |
| 4 | 4.000 | 0.2528 | 0.2658 | 51 |

**blockrot family (4x4 tiles):** depth 0 -> 0.5041, depth 1 -> 0.4563, depth 2 -> 0.4565,
depth 3 -> 0.4570, while decode FLOPs/weight go 1 -> 7 -> 13 -> 19.

**Interpretation.** Each level of depth costs 0.5 bits/weight and returns **nothing**: the
distortion is flat to within noise from depth 0 to depth 4, and slightly worse at depth 2 than
at depth 0. `blockrot` improves once (depth 0 -> 1) and then saturates completely while its
decode cost grows linearly. At the matched rate of 3.0 bits/weight, residual VQ reaches act
NMSE 0.0406 against depth-2 composition's 0.2697 -- **6.6x worse**.

**Kill criteria fired: R1** (depth beats no larger codebook) and **R5** (decode FLOPs linear in
depth, distortion benefit zero) for `blockrot`.

---

## E9 — IFS branch: grammar statistics (2026-08-31)

**Hypothesis (Branch D).** If a real compositional grammar exists, a few transforms and a few
subsequences should be reused by very many tiles, and the path distribution should be far from
uniform, so entropy coding would recover a lot.

**Config.** signperm, d=8, R=256, K=16, depths 1-3, programs collected over q_proj and up_proj
in layers 1, 11, 22; 80k probe tiles per matrix.

| depth | programs used / reachable | fixed-width bits | program entropy | wasted bits | transform-usage entropy (max) | median reuse | cross-group TV |
|---|---|---|---|---|---|---|---|
| 1 | 2862 / 4096 (0.699) | 12.0 | 10.01 | 1.99 | 3.918 (4.0) | 42 | 0.0135 |
| 2 | 12510 / 65536 (0.191) | 16.0 | 11.56 | 4.44 | 3.901 (4.0) | 7 | 0.0381 |
| 3 | 32881 / 1048576 (0.031) | 20.0 | 12.81 | **7.19** | 3.892 (4.0) | **3** | 0.1047 |

**Interpretation.**

1. **Transform usage is almost perfectly uniform** -- 3.89 bits of entropy against a 4.0-bit
   maximum for K=16. No transform is preferred; no subsequence recurs. A grammar would show a
   sharply peaked usage histogram. This one is flat.
2. **Reuse collapses with depth**: the median program serves 42 tiles at depth 1 and **3 tiles
   at depth 3**. Nearly every tile ends up with its own program. **Kill criterion R3 fires.**
3. **The fixed-width path encoding overpays by 7.19 bits/tile at depth 3** (0.9 bits/weight at
   d=8). Entropy coding the paths would recover that, but it would not help: the distortion is
   flat across depth anyway, so the recovered bits buy nothing.
4. Cross-group total-variation distance between path histograms is 0.0135 at depth 1 -- the
   "grammar" is essentially identical for q_proj and up_proj across layers 1, 11 and 22. That
   sounds like good global sharing, but it is the null's signature: the distribution is
   layer-agnostic because it is covering an isotropic Gaussian, not capturing anything about
   the model.

---

## E10 — IFS branch: permutation modulation, and an under-counted decode cost (2026-08-31)

**Hypothesis.** Composition search is the wrong way to navigate the signed-permutation group,
because that group's optimal element for a given tile has a closed form: sort. Permutation
modulation computes it directly -- an implicit codebook of `R * d! * 2^d` entries from `R*d`
stored numbers.

**Result on real weights (entropy-coded permutation, log2(d!) bits):**

| method | bits/weight | act NMSE | stored numbers |
|---|---|---|---|
| permmod d=8 R=1 | 3.912 | 0.0473 | 8 |
| permmod d=8 R=16 | 4.412 | 0.0112 | 128 |
| permmod d=16 R=256 | 4.766 | 0.00477 | 4096 |
| permmod d=8 R=256 | 4.912 | 0.00414 | 2048 |
| residual VQ K=1024 s=4 | 5.001 | 0.00670 | 32768 |
| residual VQ K=256 s=5 | 5.000 | 0.00821 | 10240 |
| gain-shape VQ K=4096 s=3 | 5.504 | 0.00389 | 98304 |
| INT4 g64 | 4.500 | 0.00944 | 0 |

At 4.77-4.91 bits/weight **permutation modulation reached the Pareto front**, worth about
**+0.2 bits/weight against INT4** and **+0.44 against 4-stage residual VQ**, and the advantage
survives against gain-shape VQ, so it is not merely the per-tile scale.

**Then the honest correction.** The rate above charges `log2(8!) = 15.30` bits for the
permutation -- its true entropy. Decoding that requires either a Lehmer-code unranking loop
(~4 ops/weight, sequential and branchy) or an unranking table of `8! * 8 = 320 KiB`, which does
not fit in shared memory. The kernel-friendly alternative stores one index per position,
`d*ceil(log2 d) = 24` bits, decoding as a plain gather. Re-run with that encoding:

| method | bits/weight | act NMSE |
|---|---|---|
| permmod d=8 R=256, entropy-coded perm | 4.912 | 0.00414 |
| **permmod d=8 R=256, direct perm (kernel-realistic)** | **6.000** | **0.00414** |
| gain-shape VQ K=4096 s=3 | 5.504 | 0.00389 |

The identical reconstruction costs **1.09 more bits/weight** once the permutation is stored in a
form a kernel can actually decode, and at 6.0 bits/weight it is **strictly dominated** by
gain-shape VQ at 5.5 bits/weight on both axes.

**Interpretation.** This is the one place in the whole study where a procedural representation
touched the Pareto front, and it did so only under an encoding whose decode cost I had not
charged. It is also confined to 4.7-5.0 bits/weight, a region where nobody needs compression --
INT4 is already 4.25. Permutation modulation cannot reach the interesting 2-3 bits/weight band
at all: its rate floor at d=8 is `log2(8!)/8 + 1` = **2.91 bits/weight** before any scale, and
the fixed magnitude profile caps its distortion there (d=8 R=1 with a 4-bit scale sits at
3.41 bits/weight and act NMSE 0.0696, versus 0.0406 for residual VQ at 3.00).

**Decision.** Recursive composition: dead (R1, R3, R5). Permutation modulation: a real but
rate-limited effect that disappears under honest decode accounting. Branch verdict **IFS-C**
with a documented **IFS-D** caveat.

---

## E11 — Kill criterion E, and a corrected scalar baseline (2026-08-31)

**Trigger.** End-to-end perplexity contradicted the activation-error ranking. Lloyd-Max NF4 at
4.125 bits/weight had act NMSE 0.0212 -- essentially the same as residual VQ's 0.0202 at
4.0 -- yet its wikitext-2 perplexity was **1662.7 against VQ's 21.2 and fp16's 11.81**. A 78x
perplexity gap behind a 1.05x activation-error gap means one of the metrics is lying.

**Diagnostic.** Per-codec error restricted to the weight tail, Qwen2.5-0.5B layer 11:

| proj | codec | NMSE | max abs err / max abs W | rel. err on top 0.1% of weights |
|---|---|---|---|---|
| q_proj | INT4 g128 | 0.01299 | 0.055 | **0.00056** |
| q_proj | Lloyd-Max 4b, RMS scale | 0.01628 | **0.610** | **0.08900** |
| q_proj | Lloyd-Max 4b, absmax scale | 0.01045 | 0.092 | 0.00652 |
| up_proj | INT4 g128 | 0.01185 | 0.036 | 0.00053 |
| up_proj | Lloyd-Max 4b, RMS scale | 0.01483 | **0.743** | **0.16203** |
| up_proj | Lloyd-Max 4b, absmax scale | 0.00988 | 0.089 | 0.00664 |
| down_proj | Lloyd-Max 4b, RMS scale | 0.01721 | **0.744** | **0.18008** |
| down_proj | Lloyd-Max 4b, absmax scale | 0.01049 | 0.089 | 0.00695 |

**Interpretation.** RMS scaling puts the outermost Lloyd-Max level at 2.73 sigma, so the heavy
tail of the weight distribution is clipped. The largest weight in a group is reconstructed with
up to **74% error**, and the error energy on the top 0.1% of weights is **150x** INT4's -- while
the aggregate NMSE is only 1.3x worse. Aggregate MSE cannot see this. Neither could per-layer
activation NMSE, which moved by only 1.7x. Only end-to-end perplexity exposed it, and only
because the damage compounds across 24 layers.

**Two consequences, both important.**

1. **The brief's insistence on three metric levels was justified, and even that was not enough.**
   A fourth diagnostic -- error on the weight tail -- is what actually explains the failure.
   Recommendation for any follow-up: report `max|W - What| / max|W|` and the relative error on
   the top 0.1% of weights alongside NMSE. It is nearly free and it catches clipping.
2. **The scalar baseline had to be re-run.** With absmax scaling the Lloyd-Max codec is both
   better in MSE and safe on the tail. Corrected head-to-head at matched rate (activation NMSE):

| bits/weight | best scalar (Lloyd-Max, absmax) | residual VQ | VQ advantage |
|---|---|---|---|
| ~2.1 | 0.14517 @ 2.250 | 0.13792 @ 2.000 | **+0.29 bits/weight** |
| ~3.1 | 0.04052 @ 3.125 | 0.04064 @ 3.003 | **+0.12 bits/weight** |
| ~4.1 | 0.00932 @ 4.125 | 0.02016 @ 4.000 | **-0.43 bits/weight (scalar wins)** |

**This strengthens the overall negative verdict.** Even the classic, structure-free
space-filling gain of vector quantization -- the one effect that genuinely exists -- is worth
only about +0.1 to +0.3 bits/weight against a properly tuned scalar quantizer in this setting,
and it reverses sign at 4 bits/weight. There is correspondingly even less room for a procedural
representation to find an advantage.

**Caveat on absolute perplexities.** Every codec here is one-shot with no error compensation
(no GPTQ-style Hessian correction, no fine-tuning) and Qwen2.5-0.5B is small and sensitive.
Absolute perplexities at 2-3 bits/weight are therefore far worse than published SOTA, which
relies on exactly that error compensation. The comparison between codecs is internally valid --
all are handicapped identically -- but these numbers are not SOTA claims.

---

## E12 — Does the central negative replicate on a second architecture? (2026-08-31)

**Hypothesis.** The "tiles are i.i.d.-Gaussian-like" result could be specific to Qwen2.5-0.5B --
its training run, its size, or its architecture.

**Config.** Identical Phase 1A probe on **TinyLlama-1.1B-Chat-v1.0** (Llama architecture, 22
layers, hidden 2048, intermediate 5632, 32/4 GQA heads -- a different family, a different
tokenizer and a different training run). Layers {2,10,18}, projections {q, o, up, down},
tiles {1x8, 4x4, 8x8, 16x16}, K in {256, 4096}. 288 probes, 125 s, peak VRAM 1.3 GiB.

| tile | K | rho gauss_rowcol | rho real | rho shuffle | real/gauss | dependence bits |
|---|---|---|---|---|---|---|
| 1x8 | 256 | 0.3425 | 0.3440 | 0.3317 | 1.0044 | -0.0255 |
| 1x8 | 4096 | 0.1851 | 0.1867 | 0.1785 | 1.0085 | -0.0312 |
| 4x4 | 4096 | 0.4390 | 0.4390 | 0.4301 | 1.0000 | -0.0146 |
| 8x8 | 4096 | 0.8356 | 0.8339 | 0.8266 | 0.9979 | -0.0063 |
| 16x16 | 4096 | 0.9727 | 0.9705 | 0.9706 | 0.9977 | +0.0001 |

Dependence signal: **mean -0.0127, max +0.00013 bits/weight** (Qwen2.5-0.5B: mean -0.0184,
max +0.0021).

**Interpretation.** The result replicates exactly across architecture, size, tokenizer and
training run. `real/gauss` stays within 1% of unity at every tile size in both models, and the
dependence signal is negative in both. This is a property of trained transformer weights, not
of one checkpoint.

**Decision.** The central finding is safe to state generally for models of this class. It has
not been tested above 1.1B parameters, which is the main remaining caveat.

---

## E13 — Research question 3: how big must the shared dictionary be? (2026-08-31)

**Hypothesis.** There is a dictionary size at which the marginal accuracy gain justifies the
marginal rate, and beyond which it saturates -- the "knee" that would tell us how much shared
state a procedural decoder should aim to replace.

**Config.** Real weights, 20 matrices, held-out split, 600k pooled tiles, tiles 1x8 and 8x8,
K in {16, 64, 256, 1024, 4096, 16384}, 12 Lloyd iterations. 26 s total.

| tile | K | held-out rho | code bpw | codebook bpw (KiB) | marginal bits saved / doubling | marginal cost | ratio |
|---|---|---|---|---|---|---|---|
| 1x8 | 16 | 0.6482 | 0.500 | 0.0000 (0) | -- | 0.1250 | -- |
| 1x8 | 64 | 0.4896 | 0.750 | 0.0000 (1) | 0.1012 | 0.1250 | 0.81 |
| 1x8 | 256 | 0.3672 | 1.000 | 0.0001 (4) | 0.1038 | 0.1250 | 0.83 |
| 1x8 | 1024 | 0.2709 | 1.250 | 0.0004 (16) | 0.1096 | 0.1250 | 0.88 |
| 1x8 | 4096 | 0.1987 | 1.500 | 0.0015 (64) | 0.1118 | 0.1250 | 0.89 |
| 1x8 | 16384 | 0.1470 | 1.750 | 0.0059 (256) | 0.1088 | 0.1250 | 0.87 |
| 8x8 | 256 | 0.8810 | 0.125 | 0.0007 (32) | 0.0139 | 0.0156 | 0.89 |
| 8x8 | 1024 | 0.8529 | 0.156 | 0.0029 (128) | 0.0117 | 0.0156 | 0.75 |
| 8x8 | 4096 | 0.8306 | 0.188 | 0.0117 (512) | 0.0096 | 0.0156 | 0.62 |
| 8x8 | 16384 | 0.8110 | 0.219 | 0.0469 (2048) | 0.0086 | 0.0156 | 0.55 |

**Interpretation.**

1. **There is no knee, because the benefit never exceeds the cost.** Every doubling of K returns
   81-89% of the bits it costs at d=8. The curve is a straight line slightly below break-even
   over a 1024x range of dictionary sizes -- the exact signature of an i.i.d. source, where the
   rate-distortion theorem forbids any code from doing better than break-even.
2. At d=64 the ratio *degrades* with K (0.89 -> 0.55) as the amortised codebook cost grows from
   32 KiB to 2 MiB. Bigger dictionaries are actively worse there.
3. **Codebook storage was never the problem.** K=4096 at d=8 is 64 KiB = 0.0015 bits/weight over
   358M weights. This is the quantitative reason the IFS branch's central motivation fails: a
   procedural codebook saves storage that costs essentially nothing to begin with. The
   amplification only starts to matter past ~2 MiB of codebook, i.e. K >= 16384 at d=64, where
   the marginal return has already fallen to 0.55.

**Decision.** Research question 3 is answered: no dictionary size pays for itself. Combined with
E13's storage numbers, this closes the last motivation for procedural codebook generation.

---

## E14 — Final diagnostic, Stage 1: the 2x2x2 factorial per layer (2026-08-31)

**Purpose.** Separate three candidate causes of sub-4-bit failure:
H1 the value representation, H2 the placement of error relative to functional sensitivity,
H3 the coordinate system / outliers / anisotropy.

**Design.** representation {scalar min-max group-wise, residual VQ d=8} x error handling
{naive, GPTQ second-order} x coordinates {native, randomized Hadamard}. The *same*
compensation engine (`lwc/gptq.py`) and the *same* rotation machinery (`lwc/rotate.py`) drive
every cell, so each factor is isolated.

**Correctness checks before measuring anything.**

* Rotation round-trip `||P^T(P W Q)Q^T - W||^2/||W||^2 = 5e-14` -- exactly function preserving.
* `rotate_hessian` matches an explicitly formed `Q^T H Q` to `5.5e-14`. This matters: GPTQ's
  loss uses `H = E[x x^T]`, and in the rotated basis the layer sees `x' = Q^T x`, so failing to
  rotate H would silently optimise the wrong objective.
* GPTQ's signature behaviour is present: at 3 bits it *raises* weight NMSE 0.052 -> 0.129 while
  *lowering* activation error 0.058 -> 0.012. Trading weight fidelity for functional fidelity is
  exactly what second-order compensation is supposed to do.
* Incoherence works as advertised: kurtosis 7.84 -> 3.04 (Gaussian is exactly 3.0),
  max/RMS 36.8 -> 5.2 on up_proj.

**Config.** Qwen2.5-0.5B, 20 matrices (q/o/up/down x layers 1,5,11,17,22), H from the cached
2048-token calibration activations, blocksize 128, percdamp 0.01, actorder off, seed 0.
5 rate points per cell. 52 runs, 559 s, **peak VRAM 758 MiB**.

**Per-cell rate-distortion fits**, `log2(activation NMSE) = a + slope * bpw`:

| cell | intercept | slope |
|---|---|---|
| scalar/naive/native | 4.077 | -2.469 |
| scalar/naive/hadamard | 3.571 | -2.384 |
| scalar/gptq/native | 3.404 | -2.875 |
| scalar/gptq/hadamard | 1.887 | -2.632 |
| vq/naive/native | -0.095 | **-1.355** |
| vq/naive/hadamard | 0.018 | -1.530 |
| vq/gptq/native | -1.470 | -1.512 |
| vq/gptq/hadamard | -2.154 | -1.636 |

Predicted activation NMSE at matched rates:

| cell | 2.25 bpw | 3.00 | 3.50 | 4.00 |
|---|---|---|---|---|
| scalar/naive/native | 0.3588 | 0.0994 | 0.0422 | 0.0180 |
| scalar/gptq/native | 0.1195 | 0.0268 | 0.0099 | 0.0037 |
| scalar/gptq/hadamard | 0.0610 | 0.0155 | 0.0062 | 0.0025 |
| vq/naive/native | 0.1132 | 0.0560 | 0.0350 | 0.0219 |
| vq/gptq/native | 0.0342 | 0.0156 | 0.0092 | 0.0055 |
| **vq/gptq/hadamard** | **0.0175** | **0.0075** | **0.0043** | **0.0024** |

**Main effects, in equivalent bits per weight** (converted through each cell's own slope):

| effect | 2.25 bpw | 3.00 | 3.50 | 4.00 |
|---|---|---|---|---|
| **compensation (GPTQ - naive)** | **+1.054** | **+1.135** | **+1.188** | **+1.241** |
| representation (VQ - scalar) | +0.843 | +0.448 | +0.184 | **-0.079** |
| rotation (Hadamard - native) | +0.322 | +0.337 | +0.346 | +0.356 |

**Two-way interactions, all positive** (equivalent bits/weight, at 3.0 bpw):

| interaction | value |
|---|---|
| representation x compensation | +0.56 to +0.60 |
| compensation x rotation | +0.27 to +0.30 |
| representation x rotation | +0.17 (naive) -> +0.35 (gptq) |

**Tail error by cell** (this is where the mechanism shows):

| cell | rel err top 1% | top 0.1% | top 0.01% | max err / max abs W | sign flips top 0.1% |
|---|---|---|---|---|---|
| scalar/naive/native | 0.0112 | 0.0079 | 0.0044 | 0.142 | 0.00003 |
| scalar/gptq/native | 0.0520 | 0.0340 | 0.0192 | 0.467 | 0.00228 |
| scalar/gptq/hadamard | 0.0205 | 0.0094 | 0.0037 | 0.179 | 0.00000 |
| **vq/naive/native** | 0.0373 | **0.0608** | **0.1095** | **0.579** | 0.00021 |
| **vq/naive/hadamard** | 0.0075 | **0.0055** | **0.0046** | **0.070** | 0.00000 |
| vq/gptq/native | 0.0536 | 0.0774 | 0.1262 | 0.604 | 0.00275 |
| vq/gptq/hadamard | 0.0120 | 0.0074 | 0.0046 | 0.117 | 0.00000 |

**Interpretation.**

1. **Compensation is the dominant factor.** +1.14 bits/weight at 3 bpw, about 2.5x the
   representation effect and 3.4x the rotation effect. It is also the only effect that *grows*
   with rate.
2. **The representation effect decays and reverses.** VQ is worth +0.84 bits at 2.25 bpw,
   +0.45 at 3.0, +0.18 at 3.5, and **-0.08 at 4.0**. The reason is visible in the slopes: VQ's
   rate-distortion curve is far flatter (-1.36 to -1.64) than scalar's (-2.38 to -2.88). Each
   extra residual-VQ stage costs a full bit but buys less than the 4x distortion reduction an
   ideal quantizer would give, so VQ's advantage is a low-rate phenomenon that erodes as bits
   are added.
3. **The tails explain VQ's behaviour.** In native coordinates VQ's error on the top 0.01% of
   weights is **0.1095** and it reconstructs the largest weight with **58% error** -- a single
   8-dimensional codeword cannot represent a tile containing one huge outlier. Under Hadamard
   the same numbers become **0.0046** and **7%**, a **24x improvement**. The rotation is not
   helping VQ compress better on average; it is removing the outliers that VQ cannot represent
   at all.
4. **GPTQ deliberately damages the tails.** Scalar/native tail error rises 0.0079 -> 0.0340
   under GPTQ, and sign flips on large weights appear (0.00228). That is the algorithm working
   as designed -- it pushes error into weights the Hessian says are cheap -- and it is why tail
   error alone is not a sufficient quality proxy either. Under Hadamard the sign flips vanish.
5. **Every interaction is positive**, so the factors are synergistic rather than substitutable.
   VQ gains more from compensation than scalar does (+1.29 vs +0.71 bits at 3 bpw, native), and
   more from rotation (+0.67 vs +0.29 with GPTQ). This is direct evidence for E4: a
   representation cannot be evaluated in isolation from the loss geometry it sits in.

**Decision.** H2 is the leading explanation, H3 second, H1 real but rate-limited and reversing
by 4 bits. Proceed to Stage 2 (full-model sequential GPTQ + perplexity) to confirm the ordering
survives end to end.

---

## E15 — Final diagnostic, Stage 2: full-model sequential GPTQ and perplexity (2026-08-31)

**Hypothesis.** The Stage 1 ordering (compensation > rotation > representation) should survive
end to end, where GPTQ is stronger because each layer is compensated against activations that
already carry every upstream layer's error.

**Config.** All 168 linear layers of Qwen2.5-0.5B. Calibration 32 x 512 wikitext-2 tokens
propagated layer by layer through the *quantized* prefix; layer kwargs captured from a real
forward pass. Perplexity on 24 x 2048 wikitext-2 test tokens. 24 runs, 48 min, peak VRAM
4573 MiB (dominated by the 151936-vocab logits, not by the method).

**Cross-check.** The new pipeline reproduces the earlier independent implementation exactly:
INT3 g128 -> ppl 56.959 (previously 56.9594), INT4 g128 -> 13.928 (previously 13.9284).

**Result (wikitext-2 perplexity, fp16 = 11.813):**

| bits/weight | scalar naive/native | scalar gptq/native | scalar naive/had | scalar gptq/had |
|---|---|---|---|---|
| 2.25 | 274426 | 401379 | 541127 | **25891** |
| 3.25 | 56.96 | 23.25 | 38.51 | **16.76** |
| 4.25 | 13.93 | 13.21 | 14.23 | **12.57** |

| bits/weight | VQ naive/native | VQ gptq/native | VQ naive/had | VQ gptq/had |
|---|---|---|---|---|
| 2.00 | 54634 | 1280 | 4040 | **49.12** |
| 3.00 | 277.68 | 129.21 | 36.89 | **16.49** |
| 4.00 | 35.77 | 28.90 | 16.20 | **13.01** |

**End-to-end decomposition** (improvement factor on delta-ppl vs fp16):

| representation | rate | compensation alone | rotation alone | both |
|---|---|---|---|---|
| scalar | 3.25 | **3.95x** | 1.69x | 9.1x |
| VQ | 3.00 | 2.26x | **10.6x** | 56.8x |

**Interpretation.**

1. **The two representations are limited by different factors.** Scalar is limited mainly by
   error placement (compensation 3.95x vs rotation 1.69x); VQ mainly by the coordinate system
   (rotation 10.6x vs compensation 2.26x). In both cases the combination beats the product of
   the individual effects (9.1 > 6.7, 56.8 > 24.0): the interaction is positive end to end.
2. **Activation NMSE misranks the representations.** VQ at 3.00 bpw and scalar at 3.25 bpw have
   activation NMSE 0.0584 vs 0.0576 -- within 1.4% -- but perplexity **277.7 vs 57.0**, a 4.9x
   gap. The top-0.01% weight error differs by 25x (0.1095 vs 0.0044) and predicts the ordering
   correctly. Third time in this project that a mean-squared metric has misranked codecs.
3. **At 2 bits, single factors can hurt.** Scalar 2.25: GPTQ alone 401379 and Hadamard alone
   541127 are both *worse* than naive 274426; only the combination (25891) helps. Nothing
   reaches usable quality at 2 bits -- the best is VQ/GPTQ/Hadamard at 49.12, 4.2x worse
   than fp16.
4. After the full treatment the representation gap is <= 0.25 bits/weight and reverses by
   4 bits (VQ 16.49 @ 3.00 vs scalar 16.76 @ 3.25; VQ 13.01 @ 4.00 vs scalar 12.57 @ 4.25).

---

## E16 — Controls: matched Gaussian, act-order, damping (2026-08-31)

**Matched-Gaussian control** (40 runs, `gauss_rowcol`, identical row/column RMS):

| representation | coordinates | weight-space real/null | activation-space real/null |
|---|---|---|---|
| scalar | native | 1.077 | 0.871 |
| scalar | hadamard | **1.000** | 0.787 |
| VQ | native | 1.041 | 0.899 |
| VQ | hadamard | 1.006 | 0.793 |

In weight space -- the pure rate-distortion geometry -- the real matrix has **no advantage**
over matched noise; it is 3% *harder* to compress on average and exactly break-even under
Hadamard. This reproduces the earlier study's central finding through an entirely different
code path (GPTQ engine + rotation module rather than the k-means probe). The activation-space
ratio of ~0.84 is confounded: the denominator ||WX|| differs between real and null because
training aligns W with the activation distribution, so it measures signal power, not
compressibility.

**Act-order / damping control.** The 2-bit GPTQ divergence in native coordinates (weight
NMSE > 1) might have been a tuning artifact, so it was checked (activation error):

| bits | percdamp | act-order | native | hadamard |
|---|---|---|---|---|
| 2 | 0.01 | off | 0.1613 | 0.0587 |
| 2 | 0.01 | on | **0.0649** | 0.0568 |
| 2 | 0.05 | on | 0.0638 | 0.0517 |
| 2 | 0.20 | off | 0.1043 | 0.0553 |
| 2 | 0.20 | on | 0.0755 | 0.0539 |

**Act-order is worth 2.5x in native coordinates and essentially nothing under Hadamard**
(0.0587 -> 0.0568). Exactly the predicted mechanism: act-order sorts columns by Hessian
diagonal magnitude, which only helps while the Hessian is anisotropic. After rotation the
result is insensitive to both damping and ordering -- the rotation makes the procedure robust.

End-to-end re-run of the 2-bit cells with act-order (its per-column index counted, +0.008 bpw):
scalar/gptq/native 401379 -> 285774; scalar/gptq/hadamard 25891 -> **17911**;
vq/gptq/native 1280 -> 5035; vq/gptq/hadamard 49.12 -> 55.05. Act-order helps scalar and
*hurts* VQ, because permuting input columns breaks the channel grouping the codebook was fit
on. The best 2-bit configuration is unchanged. The main factorial is reported without
act-order, which slightly understates the native-GPTQ cells and does not change any ordering.

**Decision.** Verdict **E2 jointly with E3**, with a real but already-exploited **E4**
interaction, and an honest **E5** caveat that 2-bit quality is not reached. The kill criterion
for the research direction is met: see docs/FINAL_DIAGNOSTIC.md. **Stop.**

---

# Programmatic quantization layout (E17-E21)

New hypothesis, not in conflict with the negative results above: the weight matrix may be
locally structureless while the map of *how expensive it is to damage each weight* is not.
Full report in `docs/PROGRAMMATIC_LAYOUT.md`.

## E17 — Sensitivity maps and what they actually predict (2026-08-31)

**Hypothesis.** Cheap sensitivity maps (magnitude, activation-weighted, Hessian diagonal, OBS
saliency, realised quantization damage) predict functional damage well enough to drive a bit
allocator.

**Config.** 12 matrices (layers 1/11/22 x q/o/up/down), INT3 g128. Two levels of ground
truth: (1) exact `||E_u X||^2` per unit using the **full** Hessian, full population;
(2) end-to-end NLL change from perturbing one output channel to 1 bit while the rest of the
model stays fp16, 96 stratified channels, 8 x 1024 held-out tokens.

**Result.**

The diagonal proxy `S5 = H_jj E^2` -- the allocator's objective -- captures **97.8%** of the
exact full-Hessian damage and ranks units at Spearman 1.000 (column), 0.990 (tile), 0.929
(group), 0.732 (row). Off-diagonal terms are worth 2.2%.

| map | per column | per group | per row | per tile |
|---|---|---|---|---|
| S1 abs(W) | **-0.245** | 0.501 | 0.523 | 0.190 |
| S2 W^2 | -0.203 | 0.569 | 0.565 | 0.278 |
| S3 W^2 E[x^2] | 0.948 | 0.571 | 0.510 | 0.781 |
| S4 H_jj | 0.989 | 0.354 | 0.000 | 0.820 |
| S5 H_jj E^2 | 1.000 | 0.929 | 0.732 | 0.990 |
| S7 W^2/[H^-1]_jj | 0.616 | 0.562 | 0.559 | 0.476 |

End-to-end: exact layer damage vs dNLL Spearman **+0.351 +- 0.104**, log-Pearson +0.537;
S2 +0.380, S5 +0.325, S3 +0.210, S1 +0.302. All good maps place the single worst channel
(dNLL 1.6e-2, 20x the next) in their top 1%.

**Interpretation.**

1. **Magnitude is an anti-predictor on the input axis** (-0.245): which column is expensive to
   damage is decided by activation scale, and high-magnitude columns tend to have small
   activations. The axes are mutually blind -- `H_jj` is exactly uninformative about rows.
2. Even *exact* layer damage predicts end-to-end loss only moderately. The tail is
   predictable, the bulk ordering is near noise -- the same pattern this project found for
   weight values, now for sensitivities.
3. A map fitted to naive damage predicts GPTQ damage much worse (S5: 1.000 -> 0.704 per
   column). Every allocator downstream is therefore recomputed inside its own cell.

**Correctness note.** The first version of `agreement` ranked ties by argsort order, which
manufactured a spurious -0.351 Spearman for `S4` at row granularity (a fully tied map).
Fixed to average ranks; `S4`/row is now exactly 0.000, as it must be.

**Decision.** Maps are good enough to drive an oracle. Proceed to the gate.

## E18 — The oracle gate: what is a free layout worth? (2026-08-31)

**Hypothesis.** If an oracle with the complete damage table and no metadata charge cannot
beat uniform quantization at the same code budget, no layout can, and the branch ends.

**Config.** Lagrangian (BFOS) allocation over {0,1,2,3,4,5,6,8,16} bits per unit at exactly
the code budget of uniform b0 bits. Granularities: per-weight / per-group (row x 128) /
per-tile (128x128) / per-row / per-column. All four cells of the earlier factorial.
12 matrices, 72 configs, 621 s, peak VRAM 1101 MiB. `MixedScalarQuantizer` at a constant bit
map reproduces `ScalarQuantizer` exactly (max abs diff 3.8e-6, fp32 rounding).

**Result (gain in bits/weight, 3-bit budget):**

| cell | per-column | per-row | per-tile | per-group | per-weight |
|---|---|---|---|---|---|
| naive/native | +0.366 | +0.084 | +0.001 | +0.138 | +2.028 |
| gptq/native | +0.216 | -0.022 | -0.000 | +0.027 | +1.181 |
| naive/hadamard | +0.000 | +0.000 | +0.000 | +0.000 | +1.585 |
| **gptq/hadamard** | **-0.002** | **+0.000** | **+0.000** | **+0.000** | **+0.777** |

Bit histogram of the per-weight oracle at 3 bits (gptq/hadamard): 13% at 0 bits, 2% at 1,
16% at 2, 27% at 3, 26% at 4, 14% at 5, 1% at 6, and **0.00% at 16**.

**Interpretation.** The gate passes, but only at a granularity that cannot be described
cheaply. Every hardware-friendly unit is worth 0.00 +- 0.03 bits/weight once GPTQ and a
rotation are in place. The absence of a 16-bit class rules out the "it is really an outlier
codec" reading: the gain is reverse water-filling on the realised rounding residual. That
observation motivated E19.

**Decision.** Do not proceed to a recursive layout yet. First ask whether the per-weight map
contains any information at all.

## E19 — Matched null and the price of the map (2026-08-31)

**Hypothesis.** If the per-weight oracle is exploiting the realisation of rounding noise
rather than structure, an i.i.d. Gaussian with the same row/column scales will earn the same
gain, and the resulting map will be incompressible.

**Config.** (a) The full oracle pipeline rerun on `gauss_rowcol` against the **real**
activation statistics, paired on the same 8 matrices, 96 configs, 597 s. (b) Held-out
cross-entropy of the saved bit maps under decoder-realisable models (marginal, row, column,
group, left/up neighbour contexts), fitted on a random half of positions and scored on the
other half, against the same map shuffled within rows.

**Result.**

| cell | granularity | gain real | gain null | real/null |
|---|---|---|---|---|
| gptq/hadamard | per-weight | +0.811 | +0.810 | **1.00** |
| naive/hadamard | per-weight | +1.582 | +1.576 | **1.00** |
| gptq/native | per-weight | +1.309 | +1.262 | 1.04 |
| naive/native | per-weight | +2.112 | +2.061 | 1.02 |
| gptq/native | per-column | +0.280 | +0.260 | 1.08 |

Description length (bits/weight, held-out):

| map | marginal | best model | vs its own shuffle | gain it buys | net |
|---|---|---|---|---|---|
| per-weight, hadamard | 2.4266 | **2.4024** (column) | -0.034 | 0.777 | **-1.63** |
| per-weight, native | 2.5564 | 2.4095 (column) | -0.157 | 1.181 | **-1.23** |
| per-group, native | 1.1138 | **0.0225** (left) | -0.884 | 0.027 | +0.00 |
| per-group, hadamard | 0.0111 | 0.0002 (left) | -0.010 | 0.000 | +0.00 |

**Interpretation.**

1. **The entire oracle gain is reproduced by matched noise.** Real weights are worth 0-4%
   more than a Gaussian with the same row and column scales. Even the per-column gain in
   native coordinates is 92-97% reproduced, because `gauss_rowcol` preserves exactly the
   column scales that gain uses. The mechanism: under min-max quantization each weight's
   rounding residual is nearly uniform, and the oracle reads off which weights happened to
   land near a level. Same class of error as the in-sample k-means in E0, in a new costume.
2. **The only map worth transmitting is the one that cannot be compressed** (2.40 bits to buy
   0.78), and the only map that compresses beautifully -- the native per-group map, 50x below
   its marginal with a *depth-1* row context -- is worth +0.027 bits and flattens to a
   constant under rotation.
3. This bounds the recursive/IFS branch without building it: a two-neighbour context model is
   a strictly more general predictor of local arrangement than a quadtree at the same scale,
   and it recovers 0.003 bits/weight on the per-weight map. Kill criteria **L2** and **L4**.

## E20 — End-to-end: layouts under full-model sequential GPTQ (2026-08-31)

**Config.** All 168 linear layers, sequential GPTQ, wikitext-2 perplexity on 24 x 2048 tokens,
fp16 = 11.813. 13 configs, 48 min, peak VRAM 4573 MiB. Each row charged the *measured*
held-out cost of its own map.

| coords | alloc | total bpw | ppl | uniform equivalent | net |
|---|---|---|---|---|---|
| hadamard | uniform | 3.250 | 16.859 | 3.250 | +0.000 |
| hadamard | per-column | 3.252 | 17.324 | 3.247 | -0.006 |
| hadamard | per-group | 3.250 | 16.744 | 3.273 | **+0.023** |
| hadamard | per-tile | 3.250 | 16.859 | 3.250 | -0.000 |
| hadamard | per-weight | 5.652 | **12.470** | 4.277 | **-1.375** |
| native | uniform | 3.250 | 23.568 | 3.250 | +0.000 |
| native | per-column | 3.252 | 18.310 | 3.686 | **+0.434** |
| native | per-group | 3.272 | 24.160 | 3.207 | -0.065 |
| native | per-weight | 5.659 | 12.302 | 4.373 | -1.286 |

At 2 bits under Hadamard: uniform 55322.5, per-column 18028.6, per-group 24644.2, per-weight
**16.227** at 4.652 total bpw.

**Interpretation.**

1. **The free per-weight oracle is worth much more end to end than in activation NMSE.** At
   3.25 bpw it improves activation NMSE 3.4x but cuts d-ppl **7.7x** (5.045 -> 0.657), beating
   uniform INT4 at 4.25 bpw (12.57). Charged for its map it is still 1.29-1.38 bits/weight
   worse than doing nothing.
2. **Activation NMSE misranks the coarse layouts again** -- the fourth time in this project.
   The native per-group oracle improves activation NMSE 1.04x and makes perplexity *worse*
   (24.16 vs 23.57).
3. **Adaptive layout and rotation are substitutes, not complements.** Both attack coordinate
   anisotropy. Uniform + Hadamard (16.86) beats the best native adaptive layout (18.31), and
   stacking them gives -0.006. Kill criterion **L6**.

## E21 — Local codebook shapes, and the verdict (2026-08-31)

**Hypothesis (Phase 8).** Groups already own their scale and offset; if they are heterogeneous
enough, letting each pick a *level shape* from a shared dictionary for log2(T)/128 bits/weight
should pay.

**Config.** T in {uniform, 1, 2, 4, 8, 16} shapes learned by alternating assignment and 1-D
Lloyd-Max on min-max-normalised groups; 8 matrices, GPTQ, both coordinate systems, with the
matched-Gaussian arm. 364 s, peak 870 MiB.

**Result (activation NMSE).**

| coords | bits | uniform levels | T=1 | T=16 | T=16 vs T=1 | null T=16 vs T=1 |
|---|---|---|---|---|---|---|
| hadamard | 3 | 0.010197 | 0.007112 | 0.006519 | 1.091x | **1.093x** |
| hadamard | 2 | 0.068074 | 0.029782 | 0.027568 | 1.080x | 1.078x |
| native | 3 | 0.018507 | 0.012162 | 0.010316 | 1.179x | 1.104x |

**Interpretation.** Nearly all the benefit is the step from uniform levels to *one* learned
shape (1.43x at 3 bits) -- the classical companding gain of a Gaussian-like marginal, not a
layout effect. Local adaptivity adds 1.09x for 0.031 bits/weight, net about +0.02, and the
matched Gaussian gets the same 1.09x. Consistent with E11, learned shapes also make the tail
worse (top-0.01% relative error 0.0052 -> 0.0196 at 3 bits native), so no perplexity claim is
made for this family on activation NMSE alone.

**Decision. Verdict PL-D**, with a real PL-C component. Conditions 2, 5 and 7 of the brief's
positive-result criteria fail; kill criteria **L2**, **L4** and **L6** are all met. The map of
the optimal bit budget is not compressible enough to be worth replacing with a program: its
usable part costs 2.40 bits/weight to transmit and buys 0.78, and the part that is cheap to
describe is worth +0.00 once GPTQ and an incoherence rotation are in place. **Close the
branch.** See `docs/PROGRAMMATIC_LAYOUT.md`.

---

# Graph-aware quantization (E22-E26)

Third alternative hypothesis: the structure is neither in weight values nor in individual
sensitivities, but in the *topology of functional coupling* between channels. Full report in
`docs/GRAPH_AWARE_QUANTIZATION.md`.

## E22 — Functional geometry: A, G, and why interactions are exact (2026-09-01)

**Setup.** For `y = W x` under the K-FAC factorisation `H ~ G (x) A` with `A = E[x x^T]`,
`G = E[g g^T]`, `g = dL/dy`, the damage of a perturbation is `tr(dW A dW^T G)`. The
interaction between two channel perturbations is therefore an identity rather than a
hypothesis:

    input channels  i, j :  I_ij = 2 A_ij (e_i^T G e_j)     e_i = dW[:, i]
    output channels i, j :  I_ij = 2 G_ij (r_i^T A r_j)     r_i = dW[i, :]

Both are an elementwise product of two matrices, so the whole population is available at once
and there is no pair-sampling bias. `A` is the natural graph over input channels, `G` over
output channels.

**Config.** `G` from the model's own next-token loss, all parameters frozen, graph anchored by
detaching the embedding output -- so backward allocates activation gradients only and the run
fits in 3630 MiB. 12 modules (layers 1/11/22 x q/o/up/down), 32 x 256 tokens.

**Correctness.**

* `register_full_backward_hook` silently returned an **all-zero** gradient for `q_proj`, whose
  output is reshaped and consumed by the rotary/attention path. Replaced by a hook on the
  output tensor itself; validated against `autograd.grad` at relative error **0.0**.
* Caching A/G in fp16 broke positive-definiteness and the GPTQ Cholesky failed at the leading
  minor of order 575. Fixed by storing fp32 and symmetrising; `cholesky_inverse_upper` now
  also escalates damping instead of crashing, which cannot affect any path that already
  succeeded.
* The in-axis and out-axis decompositions are of the *same* scalar and agree to **1.7e-6**.

## E23 — Kill gate G1: do edges predict real interaction? (2026-09-01)

**Result** (3 bits, group 128, full population, 12 matrices):

| coords | comp | axis | additive/total | signed I/total | \|I\|/total | Spearman | held-out R2 |
|---|---|---|---|---|---|---|---|
| native | naive | in | 0.989 | +0.011 | 4.7 | 0.757 | 0.606 |
| hadamard | naive | in | 1.017 | -0.017 | 15.2 | 0.718 | 0.542 |
| native | gptq | in | **7.725** | **-6.725** | 39.9 | 0.743 | 0.590 |
| hadamard | gptq | in | **10.795** | **-9.795** | 197.0 | 0.713 | 0.536 |
| hadamard | gptq | out | 1.001 | -0.001 | 3.4 | 0.688 | 0.509 |

**G1 passes**: Spearman 0.64-0.76, held-out R2 0.45-0.61. Under Hadamard the scale-free
correlation edge does as well as the covariance edge (0.704 vs 0.713); in native coordinates
the covariance edge is better (0.757 vs 0.635), so there the edge is partly just scale.

**Two findings that matter more than the gate.**

1. **Without compensation, interactions are zero-mean noise.** Summed |I| is 4.7-19x the total
   damage while the signed sum is +-1%. Errors interact enormously and cancel almost exactly.
   Choosing which channels share a block cannot systematically exploit that; only changing the
   errors can.
2. **GPTQ works entirely by manufacturing negative interactions.** With compensation the sum
   of independent per-channel damages is 7.7x (native) and 10.8x (Hadamard) the actual damage.
   This is a clean quantitative statement of what second-order compensation *is*, and it
   explains the +1.14 bits/weight that compensation was worth in E14-E15. Along the output
   axis GPTQ does nothing (additive/total = 1.001) -- correct, it treats rows independently.

## E24 — Kill gate G2: topology against matched nulls (2026-09-01)

**Config.** Correlation and covariance affinities, top-32 sparsified, balanced communities of
32/64/128 from spectral (Fiedler) sequencing. Four surrogate matrices, each re-run through the
whole pipeline: vertex permutation, weighted configuration model, degree-preserving rewiring,
and an eigenvalue-preserving random-eigenvector matrix.

**Result** (block 128, off-diagonal edge energy inside blocks):

| side | null | block energy | lift vs random | modularity |
|---|---|---|---|---|
| A-corr | **real** | 0.2696 | 2.67 | 0.1213 |
| A-corr | spectral | 0.2236 | 2.11 | 0.0964 |
| A-corr | rewire | 0.1620 | 1.48 | 0.0433 |
| G-corr | **real** | 0.3783 | 3.81 | 0.2302 |
| G-corr | spectral | 0.2164 | 2.04 | 0.0885 |

**G2 passes modestly**: real / spectral null = **1.21x** (input) and **1.75x** (output).

**G7 fires for the covariance graphs.** Sorting channels by their own second moment captures
0.7130 of block energy on A-cov against 0.3263 for spectral communities -- 2.2x more. Only the
correlation graphs carry structure that is not scale.

**Communities are unstable and not reused.** Adjusted Rand index between partitions at k=8 and
k=32 is **0.07-0.30**. And because q/k/v/gate/up in every block read the *same* residual-stream
channels, their partitions are directly comparable: the ARI between layers 1, 11 and 22 is
**-0.002 to +0.011**, i.e. chance. The same channels are grouped completely differently in
every layer -- no reusable template, nothing for a graph grammar or recursive coarsening to
capture. Top-16 eigenmodes span 20-22% of channels (spectral null: 27-29%), so the modes are
delocalised, which is the regime where a block-diagonal description cannot work.

## E25 — Kill gates G3/G4: does graph grouping improve quantization? (2026-09-01)

**Config.** Communities converted to a permutation, matrix physically reordered, ordinary
contiguous quantization, then un-permuted. Two block shapes at identical scale budget: 1x128
row group and a 16x8 2-D tile (Phase 5, the smallest block that makes an output partition
mean anything). `Tile2DQuantizer` at row_block=1 reproduces `ScalarQuantizer` exactly.
528 runs, 315 s, peak 913 MiB.

**Layer-wise result** (3.25 bpw, `tr(dW A dW^T G)` relative, lift vs contiguous):

| block | coords | in order | fisher | lift |
|---|---|---|---|---|
| 1x128 | native | contiguous | 0.01679 | 1.000 |
| 1x128 | native | **scale-sorted** | **0.01204** | **1.394** |
| 1x128 | native | A-corr communities | 0.01616 | 1.039 |
| 1x128 | hadamard | contiguous | 0.00878 | 1.000 |
| 1x128 | hadamard | A-corr communities | 0.00871 | **1.008** |

**End-to-end** (all 168 layers, sequential GPTQ, fp16 = 11.8134):

| coords | order | ppl | d ppl |
|---|---|---|---|
| hadamard | contiguous | 17.2459 | +5.4324 |
| hadamard | random, seed 2 | **16.5401** | +4.7266 |
| hadamard | random, seed 3 | 16.6445 | +4.8311 |
| hadamard | scale-sorted | 16.7034 | +4.8900 |
| hadamard | random, seed 1 | 16.9167 | +5.1033 |
| hadamard | **A-corr communities** | 16.9203 | +5.1069 |
| native | contiguous | **23.3339** | +11.5204 |
| native | scale-sorted | 23.6484 | +11.8350 |
| native | random | 24.0975 | +12.2840 |
| native | A-corr communities | 24.3299 | +12.5165 |

**G3 and G4 both fire.**

1. Where adaptive ordering does anything at all (native + GPTQ, layer-wise), plain scale
   sorting is worth 1.394x and the best graph ordering 1.085x -- the graph loses to the trivial
   baseline by 4x.
2. **The graph ordering is statistically indistinguishable from a random permutation.** Under
   Hadamard it lands at the bottom of the three-seed random range (4.727-5.103), and the spread
   between random seeds (0.38 d ppl) exceeds the gap the graph produces (0.33). Everything that
   looked like a graph benefit is "any permutation other than the identity helps GPTQ slightly
   under rotation".
3. **Fifth metric misranking in this project.** Layer-wise Fisher error said native scale
   sorting was 1.394x better; end to end it is 0.973x, i.e. worse. In native coordinates the
   identity ordering is genuinely good -- adjacent residual channels have similar scales and
   make tight min-max groups -- and reordering by correlation breaks that.
4. The 16x8 2-D tile is worse in absolute terms than the 1x128 row group everywhere, and
   output communities move it by 1.1-1.4%.

## E26 — G6, permutation legality, and the verdict (2026-09-01)

**Permutation legality (verified, not asserted).** Permuting the MLP intermediate of all 24
blocks -- `down_proj` input columns together with `up_proj` and `gate_proj` output rows --
changes the fp32 logits by **1.1e-6**, i.e. round-off. That permutation is free, runtime class
H0. Nothing else is absorbable per layer: q/k/v/gate/up share the residual stream and would all
need one common permutation; `o_proj` is tied to attention head structure. An explicit index
costs 0.008 bits/weight against a gain of about 0.005 by the layer-wise metric and zero end to
end, so **G5** fires.

**G6 — cross-layer error interaction.** Layer a alone, b alone, both, at 3 bits, in end-to-end
perplexity:

| distance | mean interaction, % of additive damage |
|---|---|
| 1 | **5.3%** |
| 2 | 8.4% |
| >=5 | **3.3%** |

Interaction is small, *positive* (damage compounds rather than cancels), and does not
concentrate on adjacency -- distance-2 pairs interact more than distance-1, and distance-21
still shows 2.4%. There is no locality for a cross-layer graph to organise, and what
compounding exists is already handled by sequential GPTQ. Combined with the zero cross-layer
community reuse in E24, Phases 6, 7 and 9 have no target and were not built.

**Decision. Verdict GA-B** -- the topology is real (G1 and G2 pass) but everything practically
relevant about it is first-order anisotropy, which rotation removes more cheaply. G3, G4, G5,
G6 and G7 all fire. **Close the branch.**

The finding worth carrying forward is not a method but a measurement (**GA-F**): compensation
is not a refinement on top of independent quantization, it is a different regime -- the
additive damage is 7.7-10.8x the achieved damage, so GPTQ destroys an order of magnitude of
error through interactions it creates itself. Future work on sub-4-bit quality should target
that mechanism (better compensation objectives, joint or iterative compensation, distillation)
rather than deciding which weights share a block. Every grouping study in this repository has
now reached the same conclusion from a different direction.

## E27 — Re-prioritisation to four surviving branches (2026-09-01)

Directive: no new branches without a very strong data signal. Surviving: (A) wider-scope /
cross-output compensation, (B) exact CVP oracle + adaptive search, (C) the GPTQ-worse-than-RTN
failure mode, (D) post-GPTQ rotation landscape. Calibration robustness (E) is a pending
decision-finish only: the 262k/524k budgets on `down_proj` decide it.

**Infrastructure.** The scope arms crashed in `block_G`: the captured decoder kwargs carry a
`DynamicCache`, harmless under `no_grad` (repeat calls are bit-identical -- verified), but
under `enable_grad` the cached K/V keep the first probe's graph alive and the second probe's
backward walks a freed graph. `block_G` now strips the cache from its own calls. Verified on
CPU for all three horizons. The end-to-end queue was cut to the scope ladder only
(`--only had_cd_module,had_cd_block1,had_cd_block2,had_cd_block1_d01,nat_cd_block1,
nat_cd_block2`); `had/nat_cd_layer` are configuration-identical to the already-run
`had/nat_gptq_cd` (pipeline is exactly deterministic), and the naive/adamp diagnostics are
parked.

**Branch C has a second candidate mechanism, found by reading the engine.** The canonical
`gptq()` calls `find_params` on `W1[:, j:g_end]` where `W1` is the current *block* (width
128), so with `group=256` every group's scale is fitted on its first 128 columns only; the
original GPTQ slices the full working matrix. That predicts the whole observed signature of
the b4/g256 failure: only the one ladder rung with group > blocksize, ranges *shrinking*
(0.900 measured -- the falsified "inflation" story pointed the right finger at the wrong
direction), native-only (heavy tails make half-ranges differ; Hadamard equalises them).
`gptq_scalefeed.py` separates this (M1) from the sequential-contamination hypothesis (M2)
with arms rtn / block / full / bs=g / frozen at identical storage, engine-verified
bit-identical to `gptq()` in the `block` arm.

First rows (layer 1 `down_proj`) say the answer is *rate-dependent*: at b4/g256 native all
clipping sits in second halves of groups (0.0000/0.0104) and `full`~`frozen` fix an 8.7% gap
(M1, contamination negligible); at b3/g256 the truncated fit accidentally *helps* (implicit
range shrinkage, the AWQ-clipping effect); at b2 frozen-original scales beat working-W scales
by 24-43% -- M2 is real at 2 bits and `frozen` is the best arm. Also notable: `frozen` improves
the standard b3/g128 rung by 5.7% on this matrix.

**Branch B predictor.** `oracle_pred.py` collects per-row cheap features (boundary-code
fractions, normalised margin statistics, kurtosis, step-vs-curvature, cheap-CD outcome)
against the strongest practical searcher (alternating k=2 + CD, then k=8, then k=24 exact
blocks) on 9 matrices x {2-bit native, 2-bit Hadamard, 3-bit native}. Question: what fraction
of the CD-to-strong gap do the top rows by each cheap feature capture.

**Transform accounting reworked** (directive item 3): deployment classes fused / online /
stored-per-matrix in `analyze_transform.py`; all learned arms were trained per matrix, so
their gains are upper bounds for the fused deployment that would have to share one R
stream-wide. At 3 bits on `q_proj` the fused T3 gain (+0.11 equivalent bits, zero cost)
is comparable to the group-size lever (+0.157 bits per 0.25 bpw spent); the stored deployment
is dominated by both.

**Calibration (E), interim.** `down_proj` fit/held gap closes with tokens/dim exactly as the
finite-sample story predicts: 1.29 at 32k (6.7 t/d) -> 1.15 at 64k (13.5 t/d); q_proj 1.07 ->
1.04. Held-out spread across disjoint draws stays 0.2-1.0% with **no bitrate trend** while
~20% of codes differ: non-unique, not unstable, at every rate including 2-bit.

### E27b — Branch C closed on mechanism; the fix is prior art (2026-09-01)

Small-group extension confirms M2 is not a g256 artefact: at b2/g128 (where block == full, no
M1) frozen-original scales beat the standard fit by 37% (0.11273 -> 0.07105) and at b2/g64 by
29%, transferring to held-out A (0.29425 -> 0.20782). Working-W scales at b2 are ~15-19%
*larger* than original-W scales: compensation inflates the working range, the min-max grid
coarsens, later errors grow -- a feedback loop that freezing breaks. Group order matters only
where M1 is active (native b4/g256: 0.01527 fwd vs 0.01098 rev) and not otherwise.

Prior art check: **the fix is published.** Two-Stage Grid Optimization (arXiv 2602.02126)
initialises group scales before GPTQ's sweep (their Stage 1, stronger than plain min-max:
reconstruction-loss-optimal per group) and refits post-hoc at fixed codes (their Stage 2 ~=
our `refit_scales`), reporting 4-6pp at INT2 -- consistent with our -29..-37%. The GPTQ repo's
`--static-groups` flag has always precomputed group grids in advance, motivated by act-order
compatibility, not quality. Rethinking Residual Errors (2604.07955) targets objective
alignment, not scale contamination.

**Branch C verdict input:** the reproduced GPTQ-worse-than-RTN point is fully explained (M1,
our engine's group>blocksize truncated fit -- an implementation trap, not a GPTQ property);
the general low-bit mechanism (M2 scale feedback) is real, measured, and its remedy is prior
art. What remains ours is the forensic decomposition (block/full/frozen at identical storage,
clip-signature per group half) and the reconciliation. Not a headline; a solid diagnostic.
The engine fix (full-group fit) goes in after `calib_size` finishes, then the g256 rows of
`comp_capacity` get re-run.

### E27c — Branch C data complete (2026-09-01)

Final aggregate over 6 matrices x 2 coords x 8 rungs: **at 2 bits frozen-original scales win
24/24 cells** (median frozen/block: g64 0.902, g128 0.884, g256 0.762-0.827) -- the effect
persists at every group size and under Hadamard, growing with group width. At 3 bits frozen
wins 24/27 by 0.5-1.6%; at 4 bits it is neutral except where M1 is active. Monotone in
bitrate, exactly what the range-inflation feedback mechanism predicts (compensation mass
grows as the grid coarsens). Remaining branch-C work is bookkeeping: fix the engine's
group>blocksize fit after calib_size lands, re-run comp_capacity's g256 rows.

### E27d — Branch B predictor complete (2026-09-01)

1440 rows, 9 matrices, 3 cells. Gap concentration exceeds the hypothesis (50% of the mass in
the top 1-3% of rows, 80% in 4-7%), a cheap trigger exists (d_gptq captures 3-5x random;
ambig x damage works under Hadamard), but the mean CD->strong prize is 0.25-0.49% of the local
objective -- nothing that survives to PPL. Adaptive search: working trigger, no prey. The
durable result is the certification side: converged CD is row-wise near-optimal everywhere
measured. FINDINGS.md section 7 updated with the final table.

### E27e — Calibration 262k: the closure conditions are met (2026-09-01)

At the community-standard 262144-token budget: down_proj gap 1.041-1.042 across ALL nine
bitrates (2.125-4.25 bpw identical to three decimals), spread <=0.28%, codes differing 17%;
q_proj 1.014 flat. Excess gap has halved on every doubling exactly (1/N), matched tokens/dim
collapses the two matrices onto one curve. The brief's closure conditions at ~50-100
samples/dim (fit/held -> 1, held-out variance <1%, eigenspaces stabilising: down_proj
0.965/0.975/0.879 rank-8/32/128 and rising) all hold. 2-bit shows no excess sensitivity over
4-bit anywhere. Pending: 524k confirmation + the seqlen-2048 pass, then the branch closes as
"finite-sample covariance estimation, not a robust-GPTQ research problem", keeping the
diagnostic finding that ~20% of codes differ across draws with <1% quality consequence.

### E27f — 524k confirmation; canonical engine fixed (2026-09-01)

524k @ seqlen 512: down_proj gap 1.024-1.025, q_proj 1.010, both flat across bitrates,
spread <=0.35%, codes 16-19% different. The seqlen-512 pass of the calibration sweep is
complete; branch E's empirical question is answered (2048 control still running).

The canonical `gptq()` now fits each group's scale on the FULL group, crossing the block
boundary from the working matrix when group > blocksize (original GPTQ semantics). Verified
bit-exact: g256 == the scalefeed `full` arm (0.0), g128 == the old engine (0.0) -- nothing
else in the project shifts. `comp_capacity` down_proj rows re-running with the fixed engine
into comp_capacity_fixed.jsonl (CPU); the running calib_size/scope processes hold the old
module in memory, so their sweeps stay internally consistent.

### E27g — Capacity tables corrected with the fixed engine (2026-09-01)

down_proj rows re-run (CPU, fixed engine) and swapped into comp_capacity.jsonl (old rows in
comp_capacity_prefix_bug.jsonl.bak; non-g256 rungs differ only by device float noise,
max 3.4e-04). The "GPTQ worse than RTN" table section is now empty: b4/g256 native
-7.1% -> +62.5% removed (L11), +62.4% -> +79.5% (L22). Ironic footnote: at b2/g256 the
correct full-group semantics is ~35% WORSE than the truncated-fit bug was -- the bug acted as
an accidental clip-search, which is independent confirmation that min-max scale estimation is
the weak link at 2 bits. FINDINGS section 4 updated.

## E28 — First end-to-end scope result: module horizon transfers (2026-09-01)

The decisive arm of branch A landed. At 3.25 bpw, Hadamard, bit-identical storage, starting
from the GPTQ+CD layer optimum (cd_pre 6):

    had_gptq          ppl 16.7968   gap 4.9834
    had_gptq_cd       ppl 16.7700   gap 4.9566   (closes 0.54% of the gap)
    had_cd_module     ppl 16.2785   gap 4.4650   (closes 10.4% of the gap)

The first local-to-global transfer success in the program, ~19x what layer-scope CD bought.
Internal consistency: extra code movement occurs ONLY on matrices with non-identity G
(q/k/v/gate/up 2.4-3.9% vs 1.1-1.6% under layer CD); o_proj and down_proj, whose outputs ARE
the module endpoints, flip identically to layer CD (1.05%/1.50%). Wider objective improved
10.7% over its start (obj 0.893). Cost 9,497s optimizer + 86s G-probes for the full model.
Caveats: single arm/model/rate/coord; sweep-count confound unlikely (layer CD converges and
breaks early) and will be settled by the block arms' horizon shape. had_cd_block1 now running.

### E28b — Overnight stall resolved (2026-09-02)

calib_size froze after the 131k @ seqlen-2048 budget (11h at zero CPU, likely blocked on GPU
memory while joint_ppl held the device at 100%). Killed. The decisive seqlen-512 pass had
already completed in full; the 2048 control pass keeps its 32k/64k/128k cells, which are
enough for the sequence-length finding (excess-gap ratio 2048/512 = 1.2-1.8, shrinking with
budget). The 262k/524k @ 2048 confirmation cells are foregone -- noted, not rerun, since they
would only re-confirm a law already measured at five budgets. Branch D's plateau chain
(basin -> dfo -> sweep) relaunched on CPU, where it does not contend with the scope ladder;
had_cd_block1 now has the GPU alone.

### E28c — Branch D gating item complete: the rotation landscape is a flat plateau (2026-09-02)

Basin (10 starts, fresh hard requantization, code mobility 24-39%/outer): every rotation-
family start converges into a +-0.5% band; identity converges to a separate basin 2x worse.
DFO (400 hard-pipeline evaluations, 16 accepted): -0.65%. Best-of-200 random rotations:
-0.65% -- the identical objective value (0.006103), a common floor. The 200-rotation plateau
spans +-1.1%; Hadamard sits slightly above the median. Unguided hard-objective search is
closed in the weak band (<1-2%) by four independent methods; the only demonstrated escape to
a substantially better basis remains surrogate-guided (T3, -14% hard metric). Open question
reframed: fused/shared-constraint survival of T3-class bases + end-to-end -- a new experiment.
FINDINGS section 5 updated. Three of four verdict gates are now closed; the last is the
scope ladder (module arm in, block1 finishing).

### E28d — Second radius point: the curve flattens at the module (2026-09-02)

had_cd_block1: ppl 16.3174 (obj 0.9013) vs module 16.2785 (0.8925) vs layer 16.7700. The
layer->module jump (10.4% of the gap) is followed by a flat/slightly-negative module->block1
step (+0.04 ppl) even though block scope adds non-identity G to o_proj and down_proj. The
exploitable structure sits inside the module (QKV softmax mixing, gate/up GLU product); the
residual stream past the module endpoint contributes nothing at this horizon. block2 (running)
separates radius saturation from probe-noise growth. Practical recipe so far = the cheap end:
module-G, one probe pass, ~10% of the quantization gap at bit-identical storage.

### E28e — Redirect: module scope at 2.25 bpw (2026-09-02)

Per user instruction the block2 arm was stopped mid-run (block1_d01 and the native controls
fell out of the queue with it; re-queue later if wanted) in favour of the sharper question:
does the module-scope gain grow at 2 bits, where local compensation capacity is exhausted?
New run configs/joint_ppl_scope_b2.json: had_gptq_b2, had_cd_layer_b2, had_cd_module_b2 --
all b2/g128 = exactly 2.25 bpw, Hadamard, same pipeline and eval. Runs with the fixed engine
(bit-identical to the old at g128, so comparability holds).

## E29 — Phase 2 opens: representation vs scope (2026-09-02)

New directive: scalar Hadamard+GPTQ+CD becomes the control baseline; the question is whether
module-scope compensation survives or grows on a modern vector representation.

**P1 closed — the scalar INT2 cliff diagnostic.** At 2.25 bpw (b2/g128 Hadamard):
GPTQ 121567 -> layer-CD 13914 -> module-CD 20654. Module scope does NOT rescue the model and
is 1.48x WORSE than layer scope end-to-end while improving its own wider objective (0.868) --
the G linearisation is measured through already-degenerate hidden states, so the metric no
longer reflects the true loss. Two conclusions: (1) scalar b2/g128 is representation-limited
on this 0.5B model; stop pushing it. (2) Module scope requires a functioning network: it
closed 10.4% of the gap at 3.25 bpw and inverts to harmful at 2.25 bpw. The cliff between
those rates is where the linearisation breaks.

**P2 running:** external VQ baseline feasibility (VPTQ/GPTVQ/AQLM/QuIP#) via background agent.
**P3 written:** src/lwc/vq.py -- GPTVQ-like minimal VQ-2D (d=2, K=32, 5-bit index = 2.5 bpw
pre-metadata; frozen amplitudes per E27; diag-Hessian-weighted k-means codebook per matrix;
variant A group scales / variant B row scales) plus vq_refine: exact CD over codeword indices
under tr(D A D^T G) with the same dense-G backoff as cd_refine. Next: joint_ppl integration.

## E30 — Reframing after the prior-art sweep (2026-09-02)

Directive: YAQA (H_O x H_I from full-model KL, two-sided, quantizer-agnostic), BaKron,
GuidedQuant (end-loss aware, explicitly ignores cross-output interactions), CBQ/ACBQ, PCDVQ,
LLVQ, WaterSIC are known -- no "first module/downstream/cross-output/VQ+functional" claims.
Surviving questions: (A) functional compensation radius -- is there an optimal horizon
strictly between layer and full-model? (B) scope x representation interaction; (C) module-
aware rate allocation under a fixed budget (WaterSIC-style, but two-sided A+G, at hardware-
realistic granularity only).

**S3 implemented natively**: `block_G(horizon="model", tail=(norm, lm_head))` -- Fisher of
the remaining network via y ~ p(model) sampling and token-subsampled lm_head tail, giving the
full-model KL geometry as the SAME dense per-matrix G the narrower horizons use (YAQA's H_O
is a Kronecker approximation of exactly this object; ours is the dense per-matrix version
inside the same pipeline and solver, so the radius comparison is geometry-vs-geometry, not
algorithm-vs-algorithm). CPU smoke passes; loss-sensitivity ordering across the 7 matrix
types is sensible (k/o > v > down > q > gate/up per unit output).

Feasibility agents running: YAQA/GuidedQuant/QTIP-quantization-side/PCDVQ portability.
GPU order per section 12: had_cd_model @ 3.25 (completes the radius curve) -> VQ-2D factorial
overnight (mechanistic ablation for question B; costs nothing it would otherwise block).
GPTVQ external env build in progress in ../gptvq-ext (agent verdict: only Windows-viable
external VQ baseline; VPTQ excluded -- cuML/flash-attn, no small-model Hessians).

### E30b — Feasibility results for the geometry comparison (2026-09-02)

Agent findings: YAQA's Hessian collection is the identical Fisher mechanism our S3 horizon
implements natively (y ~ model distribution, CE backward); their H_O/H_I are Kronecker
sketches of what we measure as dense per-matrix G, and their LDLQ_2hess solver is pure torch
with a generic cb.quantize interface (portable later for the representation phase). Full YAQA
repo needs FSDP/NCCL surgery on Windows -- native S3 stays the instrument for the radius
curve. GuidedQuant's geometry (per-output-channel-group token-reweighted input Hessians,
explicitly block-diagonal Fisher: "interactions between ... output channels of the same layer
are ignored") is reproducible from the same probe pass as S3, enabling the clean three-way
GuidedQuant-geometry vs module-G vs model-G comparison in one pipeline and solver. QTIP has a
verified pure-PyTorch quantization+ppl path (patch one fast_hadamard import, single-process
Hessians, Qwen2 bias handling; ~1-2 days) -- the strong-representation baseline. PCDVQ: no
code; faithful reimplementation ~2-4 days if the shape-gain question ever earns it.

## E31 — Attention-decomposition branch opens (2026-09-02)

Directive: mechanistically decompose WHY module scope beats layer scope, via a controlled
ladder inside attention -- layer -> QK^T logits -> softmax -> attention output -> module/
block/model -- at identical representation, rate, rotation and solver. Prior-art agent
findings that sharpen the framing: BoA derives the softmax-Jacobian objective but its
shipped relaxation DROPS J_sigma (practical BoA Q/K objective == our logit rung), never
measuring the cost; APTQ/BoA establish attention-output > layer-local pairwise (so A5-vs-A0
alone is not novel); the genuinely open items are the multi-rung isolation, attention-level
margin/top-k objectives for weight PTQ, the attention-output-vs-softmax-KL verdict, and any
value-aware sensitivity theory (VATP's score-x-value argument exists only for KV-cache token
pruning).

Implementation: `attnscope.py` -- G at three intra-attention endpoints from the same probe
machinery (logits: exact given the partner matrix; softmax: Fisher sampling j ~ p, so
E[gg^T] realises diag(p)-pp^T; attn_out: adds the V-weighting). Q/K map rebuilt from layer
weights with transformers' own rotary/GQA helpers; fidelity vs the module's real forward =
2.1e-06. k_proj is ~20x more loss-sensitive per unit output than q_proj at every endpoint --
GQA: each k output serves 7 query heads.

Running: (CPU) `attn_decomp.py` layer-level study -- 4 arms x layers {5,11,17}, ladder
metrics on held-out text + H-high/H-margin/H-entropy/H-value Spearman diagnostics + the
section-9 fixed-KL value-dispersion test. (GPU queue) S3 model-horizon arm -> e2e attention
ladder (attn_logits / attn_smax / attn_out / module_attn at 3.25 bpw, all refining only
q/k(/v), everything else at layer-CD) -> VQ-2D factorial.

### E31b — Layer-level attention ladder + sensitivity theory (2026-09-02)

Ladder at b3/g128 Hadamard, q/k quantized, v/o fp, held-out metrics, 3 layers (5/11/17),
monotone with no inversions: layer -> logits cuts every downstream metric 40-50% (KL 0.020 ->
0.011, attn-out MSE 1.6e-3 -> 0.93e-3 at L17); softmax adds a few % (KL 0.0114 -> 0.0108);
attn_out adds ~7-13% more on the functional metric (dY 0.93 -> 0.81e-3). Each rung wins its
own objective. Preliminary answer to the mechanism question: MOST of the intra-attention
prize is plain Q/K cooperation on QK^T (the term BoA's relaxation keeps); the softmax
Jacobian and V-weighting are real but second-order refinements. e2e PPL ladder queued.

Sensitivity diagnostics (Spearman, 20k rows/layer): H-high falsified (top-logit magnitude
rho ~ 0); H-margin confirmed (-0.61/-0.67); competition measures strongest (jtrace +0.69,
pmax -0.69); H-value confirmed and strongest overall (v_disp +0.70/+0.64). Section-9 test:
at FIXED softmax-KL decile, p-weighted value dispersion still predicts attention-output
damage (rho +0.13/+0.13/+0.19 across layers) -- the VATP score-x-value argument, measured
for the first time as a weight-PTQ objective property: distribution preservation is the
wrong target exactly where competing values disagree, though the effect is modest at 3 bits.

## E32 — The compensation-radius curve is complete and sharply non-monotone (2026-09-02)

had_cd_model (S3: dense per-matrix Fisher of the remaining network, measured in-pipeline,
same solver/rate/probe budget as every other horizon): ppl 16.7883 -- worse than LAYER scope
(16.7700), while improving its own objective the most of any arm (obj 0.8547 vs module
0.8925). Full curve at 3.25 bpw: layer 0.54% of the gap closed, module 10.4%, block1 9.6%,
full-model 0.17%. R* = module, strictly between layer and full-model -- the H2 scenario.

Mechanism reading: the wider the horizon, the more of the measured curvature is estimation
noise at fixed probe budget (2048-token sinks, 2 probes), and the discrete optimizer walks
into it -- the same failure family as the rank-64 surrogate, in milder form. Honest YAQA
positioning: NOT "full-model objectives lose" -- YAQA sketches on ~134M tokens; ours says at
matched cheap estimation the module endpoint is the sweet spot, and the full-model geometry
demands orders of magnitude more estimation before it can act. The radius curve is therefore
really a horizon-vs-estimation-noise frontier, which is a sharper claim than "module wins".

Caveats: one model, one rate, one probe budget; a g_tokens/n_probe scaling test on the model
horizon would separate "noise-limited" from "fundamentally unusable directions" -- queued as
a cheap follow-up. e2e attention ladder now running (auto-chain).

### E32b — First external reference point + first e2e ladder rung (2026-09-02)

GPTVQ (Qualcomm repo, Qwen2 loader patch, columns-per-group 128) on Qwen2.5-0.5B at
W2/vq-dim2/g2048 (~2.15-2.25 effective bpw, exact accounting TBD from quant.py): wikitext-2
full-test ppl 90.9 at seqlen 2048, CPU-only run. Degraded but alive -- confirms the scalar
INT2 collapse (13914 at 2.25 bpw) was representation-limited, and anchors the VQ branch.
Not directly comparable to our 24-window eval protocol; unified re-eval planned.

e2e attention ladder rung 1: had_cd_logits (q/k only, everything else layer-CD) ppl 16.6713
-- 2.5% of the quantization gap vs GPTQ, ~4.7x the whole layer-CD gain, ~1/4 of the full
module gain. Q/K logit cooperation transfers end-to-end but is not the whole module story.

### E32c — e2e attention ladder, rungs 2-3 (2026-09-02 evening)

had_cd_smax (softmax-Fisher G, q/k only): ppl 16.9164 -- WORSE than plain GPTQ (16.7968).
The third instance of the same law: transfer tracks the estimation quality of the geometry.
One sampled key per query row per probe is too noisy, and 24 layers compound it; the
layer-level study (where softmax scored well on held-out ladder metrics) did not see the
compounding. A2 is dead as a mechanism explanation at this budget.

had_cd_attnout (V-weighted pre-o_proj endpoint, q/k/v refined): ppl 16.6066 -- best
attention-only arm so far (3.8% of the gap vs logits' 2.5%). Confound to resolve: attnout
refines v_proj too, logits does not; the fair same-set comparison is had_cd_mod_attn
(q/k/v at the post-o module endpoint), now running as the last ladder arm.

Ladder so far (gap closed vs GPTQ): layer 0.54% | logits(qk) 2.52% | smax(qk) -2.13% |
attnout(qkv) 3.82% | mod_attn(qkv) pending | module(all 7) 10.4%.

### E32d — The ladder's surprise: post-o endpoint hurts the attention side (2026-09-02)

had_cd_mod_attn (q/k/v at the POST-o_proj module endpoint -- the exact G the winning module
arm used for those matrices): ppl 16.8886, worse than layer-CD. Same refined set at the
PRE-o endpoint (attnout) gave 16.6066. So at matched refined matrices, moving the endpoint
past o_proj flips the sign of the transfer, and the full module arm's +10.4% cannot be
coming from its attention side. Implication: the MLP (gate/up at the mlp-output endpoint,
i.e. through the GLU product) must carry the bulk of the module gain. The missing
decomposition arm -- module-G on gate/up ONLY (had_cd_mod_mlp) -- was inserted ahead of the
VQ factorial in the GPU queue; prediction if the MLP story is right: ppl ~16.3-16.4.

e2e ladder standings (gap closed): layer 0.54 | logits(qk) 2.52 | smax(qk) -2.13 |
attnout(qkv) 3.82 | mod_attn(qkv) -1.84 | module(all) 10.4.

### E32e — Superadditivity: no isolated sub-mechanism carries the module gain (2026-09-02)

had_cd_mod_mlp (module-G on gate/up only): ppl 16.7111 (+1.72% of gap) -- prediction of
16.3-16.4 falsified. The decomposition sums to nothing: qkv@post-o -1.84%, gate/up +1.72%,
union of the same sets refined jointly +10.4%. The module gain is strongly superadditive:
an emergent property of refining BOTH halves of every block together, not attributable to
QK logits, softmax, V-aggregation, or the MLP alone. Leading hypothesis: cross-layer
compounding -- jointly cleaner block outputs give later layers cleaner A and G, and 24
layers multiply it; partially-refined states leave a dirty component that poisons the other
half's geometry. Direct test inserted before the VQ factorial: module-G restricted to layers
0-11 vs 12-23 (glayers key). If compounding dominates, early >> late and early+late gains
should undershoot the full arm.

FINDINGS 5b updated with the complete ladder. Answer to the phase's final question, current
form: neither better QK logits, nor softmax nonlinearity, nor V-dependent functional
equivalence alone -- the module endpoint wins because joint refinement of attention AND MLP
against their own functional outputs compounds through depth. The framing shift the brief
anticipated ("functional-equivalence-aware quantization") holds, but at the BLOCK-pipeline
level rather than any single endpoint.

### E32f — Half-depth test: the gain is superlinear in coverage (2026-09-03 night)

mod_early (G in layers 0-11): +1.83% of the gap; mod_late (12-23): +2.25%; sum 4.08% vs the
full arm's 10.4%. Each half captures ~20% of the full effect at 50% layer coverage -- the
gain is superlinear in coverage and superadditive across BOTH partitions tested (across
depth: joint = 2.5x sum; within block: sum ~ 0, extreme case with the attention side alone
negative). Simple early-layer compounding is refuted (halves nearly symmetric). Standing
phenomenological summary: the module-scope benefit requires consistent joint refinement
everywhere -- partial coverage of any kind forfeits most of it, and attention-side-only
coverage is actively harmful. The within-block asymmetry (qkv@post-o negative alone,
positive jointly) remains the one unexplained detail worth a targeted follow-up.

VQ factorial (6 arms) now running via the chain.

## E33 — Scope x representation: R2, the scope effect is representation-independent (2026-09-03)

VQ-2D factorial at 2.514 bpw (d=2, K=32, row scales, frozen amplitudes, full calibration):
vq_gptq 19.1554 -> vq_layer 18.7567 (5.4% of the VQ gap; layer refinement DOES transfer on
the richer discrete space, unlike scalar's 0.54%) -> vq_module 18.0595 (14.9%). The module
increment over layer is +9.5pp of gap on VQ vs +9.9pp on scalar at 3.25 bpw -- nearly
identical. Verdict R2: the functional-scope effect is representation-independent and sits
additively on top of the representation gain. Combined with E29 (VQ un-collapses the 2.25
cliff) and E32d-f (the gain is a global cooperative phenomenon), the program's two central
claims are now: (1) representation sets the usability floor; (2) consistent module-scope
joint refinement adds a further ~10% of the quantization gap regardless of representation.
Scalar b2/g64 matched-rate controls run next.

### E33b — Factorial complete; phase synthesis (2026-09-03)

Scalar b2/g64 controls (after fixing the per-config group override, bogus 2.25 duplicates
purged; the mis-run incidentally reproduced the old b2/g128 number bit-exactly): gptq 4197,
layer 3288, module 2054. The 2.25-bpw module inversion is gone at 2.5 (module -38% even in
a broken model) -- the G-usability boundary sits between 2.25 and 2.5 bpw. Matched-rate
representation gap: ~200x (scalar 4197 vs VQ-2D 19.2).

GPTVQ external curve so far: ~2.2 bpw -> 90.9, ~2.3 bpw -> 61.5 (own protocol; W3 running,
dim4/g65536 asserted -- retry later with a divisor-safe groupsize).

Phase answers written into FINDINGS 5c: (1) INT2 cliff was representation-limited; (2)
module-scope survives VQ (+9.5 vs +9.9 pp); (3) independent additive gains; (4) 4D deferred;
(5) best <3 bpw result: 18.06 @ 2.514 bpw = VQ-2D + module scope.

### E34 — Probe-budget scaling: the full-model failure was noise-limited (2026-09-03)

had_cd_model_hiprobe (g_tokens 16384, n_probe 4 = 8x the standard budget): ppl 16.5275,
5.40% of the gap -- vs 0.17% at 1x budget and module's 10.4% at 1x. The full-model geometry
is not fundamentally unusable; it is estimation-starved, and even 8x the budget leaves it at
half the module effect. The radius curve is now a quantified two-axis frontier (horizon x
estimation budget): at cheap estimation the module endpoint dominates outright; the
full-model direction only pays with orders-of-magnitude more measurement, which is exactly
the regime YAQA operates in (134M-token sketches). FINDINGS section 3 updated.

## E35 — Rate-allocation oracle: branch killed by its own criterion (2026-09-03)

RA-A per-row Lagrangian allocation from an 8-rung menu at matched total bits, damages
measured post-GPTQ+CD (what deployment would see), 12 matrices (print had the sign flipped;
real gain = eq - budget). At 2.5 bpw: q/o_proj +0.13..0.17 eq-bpw, up_proj +0.11, down_proj
+0.03; per-matrix granularity -6.3% damage. At 3.25 bpw: exactly zero everywhere (the
allocator degenerates to uniform). Mean ~0.11 eq-bpw, low-bit-only, vanishing on the widest
matrix -- below the 0.25 continue threshold and barely above the 0.1 kill line only on
attention matrices. RA-AG would differ only by diag-G row weights and cannot plausibly
bridge 0.11 -> 0.25. **Branch killed** as a standalone direction; kept as a diagnostic
(a real but small per-row allocation prize exists only in the deep low-bit regime).
Question C answered; all three surviving-question branches now have verdicts.

### E35b — External GPTVQ curve complete (2026-09-03)

W3/dim2/g8192 (~3.05 bpw): wikitext-2 full-test ppl 16.45. Final external reference curve
(GPTVQ's own protocol, CPU runs): ~2.2 bpw -> 90.9, ~2.3 -> 61.5, ~3.05 -> 16.45. At ~3 bpw
it lands in the same band as our scalar+module (16.28 @ 3.25, our 24-window protocol);
cross-protocol caveat stands until the unified re-eval. All commissioned background work is
now complete; the machine is idle for the first time in the phase.

## E36 — Validation phase opens: P0 freeze + P1 audit launched (2026-09-03)

P0: method and protocol frozen at commit b57c9acf58e4b1f48c9322134f0752b902449776
(docs/FROZEN_PROTOCOL.md): five arms (fp16/gptq/layer/module/block at exact 3.25 bpw), all
engine constants pinned, no tuning on any new model permitted; primary metric mean NLL per
token; exact physical bpw. Evaluation hardened for the audit: full contiguous wikitext-2
test (~137 windows), 512 pre-fixed C4 validation windows (immutable cache), MC estimate of
KL(P_fp||P_q) (8 samples/position, 64 anchored windows), per-window NLLs for paired
bootstrap, and fresh-G verification (16 extra windows drawn with an offset seed re-score
the wider objective after refinement; decisions provably never see them -- decision windows
are drawn byte-identically to all historical runs).

P1 launched: three calibration draws (seeds 0/1/2) x five arms on Qwen2.5-0.5B, local GPU
(peer session yielded VRAM), ~20 h. Pre-registered pass: module < layer on wt2 AND C4 in
>=2/3 draws with positive bootstrap interval; C4 or fresh-G failure blocks P2/P3.

## E37 — P1 verdict: CLEAR PASS (2026-09-04)

Three independent calibration draws x five frozen arms, full-wt2 + 512 C4 windows + MC-KL:
module beats layer AND gptq in all three draws on both corpora with every paired-bootstrap
95% interval strictly negative (module-layer wt2: -0.032/-0.029/-0.018; C4: -0.042/-0.030/
-0.020 NLL). KL best for module in every draw. Layer-CD itself is ~neutral-to-negative on
the hardened protocol. Block ~ module (ordering draw-dependent: block won draw 2), both far
above layer -- module stays the practical sweet spot. Practicals: 2.5-2.6% codes moved,
~3000 s opt on 0.5B, 3.2 GiB peak VRAM, bit-deterministic (fp16/gptq numbers reproduced
exactly across process restarts). Fresh-G held in all draws (improvement larger under the
unseen-text G). The module effect on Qwen2.5-0.5B is now a REPRODUCED result. Proceeding to
locked P2 per the pre-registered plan.

## E38 — P2 draw 0 FAILS: the effect does not replicate on Qwen2.5-1.5B (2026-09-04)

Frozen protocol, no tuning, same commit, exact 3.25 bpw, 5090. fp16 wt2 2.22623 / c4 2.74330
(ppl 9.26 / 15.54, matches literature).

    arm     NLL wt2   gap closed    NLL c4    gap closed    KL
    gptq    2.55340   --            3.05028   --            0.35106
    layer   2.55930   -1.8%         3.06778   -5.7%         0.35868
    module  2.56162   -2.5%         3.06280   -4.1%         0.35835
    block   2.57089   -5.4%         3.06560   -5.0%         0.36364

EVERY refinement arm is worse than plain GPTQ on every metric. Module vs layer is a wash
(better on c4/KL, worse on wt2). The pre-registered P2 bar (median >= ~5% of the NLL gap
closed) is missed by a wide margin and with the wrong sign.

The decisive diagnostic rules out the obvious excuse: the module arm DOES improve its own
objective (0.871 on the decision G) and improves it MORE on a G estimated from unseen text
(0.807; absolute 646.5 -> 576.5). So the wider objective is genuinely reduced, out of sample,
by these code changes -- and end-to-end NLL still gets worse. This is not G-overfitting and
not probe noise in the estimate; it is the objective itself failing to correspond to model
quality at this scale. Same failure family as the five proxy misrankings this program has
logged, now hitting our own headline mechanism.

Mechanically nothing differs: bpw exact, flips 2.0% (as on 0.5B), guards fine (G for the
8960-wide MLP = 321 MiB < 512 limit), pipeline deterministic. No bug found; treating it as a
real negative. Draws 1-2 are running to establish whether draw 0 is representative. No
tuning, no hyperparameter change: per the pre-registered rule, a fix (e.g. width-scaled probe
budget) would require fresh pre-registration and validation on a model not used to design it.

### E38b — P2 interrupted by external VRAM exhaustion (2026-09-04)

Draw 1's module arm died with CUDA OOM on the shared 5090: two peer-session processes held
12.1 + 12.1 GiB while ours used 7.6, filling 31 of 31.4 GiB. Draw 0 is complete (4 arms),
draw 1 has gptq+layer only, draw 2 never started. No data corrupted (jsonl is append-only,
per-arm). Not restarting blind: waiting on the peer to either cap its GPU footprint or give
a finish time. Note for the record: the interruption is an infrastructure event, not a
result -- draw 0's negative finding stands on its own and is not affected.

### E38c — Planning note: the dense-G module method has a width wall at 8B (2026-09-04)

Noticed while discussing memory with the peer session. G is (out x out) dense, so it grows
quadratically in width while our probe budget is a constant:

    model        MLP out    G size (fp32)   rank/dim at 2048 samples
    0.5B          4864       95 MiB          0.42
    1.5B          8960      321 MiB          0.23
    8B (Qwen3)   12288      604 MiB          0.17

Two consequences for P5. (1) The 512 MiB per-matrix guard would reject the MLP matrices at
8B outright -- the arm cannot run as frozen. (2) Even ignoring memory, the rank/dim ratio
keeps falling, which is exactly the H1 mechanism under test in D1. So if D1 confirms H1, the
honest conclusion is not just "scale the budget": at 8B a dense G is the wrong object and the
method would need a structured metric (Kronecker factorisation as in YAQA/BaKron, or
per-head block-diagonal), which is a different method requiring its own validation. This
belongs in the P5 decision, not in a quick patch.

### E38d — Hypothesis: G's estimability is determined by the endpoint's algebra (2026-09-04)

Prompted by the peer session's remark that block-diagonal-per-head is the one structured
simplification that respects the actual computation. Working out where heads/channels
actually mix gives an exact structure for G at each endpoint we have already tested:

    endpoint                   true structure of G        samples/dim at 2048 (0.5B / 1.5B)
    attention output, PRE-o    block-diagonal per head    32 / 16     (blocks = head_dim)
    attention output, POST-o   dense (o_proj mixes)       2.3 / 1.33
    MLP intermediate, PRE-down diagonal (elementwise GLU) n/a -- diagonal G is inert
    MLP output, POST-down      dense (down_proj mixes)    0.42 / 0.23

Y_h = P_h V_h depends on q_h, k_h, v_h of that head only, so for the pre-o endpoint the
block-diagonal form is EXACT, not an approximation -- and estimating 64- or 128-wide blocks
from the same 2048 samples is an order of magnitude better conditioned than estimating an
896- or 1536-wide dense matrix.

This lines up with every intra-attention result we have: pre-o was our best attention arm
(+3.8% of the gap), post-o was negative (-1.8%) at identical refined matrices -- the
unexplained "o_proj sign flip" of E32d. The hypothesis is now that the flip is an
ESTIMATION-QUALITY artefact (dense, rank-starved G) rather than a functional one, and
likewise that the MLP side carries the real cross-output structure only because down_proj is
the only place channels mix at all.

Testable, and it makes the structured-G route for 8B concrete rather than speculative:
per-head blocks for the attention side (exact + cheap + well-conditioned), dense or
Kronecker only for the post-down MLP endpoint. NOT acting on this now -- D1 is running and
this is a post-hoc hypothesis that would need its own pre-registration.

## E39 — D2: the structural fix does not rescue 1.5B (2026-09-04)

Exact per-head block-diagonal G at the pre-o endpoint, everything else frozen, 1.5B seed 0:

    arm         NLL wt2   gap      NLL c4    gap      KL
    gptq        2.55340   --       3.05028   --       0.35106
    layer       2.55930   -1.8%    3.06778   -5.7%    0.35868
    module      2.56162   -2.5%    3.06280   -4.1%    0.35835
    module_bd   2.55952   -1.9%    3.06488   -4.8%    0.35413

module_bd beats module on wt2 (+0.6pp) and on KL, loses slightly on C4 -- a wash, and still
below plain GPTQ everywhere. Per the pre-registered D2 table this reads "still negative:
endpoint algebra is not the explanation". Removing 18% of provably-noise energy from the
attention metric moved end-to-end by well under a percentage point, which is itself a
result: the attention side's conditioning was not what broke 1.5B.

Caveat recorded rather than hidden: obj_ratio/obj_fresh came back NaN for this arm, so the
objective bookkeeping is unverified (the end-to-end number stands -- fp16 baseline reproduced
bit-exactly, flips 2.05% i.e. normal, weights evaluated). Reproduced the same fp16 GPU path
locally on 0.5B for both scopes with clean finite ratios (0.888 / 0.907), so the NaN is
specific to the 28-layer 1.5B run and remains unexplained; the 0.5B module_bd run now in
flight will show whether it recurs.

Dropped from the queue as "autopsy of a failure": P2 draws 1-2, and module_pre (which would
only decompose a null into two nulls).

## E40 — Mechanism found: the wide-matrix metric is mostly noise (2026-09-04)

A 20-minute single-layer probe replaced the planned 4-hour D3 run and answered the question
directly. Refine under the frozen-budget G (2048 samples), then score the SAME weight change
under a G estimated with 8x the probes. Qwen2.5-1.5B, layer 13:

    matrix     out    samp/dim   optimizer believed   scored under 8x-G
    gate_proj  8960     0.23        0.615               1.065   (inverted: worse)
    up_proj    8960     0.23        0.636               1.008   (nothing)
    q_proj     1536     1.33        0.648               0.667   (real)
    k_proj      256     8.0         0.356               0.415   (real)
    v_proj      256     8.0         0.808               0.986   (mostly illusory)

The discrepancy tracks matrix width. On the MLP matrices -- which hold most of the weights --
the entire measured improvement is an artefact of the cheap estimate; under a better metric
the move is slightly harmful. H1 is confirmed and, more usefully, LOCALISED: it is the wide
matrices, not the method.

This retro-explains the whole sequence: 0.5B worked (MLP 4864, 0.42 samp/dim); 1.5B failed
(8960, 0.23); the block-diagonal fix did not rescue it because attention was never the
problem (q_proj 0.648 -> 0.667 is honest); and the fresh-G check missed it because it used
the SAME budget and therefore shared the same blind spots -- two independent draws agree on
their dominant directions and are wrong about the tail in the same way. Recording that as a
methodological lesson: an out-of-sample check at the same estimator resolution validates
sampling, not resolution.

Now running (30 min, both models): score the same move under budgets 4k/8k/16k/32k to find
where the inversion disappears, on 1.5B and on the 0.5B control. That measures the rule
instead of guessing it -- if 0.42 samples/dim is sufficient, the rule is N ~ 0.4 * out and
1.5B needs ~3.8k probes, not the 16k a blind fix would have used.

### E40b — D2 closed: the block-diagonal variant regresses where the method works (2026-09-04)

0.5B seed 0, no-regression check: module_bd closes 6.40% (wt2) / 1.44% (C4) against module's
8.66% / 7.24%; KL also worse (0.3464 vs 0.3347). Combined with the 1.5B result (-1.9%, still
below GPTQ), module_bd fails on both models. Pre-registered reading: no claim, variant
dropped.

Where the reasoning went wrong, recorded because it is instructive: the measurement was right
(off-block energy at the pre-o endpoint IS provably noise -- 18% of q_proj's estimate), but
module_bd bundled a second change, moving the attention endpoint from post-o to pre-o. That
breaks ENDPOINT CONSISTENCY between the two halves of the block: attention then aims at a
different target than the MLP. E32e already showed the effect is superadditive and dies under
inconsistent coverage (parts ~ 0, whole 10.4%), so the denoising gain was smaller than the
cooperation loss. The standalone ladder result (pre-o +3.8% vs post-o -1.8%) does not carry
over to the full-module setting for exactly this reason.

What survives: E40's probe is an independent measurement and stands. The fix indicated is
budget, not structure -- keep the original module scope with both endpoints consistent, and
scale the probe count with matrix width. That is the next and only test.

## E41 — P3: the method works on a second family. Two of three models positive (2026-09-04)

Llama-3.2-1B, frozen protocol, seed 0, gap closed vs GPTQ:

    model            fp16    GPTQ   module | wt2      c4       KL
    Qwen2.5-0.5B    13.07   18.57   18.01  | +8.7%   +7.2%   +8.0%
    Llama-3.2-1B     9.75   15.89   15.54  | +4.6%   +3.4%   +2.4%
    Qwen2.5-1.5B     9.26   12.85   12.96  | -2.5%   -4.1%   -2.1%

All three metrics agree in sign within each model. The 0.5B result is therefore not a
small-model artefact: it replicates on a different family, a different tokenizer, half the
depth and 2.2x the width, with no tuning of any kind.

What this kills: the failure is NOT explained by size (0.5B and 1B work, 1.5B does not),
family (Llama works), width (Llama's MLP is 8192 vs Qwen-1.5B's 8960 -- nearly identical),
or G conditioning (0.25 vs 0.23 samples per dimension -- also nearly identical, and the
sweep control already showed the same illusory-improvement pattern on the model where the
method works). Every mechanical explanation offered so far is refuted by this pair.

What is left as candidate distinguishing features of the one failure, unranked and untested:
depth (16 and 24 layers work, 28 fails) and head_dim (64 and 64 work, 128 fails). With n=3
these are hypotheses, not findings, and I am not going to fit a story to one negative.

Running: Llama-3.2-1B calibration draws 1 and 2 (gptq + module only -- layer and block are
closed, mildly harmful on all three models). If the effect holds, the position becomes two
model families x three draws positive with one unexplained negative, which is an honest and
reportable result.

## E42 — P3 verdict: conditional PASS on a second model family (2026-09-05)

Llama-3.2-1B, three independent calibration draws, frozen protocol, paired bootstrap over
evaluation windows (module - gptq, 10k resamples):

    draw   wikitext-2                     C4
    0      -0.02256 [-0.02731, -0.01773]  -0.02456 [-0.03090, -0.01792]
    1      -0.01543 [-0.02028, -0.01046]  +0.00362 [-0.00169, +0.00930]
    2      -0.02077 [-0.02564, -0.01597]  -0.08714 [-0.09164, -0.08251]

wikitext-2: improves in 3/3 draws, every interval strictly negative (+3.1 to +4.6% of the
quantization gap). C4: improves in 2/3, one draw neutral (interval straddles zero). KL:
2/3. Pre-registered reading: CONDITIONAL PASS.

Caveat that must travel with these numbers: on Llama the GPTQ baseline itself moves 0.064
NLL across calibration draws on C4, roughly 2.5x the size of our effect, so C4 on this model
is a noisy measurement and single draws must not be read individually. The within-draw
paired bootstrap is the only sound comparison, which is why it was pre-registered.

Program status: 2 of 3 models positive (Qwen2.5-0.5B clear pass 3/3 draws both corpora;
Llama-3.2-1B conditional pass; Qwen2.5-1.5B negative and unexplained). Cost per model at
3.25 bpw: ~2.7 h optimizer on a 5090, 8.5 GiB peak, 2.2% of codes moved, bit-identical
storage. The GPU box was released to the peer session.

### E42b — RMSNorm-extremity hypothesis tested and refuted (2026-09-05)

A neighbouring embedding-quantisation project hit a similar shape on its own models and
proposed that models with more extreme RMSNorm scales break
compensation/rotation methods -- testable from the weights alone, no run required. Median
over layers of max/median |RMSNorm weight|:

                        input_ln   post_attn_ln   model.norm
    Qwen2.5-0.5B (+8.7%)   4.51        2.17          2.45
    Llama-3.2-1B (+4.6%)   2.27        1.30          1.18
    Qwen2.5-1.5B (-2.5%)   3.65        1.78          2.61

Refuted, and in the wrong direction: the model with the MOST extreme scales is the one where
the method works best, and there is no monotone relationship. There is also a structural
reason it should not apply here -- we never fuse norm scales into the weights (their
measurement is about that fusion, i.e. a pipeline choice, not a model property).

Useful by-product: our two projects share the same confound. Every large model either of us
has tested is a Qwen, and every non-Qwen point is small. "Bigger models" and "Qwen models"
are therefore not separable from the combined evidence. The cheapest decisive experiment for
both projects is the same: a LARGER NON-QWEN model (for us, Llama-3.2-3B: 28 layers, hidden
3072). Positive there => the failure is Qwen-specific; negative => it is a scaling
mechanism. Proposed to the user; not started, and the peer's run has the box.

### E43 — Pre-registered prediction for Qwen3-0.6B (written before the run, 2026-09-05)

Next test, on the local card once the peer's sweep finishes: **stock Qwen3-0.6B**, frozen
protocol, gptq + module, 2 calibration draws. Chosen because it isolates the two remaining
architectural candidates from size: 28 layers and head_dim 128 (matching the failing
Qwen2.5-1.5B on both) at 0.6B (matching the working models on size). It also adds QK-norm,
which Qwen2.5 does not have -- a confound to state now rather than discover later.

Not using microsoft/harrier-oss-v1-0.6b despite it being the peer's exact checkpoint: its
architecture is `Qwen3Model`, a bare encoder with no LM head, so NLL/KL cannot be computed
and the frozen protocol does not apply. An identical-checkpoint cross-check is therefore not
available to us; stock Qwen3-0.6B answers the architectural question instead.

**Prediction, logged before looking (peer's suggestion, and a good one): I expect it to
WORK, ~60/40.** Reasoning: both working models are small and the single failure is the
largest model; and the peer's own data already shows that shape working on their side, so it is not
universally fatal. Against my prediction: it is the only test
where both architectural axes match the failure, and QK-norm is a genuine unknown.

If it works -> the architectural axes are dead and the axis is size or something tracking it,
which implies a ceiling near 1B and is the commercially worse outcome. If it fails -> a
within-method architectural predictor exists, and the failure is not about size.

### E43b — Peer correction and a cross-project pattern that does NOT hold for us (2026-09-05)

Two things from the peer session worth recording.

**Correction to a number we had cited.** Their earlier ~1% figure for the runtime cost of the
online rotation was an operation count quoted as a time. Measured properly (hand-written AVX2
kernel running at the vector-add throughput limit): **~10% latency overhead**, linear in the
number of rotation rounds. BRIEFING.md corrected; we had propagated their estimate into a
document meant for outside readers, which is exactly the sort of borrowed number that needs a
source and a date attached.

**Their pattern, and why it does not transfer.** Across their settings the rotation gain
tracks how broken the baseline is (their numbers, not reproduced here). Tempting to read as
a law. It does not hold for our module-scope gain, sorted by damage:

    Qwen2.5-1.5B   gap 0.327 NLL   -2.5%
    Qwen2.5-0.5B   gap 0.351 NLL   +8.7%
    Llama-3.2-1B   gap 0.488 NLL   +4.6%

The most damaged model is not the biggest winner and the least damaged is not the loser;
there is no monotone relationship. Noted so we do not import a neighbouring project's
regularity as if it were ours. Their result on a non-Qwen encoder does however settle for
their branch that the effect is not a Qwen artifact -- which is the question our
Qwen3-0.6B run is asking on our side, for our method.

## E44 — Qwen3-0.6B: the architectural hypothesis is dead (2026-09-05)

Prediction logged in E43 before the run: "expect it to WORK, ~60/40". Result on wikitext-2,
draw 0 (the module arm then crashed on CUBLAS during C4 eval -- tight 6 GB card, and the
local launch was missing the expandable_segments setting the box runs use; relaunched):

    fp16 3.04236 | gptq 3.55765 | module -0.04215 NLL [-0.05041, -0.03381]
    = 8.2% of the quantization gap closed, interval strictly negative.

Qwen3-0.6B has 28 layers and head_dim 128 -- both features it shares with the failing
Qwen2.5-1.5B -- and it is our second-best result. **Both remaining architectural candidates
are therefore dead**, and with them the last mechanical hypothesis on the list. It also adds
QK-norm, so that is not protective or harmful in any obvious way either.

Running tally of what does NOT explain the Qwen2.5-1.5B failure: size, family, matrix width,
metric conditioning (samples per output dimension), RMSNorm scale extremity, depth, head_dim,
depth-to-width ratio, and -- per the peer's finding on their side -- baseline brokenness.

Two corrections made while checking this:

1. **Verdict-logic bug in scripts/analyze_valid.py.** `neg >= n - 1` classified 0 improving
   draws out of 1 as "conditional PASS". Fixed to `neg >= max(1, n - 1)`; Qwen2.5-1.5B now
   correctly reads "NEEDS MORE DRAWS". No other verdict changed.
2. **Our headline negative rests on n=1 calibration draw.** The 1.5B module arm exists for
   draw 0 only (draw 1 died in the shared-GPU OOM, draw 2 never ran). Its bootstrap interval
   is tight -- +0.00822 [+0.00375, +0.01266] on wt2, +0.01252 [+0.01028, +0.01473] on C4 --
   but that is *evaluation* noise. Calibration-draw noise is a separate and, on Llama,
   larger term: the C4 effect there ranged +3.4%, -0.5%, +11.9% across draws. So the honest
   status of the failure is: significant against evaluation noise, untested against
   calibration noise, n=1. A second draw is now worth running -- not to re-examine a settled
   negative, but because we have since learned that the dominant variance term is untested.

## E45 — The Qwen2.5-1.5B failure replicates on a second calibration draw (2026-09-05)

The decisive run. Draw 1, frozen protocol, module vs gptq, paired bootstrap over windows:

    draw   wikitext-2                      C4
    0      +0.00822 [+0.00375, +0.01266]   +0.01252 [+0.01028, +0.01473]
    1      +0.00333 [-0.00089, +0.00763]   +0.01473 [+0.01240, +0.01703]

(positive = worse). Both draws negative on both corpora and on KL; C4 intervals strictly
positive in both. wt2 draw 1 straddles zero, so the honest reading is: **the failure
replicates, clearly on C4 and directionally on wikitext-2.** It is not a single-draw
artefact, which is what the peer session's analogous "anomaly" turned out to be on their
side. Ours is real.

So the program's position is now four models, two of which the method helps, one it does not
touch measurably, and one it actively harms -- with no property that separates them:

    Qwen2.5-0.5B   24L  head_dim 64   +8.7%  (3/3 draws)
    Qwen3-0.6B     28L  head_dim 128  +8.2%  (draw 0; draw 1 running)
    Llama-3.2-1B   16L  head_dim 64   +4.6%  (3/3 wt2, 2/3 C4)
    Qwen2.5-1.5B   28L  head_dim 128  -2.5%  (2/2 draws negative)

Qwen3-0.6B and Qwen2.5-1.5B share both architectural features and land at opposite ends, so
the architecture story is dead twice over. Ruled out to date: size, family, width, samples
per output dimension, RMSNorm extremity, depth, head_dim, depth/width ratio, baseline
damage, and single-draw noise.

## E46 — Qwen3-0.6B, both draws: strong on wikitext-2, split on C4 (2026-09-05)

    draw   wikitext-2                       C4
    0      -0.04215 [-0.05041, -0.03381]    -0.01272 [-0.01663, -0.00879]
    1      -0.03379 [-0.04142, -0.02626]    +0.02801 [+0.02386, +0.03216]

wikitext-2: 2/2 draws improve, both intervals strictly negative, 8.2% and 6.6% of the gap.
C4: one clear improvement, one clear degradation -- and both intervals are tight, so this is
not evaluation noise, it is genuine calibration-draw sensitivity. Same pattern as Llama-3.2-1B
(3/3 on wt2, one neutral draw on C4) but more pronounced.

The honest reading of the C4 column across the whole program: calibration is drawn from
wikitext, so wikitext-2 is in-domain and C4 is out-of-domain, and the out-of-domain transfer
of the post-pass is **draw-dependent in a way the in-domain result is not**. Two models now
show it. That belongs in the limits, not the headline: the method's in-domain gain is solid
and its out-of-domain gain is a coin-flip on which calibration sample you drew.

Model tally on wikitext-2 (the metric with consistent behaviour):
    Qwen2.5-0.5B  +8.7%  3/3 draws
    Qwen3-0.6B    +8.2%  2/2 draws
    Llama-3.2-1B  +4.6%  3/3 draws
    Qwen2.5-1.5B  -2.5%  0/2 draws

## E47 — Sharpened variance test costs us two claims (2026-09-06)

A peer session's reviewer sharpened a test we had both been getting half-right: an effect
must be compared not to the spread of *scores* across calibration draws, but to the spread of
the **paired method-minus-control difference** across draws. Our primary analysis was already
paired within draw, so the method is sound -- but we had never computed the second-order
quantity, and it changes what we may claim. Effect vs spread-of-paired-difference:

    model            eval   effect     spread    ratio
    Qwen2.5-0.5B     wt2    -0.0274    0.0108    2.5x
    Qwen3-0.6B       wt2    -0.0380    0.0084    4.5x
    Llama-3.2-1B     wt2    -0.0196    0.0071    2.7x
    Qwen2.5-1.5B     wt2    +0.0058    0.0049    1.2x   <- weak
    Qwen2.5-0.5B     c4     -0.0272    0.0143    1.9x
    Qwen2.5-1.5B     c4     +0.0136    0.0022    6.2x
    Qwen3-0.6B       c4     +0.0077    0.0407    0.2x   <- meaningless
    Llama-3.2-1B     c4     -0.0360    0.0908    0.4x   <- meaningless

Two claims have to be weakened.

1. **The C4 column is not reportable on two of four models.** On Qwen3-0.6B and Llama-3.2-1B
   the across-draw spread of the paired difference exceeds the mean effect. Those C4 numbers
   say nothing, and the headline table has been presenting them as if they did. Only the
   in-domain (wikitext-2) result is stable, which makes sense: calibration is drawn from
   wikitext, so C4 measures out-of-domain transfer, and that transfer depends on which
   calibration sample was drawn.
2. **The 1.5B failure is supported by C4 and only weakly by wikitext-2** (1.2x its own draw
   spread, n=2). It is the mirror image of the positive results, which are strong in-domain
   and unstable out-of-domain. Honest statement: the failure is real on the out-of-domain
   metric and directional in-domain -- not the flat "it fails" we have been writing.

Caveat on the caveat: n is 2-3 draws, so these spread estimates are themselves noisy and
should be read as order-of-magnitude.

Also checked, since the same review found a calibration/eval leak on their side: our
calibration is wikitext-2 **train** and evaluation is wikitext-2 **test** plus C4
**validation** -- disjoint by construction, no document-level overlap possible. Recorded as
verified rather than assumed.

## E48 — Rate–distortion curve: the effect converts to ~0.08 equivalent bits at best (2026-09-06)

Qwen2.5-0.5B, frozen protocol, one draw per rate, wikitext-2:

    bpw    gptq gap   module gap   gap closed   equivalent bits saved
    2.50    6.47051     6.56426       -1.4%     (model destroyed; module harmful)
    3.25    0.35096     0.32057       +8.7%     +0.084
    3.50    0.26104     0.25562       +2.1%     +0.015
    4.25    0.06736     0.06138       +8.9%     +0.023

Local R-D slope: 0.360 NLL/bpw between 3.25 and 3.50, 0.258 between 3.50 and 4.25.

**The commercially meaningful number is ~0.08 equivalent bits at best, and 0.015-0.023 at
the other usable rates.** The pre-registered product threshold for this branch was
0.1-0.2 bpw. We are below it, at every rate measured, on the model where the method performs
best. That is the honest headline for anyone assessing this as a deployable technique rather
than as a scientific result.

Two further observations. The percentage-of-gap-closed figure is *not* monotone in rate
(8.7 / 2.1 / 8.9), which means single-rate quotes of "closes X% of the gap" are unstable and
should not be used without the rate attached -- our own headline number is the best of three.
And the equivalent-bits figure is small precisely because the R-D curve is steep: a quarter
of a bit buys 0.09 NLL, while the whole post-pass buys 0.03. Recovering a fixed fraction of
quantization damage is worth little when damage falls fast with rate.

This does not weaken the scientific claims (the objective-geometry results stand on their own
measurements) but it does settle the product question in the negative at these rates, and
that belongs in the pitch rather than in a footnote.

## E47 — Instance shutdown before final pull (2026-09-07)

The rented instance was shut down after the R-D sweep and the Llama-3.2-3B run were launched.
Results had been pulled once (after P6 point 2.50 bpw), not after each stage. Lost with the
instance: per-window NLLs for the 3.50 and 4.25 bpw R-D points (aggregate NLLs preserved in
`evidence/lost_to_shutdown/monitor_rd_and_llama3b.log`), and the Llama-3.2-3B module arm
(its GPTQ baseline wt2 2.36377 / fp16 2.05586 was captured before the shutdown). Nothing
downstream of these was claimed beyond the aggregates. Lesson: on ephemeral infrastructure,
pull after every stage, not once at the end. A brief power-on is planned solely to check
whether `/root/lwc/results/rd_q05_b3g64.*`, `rd_q05_b4g128.*`, `p6_llama3b_s0.*` survived
on disk; no computation will be run.

**Outcome of the power-on (same day):** the disk survived. All three files pulled and
committed to `evidence/validation/`. The 3.50 and 4.25 bpw points now have full per-window
data and are bootstrapped (`scripts/analyze_rd.py`). The Llama-3.2-3B module arm had reached
layer 8/27 after 8053 s of optimisation when the instance went down, so it could not have
finished in the time available and is not recoverable without a ~9 h re-run; only its fp16
and GPTQ references exist. The earlier `evidence/` copy of the 2.50 bpw file turned out to
be **truncated** (2 of 8 records, fp16 only) — pulled mid-run — so that row, previously
described as "full data", was not; it is now complete. Every other file on the instance
matched its `evidence/` copy by md5, including the eight launch scripts. Nothing of ours was
running; the instance was shut down again immediately after the pull.

## E48 — External review: prior-art correction and addendum D3 (2026-09-07)

An external review of `PITCH.md` / `REVIEWER.md` named BRECQ (2021), MREM (2022), YAQA
(2025) and BaKron (Aug 2026, arXiv 2608.06291) as direct prior art for the "non-monotone
objective horizon" story. All four read from the primary PDFs; `docs/PRIOR_ART_LADDER.md`
holds the module-by-module comparison. Verified: BaKron App. A.1 derives the gated-MLP
output Hessian `E[xxᵀ ⊗ (ggᵀ ⊙ DᵀD)]`, which is our `module` metric for gate/up in closed
form; BaKron App. D compares GPTQ / MLP-local / fully-local / backprop-Fisher at one solver
on Llama-3 and Qwen3 1B–8B, and the ordering flips by model (global worse than layer-local
on Llama-3.2-3B with 524k calibration tokens) and by factorization (K-FAC vs Shampoo:
60.97 vs 20.26 ppl on Qwen3-1.7B). BRECQ Table 1 has net-wise far below layer-wise with
the bias–variance explanation; MREM Fig. 5(b) has layer-wise ahead below 128 calibration
samples and module-wise ahead above.

Consequences applied: "nobody sweeps the ladder / nobody has stated / that framing is
ours" removed from PITCH; positioning in BRIEFING §4, FINDINGS §5b and REVIEWER
rewritten; perplexity-change column and preparation cost (20–30× GPTQ time, zero
inference overhead) added; "C4 systematically pessimistic" withdrawn (it rested on a
retrieval-embedding result); certificates for the rotation plateau, rank-64 and
diagonal-G negatives scoped to the settings they were run in (GuidedQuant's per-sample
weighting is not covered by the diagonal-G control); "structural rather than fixable"
replaced by "unfavourable conversion at today's effect size, ceiling unmeasured".

Also found while checking the review: our 8× estimation-budget run raised the metric's
token subsample (2048 → 16384) *and* the probe count (2 → 4) together, so it never
separated probe variance from text coverage — the exact split the review asked for. And
the exploratory `joint_ppl` records carry no per-window NLLs, so the attention/MLP
interaction (−1.8 / +1.7 / +10.4%) has no interval.

Addendum D3 pre-registered in `FROZEN_PROTOCOL.md` with interpretation tables: D3a the
four-arm interaction with intervals (`module_attn`, `module_mlp` under the full
evaluation); D3b the full-model horizon at three budgets — (2 probes, 2k tokens), (8
probes, 2k), (2 probes, 8k) — with the arm's own KL-to-fp16 logged next to NLL; D3c
`--calib-source c4` for `gptq` and `module`. Eval-side changes only: three diagnostic
arms, a calibration-source switch whose default path is byte-identical (checked), and
`calib_source` / `n_probe` / `g_tokens` recorded per row. Queue: `scripts/local_d3.sh`,
Qwen2.5-0.5B seed 0 on the shared local GPU when the peer's job releases it; the failing
model (1.5B) waits for a ≥12 GB GPU.

## E49 — Addendum D3a and D3c results (2026-09-08, local RTX 4050, Qwen2.5-0.5B, draw 0)

Run under the frozen evaluation (`valid_eval`, full wt2 test, 512 C4 windows, MC-KL); the
gptq/module rows are the frozen draw-0 validation rows; fp16 rows reproduced to 1e-6.
Read-out: `python scripts/analyze_d3.py`. Files: `evidence/validation/d3a_*`, `d3c_*`.

**D3a — four-arm interaction, now with intervals.** wt2: attention half +0.6% of the gap
(−0.002 [−0.007, +0.002], not distinguishable from 0), MLP half +2.4% (−0.008 [−0.013,
−0.003]), whole +8.7% (−0.030 [−0.035, −0.026]); interaction (whole − sum of halves)
−0.020 [−0.026, −0.014], strictly negative → **superadditive, pre-registered reading
"claim stands"**. On C4 the attention half is harmful (+0.006 [+0.003, +0.008]), the MLP
half is zero, the whole is +7.2%, interaction −0.031 [−0.035, −0.028]. The exploratory
24-window numbers (−1.8 / +1.7 / +10.4%) were directionally right and over-stated the
halves' spread; the honest statement is "each half alone ≤2.4%, together 8.7%".
Halves: module_attn opt 984 s, module_mlp 1971 s (vs 3346 s for the whole), so the
whole is not merely the sum of the halves' compute either.

**D3c — calibration on C4 train instead of wikitext-2 train (same seed).** Both arms lose
≈0.1 NLL on wt2 (gptq +0.099, module +0.094; their difference-of-differences −0.006
[−0.013, +0.001] covers 0). On the C4 evaluation, C4 calibration improves gptq by 0.006
[0.003, 0.009] and module by 0.011 [0.008, 0.014]; the post-pass's advantage grows by
0.005 [0.002, 0.009] → **pre-registered reading: "the post-pass gains more than GPTQ from
matched domain"**, gap closed on C4 7.2% → 8.9%. The effect of matched domain is small
for both arms on this model (0.006 NLL for GPTQ), so the earlier "C4 numbers are
systematically pessimistic" line would have been right in sign and wrong in size; the
corrected wording ("untested at matched domain") is now replaced by the measurement.
KL to fp16 (C4-calibrated): gptq 0.401, module 0.371.

D3b (full-model horizon at three budgets) is queued; the GPU was yielded to the peer at
08:26 with stage 1 restarted from scratch later (30 min lost).

## E50–E51 — External application on an encoder (2026-09-08; aggregate only)

The post-pass (`vq_refine`, through `scripts/vq_external_refine.py`, extended with per-block
codebook slices for the peer's container) was applied by the neighbouring
embedding-quantisation project to its own per-block VQ query encoders at 1.8 bpw, first
post-hoc on a frozen capture and then inside its sequential quantisation loop, with a
proper calibration-draw control. Aggregate result, by agreement with that project:
in a neighbouring encoder-quantisation project the post-pass gave no effect resolvable above calibration-draw variance at 1.8 bpw. Two methodological points from that series are recorded because they bear on our
claims: (1) a query-paired confidence interval does not contain calibration-draw variance —
two equally good quantisations of the same model differed by more than the interval
"confirmed" — which is why we report every draw and require a consistent sign; (2) our
docs had described the post-pass as composing with an existing checkpoint, whereas the
validated form is *sequential* (downstream blocks are rounded against the refined
upstream); the post-hoc-on-frozen-capture variant was never validated by us and is the
one that project tried first. Corrected in PITCH, REVIEWER and PRIOR_ART_LADDER. The
per-arm numbers of that series belong to that project and are not reproduced here.

## E52 — Addendum D3b: the full-model horizon is text-limited, not probe-limited (2026-09-08)

Qwen2.5-0.5B, draw 0, 3.25 bpw, frozen evaluation, arm `model` (full-model Fisher, K-FAC
form, damp 1.0, cd_pre 6 + cd 6) at three estimation budgets that differ only in the
metric estimator; local RTX 4050. Files `evidence/validation/d3b_*`; read-out
`scripts/analyze_d3.py`.

| probes | G tokens | gap closed wt2 | model − gptq [CI] | KL fp16 | obj_ratio | scope s | opt s |
|---|---|---|---|---|---|---|---|
| 2 | 2048 (standard) | +0.3% | −0.001 [−0.005, +0.003] | 0.359 | 0.855 | 215 | 7089 |
| 8 | 2048 (4× probes) | −1.1% | +0.004 [−0.001, +0.009] | 0.365 | 0.854 | 1327 | 10234 |
| 2 | 8192 (4× text) | **+5.5%** | **−0.019 [−0.024, −0.015]** | 0.349 | 0.880 | 215 | 5329 |
| reference: gptq KL 0.364; module arm +8.7%, KL 0.335 | | | | | | | |

Pre-registered reading (FROZEN_PROTOCOL D3b, row "(ii) ≈ same, (iii) improves"): **token
coverage is the operative error; "noise-limited" should read "text-limited".** The
exploratory 8× result (E34: 0.2% → 5.4%, which raised tokens 8× *and* probes 2×) is
reproduced by the token increase alone (+5.5% at 4× tokens) and not at all by the probe
increase. The KL to fp16 — the quantity the arm's objective approximates — moves with the
NLL (0.359 → 0.349), so the objective/metric mismatch raised in review is not the
explanation of the global arm's poor rank; under-sampled calibration text in the metric
is. Consistent with MREM's finding that wider objectives need more calibration data and
with BRECQ's over-fitting reading; the "estimation-budget frontier" we proposed is
therefore a *data*-budget frontier, which is prior art in kind, with our contribution
being the controlled measurement (probes vs tokens separated at one solver/rate).

Two secondary observations. (1) The better-estimated objective is *harder* to improve
(obj_ratio 0.880 at 8k tokens vs 0.855 at 2k): the cheap estimate was being over-fitted,
which is the rank-64 failure family in mild form. (2) Cost: 4× probes costs 6× metric time
and buys nothing; 4× tokens costs no extra metric time here (the tokens are a subsample of
the already-propagated calibration set) and buys 5.5% of the gap — still 60% of the
module arm's 8.7% at the same calibration set, so R\* = module stands at this budget.
Open: does the full-model arm overtake the module arm at 16k+ tokens (the whole
calibration set), and does the same text-limitation govern the module arm? Not run.

## E53 — D3b-ext: the attention half goes from nothing to +6.6% with 4× metric tokens (2026-09-09)

Pre-registered in FROZEN_PROTOCOL D3b-ext before running. Qwen2.5-0.5B, draw 0, 3.25 bpw,
frozen evaluation, arm `module_attn` (q/k/v under the attention-module G, o identity; no
MLP refinement) with `--g-tokens 8192 --n-probe 2`, otherwise identical to the D3a run at
2048 tokens. Local RTX 4050, 56 min wall (fp16 + gptq eval 12 min, refinement 28 min, eval
16 min; I had promised 35). File `evidence/validation/d3bx_qwen05_attn_t8k_s0.*`.

| metric tokens | wt2 gap closed | arm − gptq wt2 [CI] | C4 gap closed | arm − gptq C4 [CI] | KL fp16 | obj_ratio | obj_fresh |
|---|---|---|---|---|---|---|---|
| 2048 (D3a) | +0.6% | −0.002 [−0.007, +0.002] | −1.6% | +0.006 [+0.003, +0.008] | 0.365 | 0.911 | 0.704 |
| 8192 | **+6.6%** | **−0.023 [−0.028, −0.019]** | **+4.8%** | **−0.017 [−0.019, −0.015]** | 0.341 | 0.915 | 0.697 |
| 8k − 2k, paired | | −0.021 [−0.025, −0.017] | | −0.023 [−0.025, −0.021] | | | |

Reference: the whole module arm at 2k tokens closes +8.7% (wt2) / +7.2% (C4).

Pre-registered reading (D3b-ext, first row): **the text-limitation found for the full-model
metric extends to the intra-module metric; the module arm itself must be re-run at 8k
tokens.** The attention half alone, given 4× the calibration tokens in its metric, reaches
three quarters of the whole 2k-token module effect, and turns from harmful to clearly
positive on C4. The objective ratios barely move (0.911 → 0.915 on the decision G, 0.704
→ 0.697 on fresh G): the better-estimated metric is not "easier to improve", it points the
same amount of improvement in directions that transfer. This is the end-to-end
confirmation of E40's rescoring probe (the cheap estimate's gains on wide matrices were
partly illusory), now on the model where the method works and on the half where D3a
found nothing.

Consequences, stated carefully. (1) The frozen constant `g_tokens 2048` under-serves the
metric even on the 0.5B model; the validated +8.7% is a *lower* bound of what the objective
delivers with a properly estimated G. (2) This is one draw, one model, one half; it is a
diagnostic and changes no headline number. (3) The D1 rule applies: if the whole module
arm at 8k tokens beats 8.7% on 0.5B draw 0, `g_tokens` becomes a candidate re-freeze whose
value is taken from the 0.5B regime and must then be validated unchanged on Qwen3-0.6B,
Llama-3.2-1B and — the interesting one — Qwen2.5-1.5B, where E40 measured the widest
matrices at 0.23 samples per output dimension. (4) Cost is nil: the 8k tokens are a
subsample of the calibration set already propagated, scope time 36 s vs 33 s.

## E54 — D3b-ext2: the whole module arm does NOT improve with 4× metric tokens (2026-09-09)

Pre-registered (FROZEN_PROTOCOL D3b-ext2) before running. Qwen2.5-0.5B, draw 0, 3.25 bpw,
frozen evaluation, arm `module` with `--g-tokens 8192 --n-probe 2`, otherwise frozen.
Local RTX 4050, 75 min wall. File `evidence/validation/d3bx2_qwen05_module_t8k_s0.*`.

| metric tokens | wt2 gap closed | module − gptq wt2 [CI] | C4 gap closed | module − gptq C4 [CI] | KL fp16 | obj_ratio | obj_fresh |
|---|---|---|---|---|---|---|---|
| 2048 (frozen, P1) | +8.7% | −0.030 [−0.035, −0.026] | +7.2% | −0.026 [−0.028, −0.023] | 0.335 | 0.893 | 0.767 |
| 8192 | +7.3% | −0.026 [−0.031, −0.021] | +4.4% | −0.016 [−0.018, −0.013] | 0.342 | 0.907 | 0.743 |
| 8k − 2k, paired | | **+0.005 [+0.001, +0.009]** (worse) | | **+0.010 [+0.008, +0.012]** (worse) | | | |

Pre-registered reading (D3b-ext2, second row): **the frozen budget stands; no re-freeze.**
The result is in fact one step stronger than "does not beat 8.7%": the whole module arm
is *worse* at 8k tokens on both evaluation sets, with CIs excluding zero. Put next to E53
(attention half alone: +0.6% → +6.6% at 8k), the picture is: the attention-half metric is
text-limited, the whole-module refinement is not helped by the same budget, so the extra
tokens change what the MLP half does or how the two halves cooperate (E49: halves ≤2.4%,
whole 8.7%). Which of the two is not identifiable from these runs; the MLP half at 8k
tokens is the one diagnostic that would separate them (not run; ~1 h).

Corrections to the record. (1) E53's sentence "the validated +8.7% is a lower bound" was
an over-reach and is withdrawn here and in PITCH/REVIEWER/BRIEFING/FINDINGS/
PRIOR_ART_LADDER: the budget helps one half and not the whole. (2) "The metric is
text-limited inside the module" is true of the attention endpoint only, as measured. (3)
The frozen `g_tokens 2048` is not shown to be under-serving the validated arm; the
validated table is unchanged and no re-validation on other models is triggered.
Secondary: at 8k the arm improves its own objective less (0.907 vs 0.893) and its fresh-G
objective less (0.743 vs 0.767), i.e. the better-estimated metric is not over-fitted more;
it simply points the whole-module refinement somewhere slightly worse for the end metric.

## E55 — P4: peer's per-block VQ at 2.12 bpw, with and without our post-pass (2026-09-09)

The "external strong VQ checkpoint + our post-pass" test planned since P0, finally run,
with the neighbouring project's quantiser (per-256-column codebooks, d=4, K=256, frozen
row scales, input-side rotation, GPTQ-style propagation; `scripts/vq_decoder_driver.py` in
their repo) on Qwen/Qwen2.5-0.5B with our frozen calibration windows (wikitext-2 train
32×512, seed 0). Two checkpoints from one quantisation: plain, and plain + our sequential
`vq_refine` inside their loop (all 168 linears, 6 identity sweeps then 6 under the module
G from 2 probes × 8192 tokens, damping 1.0; 9.8% of codes changed, objective I 0.909,
G 0.853, fresh-G 0.862, 0 matrices worse). Blocks 2.1161 bpw (codes 2.000 + scales +
codebooks); embeddings, tied lm_head, norms, biases fp16. Evaluated as dequantised fp16
under the frozen protocol with `scripts/eval_external.py` (fp16 reference reproduced to
1e-6). Files `evidence/validation/p4_qwen05_vq20_{plain,refine}.*`.

| arm | bpw | NLL wt2 | gap wt2 | NLL C4 | gap C4 | KL fp16 |
|---|---|---|---|---|---|---|
| fp16 | 16 | 2.57033 | — | 3.04346 | — | 0 |
| scalar GPTQ (frozen) | 3.25 | 2.92129 | 0.351 | 3.39860 | 0.355 | 0.364 |
| scalar GPTQ (R-D) | 2.50 | 9.04084 | 6.471 | 9.52212 | 6.479 | — |
| **VQ plain** | 2.12 | 3.26245 | 0.692 | 3.99321 | 0.950 | 0.786 |
| **VQ + post-pass** | 2.12 | 3.28906 | 0.719 | 3.92619 | 0.883 | 0.712 |

Paired refine − plain over windows: wt2 **+0.027 [−0.038, +0.098]** (−3.8% of the gap,
n.s.); C4 **−0.067 [−0.088, −0.045]** (+7.1% of the gap); KL −9.5%.

**Pre-registered P4 criterion** ("the module post-pass must add ≥ ~3% of the NLL gap" on
the primary metric, wt2): **not met.** C4 and KL are positive; one draw.

The wt2 interval is four times wider than anything at 3.25 bpw, and the per-window
distribution says why: the post-pass is better on **127 of 146** wt2 windows (median
−0.070 nats, ≈10% of the gap) and on **467 of 512** C4 windows (median −0.092), but three
wt2 windows get catastrophically worse (+1.65, +1.68, +1.73 nats; plain NLL 3.4–3.5 there,
i.e. ordinary text) and two C4 windows swing by ±1.7. Dropping the two largest-|Δ| wt2
windows moves the mean from +0.027 to +0.003. So at 2.1 bpw the refinement improves
typical text by about a tenth of the gap and *breaks the model on a few inputs*; the
mean-NLL criterion, correctly, does not forgive that. This is the 2.50-bpw scalar
inversion (E48) in a milder, input-localised form: the quadratic model around a heavily
quantised network is right on average and wrong in a tail. Reported as: criterion not
met on the primary metric; a robustness failure, not an absence of effect; and the
representation, not the post-pass, is what makes 2 bits usable at all (wt2 gap 0.69 vs
6.47 for scalar at 2.50 bpw). Attribution: peer's quantiser, our post-pass in their loop,
our evaluation.

## E56 — D4: the 2-bit tail is a first-token (attention-sink) failure; with a fixed prefix the post-pass wins on both sets (2026-09-09)

Pre-registered (FROZEN_PROTOCOL D4) before running. Diagnosis first (CPU/GPU, minutes):
the ten worst wt2 windows under the refined 2.12-bpw VQ checkpoint all begin with a
digit-like fragment ("0 ,", "06", "20", "0 @"), the ten best with ordinary continuations
(", the", ". The"); rare-token fraction of the bad windows is ordinary (0.03–0.06 vs
median 0.041, corr with Δ = 0.03); the damage is uniform from position 3 onward (every
256-token segment +1.4 to +2.5 nats, 36% of tokens worse by >2 nats), which for a causal
model locates the cause in the first tokens. On the three worst windows, dropping the
first token, prepending "\n\n", or swapping the first token for a good window's brings the
refined model from 5.20 to 3.33–3.38 nats, below plain (3.47). The refined 2-bit model
fails to form an attention sink on some first tokens; the frozen evaluation (contiguous
windows, no BOS — Qwen has none) exposes it, and nothing in the objective can see it
(fresh-G ≈ decision-G, 0 matrices worse; per type q 0.808 / k 0.804 / v 0.851 / gate
0.903 / up 0.897). Peer's mechanism, recorded: calibration windows start mid-text too, so
the sink-forming rows for digit-first tokens are outside the metric's support and the
refine moves those codes at zero measured cost.

**D4 run**: both checkpoints, every window = "\n\n" (token 271) + first 2047 tokens, NLL
over the positions after the prefix. `evidence/validation/d4_qwen05_vq20_prefix.*`.

| eval | plain | refine | refine − plain [CI] | better | windows Δ>1 | max Δ |
|---|---|---|---|---|---|---|
| wt2, prefixed | 3.20277 | 3.17293 | **−0.030 [−0.034, −0.025]** | 125/146 | 0 | +0.054 |
| C4, prefixed | 3.94439 | 3.91441 | **−0.030 [−0.034, −0.026]** | 391/512 | 0 | +0.333 |
| wt2, frozen (no prefix) | 3.26245 | 3.28906 | +0.027 [−0.038, +0.098] | 127/146 | 10 | +1.73 |

Prefixing also helps the plain checkpoint (−0.060 wt2, −0.049 C4): the sink fragility is
a property of 2-bit VQ on this model that the refinement aggravates, not one it creates.
Relative to the unprefixed fp16 reference the post-pass now closes ≈4.7% (wt2) / 3.3%
(C4) of the plain gap.

**Pre-registered reading, first row: the tail is a first-token/sink effect.** The P4
verdict on the frozen protocol stands as reported (criterion not met on wt2, E55); this
addendum says why and that the effect underneath is a consistent ≈3–5% of the gap on
both sets with a fixed prefix, at 2.12 bpw, one draw. Two things follow. (1) For any 2-bit
deployment of this model, a fixed BOS/prefix is not optional; the protocol without it
measures a real fragility that a product would hit on inputs starting with numbers or
symbols (the peer's encoder clients send exactly such queries). (2) The metric-support
fix — a few calibration windows starting with digit/punctuation tokens in the G batches —
is a hypothesis for the peer's loop, not run.
