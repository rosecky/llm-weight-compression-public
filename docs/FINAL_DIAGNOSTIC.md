# Final diagnostic — why sub-4-bit compression fails

**The one-sentence answer.**

> Our sub-4-bit representations failed **not** because they could not represent weight values
> well enough, but because they put the quantization error in functionally wrong places and in
> a badly conditioned coordinate system: at 3 bits/weight, second-order error compensation is
> worth **+1.14 bits/weight** and incoherence rotation **+0.34**, while switching the
> representation from scalar to vector quantization is worth **+0.45 and falls to −0.08 by
> 4 bits/weight**.

**Verdict: E2 (error allocation dominates), jointly with E3, plus a real but already-exploited
E4 interaction. Recommendation: close this research direction.**

Model: Qwen2.5-0.5B. Hardware: one RTX 4050 Laptop (6 GB), Core Ultra 7 155H, 32 GB RAM.
Stage 1: 52 runs, 559 s, peak VRAM **758 MiB**. Stage 2: 24 runs, 48 min, peak VRAM
**4573 MiB** (dominated by the 151936-vocab logits during perplexity, not by the method).

---

## 1. Experimental matrix

| factor | level 0 | level 1 |
|---|---|---|
| representation | **scalar**: asymmetric min-max, group 128, per-group scale + zero | **VQ**: residual (additive) VQ, d=8 (1×8 row tiles), K∈{256,1024}, 2–4 stages |
| error handling | **naive**: independent quantization | **GPTQ**: second-order compensation against `H = 2·E[x xᵀ]` |
| coordinates | **native** | **Hadamard**: `W' = P W Q`, randomized block-Hadamard + signs + permutation on both axes |

All eight cells are technically meaningful and all eight were run. The *same* engine
(`src/lwc/gptq.py`) drives every cell; `Quantizer.atom` declares how many columns the
representation needs jointly (1 for scalar, 8 for VQ), and error is propagated only to columns
after the atom. Setting `atom=1, group=0` reduces the engine exactly to textbook GPTQ.

**Scalar baseline choice.** Asymmetric min-max, not Lloyd-Max. EXPERIMENT_LOG E11 showed the
earlier Lloyd-Max/RMS setup reconstructed the largest weight in a group with up to 74 % error
and cost 98× perplexity. Min-max spans each group's observed range, so it *cannot* clip.
Group size is fixed at 128 for every scalar cell.

**VQ baseline choice.** The best representation from the earlier study: realistically decodable
(M lookups + M−1 adds per weight), codebook 64–256 KiB = 0.0015–0.006 bits/weight amortised
over the model's 357.8 M target weights, and no permutation or unranking overhead. No new
representation was developed for this experiment.

### Correctness checks made before measuring anything

| check | result |
|---|---|
| rotation round-trip `‖PᵀP W Q Qᵀ − W‖²/‖W‖²` | **5 × 10⁻¹⁴** — exactly function preserving |
| `rotate_hessian` vs an explicitly formed `Qᵀ H Q` | **5.5 × 10⁻¹⁴** |
| GPTQ signature: weight error up, activation error down | 3-bit: NMSE 0.052 → 0.129, act 0.058 → **0.012** |
| incoherence actually achieved | kurtosis 7.84 → **3.04** (Gaussian = 3.0); max/RMS 36.8 → **5.2** |
| new pipeline vs the earlier independent implementation | INT3 g128 ppl **56.959** vs 56.9594; INT4 g128 **13.928** vs 13.9284 |

The Hessian rotation matters and is easy to get wrong: in the rotated basis the layer sees
`x' = Qᵀx`, so the correct local Hessian is `Qᵀ H Q`. Using the unrotated `H` would silently
optimise the wrong objective.

---

## 2. Effective bits per weight

Everything is counted: codes, per-group scales and zeros, the VQ codebook amortised over all
357.8 M target weights, the rotation (a 32-bit seed — it is generated, not stored), and, where
act-order is used, a per-column index at `⌈log₂ in_features⌉` bits.

| cell | configs | effective bits/weight |
|---|---|---|
| scalar (group 128) | 2/3/4 bit | 2.250, 3.250, 4.250 |
| scalar (group 64) | 2/3 bit | 2.500, 3.500 |
| VQ K=256 | 2/3/4 stages | 2.000, 3.000, 4.000 |
| VQ K=1024 | 2/3 stages | 2.501, 3.751 |
| + Hadamard | any | +0.000002 (32 bits per matrix) |
| + act-order | any | +0.008 (scalar 2-bit, measured) |

---

## 3. Stage 1 — per-layer rate–distortion, all metrics

Fits of `log₂(activation NMSE) = a + slope · bpw`, 5 rate points per cell:

| cell | intercept | slope |
|---|---|---|
| scalar/naive/native | 4.077 | −2.469 |
| scalar/naive/hadamard | 3.571 | −2.384 |
| scalar/gptq/native | 3.404 | −2.875 |
| scalar/gptq/hadamard | 1.887 | −2.632 |
| vq/naive/native | −0.095 | **−1.355** |
| vq/naive/hadamard | 0.018 | −1.530 |
| vq/gptq/native | −1.470 | −1.512 |
| vq/gptq/hadamard | −2.154 | −1.636 |

Predicted activation NMSE at matched rates:

| cell | 2.25 | 3.00 | 3.50 | 4.00 |
|---|---|---|---|---|
| scalar/naive/native | 0.3588 | 0.0994 | 0.0422 | 0.0180 |
| scalar/naive/hadamard | 0.2887 | 0.0836 | 0.0366 | 0.0160 |
| scalar/gptq/native | 0.1195 | 0.0268 | 0.0099 | 0.0037 |
| scalar/gptq/hadamard | 0.0610 | 0.0155 | 0.0062 | **0.0025** |
| vq/naive/native | 0.1132 | 0.0560 | 0.0350 | 0.0219 |
| vq/naive/hadamard | 0.0932 | 0.0421 | 0.0248 | 0.0146 |
| vq/gptq/native | 0.0342 | 0.0156 | 0.0092 | 0.0055 |
| **vq/gptq/hadamard** | **0.0175** | **0.0075** | **0.0043** | 0.0024 |

**The slopes are the most informative single row of this table.** VQ's rate–distortion curve is
far flatter (−1.36 to −1.64) than scalar's (−2.38 to −2.88). An ideal quantizer gives −2. Each
additional residual-VQ stage costs a full bit but returns less than the 4× distortion reduction
that bit is worth, so VQ's advantage is intrinsically a *low-rate* phenomenon that erodes as
bits are added. This, not any subtlety of the codebook, is why the representation effect
reverses sign by 4 bits.

Figure: [`diag_activation.png`](../results/figures/diag_activation.png).

---

## 4. Factorial effects

Expressed in **equivalent bits per weight** — each distortion ratio is converted through the
cell's own measured slope, so "+1.14 bits" means the treatment reaches the same quality while
spending 1.14 fewer bits per weight.

### Main effects

| effect | 2.25 bpw | 3.00 | 3.50 | 4.00 |
|---|---|---|---|---|
| **compensation (GPTQ − naive)** | **+1.054** | **+1.135** | **+1.188** | **+1.241** |
| representation (VQ − scalar) | +0.843 | +0.448 | +0.184 | **−0.079** |
| rotation (Hadamard − native) | +0.322 | +0.337 | +0.346 | +0.356 |

### Main effects broken out by the level held fixed

| effect | holding | 2.25 | 3.00 | 3.50 | 4.00 |
|---|---|---|---|---|---|
| compensation | scalar/native | +0.593 | +0.707 | +0.783 | +0.859 |
| compensation | scalar/hadamard | +0.895 | +0.969 | +1.019 | +1.068 |
| compensation | vq/native | +1.206 | +1.289 | +1.344 | +1.398 |
| compensation | **vq/hadamard** | **+1.523** | **+1.573** | **+1.607** | **+1.640** |
| representation | naive/native | +0.870 | +0.433 | +0.142 | −0.150 |
| representation | gptq/native | +0.824 | +0.358 | +0.047 | **−0.263** |
| representation | gptq/hadamard | +0.843 | +0.493 | +0.259 | +0.026 |
| rotation | scalar/naive | +0.129 | +0.103 | +0.085 | +0.067 |
| rotation | scalar/gptq | +0.353 | +0.287 | +0.243 | +0.199 |
| rotation | vq/naive | +0.195 | +0.286 | +0.347 | +0.408 |
| rotation | **vq/gptq** | **+0.612** | **+0.671** | **+0.710** | **+0.749** |

### Two-way interactions — all positive

| interaction | at | 2.25 | 3.00 | 3.50 | 4.00 |
|---|---|---|---|---|---|
| representation × compensation | native | +0.592 | +0.564 | +0.546 | +0.527 |
| representation × compensation | hadamard | +0.621 | +0.599 | +0.584 | +0.569 |
| compensation × rotation | scalar | +0.301 | +0.266 | +0.243 | +0.219 |
| compensation × rotation | vq | +0.330 | +0.301 | +0.281 | +0.261 |
| representation × rotation | naive | +0.052 | +0.165 | +0.240 | +0.316 |
| representation × rotation | gptq | +0.220 | +0.345 | +0.429 | +0.513 |

No factor is substitutable for another. VQ gains roughly twice as much from compensation as
scalar does (+1.29 vs +0.71 bits at 3 bpw, native) and more than twice as much from rotation
(+0.67 vs +0.29 with GPTQ).

---

## 5. Tail-error analysis

Mean over the 20 evaluated matrices, relative error restricted to the largest-magnitude weights.

| cell | top 1 % | top 0.1 % | top 0.01 % | max err / max\|W\| | sign flips (top 0.1 %) |
|---|---|---|---|---|---|
| scalar/naive/native | 0.0112 | 0.0079 | 0.0044 | 0.142 | 0.00003 |
| scalar/naive/hadamard | 0.0089 | 0.0043 | 0.0018 | 0.088 | 0.00000 |
| scalar/gptq/native | 0.0520 | 0.0340 | 0.0192 | **0.467** | **0.00228** |
| scalar/gptq/hadamard | 0.0205 | 0.0094 | 0.0037 | 0.179 | 0.00000 |
| **vq/naive/native** | 0.0373 | **0.0608** | **0.1095** | **0.579** | 0.00021 |
| **vq/naive/hadamard** | 0.0075 | **0.0055** | **0.0046** | **0.070** | 0.00000 |
| vq/gptq/native | 0.0536 | 0.0774 | **0.1262** | **0.604** | **0.00275** |
| vq/gptq/hadamard | 0.0120 | 0.0074 | 0.0046 | 0.117 | 0.00000 |

Figure: [`diag_tail.png`](../results/figures/diag_tail.png).

Three findings, all of which aggregate MSE hides:

1. **VQ destroys outliers in native coordinates.** Its error on the top 0.01 % of weights is
   **0.1095** and it reconstructs the single largest weight with **58 % error**. A single
   8-dimensional codeword cannot simultaneously represent one huge outlier and seven ordinary
   values. Under Hadamard these become **0.0046** and **7 %** — a **24× improvement**. The
   rotation is not helping VQ compress better on average; it is *removing the objects VQ cannot
   represent at all*.
2. **GPTQ deliberately damages the tails**, and that is correct behaviour: it pushes error into
   whichever weights the Hessian says are cheap. Scalar/native tail error rises 0.0079 → 0.0340
   and sign flips on large weights appear (0.00228). So tail error is not a sufficient quality
   proxy either — it is a *diagnostic*, not an objective. Under Hadamard the sign flips vanish
   entirely.
3. **Tail error, not activation NMSE, tracks end-to-end quality across representations.** VQ at
   3.00 bpw and scalar at 3.25 bpw have essentially identical activation NMSE (0.0584 vs 0.0576)
   but perplexity **277.7 vs 57.0** — a 4.9× gap. The top-0.01 % error differs by 25×
   (0.1095 vs 0.0044) and predicts the ordering correctly. This is the third independent
   occasion in this project where a mean-squared metric misranked codecs.

---

## 6. Stage 2 — end-to-end perplexity, full-model sequential GPTQ

All 168 linear layers quantized, calibration activations propagated layer by layer through the
*already quantized* prefix. wikitext-2, 24 × 2048 tokens. fp16 baseline **11.813**.

| bits/weight | scalar naive/native | scalar gptq/native | scalar naive/had | scalar gptq/had |
|---|---|---|---|---|
| 2.25 | 274 426 | 401 379 | 541 127 | **25 891** |
| 3.25 | 56.96 | 23.25 | 38.51 | **16.76** |
| 4.25 | 13.93 | 13.21 | 14.23 | **12.57** |

| bits/weight | VQ naive/native | VQ gptq/native | VQ naive/had | VQ gptq/had |
|---|---|---|---|---|
| 2.00 | 54 634 | 1 280 | 4 040 | **49.12** |
| 3.00 | 277.68 | 129.21 | 36.89 | **16.49** |
| 4.00 | 35.77 | 28.90 | 16.20 | **13.01** |

Figure: [`diag_perplexity.png`](../results/figures/diag_perplexity.png),
per-cell bars: [`diag_cells.png`](../results/figures/diag_cells.png).

**Degradation reduction from the full treatment** (Δppl vs fp16, naive/native → gptq/hadamard):

| representation | rate | Δ naive/native | Δ gptq/hadamard | factor |
|---|---|---|---|---|
| scalar | 3.25 | +45.15 | +4.94 | **9.1×** |
| scalar | 4.25 | +2.115 | +0.752 | 2.8× |
| VQ | 3.00 | +265.87 | +4.68 | **56.8×** |
| VQ | 4.00 | +23.96 | +1.19 | **20.1×** |

**End-to-end effect decomposition** (Δppl improvement factor, single factor vs both):

| representation | rate | compensation alone | rotation alone | both |
|---|---|---|---|---|
| scalar | 3.25 | **3.95×** | 1.69× | 9.1× |
| VQ | 3.00 | 2.26× | **10.6×** | 56.8× |

The two representations are limited by *different* factors: scalar mainly by error placement,
VQ mainly by the coordinate system. In both cases the combination beats the product of the
individual effects (9.1 > 6.7 and 56.8 > 24.0), i.e. the interaction is positive end to end.

**The 2-bit regime.** No combination reaches usable quality. The best is VQ/GPTQ/Hadamard at
2.00 bpw with ppl 49.12 against fp16's 11.81 — 4.2× worse. At 2.25 bpw scalar's best is
25 891. Two factors *individually* make 2-bit scalar worse than naive (GPTQ alone 401 379,
Hadamard alone 541 127, vs naive 274 426); only their combination helps.

---

## 7. Controls

### 7.1 Matched-Gaussian control

The same pipeline on an i.i.d. Gaussian with identical per-row and per-column RMS
(`gauss_rowcol`), 40 runs:

| representation | coordinates | weight-space real/null | activation-space real/null |
|---|---|---|---|
| scalar | native | **1.077** | 0.871 |
| scalar | hadamard | **1.000** | 0.787 |
| VQ | native | **1.041** | 0.899 |
| VQ | hadamard | **1.006** | 0.793 |

**In weight space — the pure rate–distortion geometry the question is about — the real matrix
has no advantage over matched noise; it is if anything 3 % harder to compress, and exactly
break-even under Hadamard.** This reproduces the central finding of the earlier study through
a completely different code path.

The activation-space ratio of ~0.84 should *not* be read as a real-weight advantage: the
denominator ‖WX‖ differs between the real matrix and the null, because training aligns W with
the activation distribution while the null does not. That is a signal-power effect, not a
compressibility effect.

### 7.2 Act-order and damping control

The 2-bit GPTQ divergence in native coordinates (weight NMSE > 1) could have been a tuning
artifact, so it was checked directly (3 layers × 3 projections, activation error):

| bits | percdamp | act-order | native | hadamard |
|---|---|---|---|---|
| 2 | 0.01 | off | 0.1613 | 0.0587 |
| 2 | 0.01 | **on** | **0.0649** | 0.0568 |
| 2 | 0.05 | on | 0.0638 | 0.0517 |
| 2 | 0.20 | off | 0.1043 | 0.0553 |
| 2 | 0.20 | on | 0.0755 | 0.0539 |

**Act-order matters enormously in native coordinates (0.161 → 0.065) and almost not at all
under Hadamard (0.0587 → 0.0568).** That is exactly the predicted mechanism: act-order sorts
columns by Hessian diagonal magnitude, which is only useful while the Hessian is anisotropic.
Once the rotation has made it near-isotropic there is no ordering left to exploit — and the
result becomes insensitive to both damping and ordering, i.e. the rotation makes the whole
procedure robust.

Re-running the 2-bit cells end to end with act-order (storage counted, +0.008 bpw):

| cell | without act-order | with act-order |
|---|---|---|
| scalar/gptq/native | 401 379 | 285 774 |
| scalar/gptq/hadamard | 25 891 | **17 911** |
| vq/gptq/native | 1 280 | 5 035 |
| vq/gptq/hadamard | **49.12** | 55.05 |

Act-order helps scalar and *hurts* VQ — permuting the input columns breaks the channel grouping
the codebook was fit on. The best 2-bit configuration is unchanged. The main factorial is
reported without act-order, which slightly understates the native-GPTQ cells; the effect
ordering is unaffected.

---

## 8. Interpretation of H1 / H2 / H3

**H1 — representation bottleneck: rejected as the dominant cause.** A significant scalar↔VQ gap
does *not* survive compensation and rotation. It is +0.84 bits at 2.25 bpw, falls to +0.45 at
3.0, +0.18 at 3.5, and **−0.08 at 4.0**. After the full treatment the two representations land
within 0.25 bits of each other end to end (VQ 16.49 @ 3.00 vs scalar 16.76 @ 3.25; VQ 13.01 @
4.00 vs scalar 12.57 @ 4.25). The mechanism is visible in the slopes: residual VQ buys less
than an ideal quantizer per added stage, so its advantage is structurally confined to low rate.

**H2 — error-placement bottleneck: supported, and the largest single effect.** Compensation is
worth +1.05 to +1.24 bits/weight across the whole range, 2.5× the representation effect and
3.4× the rotation effect at 3 bpw. It is also the only main effect that *grows* with rate. End
to end it reduces scalar's 3.25-bit degradation by 3.95×. And it does exactly what the theory
says: it *raises* weight-space error (NMSE 0.052 → 0.129) in exchange for lower functional
error (0.058 → 0.012).

**H3 — coordinate-system bottleneck: supported, and dominant specifically for VQ.** Rotation is
worth +0.32 to +0.36 bits/weight on average, but the average hides the important part: +0.10
for scalar/naive versus **+0.75 for vq/gptq**. End to end, rotation alone improves VQ's 3-bit
degradation by **10.6×** against compensation's 2.26×. The mechanism is measured, not assumed:
kurtosis 7.84 → 3.04, max/RMS 36.8 → 5.2, and VQ's top-0.01 % weight error 0.1095 → 0.0046.

---

## 9. Verdict

### **E2 — Error allocation dominates**, jointly with **E3**, with a genuine but already-exploited **E4** interaction.

Compensation is the single largest factor (+1.14 bits/weight at 3 bpw), the coordinate system is
second and is decisive for VQ specifically, and the representation is third and reverses sign by
4 bits/weight. Every interaction is positive, and the two representations are limited by
different factors — so a representation genuinely cannot be evaluated in isolation from the loss
geometry it sits in (the E4 claim is true).

**But the E4 interaction does not justify continuing.** It is fully explained — VQ cannot code
outliers, rotation removes outliers — and it is precisely the combination that QuIP# and AQLM
already ship. It points into occupied territory, not new territory.

**An E5 caveat is also true and must be stated:** this prototype does *not* reproduce the
mechanism of successful sub-3-bit methods. The best 2-bit configuration reaches ppl 49.12
against fp16's 11.81. What is missing, in the order I would expect it to matter:

1. **fine-tuning / QAT** — end-to-end or layer-wise distillation after quantization, which is
   how AQLM and QuIP# actually recover 2-bit quality and is the single biggest absent mechanism;
2. **better codebooks** — E8-lattice or trellis-coded codebooks rather than plain k-means,
   worth a few tenths of a bit;
3. **learned rather than random rotations** (SpinQuant-style);
4. **finer group structure and clipping search** in the scalar quantizer;
5. explicit outlier/residual handling for the small set of weights that dominate quality.

None of these is a *representation* question in the sense this project set out to study.

---

## 10. Recommendation: **close this direction**

The kill criterion stated for this experiment is met: the results are E2/E3, and the one
interaction found is real but neither surprising nor unoccupied. Concretely:

* **Stop** designing procedural, recursive, grammar-based, or tile-dictionary weight
  representations. Three independent lines of evidence now say the raw local statistics of LLM
  weights are not the right object: tiles are indistinguishable from matched i.i.d. Gaussian
  (earlier study, replicated on two architectures), the matched-Gaussian control here reproduces
  it through a different code path (weight-space ratio 1.00–1.08), and the factorial shows the
  representation axis is the weakest of the three and reverses sign by 4 bits.
* **The dominant levers are already-known ones**: second-order error compensation and
  incoherence transforms. Anything further in this space is competing directly with
  GPTQ/QuIP#/QuaRot/AQLM on their own ground.
* **If any work continues**, the highest-information next step is not a codec at all — it is
  adding layer-wise distillation on top of the existing `vq/gptq/hadamard` pipeline and
  measuring how much of the 2-bit gap (49.12 → 11.81) it closes. That is a one-day experiment on
  this hardware and it would settle whether the residual gap is a representation problem at all.
  My expectation, given everything above, is that it is not.

### Methodological findings worth carrying forward

1. **Report tail error alongside MSE.** `max|W−Ŵ|/max|W|` and the relative error on the top
   0.1 % / 0.01 % of weights cost nothing and caught three separate misrankings in this project
   that both weight NMSE and activation NMSE missed.
2. **Activation NMSE measured against unquantized upstream activations is not a reliable
   end-to-end proxy across representations.** VQ and scalar matched to within 1.4 % on it while
   differing 4.9× in perplexity.
3. **Rotate the Hessian when you rotate the weights.** `Qᵀ H Q`, verified to 5.5 × 10⁻¹⁴.
   Getting this wrong optimises the wrong objective silently.
4. **Check whether a low-bit failure is a tuning artifact before believing it.** The 2-bit GPTQ
   divergence was 2.5× better with act-order — and the check also revealed that act-order's
   value disappears under rotation, which is itself a clean result.
