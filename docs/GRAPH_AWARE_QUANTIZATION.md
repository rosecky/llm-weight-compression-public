# Graph-aware quantization

**Question.** Weight values are structureless and their individual sensitivities are not
compressible (see `FINDINGS.md`, `FINAL_DIAGNOSTIC.md`, `PROGRAMMATIC_LAYOUT.md`). But maybe
the relevant structure is in the *topology of functional coupling* between channels. Can
quantization blocks be defined as communities in a functional graph rather than as adjacent
regions of the weight matrix, and does that pay?

**Answer, in one sentence.**

> No: the graph's edges do predict real quantization-error interactions and its communities
> do beat an eigenvalue-preserving null, but after GPTQ and an incoherence rotation a
> graph-derived channel ordering is **statistically indistinguishable from a random
> permutation** (Δppl 5.107 against a random-ordering range of 4.727–5.103), and in native
> coordinates it is *worse* than leaving the channels alone.

**Verdict: GA-B** — the topology is real but everything practically relevant about it is
first-order anisotropy, which rotation removes more cheaply. With a **GA-F** side finding
that is worth more than the main result: GPTQ turns out to operate *entirely* through
manufactured negative error interactions, and that can now be quantified exactly.

---

## 0. Method, and why the interaction algebra is exact

For `y = W x` under the K-FAC factorisation `H ≈ G ⊗ A` with `A = E[x xᵀ]` and
`G = E[g gᵀ]`, `g = ∂L/∂y`, the damage of a weight perturbation is `tr(ΔW A ΔWᵀ G)`. That
makes the interaction between two channel perturbations an identity, not a hypothesis:

```
input channels  i, j :  I_ij = 2 A_ij (e_iᵀ G e_j)      e_i = ΔW[:, i]
output channels i, j :  I_ij = 2 G_ij (r_iᵀ A r_j)      r_i = ΔW[i, :]
```

So `A` is the natural graph over input channels and `G` over output channels, and if the
covariance entry is zero the two perturbations *cannot* interact at second order. Both forms
are computable for the whole population at once as an elementwise product of two matrices, so
the edge-validity gate needs no pair sampling and has no selection bias. The two decompositions
are of the same scalar, and they agree to **1.7e-6** — an internal consistency check on the
implementation.

`G` is taken from the model's own next-token loss with all parameters frozen and the graph
anchored at the embedding output, so backward allocates activation gradients only. The values
were validated against a direct `autograd.grad` computation (relative error **0.0**). A first
implementation using `register_full_backward_hook` silently returned an all-zero gradient for
`q_proj`; the fix was a hook on the output tensor itself, and it is the reason that check
exists.

Model: Qwen2.5-0.5B, wikitext-2, seed 0, layers 1/11/22 × {q, o, up, down} for the layer-wise
work and all 168 linear layers for perplexity. Peak VRAM 3630 MiB (gradient capture),
1321 MiB (structure), 913 MiB (quantization).

---

## 1. Graph definitions

| graph | vertices | edge | note |
|---|---|---|---|
| A-cov | input channels | `\|A_ij\|` | dominated by channel scale — kept only as the G7 control |
| A-corr | input channels | `\|corr(A)_ij\|` | primary: scale-free |
| A-pcorr | input channels | normalised negative precision | coupling with everything else regressed out |
| G-cov / G-corr | output channels | same, on the Fisher `G` | |

Every affinity is symmetrised with a zero diagonal, then sparsified to each vertex's `k`
strongest edges (`k = 32` primary). Communities are equal-size contiguous chunks of a
**spectral (Fiedler) sequencing**, because what a deployable method needs is a permutation,
not a label set — the ordering *is* the deliverable.

---

## 2. Null controls

All four nulls are surrogate matrices that the entire pipeline is re-run on, so the null gets
the same optimisation the real graph gets.

| null | preserves | destroys |
|---|---|---|
| permute | every graph statistic (isomorphism) | alignment with index order |
| config | strength sequence | everything else |
| rewire | strength sequence + edge-weight multiset | topology |
| **spectral** | the entire eigenvalue spectrum | which channels the eigenmodes live on |

Block energy = fraction of off-diagonal edge energy falling inside communities of 128
channels; lift = that over a random balanced partition.

| side | graph | null | block energy | lift | modularity | eff. support / n |
|---|---|---|---|---|---|---|
| A | corr | **real** | 0.2696 | 2.67 | 0.1213 | 0.202 |
| A | corr | spectral | 0.2236 | 2.11 | 0.0964 | 0.267 |
| A | corr | rewire | 0.1620 | 1.48 | 0.0433 | 0.372 |
| A | corr | config | 0.1435 | 1.31 | −0.2109 | 0.033 |
| G | corr | **real** | 0.3783 | 3.81 | 0.2302 | 0.222 |
| G | corr | spectral | 0.2164 | 2.04 | 0.0885 | 0.294 |
| G | corr | rewire | 0.1666 | 1.53 | 0.0460 | 0.362 |

**G2 passes, modestly.** Real over the eigenvalue-preserving null is **1.21×** on the input
graph and **1.75×** on the output graph. The `permute` null reproduces the real numbers
exactly, which is the correct behaviour — a relabelling is an isomorphism — and confirms the
pipeline is not secretly rewarding index adjacency.

The localisation number is the warning sign. The top-16 eigenmodes have effective support over
**20–22%** of all channels, against 27–29% for the spectral null. The modes are only slightly
more localised than random ones and nowhere near living on identifiable small communities,
which is the regime a block-diagonal description needs.

### G7: the covariance graph is just scale

| side | graph | spectral communities | scale-sorted | contiguous | random |
|---|---|---|---|---|---|
| A | cov | 0.3263 | **0.7130** | 0.1319 | 0.1126 |
| A | corr | **0.2696** | 0.1835 | 0.1573 | 0.1146 |
| G | cov | 0.3262 | **0.3963** | 0.2648 | 0.1073 |
| G | corr | **0.3783** | 0.1775 | 0.2915 | 0.1128 |

On the covariance graph, simply sorting channels by their own second moment captures **more
than twice** the block energy that spectral clustering does. Kill criterion **G7** fires for
the covariance graphs; only the correlation graphs carry structure that is not scale.

---

## 3. Edge validity (kill gate G1)

Exact interaction decomposition of `tr(ΔW A ΔWᵀ G)` at 3 bits, group 128, full population,
12 matrices:

| coords | comp | axis | additive/total | signed I/total | \|I\|/total | Spearman(edge, \|I\|) | held-out R² |
|---|---|---|---|---|---|---|---|
| native | naive | in | 0.989 | +0.011 | 4.7 | 0.757 | 0.606 |
| native | naive | out | 1.008 | −0.008 | 8.8 | 0.694 | 0.529 |
| hadamard | naive | in | 1.017 | −0.017 | 15.2 | 0.718 | 0.542 |
| native | gptq | in | **7.725** | **−6.725** | 39.9 | 0.743 | 0.590 |
| native | gptq | out | 0.998 | +0.002 | 3.1 | 0.722 | 0.560 |
| hadamard | gptq | in | **10.795** | **−9.795** | 197.0 | 0.713 | 0.536 |
| hadamard | gptq | out | 1.001 | −0.001 | 3.4 | 0.688 | 0.509 |

**G1 passes.** Edge strength predicts interaction magnitude with Spearman 0.64–0.76 and
held-out R² 0.45–0.61 over the full population. Under Hadamard the scale-free correlation
edge does as well as the covariance edge (0.704 vs 0.713), so this is not purely a scale
effect; in native coordinates the covariance edge is clearly better (0.757 vs 0.635), so
there it partly is.

But two things in that table matter more than the gate:

1. **Without compensation, interactions are zero-mean noise.** For naive quantization the
   summed absolute interaction is 4.7–19× the total damage while the *signed* sum is ±1%.
   Errors interact enormously and cancel almost perfectly. Choosing which channels share a
   block cannot systematically exploit that — only changing the errors can.
2. **GPTQ works entirely by manufacturing negative interactions.** With compensation the sum
   of independent per-channel damages is **7.7×** (native) and **10.8×** (Hadamard) the actual
   damage; the interaction term is −6.7 to −9.8× the total. This is a clean quantitative
   statement of what second-order compensation *is*, and it explains why the earlier factorial
   found compensation worth +1.14 bits/weight. Along the output axis GPTQ does nothing
   (additive/total = 1.001), exactly as it should — it treats rows independently.

---

## 4. Community structure

| property | measurement |
|---|---|
| sizes | balanced by construction: 32 / 64 / 128 channels |
| modularity | 0.121 (A-corr), 0.230 (G-corr), vs 0.096 / 0.089 for the spectral null |
| off-block energy | 73% (A-corr), 62% (G-corr) of edge energy falls *outside* blocks of 128 |
| spectral localisation | top-16 modes span 20–22% of channels; delocalised |
| **stability in k** | ARI between partitions at k=8 and k=32: **0.07–0.30** |
| **cross-layer reuse** | ARI between layers: **−0.002 to +0.011** (chance = 0) |

The last two rows are the ones that decide the recursive/grammar question without building
anything. Communities are not stable under a change in how many edges per vertex are kept
(an adjusted Rand index of 0.07–0.30 means the partitions largely disagree). And
`q_proj`, `up_proj` and friends in *every* block read the same residual-stream channels, so
their A-graph partitions are directly comparable — the adjusted Rand index between layers 1,
11 and 22 is **statistically zero**. The same channels are grouped completely differently in
every layer. There is no reusable community template, no shared motif, and nothing for a
graph grammar or recursive coarsening to capture.

---

## 5. Quantization results

Layer-wise, 3 bits, group 128, 3.25 bpw for every row (a 16×8 tile has exactly the same scale
budget as a 1×128 group). `fisher` is `tr(ΔW A ΔWᵀ G)/tr(W A Wᵀ G)`, lower is better.

| block | coords | comp | in order | out order | fisher | lift vs contiguous |
|---|---|---|---|---|---|---|
| 1×128 | native | gptq | contiguous | — | 0.01679 | 1.000 |
| 1×128 | native | gptq | random | — | 0.01666 | 1.008 |
| 1×128 | native | gptq | **scale-sorted** | — | **0.01204** | **1.394** |
| 1×128 | native | gptq | A-corr communities | — | 0.01616 | 1.039 |
| 1×128 | native | gptq | A-cov communities | — | 0.01548 | 1.085 |
| 1×128 | hadamard | gptq | contiguous | — | 0.00878 | 1.000 |
| 1×128 | hadamard | gptq | scale-sorted | — | 0.00853 | 1.029 |
| 1×128 | hadamard | gptq | A-corr communities | — | 0.00871 | **1.008** |
| 16×8 | hadamard | gptq | contiguous | contiguous | 0.00887 | 1.000 |
| 16×8 | hadamard | gptq | A-corr | G-corr | 0.00877 | 1.011 |
| 16×8 | native | gptq | contiguous | G-corr | 0.01834 | 1.014 |

**G3 fires.** In the only cell where adaptive ordering does anything (native + GPTQ), plain
activation-scale sorting is worth **1.394×** while the best graph ordering manages **1.085×**.
The graph loses to the trivial baseline by a factor of four.

**Phase 5 (2-D blocks) adds nothing.** Lifts for the 16×8 rows are against that shape's own
contiguous baseline. The 2-D tile is *worse in absolute terms* than a 1×128 row group
everywhere (0.00887 vs 0.00878 under Hadamard, 0.01859 vs 0.01679 native), and adding
output-channel communities on top moves it by 1.1–1.4% — which does not recover the loss
from changing the block shape in the first place.

---

## 6. End-to-end perplexity

Full-model sequential GPTQ, all 168 linear layers, 3 bits group 128, wikitext-2 on 24×2048
tokens, fp16 baseline **11.8134**. Non-contiguous orderings are charged
`ceil(log2(in))` bits per column (+0.008 bpw).

| coords | order | bpw | ppl | Δppl | vs contiguous |
|---|---|---|---|---|---|
| hadamard | contiguous | 3.2500 | 17.2459 | +5.4324 | 1.000 |
| hadamard | random, seed 2 | 3.2579 | **16.5401** | +4.7266 | 1.149 |
| hadamard | random, seed 3 | 3.2579 | 16.6445 | +4.8311 | 1.124 |
| hadamard | scale-sorted | 3.2579 | 16.7034 | +4.8900 | 1.111 |
| hadamard | random, seed 1 | 3.2579 | 16.9167 | +5.1033 | 1.064 |
| hadamard | **A-corr communities** | 3.2579 | 16.9203 | +5.1069 | **1.064** |
| native | contiguous | 3.2500 | **23.3339** | +11.5204 | 1.000 |
| native | scale-sorted | 3.2578 | 23.6484 | +11.8350 | 0.973 |
| native | random | 3.2578 | 24.0975 | +12.2840 | 0.938 |
| native | A-corr communities | 3.2578 | 24.3299 | +12.5165 | 0.920 |

This is the decisive table, and it exists because this project has now watched a
mean-squared metric misrank codecs five times.

**G4 fires unambiguously.** Under Hadamard the graph ordering (Δppl 5.107) sits at the
*bottom* of the three-seed random-permutation range (4.727–5.103). Everything that looked
like a graph benefit is "any permutation other than the identity helps GPTQ a little under
rotation" — a real but topology-free effect, and the spread between random seeds (0.38 Δppl)
is larger than the gap the graph produces (0.33).

In native coordinates the layer-wise Fisher error said scale-sorting was 1.394× better; end
to end it is **0.973×**, i.e. worse. The graph ordering is worse still (0.920×) — worse than
a random permutation. The identity ordering of input channels is genuinely good in native
coordinates, presumably because adjacent residual-stream channels have similar scales and
therefore make tight min-max groups; reordering by *correlation* breaks that without
compensating for it.

(These runs share one engine build. The ~2% offset from the `PROGRAMMATIC_LAYOUT` perplexities
comes from a Hessian symmetrisation added to `cholesky_inverse_upper` in between; comparisons
within this table are unaffected.)

---

## 7. Cross-layer results (kill gate G6)

Layer `a` quantized alone, layer `b` alone, and both, at 3 bits, measured in end-to-end
perplexity: `I = Δppl(both) − Δppl(a) − Δppl(b)`.

| distance | pairs | mean interaction, % of additive damage |
|---|---|---|
| 1 | (1,2), (5,6), (10,11), (11,12) | **5.3%** |
| 2 | (1,3), (11,13) | 8.4% |
| ≥5 | (11,16), (5,11), (1,22) | **3.3%** |

**G6 fires.** Cross-layer error interaction exists but is small (2–10% of the additive
damage), is *positive* — damage compounds slightly rather than cancelling — and does not
concentrate on adjacency: distance-2 pairs interact more than distance-1 pairs, and
distance-21 still shows 2.4%. There is no locality for a cross-layer functional graph to
exploit. What little compounding exists is already handled by sequential GPTQ, which
compensates each layer against the already-quantized prefix.

Combined with the zero cross-layer community reuse in §4, the H-VALUE and H-ERROR branches
have no target, and Phases 6–7 and 9 were not built. The brief gated them on single-layer
grouping showing a clear benefit; it showed none.

---

## 8. Hardware feasibility

Verified numerically, not asserted: permuting the MLP intermediate of all 24 blocks —
`down_proj`'s input columns together with the output rows of `up_proj` and `gate_proj` —
changes the fp32 logits by **1.1e-6**, i.e. round-off. That permutation is genuinely free
(class **H0**): it is folded into the checkpoint and no runtime work remains.

Everything else is not absorbable per layer:

* `q/k/v/gate/up` all read the shared residual stream, so a per-layer input permutation is
  illegal; they would all have to share one global channel permutation for the whole model.
* `o_proj`'s input is tied to attention head structure — only within-head permutation is safe.
* An explicit permutation index costs `ceil(log2(in))` bits per column = **0.008 bits/weight**.
  The graph ordering is worth about 0.005 bits/weight by the layer-wise Fisher error and
  nothing at all end to end, so **G5** fires for the general case.

Since the only free permutation is the MLP intermediate, and the effect there is inside the
random-permutation spread, the free case does not rescue the result.

---

## 9. Verdict

**GA-B — the structure is real, but simple anisotropy explains everything that matters.**

| gate | result |
|---|---|
| G1 edges are functional | **passes** — Spearman 0.64–0.76, held-out R² 0.45–0.61 |
| G2 topology beats null | **passes** — 1.21× (A) and 1.75× (G) over the spectral null |
| G3 beats simple baselines | **fires** — scale sorting 1.394× vs graph 1.085× |
| G4 survives GPTQ + rotation | **fires** — graph ≡ random permutation end to end |
| G5 metadata | **fires** — 0.008 bpw index vs ≈0.005 bpw of gain |
| G6 cross-layer interaction | **fires** — 5.3% adjacent vs 3.3% distant, no locality |
| G7 just scale grouping | **fires** for covariance graphs — scale sorting captures 2.2× more |

Against the seven conditions for calling the direction interesting: (1) yes, (2) yes,
(3) **no** — simple sorting explains more than the graph, (4) **no**, (5) **no** — communities
are unstable in k and have zero cross-layer reuse, (6) only the MLP intermediate, (7) **no**.

**Recommendation: close this branch too.** The one thing worth carrying forward is not a
method but a measurement: **compensation is not a refinement on top of independent
quantization, it is a different regime.** The additive damage is 7.7–10.8× the achieved
damage, so GPTQ destroys an order of magnitude of error through interactions it creates
itself. Any future work on sub-4-bit quality should be aimed at that mechanism — better
compensation objectives, joint or iterative compensation, distillation — and not at deciding
which weights sit in the same block. Every grouping study in this repository has now found the
same thing from a different direction.

---

## Reproduction

```bash
export PYTHONPATH=src HF_HOME=$PWD/cache/hf

# A and G for 12 layers (fp32; fp16 breaks the Cholesky)     ~4 min, 3630 MiB
python scripts/make_graphcal.py --layers 1,11,22 --n-seq 32 --seqlen 256

# G1 and G2                                                  43 s / 145 s, 1321 MiB
python -m lwc.experiments.graph_gates --phase edges
python -m lwc.experiments.graph_gates --phase structure

# G3 / G4 / Phase 5, layer-wise                              315 s, 913 MiB
python -m lwc.experiments.graph_quant

# permutation legality, cross-layer gate, end-to-end          ~35 min
python -m lwc.experiments.graph_endtoend --mode permcheck
python -m lwc.experiments.graph_endtoend --mode crosslayer
python -m lwc.experiments.graph_endtoend --mode ppl
python -m lwc.experiments.graph_endtoend --mode ppl --configs configs/graph_ppl_null.json

python scripts/analyze_graph.py && python scripts/plot_graph.py
```

Raw results in `results/raw/graph_{structure,edges,quant,endtoend}.jsonl`; tables in
`results/tables/graph.md`; figures `results/figures/graph_null.png` and
`results/figures/graph_ppl.png`.
