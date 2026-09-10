# Phase 0 — Prior art, scoped to "what would this experiment merely reproduce?"

Time-boxed. Purpose is to (a) place our baselines correctly, (b) mark experiments that are
already-answered so we don't spend compute on them.

## The rate–distortion frame we are actually in

Post-training weight compression of an LLM linear layer is lossy source coding of a matrix
`W ∈ R^{out×in}`. Empirically, LLM weight entries within a row/column-scaled block are close to
i.i.d. and roughly Gaussian/Laplacian. That single fact controls most of the outcomes:

* For an i.i.d. source, the entire achievable gain of *vector* quantization over *scalar*
  quantization is the **space-filling (granular) gain**, asymptotically 1.53 dB ≈ **0.254 bits/dim**,
  plus a **shape gain** from non-Gaussianity of the marginal. There is no "structure" gain.
* Any method that finds *no* structure beyond this is not failing — it is hitting the source's
  actual rate–distortion function.
* Therefore the decisive question for this project is **not** "can VQ beat scalar quant" (known: yes,
  by a fraction of a bit) but **"is there structure in LLM weight tiles beyond an i.i.d. scaled
  Gaussian?"** If not, procedural/learned decoders cannot win, by information-theoretic argument,
  and the correct verdict is C.

**Design consequence:** every structural diagnostic in Phase 1 is run *identically* on a
**synthetic i.i.d. Gaussian null model** matched to the real matrix's per-row and per-column scales.
The gap between real and null is the entire signal we care about. Without this control, PCA/k-means
numbers are uninterpretable (a random matrix also has "clusterable" tiles at finite sample size).

## Families, and what is already known

| Family | Representative work | Known result | Our use |
|---|---|---|---|
| Scalar quantization, group-wise | RTN, GPTQ, AWQ, SmoothQuant | 4-bit ≈ near-lossless with group size 64–128 + activation-aware error compensation; 3-bit degrades; 2-bit needs much more | **Baseline**, not innovation |
| Vector quantization / codebooks | QuIP#, AQLM, QTIP, GPTVQ, K-means quant | 2–3 bits/weight viable; gains come from granular gain + incoherence processing (random rotation), *not* from tile-level semantic redundancy | **Baseline**; also the main competitor to beat |
| Incoherence processing / random rotation | QuIP, QuIP#, QuaRot, SpinQuant | Hadamard/random-orthogonal rotation makes weights *more* Gaussian and *removes* outliers → improves quantization. Note the direction: it **destroys** structure and that *helps*. Strong evidence against "structure exists" | Interpreted as prior evidence for the null |
| Low-rank decomposition | SVD-LLM, ASVD, FWSVD | Whole-matrix low-rank is poor for LLM weights (spectrum is not low-rank); only helps as a *residual* term | Baseline 2 |
| Low-rank + sparse/outlier | LQ-LoRA, LoftQ, SqueezeLLM (dense+sparse), SpQR | Outlier handling matters a lot at ≤3 bits; low-rank residual on quantization error gives modest gains | Variant D relative |
| Tensor / Kronecker | KronA, Tensor-Train compression | Works for fine-tuning adapters; for base weights loses a lot | Skipped — known negative, expensive |
| Cross-layer basis sharing / weight prediction | Basis sharing, MiniCache, cross-layer parameter sharing | Works between *adjacent* layers of *some* models; usually needs training to recover | Phase 1B measures the raw signal cheaply |
| INR / hypernetwork weights | NeRF-style implicit weights, HyperNetworks, "Neural network diffusion" | Coordinate-MLP decoding of trained weights has never been shown competitive at LLM scale; decoder cost dominates | Variant C, budget-matched, expected to lose |
| Permutation / alignment | Git Re-Basin, ZipIt, model merging permutations, "weight permutation for quantization" (e.g. re-ordering channels before grouping) | Channel permutation *does* help group-wise quantization by grouping similar-scale channels. This is a real, cheap, function-preserving win | Phase 1C — the most promising alignment idea |
| Fused / reconstructive quantized matmul | Marlin, Machete, QTIP decode-in-kernel, any-precision LLM | Decode must be O(few) ops/weight and register-local, otherwise the kernel is memory-bound-to-compute-bound flipped and loses | Sets our decoder-complexity bar |

## Experiments that would only reproduce known results (deliberately NOT run)

1. "Group-wise INT4 is close to FP16" — known. We run it once as a calibration point, nothing more.
2. "Whole-matrix SVD of an LLM weight matrix is not low-rank" — known. We measure the spectrum once
   in Phase 1 as a diagnostic and move on.
3. "Random rotation improves quantization" — known (QuIP#/QuaRot). We do not re-derive it; we *use*
   the fact as evidence about the source distribution.
4. Large-scale training-aware / QAT recovery — out of scope, expensive, and would mask the
   representation question with fine-tuning capacity.
5. Kronecker/TT factorization of base weights — known negative at this quality level.

## The bar a procedural decoder must clear

From the fused-kernel literature, a weight codec that wants to live inside a GEMM needs roughly:

* **≤ ~4–8 arithmetic ops per reconstructed weight** (an INT4 GEMM already spends ~2 ops/weight on
  dequant; the matmul itself spends 2 ops/weight per output column, amortized over the tile's N).
* **shared state ≤ ~48–96 KB** so it lives in L1/shared memory per SM.
* **no dependent loads** (no pointer chasing), **no per-tile branching**, **no iteration**.
* per-tile state read must be *streamed contiguously*.

A codec saving 1 byte/weight but costing 100 FLOPs/weight is dead on arrival: at ~4 bytes/FLOP-ish
consumer-GPU arithmetic intensity, you have traded a memory saving worth ~1 byte for arithmetic worth
~25 bytes of equivalent time. This is stated as an explicit screening rule in the results table.
