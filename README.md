# llm-weight-compression

**Which objective should a post-training quantizer optimize?** A pre-registered study of
refining GPTQ's integer codes against the error at the output of the enclosing
attention/MLP *module* instead of the single linear layer — same bits, same layout, same
kernel — with every measurement, every pre-registration and every correction in the open.

Jan Rosecký, September 2026. MIT. Companion to the
[Thinletter](https://thinletter.io) embedding-quantization project
([bge-m3](https://huggingface.co/honza-rosecky/bge-m3-query-clients) and
[Qwen3-Embedding](https://huggingface.co/honza-rosecky/qwen3-embedding-0.6b-query-clients) query
clients). Evidence as a dataset:
[honza-rosecky/llm-weight-compression-evidence](https://huggingface.co/datasets/honza-rosecky/llm-weight-compression-evidence);
interactive explorer:
[honza-rosecky/llm-weight-compression-explorer](https://huggingface.co/spaces/honza-rosecky/llm-weight-compression-explorer).

## The result in one table

Frozen protocol (method and hyperparameters committed before the runs, no tuning
afterwards), Hadamard + GPTQ at exactly 3.25 bits per weight, full wikitext-2 test plus 512
fixed C4 windows, paired bootstrap inside each independent calibration draw:

| model | draws | GPTQ → GPTQ + post-pass (wt2 NLL) | quantization gap closed | perplexity |
|---|---|---|---|---|
| Qwen2.5-0.5B | 3 | 2.921 → 2.891 | **+8.7%** (3/3 draws) | −2.7% |
| Qwen3-0.6B | 2 | 3.543 → 3.505 | **+8.2%** (2/2) | −3.7% |
| Llama-3.2-1B | 3 | 2.769 → 2.749 | **+4.6%** (3/3) | −1.9% |
| Qwen2.5-1.5B | 2 | 2.549 → 2.555 | **−2.5%** (0/2) | +0.6% |

The post-pass costs 20–30× the GPTQ time offline and nothing at inference. The last row
is real, replicated, and unexplained: no property we tested separates it from the three
above it. Converted to bits at equal quality the gain is at most ~0.08 bits per weight,
below the product threshold we set ourselves before measuring.

What survives, what does not, and why, is graded claim by claim in
[`docs/PITCH.md`](docs/PITCH.md). If you want to break the claims, start with
[`docs/REVIEWER.md`](docs/REVIEWER.md), which was written for that purpose and lists the
evidence file behind each number.

## What is ours and what is not

The non-monotone "objective horizon" curve (layer < module > full model) and its
bias–variance explanation are prior art — BRECQ (2021), MREM (2022) — and BaKron (2026)
derives the MLP-output Hessian we use and compares local vs global objectives at one solver
on Llama-3/Qwen3 to 8B. [`docs/PRIOR_ART_LADDER.md`](docs/PRIOR_ART_LADDER.md) sets this
out projection by projection. What is ours:

- the rungs *inside* attention (QKᵀ logits → softmax → attention output → module) at one
  solver, rate and probe budget, where the logit objective used by BoA/BaKron is the worst;
- a clean separation of estimator variance from calibration-text coverage: the
  full-model objective is text-limited, not probe-limited (4× tokens: 0.3% → 5.5% of the
  gap; 4× probes: nothing), and the same holds for the attention endpoint but not for the
  whole module;
- integer-only refinement *inside* the sequential quantization, format preserved, with a
  certified optimality gap on the row subproblem;
- the validation record: multiple calibration draws, every draw reported, paired
  bootstraps, fresh-metric audits, pre-registered diagnostics with interpretation tables,
  and the negatives — layer-local coordinate descent is mildly harmful end-to-end on
  three models; a flat rotation plateau; rate allocation worth ≤0.11 bits; diagonal output
  metrics exactly inert; rank-64 surrogates catastrophic;
- a failure mode at 2 bits: a refined 2.1-bpw vector-quantized model fails to form an
  attention sink on inputs that start with a digit-like token, which the objective cannot
  see and a fixed BOS/prefix at inference removes.

## Documents

| file | what it is |
|---|---|
| [`docs/PITCH.md`](docs/PITCH.md) | every claim graded by its evidence; the product verdict |
| [`docs/REVIEWER.md`](docs/REVIEWER.md) | the attack surface: where we would attack first, with the evidence files |
| [`docs/BRIEFING.md`](docs/BRIEFING.md) | self-contained technical summary for outside readers |
| [`docs/FROZEN_PROTOCOL.md`](docs/FROZEN_PROTOCOL.md) | the pre-registration, with addenda D1–D4 and their outcomes |
| [`docs/PRIOR_ART_LADDER.md`](docs/PRIOR_ART_LADDER.md) | BRECQ, MREM, YAQA, BaKron vs ours, module by module |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | every measurement, including the ones that killed our own hypotheses |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | the full trail in order, E1–E56 |
| [`evidence/`](evidence/README.md) | per-window NLLs and raw diagnostics behind every table; regenerate with the scripts below |

Earlier, closed directions (procedural tile representations, adaptive layout, graph-aware
grouping) are in `docs/FINAL_DIAGNOSTIC.md`, `docs/PROGRAMMATIC_LAYOUT.md` and
`docs/GRAPH_AWARE_QUANTIZATION.md`; their one-line verdicts are in `docs/FINDINGS.md` §8.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy transformers datasets safetensors tqdm accelerate
export PYTHONPATH=src

# one validation draw, all arms (fp16 / gptq / layer / module / block), ~2 h on a 6 GB GPU
python -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --calib-seed 0 \
       --fresh-g-seq 16 --out results/raw/valid_qwen05_s0.jsonl

# the tables, from the committed evidence (no GPU)
python scripts/analyze_valid.py --print        # four-model table, paired bootstraps
python scripts/analyze_rd.py                   # rate–distortion, equivalent bits
python scripts/analyze_d3.py                   # addendum D3: halves, C4 calibration, budgets
```

The pipeline is bit-deterministic: fp16 and GPTQ numbers reproduced exactly across
machines and allocator settings. Anything not reproducible from `evidence/` should be
treated as unsupported.

## Code map

```
src/lwc/
  gptq.py          GPTQ/OBS compensation, pluggable quantizer, full-group scale fit
  rotate.py        randomized Hadamard incoherence pair + consistent Hessian rotation
  joint.py         the stored quantization state; exact coordinate descent with
                   verify-and-backoff under a dense output metric; certified sphere-decoder oracle
  scopeg.py        G = E[JᵀJ] at a chosen horizon (module / block / full-model Fisher), in-pipeline
  attnscope.py     the intra-attention endpoints (logits / softmax / attention output), exact per-head blocks
  vq.py            2-D/4-D vector quantization with frozen amplitudes; exact codeword CD; per-block codebooks
  calib.py         calibration windows (wikitext-2 train, C4 train), seeded draws
  experiments/     valid_eval.py (frozen driver), joint_ppl.py (ladder), attn_decomp.py, rate_alloc.py, ...
scripts/           analyze_*.py, eval_external.py (evaluate an outside checkpoint), eval_prefix.py,
                   vq_external_refine.py (run the VQ post-pass on an outside .npz)
```

## Acknowledgements

Experiments were run with Claude (Anthropic) acting as an autonomous research agent under
the author's direction; every pre-registration, every result and every correction — of
which there were several, recorded as such — is in `EXPERIMENT_LOG.md`. The external
review that reset the prior-art positioning is answered point by point in
`docs/PRIOR_ART_LADDER.md`. The 2-bit decoder checkpoints in E55/E56 were produced by a
per-block VQ quantiser from the embedding-quantization project (d=4, K=256, ~2.1 bpw,
input-side rotation); the quantiser is not part of this release.
