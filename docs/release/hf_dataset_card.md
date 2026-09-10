---
license: mit
pretty_name: "llm-weight-compression evidence"
language:
  - en
tags:
  - quantization
  - gptq
  - post-training-quantization
  - pre-registered
  - evaluation
size_categories:
  - n<1K
---

# llm-weight-compression: evidence

Per-window evaluation data and raw diagnostics behind every number in
[rosecky/llm-weight-compression-public](https://github.com/rosecky/llm-weight-compression-public):
a pre-registered study of refining GPTQ's integer codes against the output of the enclosing
attention/MLP module instead of the single layer, on Qwen2.5-0.5B, Qwen3-0.6B,
Llama-3.2-1B and Qwen2.5-1.5B at exactly 3.25 bits per weight, plus rate–distortion,
estimation-budget and 2-bit vector-quantized points on Qwen2.5-0.5B.

## Layout

```
validation/    frozen-protocol runs: one .jsonl per (model, calibration draw) with
               per-window mean NLL for every arm on the full wikitext-2 test (146 windows
               of 2048 tokens) and 512 fixed C4 validation windows, plus a summary row per
               arm (bpw, KL to fp16, objective ratios, flips, timings, peak VRAM); the
               matching .log is the run's stdout
diagnostics/   exploratory-phase runs (objective ladder, superadditivity, VQ factorial),
               probe-budget rescoring, rotation landscape, rate-allocation oracle, oracle
               prediction features
box_scripts/   the exact shell scripts run on the rented GPU instance
README.md      file-by-file map and the pre-registration commit for each stage
```

Each `validation/*.jsonl` row is either `{"arm", "eval": "wt2"|"c4", "nlls": [...],
"nll": mean}` or a per-arm summary with `bpw`, `nll_wt2`, `nll_c4`, `kl_fp16`, `obj_ratio`,
`obj_fresh`, `flips`, `opt_s`, `peak_vram_mib`. Windows are in corpus order, so any two
arms of the same file are paired window by window.

## Regenerate the tables

```
git clone https://github.com/rosecky/llm-weight-compression-public
python scripts/analyze_valid.py --print    # four-model table, paired bootstraps within draw
python scripts/analyze_rd.py               # rate–distortion, equivalent bits
python scripts/analyze_d3.py               # halves, calibration domain, estimation budgets
```

with the analysis scripts pointed at this dataset's `validation/` directory. No GPU is
needed.

## Provenance

Method and protocol were frozen and committed before any validation run
(`docs/FROZEN_PROTOCOL.md` in the repository); diagnostic addenda were pre-registered with
interpretation tables before running. The 2-bit checkpoints evaluated in
`validation/p4_*` and `validation/d4_*` were produced by a per-block VQ quantiser from the
embedding-quantization project (d=4, K=256, ~2.1 bpw, input-side rotation) that is not part
of this release; the dequantised evaluations are.

Author: Jan Rosecký, September 2026. Experiments run with Claude (Anthropic) as an
autonomous research agent under the author's direction. License: MIT.
