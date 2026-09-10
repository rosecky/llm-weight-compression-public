# Evidence

The raw measurements behind every claim in `docs/PITCH.md`, `docs/BRIEFING.md` and
`docs/FINDINGS.md`. `results/raw/` is scratch and git-ignored; this directory is the
committed record, because most of it was produced on a rented GPU instance that no longer
exists.

Regenerate the tables with `python scripts/analyze_valid.py --print` (point `RAW` at
`evidence/validation`), `python scripts/analyze_rd.py` for the rate–distortion curve, `python scripts/analyze_d3.py`
for addendum D3, and
`python scripts/analyze_*.py` for the rest.

## validation/

Frozen-protocol runs (method and hyperparameters committed at `b57c9ac` before any of these
ran; see `docs/FROZEN_PROTOCOL.md`). These back the four-model headline table.

| file | model | draws | arms |
|---|---|---|---|
| `valid_qwen05_s{0,1,2}` | Qwen2.5-0.5B | 3 | fp16, gptq, layer, module, block |
| `valid_qwen3_06b_s{0,1}` | Qwen3-0.6B | 2 | fp16, gptq, module |
| `p3_llama1b`, `p3_llama1b_s{1,2}` | Llama-3.2-1B | 3 | fp16, gptq, (layer), module |
| `valid_qwen15_s{0,1}`, `valid_qwen15_s1b` | Qwen2.5-1.5B | 2 | fp16, gptq, layer, module, block |
| `rd_q05_b2g64`, `rd_q05_b3g64`, `rd_q05_b4g128` | Qwen2.5-0.5B, draw 0, at 2.50 / 3.50 / 4.25 bpw | 1 | fp16, gptq, module |
| `p6_llama3b_s0` | Llama-3.2-3B | 1 | fp16, gptq only (module arm lost, see below) |
| `d3a_qwen05_halves_s0` | Qwen2.5-0.5B, draw 0 | 1 | fp16, module_attn, module_mlp (D3a interaction; local RTX 4050) |
| `d3c_qwen05_c4calib_s0` | Qwen2.5-0.5B, draw 0, **C4-train calibration** | 1 | fp16, gptq, module (D3c) |
| `d3b_qwen05_model_{p2_t2k,p8_t2k,p2_t8k}_s0` | Qwen2.5-0.5B, draw 0 | 1 each | fp16, `model` (full-model Fisher) at 2 probes/2k tokens, 8 probes/2k, 2 probes/8k (D3b) |
| `d3bx_qwen05_attn_t8k_s0` | Qwen2.5-0.5B, draw 0 | 1 | fp16, `module_attn` at 2 probes/8k tokens (D3b-ext; compare `d3a_*` at 2k) |
| `d3bx2_qwen05_module_t8k_s0` | Qwen2.5-0.5B, draw 0 | 1 | fp16, `module` at 2 probes/8k tokens (D3b-ext2; compare `valid_qwen05_s0` module at 2k — worse) |
| `p4_qwen05_vq20_{plain,refine}` | Qwen2.5-0.5B, draw 0, **peer's per-block VQ at 2.12 bpw** | 1 each | fp16 + the external checkpoint evaluated as-is (`scripts/eval_external.py`); `refine` = same quantisation + our sequential post-pass (6+6, 8k metric tokens) inside the peer's loop (P4) |
| `d4_qwen05_vq20_prefix` | same two checkpoints | 1 | both evaluated with a fixed "

" prefix on every window (`scripts/eval_prefix.py`, addendum D4): the first-token tail disappears, refine − plain −0.030 on both sets |

Each `.jsonl` holds one record per (arm, eval set) with **per-window NLLs**, so the paired
bootstrap is reproducible offline without re-running anything. `valid_qwen15_s1` is the run
that died in a shared-GPU OOM; `valid_qwen15_s1b` is its completed replacement.

The `rd_*` files are the rate–distortion sweep (P6): the frozen method at three more exact
rates, evaluated with the same driver (`--bits`/`--group` overrides only). Together with
`valid_qwen05_s0` (3.25 bpw) they give the four-point curve;
`python scripts/analyze_rd.py` bootstraps each point. `p6_llama3b_s0` holds the fp16 and
GPTQ references for Llama-3.2-3B; its module arm was at layer 8/27 after 2.2 h when the
instance was shut down (`monitor_rd_and_llama3b.log`, `p6_llama3b_s0.log`), so nothing
above 1.5B is tested. All of these were recovered from the instance disk after a brief
power-on; `E47` in `EXPERIMENT_LOG.md` records the near-loss.

## diagnostics/

Measurements behind the mechanism claims and the negative results:

- `probe_05b`, `probe_15b`, `noise_probe`, `sweep.log` — rescoring the same weight change
  under 4×–16× better-estimated metrics. Shows the "improvement" on the widest matrices is
  illusory **on the model where the method works** as well as where it fails.
- `d2_*` — the block-diagonal / pre-o_proj endpoint variants that failed on both models.
- `attn_decomp` — the intra-attention objective ladder and the sensitivity correlations.
- `joint_ppl` — the exploratory-phase end-to-end runs (radius curve, superadditivity,
  half-depth split, VQ factorial). Lighter 24-window protocol; not comparable to
  `validation/`.
- `rate_alloc` — the rate-allocation oracle that killed that branch (≤0.11 equivalent bpw).
- `transform_plateau` — tangent, basin, DFO and 200-rotation sweep of the rotation landscape.
- `gptq_scalefeed` — the scale-source ablation (frozen vs working-weight scales).
- `oracle_pred` — per-row features vs the CD→exact-search gap, 1440 rows.
- `calib_size`, `comp_capacity`, `joint_oracle_k` — calibration budget sweep, compensation
  capacity ladder, and the k-move reconciliation with CDQuant.

## box_scripts/

The exact shell scripts run on the rented instance, kept so the provenance of each result
file is auditable.
