# Frozen method & protocol (P0) — validation phase

*Effective from the commit whose message references this file. From this point on, no
parameter below may be tuned in response to results on any new model or evaluation set.
Changes to decision-making code are prohibited; changes to evaluation instrumentation are
permitted only if they provably do not alter any quantization decision (the decision
windows are drawn byte-identically regardless of eval additions).*

## Frozen method (the five arms)

| arm | definition |
|---|---|
| fp16 | unquantized reference |
| gptq | Hadamard + GPTQ, scalar INT3, group 128 |
| layer | gptq + 6 sweeps exact coordinate descent, layer objective |
| module | gptq + 6 sweeps layer-CD (`cd_pre`) + 6 sweeps CD under dense module-G, gdamp 1.0 |
| block | as module, endpoint = block output (`block1`) |

Frozen constants: GPTQ blocksize 128, percdamp 0.01, full-group scale fit (post-E27f
engine); asymmetric min-max INT3 g128 → exact 3.25 bpw = 3 + 2·16/128, plus 32 bits per
matrix for the rotation seed; randomized Hadamard rotation pair, seed 0, both sides, per
matrix; CD order forward, no refit, no ambiguity mask, no A-damping; G: dense per-matrix,
measured in-pipeline (upstream quantized, downstream fp), Rademacher probes, n_probe 2,
g_tokens 2048 sink cap, probe batch 4, trust-region damping 1.0 × mean diagonal; module
horizon endpoints = attention output (q,k,v) and MLP output (gate,up), o/down identity.

Calibration: wikitext-2 train, 32 windows × 512 tokens; the three independent draws use
`--calib-seed 0 / 1 / 2`; rotation and probe seeds stay 0. Model-family adaptations beyond
the tokenizer/model id are prohibited.

## Frozen evaluation

- Primary metric: **mean NLL per predicted token** (log-perplexity); PPL derived, secondary.
- Wikitext-2: the **full contiguous test split** at 2048-token windows (no sampling).
- C4: **512 pre-fixed windows** of 2048 tokens from allenai/c4 en validation, built once
  into `cache/c4_eval_512x2048.pt` (builder seed 17) and immutable thereafter.
- KL vs fp16: Monte-Carlo estimator E_{y~P_fp}[log P_fp(y) − log Q(y)], 8 samples/position,
  64 fixed windows (32 wt2 + 32 C4), sampling seed 0, anchored before quantization.
- Fresh-G verification: 16 extra calibration windows (seed = calib_seed + 1000), used only
  to re-score the wider objective after refinement (`obj_fresh`); decisions never see them.
- Per-window NLLs are logged so paired bootstrap over windows is computed offline.
- Exact physical bpw from the pipeline's own storage accounting (codes + scales/zeros +
  rotation seeds + codebooks where applicable).

## Pass criteria (pre-registered, from the P0–P8 brief)

- **P1 (Qwen2.5-0.5B audit):** module beats layer on wt2 AND C4, in ≥2 of 3 calibration
  draws, with a positive paired-bootstrap interval or a very stable effect sign. Failure on
  C4 or on fresh-G ⇒ fix methodology before any new model.
- **P2 (Qwen2.5-1.5B):** same, plus median ≥ ~5% of the NLL quantization gap closed.
- **P3 (Llama-3.2-1B):** same protocol; decides generality vs Qwen-specificity.
- **P4 (unified strong baseline):** on a strong VQ quantizer the module post-pass must add
  ≥ ~3% of the NLL gap.
- **P6 (rate–distortion):** ≥ ~0.1–0.2 bpw savings at equal quality, no inference overhead.

## Addendum D1 — post-hoc diagnostic after the P2 failure (pre-registered 2026-09-04)

*Written and committed BEFORE the runs. This is a diagnostic, not a rescue: it may not be
reported as a method result, and any rule derived from it must be re-frozen and validated on
a model that was not used to design it (Llama-3.2-1B) plus a no-regression re-check on
Qwen2.5-0.5B.*

**Motivation.** `G` is (out × out) but is estimated from a fixed 2048 token-gradient samples,
so its rank is capped at 2048 regardless of matrix width; the rest of the space is filled by
isotropic damping and is free for the optimizer to move into. Samples per output dimension:
0.5B MLP 0.42, 1.5B MLP 0.23 (attention 2.3 vs 1.33). The fixed constant is therefore a
hidden assumption about model size, and this is the documented failure family of the rank-64
surrogate (objective down, reality worse), in milder form.

**Runs.** The module arm only, calib-seed 0, everything identical to P2/P1 except the probe
budget raised 8x (`--g-tokens 16384 --n-probe 4`):

- **D1a** Qwen2.5-1.5B — the test.
- **D1b** Qwen2.5-0.5B — the control (was the small model already budget-saturated?).

**Pre-registered interpretation.**

| D1a | D1b | reading |
|---|---|---|
| gap closure turns positive | ~unchanged | H1 confirmed: the mechanism is real, the *budget rule* is not scale-invariant. Proceed to a principled rule (N ∝ out_features, or gdamp ∝ out/N), re-freeze, validate on Llama-3.2-1B + 0.5B no-regression. |
| stays negative | ~unchanged | H1 rejected. The module effect is likely specific to the small model (H4); the universal claim dies and only the 0.5B finding survives, as a scale-limited observation. |
| turns positive | also improves markedly | budget is the dominant variable everywhere; the original 0.5B result was itself budget-limited, and the whole scope story must be restated as a budget-vs-horizon frontier result. |

No other parameter may change in D1. If D1a is positive, the *fix* is not "use 16384": it is a
width-scaled rule whose constant is taken from the 0.5B regime, not fitted to 1.5B.

## Addendum D2 — endpoint algebra vs estimation noise (pre-registered 2026-09-04)

*Committed before the runs. Diagnostic, not a method claim.*

**Origin (important for the "no tuning on the validation model" rule):** the hypothesis comes
from the 0.5B intra-attention ladder measured long before P2 (pre-o_proj endpoint +3.8% of
the gap, post-o_proj −1.8% at identical refined matrices, E32c/E32d), not from any 1.5B
result. It was not fitted to the model that rejected the method.

**Claim.** Head h's pre-o output `Y_h = P_h V_h` depends only on head h's q/k/v, so `G` at
that endpoint is exactly block-diagonal. Measured: the dense estimator's on-block entries are
bit-identical to the block estimator's, while **18% of the dense estimate's energy (q_proj;
12% v_proj) sits in off-block entries that are exactly zero in truth** — quantified pure
estimation noise, in directions the discrete optimizer is free to exploit. Past o_proj heads
genuinely mix, so there the dense form is required and this noise cannot be removed.

**Arms** (all else frozen; `module` = current post-o dense + MLP):

| arm | attention endpoint | attention estimator | isolates |
|---|---|---|---|
| module | post-o | dense | (the P1/P2 baseline) |
| module_pre | pre-o | dense | endpoint change alone |
| module_bd | pre-o | exact per-head blocks | + removal of the measured noise |

**Runs.** calib-seed 0: 1.5B `module_pre` and `module_bd` (the failing model), 0.5B
`module_bd` (no-regression check).

**Pre-registered interpretation.**

| 1.5B module_bd | 0.5B module_bd | reading |
|---|---|---|
| gap closure positive | ≥ current 8.7% (no regression) | structure/noise is the mechanism; proceed to re-freeze and validate on Llama-3.2-1B before any claim |
| positive but 0.5B regresses | — | not a general improvement, a trade; report as such, no claim |
| still negative | — | endpoint algebra is not the explanation; H1 (budget, D1) and H4 (small-model artefact) remain |

If `module_pre` alone recovers most of the effect, the story is the endpoint, not the noise;
if `module_bd` adds substantially over `module_pre`, the noise is the operative part. Either
outcome must survive Llama-3.2-1B before it is called a method.

## Addendum D3 — three questions from the external review (pre-registered 2026-09-07)

Method unchanged. Evaluation driver gains three diagnostic arms (`module_attn`,
`module_mlp`, `model`) and a `--calib-source c4` switch; every row now records
`calib_source`, `n_probe`, `g_tokens`. All runs Qwen2.5-0.5B, calib-seed 0, 3.25 bpw, on
the shared local GPU when it is free (the rented instance is gone). The failing model
(Qwen2.5-1.5B) cannot be run locally and is deferred until a GPU with ≥12 GB is available.

**D3a — four-arm interaction with intervals.** Arms `module_attn` and `module_mlp` under the
full evaluation (per-window NLLs), alongside the existing `gptq` and `module` rows for the
same draw. Quantity: interaction = (module − gptq) − [(module_attn − gptq) + (module_mlp −
gptq)] per window, paired bootstrap. Exploratory values were −1.8% / +1.7% / +10.4% of the
gap on 24 windows.

| interaction CI | reading |
|---|---|
| strictly negative (superadditive) | the exploratory claim stands; report with its interval |
| covers 0 | "superadditive" is withdrawn; the halves are merely additive |
| strictly positive | halves are subadditive; the exploratory table was noise |

**D3b — estimator variance vs. text, for the full-model horizon.** Three `model` runs that
differ only in the metric's estimation budget: (i) `--n-probe 2 --g-tokens 2048` (the
standard budget), (ii) `--n-probe 8 --g-tokens 2048` (4× probes, same text), (iii)
`--n-probe 2 --g-tokens 8192` (same probes, 4× text from the *same* 16k calibration set).
Each row carries `kl_fp16` (the quantity the arm's objective actually approximates) as
well as NLL, and `obj_ratio`.

| (ii) vs (i) | (iii) vs (i) | reading |
|---|---|---|
| improves | ≈ same | probe variance is the operative error; the 8× result was about probes |
| ≈ same | improves | token coverage is operative; "noise-limited" should read "text-limited" |
| both improve similarly | | both matter; the frontier story survives but needs the factorization arm before it is a mechanism |
| neither improves | | the earlier 8× result is not reproducible at this protocol; the mechanism claim is withdrawn |

If KL improves while NLL does not, the objective/metric mismatch the review raised is the
explanation and the arm is not "noise-limited" at all.

*Outcome (2026-09-08, E52): (ii) +0.004 [−0.001, +0.009] ≈ same; (iii) −0.019 [−0.024,
−0.015], gap closed 0.3% → 5.5%, KL 0.359 → 0.349 moving with NLL. Row 2 of the table:
token coverage is operative, "noise-limited" reads "text-limited"; no objective/metric
mismatch.*

**D3c — matched-domain calibration.** `gptq` and `module` with `--calib-source c4`
(windows from the C4 *train* stream; evaluation stays on the pre-fixed C4 *validation*
windows and full wikitext-2 test). Compared with the wikitext-2-calibrated rows of the same
draw.

| C4-calibrated module − gptq on C4 eval | reading |
|---|---|
| more negative than the wt2-calibrated difference | the post-pass gains more than GPTQ from matched domain; the earlier "C4 is pessimistic" line was right for the wrong reason |
| about the same | calibration domain moves both arms alike; the C4 column is simply out-of-domain, as now stated |
| less negative / positive | the post-pass over-fits the calibration genre more than GPTQ; a new limitation to report |

Nothing in D3 changes a headline claim; each outcome above is reported as it lands.

## Run commands

```
python -m lwc.experiments.valid_eval --model <MODEL> --calib-seed {0,1,2} \
    --fresh-g-seq 16 --out results/raw/valid_<model>_s{seed}.jsonl
```

Addendum D3 (local, Qwen2.5-0.5B, seed 0): `scripts/local_d3.sh`.

**D3b-ext (pre-registered 2026-09-09, before running).** One run: `module_attn` (attention
half of the module arm) with `--g-tokens 8192 --n-probe 2`, otherwise identical to D3a.
Reference: `module_attn` at 2k tokens closed +0.6% of the wt2 gap (−0.002 [−0.007,
+0.002], D3a). Reading: if the 8k-token arm improves on that with a CI excluding zero, the
text-limitation found for the full-model metric (D3b) extends to the intra-module metric
and the module arm itself should be re-run at 8k tokens; if it stays ≈ 0, the attention
half is limited by something other than metric text coverage (endpoint or quadratic
model) and 2k tokens is not what holds it back.

*Outcome (2026-09-09, E53): +6.6% of the wt2 gap (−0.023 [−0.028, −0.019]), C4 from −1.6%
to +4.8%; first row of the reading applies — the module arm is to be re-run at 8k tokens.*

**D3b-ext2 (pre-registered 2026-09-09, not yet run).** One run: the full `module` arm with
`--g-tokens 8192 --n-probe 2`, Qwen2.5-0.5B, draw 0, otherwise frozen. Reference: +8.7%
(wt2) / +7.2% (C4) at 2k tokens. Reading: if it beats 8.7% with the paired 8k − 2k CI
excluding zero, `g_tokens 8192` becomes a candidate re-freeze under the D1 rule — constant
taken from the 0.5B regime, then validated unchanged (no further tuning) on Qwen3-0.6B,
Llama-3.2-1B and Qwen2.5-1.5B, draws 0–1, before any headline number changes. If it does
not beat 8.7%, the attention half's gain at 8k is absorbed by the MLP half's and the
frozen budget stands. Either way the validated table stays as published until the
re-validation completes.

*Outcome (2026-09-09, E54): +7.3% (wt2) / +4.4% (C4) vs +8.7% / +7.2% at 2k; 8k − 2k
paired +0.005 [+0.001, +0.009] and +0.010 [+0.008, +0.012] — worse on both. Second row of
the reading applies: the frozen budget stands, no re-freeze, no re-validation triggered.
E53's "lower bound" wording withdrawn.*

*P4 outcome (2026-09-09, E55): peer's per-block VQ at 2.12 bpw on Qwen2.5-0.5B; post-pass
inside their loop (6+6, 8k tokens): wt2 +0.027 [−0.038, +0.098] (−3.8% of the gap, n.s.),
C4 −0.067 [−0.088, −0.045] (+7.1%), KL −9.5%. Criterion (≥ ~3% on wt2) not met; better on
127/146 wt2 windows and 467/512 C4 windows, three wt2 windows +1.7 nats. One draw.*

## Addendum D4 — first-token (attention-sink) diagnostic for the 2-bit tail (pre-registered 2026-09-09, before running)

*Diagnostic, outside the frozen evaluation; changes no protocol number.* E55's ten worst
wt2 windows under the refined 2.12-bpw VQ checkpoint all begin with a digit-like fragment
("0 ,", "06", "20"), the ten best with ordinary continuations (", the", ". The"); on the
three worst, dropping the first token, prepending "\n\n", or swapping the first token for
a good window's brings the refined model from 5.2 to 3.3–3.4 nats — *below* plain (3.47).
Hypothesis: the refined 2-bit model fails to form an attention sink on some first tokens,
and the frozen evaluation (contiguous windows, no BOS — Qwen has none) exposes it.

**Run.** Both checkpoints (plain, refine), full wt2 test and 512 C4 windows, each window
= "\n\n" prefix + the window's first 2048 − |prefix| tokens (length kept at 2048), per-window
NLL over the same predicted positions as the frozen run minus the prefix.

| refine − plain with prefix (wt2) | reading |
|---|---|
| CI strictly negative and the +1 nat tail gone | the tail is a first-token/sink effect; P4's wt2 verdict stands as the frozen number, and the sink fragility is reported as the 2-bit failure mode with its fix (a fixed prefix / BOS at inference) |
| CI still covers 0 or tail persists | the tail is not (only) a first-token effect; the robustness failure stands as stated |

*Outcome (2026-09-09, E56): with the prefix, refine − plain wt2 −0.030 [−0.034, −0.025],
C4 −0.030 [−0.034, −0.026]; no window worse by >1 nat (max +0.05 / +0.33). First row of the
reading applies: the tail is a first-token/sink effect. P4's frozen verdict stands as the
frozen number.*
