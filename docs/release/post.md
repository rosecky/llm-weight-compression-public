# We tried to make GPTQ optimise the right thing. Here is what survived.

Jan Rosecký, September 2026. Full report: [report](./report.md). Code and evidence:
[github.com/rosecky/llm-weight-compression-public](https://github.com/rosecky/llm-weight-compression-public).

GPTQ rounds each weight matrix so that *that matrix's* output changes as little as
possible. It is a proxy, and everyone knows it. So we asked a narrow question: keep the
solver, the bits, the file format and the calibration data exactly as they are, and only
change what the objective looks at — the single layer, the attention logits, the attention
output, the whole attention/MLP module, the transformer block, or the model's own loss.
Which target makes the quantized model better where it counts, on held-out text?

We froze the protocol before running, reported every calibration draw, and asked an
outside reviewer to break it. Here is the short version.

**What works.** Refining GPTQ's integer codes against the error at the *module* output,
block by block inside the same sequential pass, recovers 5–9% of the quantization loss at
3.25 bits per weight on three of four small decoders (Qwen2.5-0.5B, Qwen3-0.6B,
Llama-3.2-1B): 2–4% lower perplexity, identical file, identical kernel, no inference cost.
On the fourth (Qwen2.5-1.5B) it is slightly harmful, replicated across draws, and nothing
we measured predicts it.

**What it is worth.** Along the rate–distortion curve the gain is at most 0.08 bits per
weight at equal quality. We had set 0.1–0.2 as the bar before measuring. It is a
scientific result, not a product.

**What is ours and what is not.** The reviewer pointed us at BRECQ (2021), MREM (2022) and
BaKron (2026): the non-monotone objective ladder and its explanation are theirs, and
BaKron has our MLP-output objective in closed form. We rewrote the positioning. What
remains ours: the rungs *inside* attention (the logit objective BoA/BaKron use is the worst
one); the finding that the global objective is limited by how much calibration text its
metric sees, not by probe noise (4× tokens: 0.3% → 5.5% of the gap; 4× probes: nothing);
superadditivity of the attention and MLP halves with a confidence interval; and a set of
negatives with their scope stated — layer-local coordinate descent is mildly harmful
end-to-end, the rotation landscape is flat, rate allocation is worth ≤0.11 bits.

**What broke at 2 bits.** On a 2.1-bit vector-quantized checkpoint the post-pass helped
on 127 of 146 test windows and destroyed ten. Every one of the ten starts with a
digit-like token. The refined model fails to form an attention sink on those first tokens,
the objective cannot see it because calibration windows start mid-text, and a fixed
`"\n\n"` prefix at inference makes the problem disappear and the post-pass win on both
test sets. If you ship a 2-bit model, inputs that begin with a number will find this for
you.

**What we would tell a colleague.** Keep the paired bootstrap, but run the calibration
draws too: a window-paired interval does not contain draw variance, and two equally good
quantisations of the same model differ by more than the interval "confirms". Audit an
estimated objective at a higher resolution, not only on fresh data: a same-budget fresh
check passed on the model where the method fails. And write the kill rule before the run.

Every pre-registration, result and correction — there were several — is in the experiment
log. The experiments were run with Claude (Anthropic) as an autonomous research agent under
my direction.
