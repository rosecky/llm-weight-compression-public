#!/bin/bash
# Overnight: (1) rate-distortion curve on the best-case model, to convert our effect into
# equivalent bits saved -- the only commercially meaningful unit; (2) Llama-3.2-3B, the
# largest model we will have tested and the only one above the failure point on a family
# that works.
cd /root/lwc/repo_rd || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python
R=/root/lwc/results
# (1) R-D curve: 2.50, 3.50, 4.25 bpw (3.25 already measured), Qwen2.5-0.5B
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --bits 2 --group 64 \
   --arms gptq,module --out $R/rd_q05_b2g64.jsonl  > $R/rd_q05_b2g64.log 2>&1
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --bits 3 --group 64 \
   --arms gptq,module --out $R/rd_q05_b3g64.jsonl  > $R/rd_q05_b3g64.log 2>&1
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --bits 4 --group 128 \
   --arms gptq,module --out $R/rd_q05_b4g128.jsonl > $R/rd_q05_b4g128.log 2>&1
touch $R/RD_DONE
# (2) the scaling point
$P -u -m lwc.experiments.valid_eval --model unsloth/Llama-3.2-3B --calib-seed 0 \
   --arms gptq,module --out $R/p6_llama3b_s0.jsonl > $R/p6_llama3b_s0.log 2>&1
touch $R/P6_DONE
