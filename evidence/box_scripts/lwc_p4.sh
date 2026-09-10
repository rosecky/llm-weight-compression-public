#!/bin/bash
# P4: Llama-3.2-3B -- matches the failing Qwen2.5-1.5B on BOTH untested candidate features
# (28 layers, head_dim 128) while differing in family and size. Frozen protocol, no tuning.
cd /root/lwc/repo_new || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python
R=/root/lwc/results
for s in 0 1; do
  $P -u -m lwc.experiments.valid_eval --model unsloth/Llama-3.2-3B --calib-seed $s \
     --fresh-g-seq 16 --arms gptq,module --out $R/p4_llama3b_s$s.jsonl > $R/p4_llama3b_s$s.log 2>&1 || break
done
touch $R/P4_DONE
