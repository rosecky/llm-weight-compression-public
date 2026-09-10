#!/bin/bash
cd /root/lwc/repo_new || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python
R=/root/lwc/results
for s in 1 2; do
  $P -u -m lwc.experiments.valid_eval --model unsloth/Llama-3.2-1B --calib-seed $s \
     --fresh-g-seq 16 --arms gptq,module --out $R/p3_llama1b_s$s.jsonl > $R/p3_llama1b_s$s.log 2>&1 || break
done
touch $R/P3B_DONE
