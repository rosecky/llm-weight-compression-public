#!/bin/bash
cd /root/lwc/repo_new || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python
R=/root/lwc/results
$P -u -m lwc.experiments.valid_eval --model unsloth/Llama-3.2-1B --calib-seed 0 \
   --fresh-g-seq 16 --arms gptq,layer,module --out $R/p3_llama1b.jsonl > $R/p3_llama1b.log 2>&1
touch $R/P3_DONE
