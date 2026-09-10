#!/bin/bash
# D3: maximum-power test of the estimation hypothesis -- BOTH fixes at once.
# Attention side: exact per-head structure (cheap, already shown to be exact).
# MLP side: 8x probe budget, the only lever available where no exact structure exists.
while [ ! -f /root/lwc/results/D2B_DONE ]; do sleep 120; done
cd /root/lwc/repo_new || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export PYTHONPATH=src
P=/venv/main/bin/python
R=/root/lwc/results
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-1.5B --calib-seed 0 \
   --fresh-g-seq 16 --arms module_bd --g-tokens 16384 --n-probe 4 \
   --out $R/d3_qwen15_bd_hi.jsonl > $R/d3_qwen15_bd_hi.log 2>&1
touch $R/D3_DONE
