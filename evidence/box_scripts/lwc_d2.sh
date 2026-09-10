#!/bin/bash
cd /root/lwc/repo_new || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export PYTHONPATH=src
P=/venv/main/bin/python
R=/root/lwc/results
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-1.5B --calib-seed 0 \
   --fresh-g-seq 16 --arms module_bd --out $R/d2_qwen15_bd.jsonl > $R/d2_qwen15_bd.log 2>&1
touch $R/D2A_DONE
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-0.5B --calib-seed 0 \
   --fresh-g-seq 16 --arms module_bd --out $R/d2_qwen05_bd.jsonl > $R/d2_qwen05_bd.log 2>&1
touch $R/D2B_DONE
$P -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-1.5B --calib-seed 0 \
   --fresh-g-seq 16 --arms module_pre --out $R/d2_qwen15_pre.jsonl > $R/d2_qwen15_pre.log 2>&1
touch $R/D2_DONE
