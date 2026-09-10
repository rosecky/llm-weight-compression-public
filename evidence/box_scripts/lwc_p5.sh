#!/bin/bash
cd /root/lwc/repo_new || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/venv/main/bin/python -u -m lwc.experiments.valid_eval --model Qwen/Qwen2.5-1.5B \
  --calib-seed 1 --arms gptq,module --out /root/lwc/results/valid_qwen15_s1b.jsonl \
  > /root/lwc/results/valid_qwen15_s1b.log 2>&1
touch /root/lwc/results/P5_DONE
