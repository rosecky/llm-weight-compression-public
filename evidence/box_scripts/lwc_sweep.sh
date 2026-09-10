#!/bin/bash
cd /root/lwc/repo_new || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python
R=/root/lwc/results
echo "=== Qwen2.5-1.5B layer 13 ==="
$P -u noise_probe.py --model Qwen/Qwen2.5-1.5B --layer 13 --out $R/probe_15b.jsonl 2>&1 | grep -avE 'Loading|Warning|it/s|longer than'
echo "=== Qwen2.5-0.5B layer 11 (control: the size where the method works) ==="
$P -u noise_probe.py --model Qwen/Qwen2.5-0.5B --layer 11 --out $R/probe_05b.jsonl 2>&1 | grep -avE 'Loading|Warning|it/s|longer than'
touch $R/SWEEP_DONE
