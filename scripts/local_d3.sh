#!/usr/bin/env bash
# Addendum D3 runs on the shared local GPU (RTX 4050, 6 GB). Waits until no other process
# holds the card (the peer has priority), then runs the queue in order. Each stage writes
# its own jsonl; a stage that already has its DONE marker is skipped, so the script can be
# re-run after an interruption. Results are appended, never overwritten.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export OMP_NUM_THREADS=8
M=Qwen/Qwen2.5-0.5B
PY=${PY:-.venv/Scripts/python.exe}
OUT=results/raw
mkdir -p "$OUT"
LOG="$OUT/local_d3.log"

echo $$ > "$OUT/local_d3.pid"
# Reliable stop: `touch results/raw/STOP` (checked every poll and before every stage).
# Killing the wrapper from a harness is NOT reliable on Windows/Git Bash: on 2026-09-07 a
# "successful" task stop left this loop alive and it auto-started a stage 10 h later.
stop_requested() { [ -f "$OUT/STOP" ]; }

wait_for_gpu() {
  # free when no compute process other than ours is listed
  while true; do
    if stop_requested; then echo "$(date +%H:%M) STOP flag seen, exiting" >> "$LOG"; exit 0; fi
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    if [ "$n" -eq 0 ]; then return; fi
    echo "$(date +%H:%M) GPU busy ($n proc), waiting" >> "$LOG"
    sleep 300
  done
}

run_stage() {  # name, args...
  name=$1; shift
  if [ -f "$OUT/$name.DONE" ]; then echo "skip $name" >> "$LOG"; return; fi
  wait_for_gpu
  if stop_requested; then exit 0; fi
  echo "$(date +%H:%M) start $name" >> "$LOG"
  "$PY" -m lwc.experiments.valid_eval --model "$M" --calib-seed 0 --fresh-g-seq 16 \
      --out "$OUT/$name.jsonl" "$@" >> "$OUT/$name.log" 2>&1 \
    && touch "$OUT/$name.DONE"
  echo "$(date +%H:%M) end $name rc=$?" >> "$LOG"
}

# D3a: halves of the module arm (gptq/module for this draw exist in valid_qwen05_s0)
run_stage d3a_qwen05_halves_s0   --arms module_attn,module_mlp
# D3c: matched-domain calibration
run_stage d3c_qwen05_c4calib_s0  --arms gptq,module --calib-source c4
# D3b: full-model horizon at three estimation budgets
run_stage d3b_qwen05_model_p2_t2k_s0  --arms model --n-probe 2 --g-tokens 2048
run_stage d3b_qwen05_model_p8_t2k_s0  --arms model --n-probe 8 --g-tokens 2048
run_stage d3b_qwen05_model_p2_t8k_s0  --arms model --n-probe 2 --g-tokens 8192
echo "$(date +%H:%M) queue finished" >> "$LOG"
