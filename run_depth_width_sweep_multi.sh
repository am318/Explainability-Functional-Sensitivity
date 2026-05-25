#!/usr/bin/env bash
set -euo pipefail

# Edit these lists to define the sweep.
deep_scripts=(
  "initial_experiment_2d_morse_sweep.py"
  "initial_experiment_2d_exp_test_sweep.py"
  "/initial_experiment_2d_vanderpol_sweep.py"
)

depths=(1 2 3 4 5 6)
widths=(16 32 64 128 256 512)
# GPU IDs as visible to CUDA. Example: (0 1 2 3)
gpus=(0 1 2 3)

# Optional overrides shared by all runs.
export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-100000}"

mkdir -p logs

pids=()
slot_busy=()
for ((i=0; i<${#gpus[@]}; i++)); do
  pids[$i]=""
  slot_busy[$i]=0
done

launch_job() {
  local slot="$1"
  local gpu="$2"
  local script="$3"
  local depth="$4"
  local width="$5"

  if [[ -n "${pids[$slot]}" ]]; then
    wait "${pids[$slot]}" || true
  fi

  local stem
  stem=$(basename "$script" .py)
  local log_file="logs/${stem}_d${depth}_w${width}_gpu${gpu}.log"

  echo "Launching ${stem} depth=${depth} width=${width} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    N_HIDDEN="$depth" \
    HIDDEN_WIDTH="$width" \
    python "$script" >"$log_file" 2>&1 &
  pids[$slot]=$!
}

for script in "${deep_scripts[@]}"; do
  for depth in "${depths[@]}"; do
    for width in "${widths[@]}"; do
      slot=$(( (depth + width) % ${#gpus[@]} ))
      gpu="${gpus[$slot]}"
      launch_job "$slot" "$gpu" "$script" "$depth" "$width"
    done
  done

  # Wait for all running jobs for this script before moving to the next one.
  for ((i=0; i<${#gpus[@]}; i++)); do
    if [[ -n "${pids[$i]}" ]]; then
      wait "${pids[$i]}" || true
      pids[$i]=""
    fi
  done
done

echo "All runs completed. Logs are in ./logs"
