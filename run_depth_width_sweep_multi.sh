#!/usr/bin/env bash
set -euo pipefail

# Edit these lists to define the sweep.
deep_scripts=(
  "initial_experiment_2d_morse_sweep.py"
  "initial_experiment_2d_exp_test_sweep.py"
  "initial_experiment_2d_vanderpol_sweep.py"
)

depths=(1 2 3 4 5 6)
widths=(8 16 32 64 128 256 512)
# GPU IDs as visible to CUDA. Example: (0 1 2 3)
gpus=(0 1 2 3)

# Optional overrides shared by all runs.
export SEED="${SEED:-0}"
export EPOCHS="${EPOCHS:-100000}"

mkdir -p logs

# One PID slot per GPU; empty string means the GPU is free.
pids=()
for ((i=0; i<${#gpus[@]}; i++)); do
  pids[$i]=""
done

# Block until a GPU slot is free, then return its index via stdout.
find_free_slot() {
  while true; do
    for ((i=0; i<${#gpus[@]}; i++)); do
      if [[ -z "${pids[$i]}" ]]; then
        echo "$i"
        return
      fi
      # If the PID has already exited, mark the slot free immediately.
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        wait "${pids[$i]}" || true
        pids[$i]=""
        echo "$i"
        return
      fi
    done
    sleep 1   # all GPUs busy — poll again shortly
  done
}

launch_job() {
  local slot="$1"
  local gpu="${gpus[$slot]}"
  local script="$2"
  local depth="$3"
  local width="$4"

  local stem
  stem=$(basename "$script" .py)

  local dataset_dir
  # case "$stem" in
  #   initial_experiment_2d_exp_test_sweep) dataset_dir="Plots/exp_test" ;;
  #   initial_experiment_2d_morse_sweep)    dataset_dir="Plots/morse" ;;
  #   initial_experiment_2d_vanderpol_sweep) dataset_dir="Plots/vanderpol" ;;
  #   *) dataset_dir="Plots/${stem}" ;;
  # esac
  case "$stem" in
    initial_experiment_2d_exp_test_sweep) dataset_dir="L1Plots/exp_test" ;;
    initial_experiment_2d_morse_sweep)    dataset_dir="L1Plots/morse" ;;
    initial_experiment_2d_vanderpol_sweep) dataset_dir="L1Plots/vanderpol" ;;
    *) dataset_dir="Plots/${stem}" ;;
  esac

  mkdir -p "$dataset_dir"
  local log_file="logs/${stem}_d${depth}_w${width}_gpu${gpu}.log"

  echo "Launching ${stem} depth=${depth} width=${width} on GPU ${gpu} (slot ${slot})"
  CUDA_VISIBLE_DEVICES="$gpu" \
    OUTPUT_DIR="$dataset_dir" \
    N_HIDDEN="$depth" \
    HIDDEN_WIDTH="$width" \
    python "$script" >"$log_file" 2>&1 &
  pids[$slot]=$!
}

for script in "${deep_scripts[@]}"; do
  echo "=== Starting sweep for $(basename "$script") ==="

  for depth in "${depths[@]}"; do
    for width in "${widths[@]}"; do
      slot=$(find_free_slot)
      launch_job "$slot" "$script" "$depth" "$width"
    done
  done

  # Drain all in-flight jobs for this script before moving to the next.
  echo "Waiting for all jobs from $(basename "$script") to finish..."
  for ((i=0; i<${#gpus[@]}; i++)); do
    if [[ -n "${pids[$i]}" ]]; then
      wait "${pids[$i]}" || true
      pids[$i]=""
    fi
  done

done

echo "All runs completed. Logs are in ./logs"