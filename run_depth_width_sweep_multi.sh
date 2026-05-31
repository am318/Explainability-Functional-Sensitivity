#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Datasets to sweep — must match keys in datasets.DATASET_REGISTRY.
# Available: parabola, power, sine, symmetric_vector_field,
#            morse_vector_field, vanderpol_vector_field, vanderpol_timeseries
# ============================================================
datasets=(
  "morse_vector_field"
  "symmetric_vector_field"
  "vanderpol_vector_field"
  "parabola"
  "power"
  "sine"
  "vanderpol_timeseries"
)

# The single entry-point script.
SWEEP_SCRIPT="Experimental_Sweep.py"

depths=(1 2 3 4 5 6)
widths=(8 16 32 64 128 256 512)

# GPU IDs as visible to CUDA. Example: (0 1 2 3)
gpus=(0 1 2 3)

# Shared overrides — any Config env-var can be added here.
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
  local dataset="$2"
  local depth="$3"
  local width="$4"

  # Output directories.
  local dataset_dir
  case "$dataset" in
    morse_vector_field)       dataset_dir="Plots/morse" ;;
    symmetric_vector_field)   dataset_dir="Plots/exp_test" ;;
    vanderpol_vector_field)   dataset_dir="Plots/vanderpol" ;;
    *)                        dataset_dir="Plots/${dataset}" ;;
  esac

  mkdir -p "$dataset_dir"
  local log_file="logs/${dataset}_d${depth}_w${width}_gpu${gpu}.log"

  echo "Launching dataset=${dataset} depth=${depth} width=${width} on GPU ${gpu} (slot ${slot})"
  CUDA_VISIBLE_DEVICES="$gpu" \
    DATASET="$dataset" \
    OUTPUT_DIR="$dataset_dir" \
    N_HIDDEN="$depth" \
    HIDDEN_WIDTH="$width" \
    python "$SWEEP_SCRIPT" >"$log_file" 2>&1 &
  pids[$slot]=$!
}

for dataset in "${datasets[@]}"; do
  echo "=== Starting sweep for dataset: ${dataset} ==="

  for depth in "${depths[@]}"; do
    for width in "${widths[@]}"; do
      slot=$(find_free_slot)
      launch_job "$slot" "$dataset" "$depth" "$width"
    done
  done

  # Drain all in-flight jobs for this dataset before moving to the next.
  echo "Waiting for all jobs from dataset '${dataset}' to finish..."
  for ((i=0; i<${#gpus[@]}; i++)); do
    if [[ -n "${pids[$i]}" ]]; then
      wait "${pids[$i]}" || true
      pids[$i]=""
    fi
  done

done

echo "All runs completed. Logs are in ./logs"