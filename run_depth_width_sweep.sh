#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/mnt/data/initial_experiment_2d_vanderpol_sweep.py"
LOG_DIR="/mnt/data/sweep_logs"
OUT_ROOT="/mnt/data/sweep_outputs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# Edit these lists as needed.
depths=(1 2 3 4 5 6)
widths=(8 16 32 64 128 256)

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$gpu_count" -le 0 ]; then
  echo "No CUDA GPUs found via nvidia-smi." >&2
  exit 1
fi
mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader)

job=0
for d in "${depths[@]}"; do
  for w in "${widths[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$gpu_count" ]; do
      wait -n
    done

    gpu="${gpu_ids[$((job % gpu_count))]}"
    run_dir="$OUT_ROOT/depth_${d}_width_${w}"
    mkdir -p "$run_dir"

    echo "Launching depth=$d width=$w on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    N_HIDDEN="$d" \
    HIDDEN_WIDTH="$w" \
    OUTPUT_DIR="$run_dir/Plots" \
    python "$SCRIPT" > "$LOG_DIR/depth_${d}_width_${w}.out" 2> "$LOG_DIR/depth_${d}_width_${w}.err" &

    job=$((job + 1))
  done
done

wait
echo "All runs completed."
