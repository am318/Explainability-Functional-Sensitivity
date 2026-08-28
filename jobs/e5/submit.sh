#!/bin/bash
#SBATCH --job-name=e5_failure_modes
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=jobs/e5/logs/%A_%a.out
mkdir -p jobs/e5/logs
CONFIGS=(jobs/e5/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
