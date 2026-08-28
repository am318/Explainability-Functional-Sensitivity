#!/bin/bash
#SBATCH --job-name=e9_coverage
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=jobs/e9/logs/%A_%a.out
mkdir -p jobs/e9/logs
CONFIGS=(jobs/e9/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
