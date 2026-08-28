#!/bin/bash
#SBATCH --job-name=e7_convergence
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=jobs/e7/logs/%A_%a.out
mkdir -p jobs/e7/logs
CONFIGS=(jobs/e7/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
