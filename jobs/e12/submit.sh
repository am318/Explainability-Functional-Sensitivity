#!/bin/bash
#SBATCH --job-name=e12_ceiling
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=jobs/e12/logs/%A_%a.out
mkdir -p jobs/e12/logs
CONFIGS=(jobs/e12/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
