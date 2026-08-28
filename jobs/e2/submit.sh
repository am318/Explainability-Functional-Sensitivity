#!/bin/bash
#SBATCH --job-name=e2_tstar_scaling
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=jobs/e2/logs/%A_%a.out
mkdir -p jobs/e2/logs
CONFIGS=(jobs/e2/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
