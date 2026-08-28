#!/bin/bash
#SBATCH --job-name=e8_long_all
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --output=jobs/e8/logs/%A_%a.out
mkdir -p jobs/e8/logs
CONFIGS=(jobs/e8/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
