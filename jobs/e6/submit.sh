#!/bin/bash
#SBATCH --job-name=e6_same_init
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=jobs/e6/logs/%A_%a.out
mkdir -p jobs/e6/logs
CONFIGS=(jobs/e6/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
