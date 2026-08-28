#!/bin/bash
#SBATCH --job-name=e4_laziness
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=jobs/e4/logs/%A_%a.out
mkdir -p jobs/e4/logs
CONFIGS=(jobs/e4/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
