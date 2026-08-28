#!/bin/bash
#SBATCH --job-name=e7b_vit_long
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=jobs/e7b/logs/%A_%a.out
mkdir -p jobs/e7b/logs
CONFIGS=(jobs/e7b/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
