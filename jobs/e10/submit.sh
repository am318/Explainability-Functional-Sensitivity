#!/bin/bash
#SBATCH --job-name=e10_lstm
#SBATCH --array=0-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=jobs/e10/logs/%A_%a.out
mkdir -p jobs/e10/logs
CONFIGS=(jobs/e10/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
