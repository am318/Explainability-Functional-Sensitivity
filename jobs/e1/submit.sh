#!/bin/bash
#SBATCH --job-name=e1_rank_stability
#SBATCH --array=0-21
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=jobs/e1/logs/%A_%a.out
mkdir -p jobs/e1/logs
CONFIGS=(jobs/e1/cfg_*.json)
python -m fsd.cli --config "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
