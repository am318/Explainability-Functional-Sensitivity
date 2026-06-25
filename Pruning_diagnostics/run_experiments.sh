#!/usr/bin/env bash
set -euo pipefail

python Pruning_Sensitivity_ViT_Tiny.py

DATASET="CIFAR100" python ViT_Sensitivity_Pruning_Experiment.py

DATASET="CIFAR100" python ViT_SYNFLOW_Pruning_Experiment.py
