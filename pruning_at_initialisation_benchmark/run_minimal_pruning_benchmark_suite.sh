#!/usr/bin/env bash
set -euo pipefail

# Minimal reviewer-facing suite:
#   ResNet-20/CIFAR-10 sanity check
#   ViT-Tiny/CIFAR-10 main transformer result
#   NanoGPT/Tiny-Shakespeare character LM language-model result
#
# Override EPOCHS/MAX_ITERS/BATCH_SIZE/LR from the shell as needed.

SPARSITY=${SPARSITY:-0.95,0.98,0.99}
METHODS=${METHODS:-all}
SEEDS=${SEEDS:-0}
OUT=${OUTPUT_DIR:-PruningBenchResults}

for seed in ${SEEDS}; do
  ARCH=resnet20 DATASET=cifar10 PRUNING_METHOD=${METHODS} SPARSITY=${SPARSITY} SEED=${seed} OUTPUT_DIR=${OUT} \
    EPOCHS=${EPOCHS_RESNET:-100} LR=${LR_RESNET:-0.001} BATCH_SIZE=${BATCH_SIZE:-128} \
    python Transformer_Prune_At_Init_Benchmark.py

  ARCH=vit_tiny DATASET=cifar10 PRUNING_METHOD=${METHODS} SPARSITY=${SPARSITY} SEED=${seed} OUTPUT_DIR=${OUT} \
    EPOCHS=${EPOCHS_VIT:-100} LR=${LR_VIT:-0.0003} BATCH_SIZE=${BATCH_SIZE:-128} \
    python Transformer_Prune_At_Init_Benchmark.py

  ARCH=nanogpt DATASET=text PRUNING_METHOD=${METHODS} SPARSITY=${SPARSITY} SEED=${seed} OUTPUT_DIR=${OUT} \
    MAX_ITERS=${MAX_ITERS_GPT:-5000} LR=${LR_GPT:-0.0003} GPT_BATCH_SIZE=${GPT_BATCH_SIZE:-64} \
    python Transformer_Prune_At_Init_Benchmark.py
done
