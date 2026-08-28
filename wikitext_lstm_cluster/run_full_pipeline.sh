#!/bin/bash
# Self-contained pipeline for the WikiText-2 AWD-LSTM sensitivity/pruning
# experiment, sized to (roughly) match Merity et al. 2017's own AWD-LSTM
# config for WikiText-2 (3-layer LSTM, 1150 hidden units, 400-dim tied
# embedding, 750 epochs) -- see the defaults below. Sets up its own venv,
# installs dependencies, runs a smoke test, then the full pipeline:
#   train.py -> rank_stability.py -> analyze_distributions.py
#            -> pruning_experiment.py -> plot_pruning_story.py
#
# Usage:
#   bash run_full_pipeline.sh
#   EPOCHS=200 CHECKPOINT_INTERVAL=10 bash run_full_pipeline.sh   # override any setting
#
# All settings are plain environment variables (see Config in train.py for
# the full list) -- anything not set here falls back to train.py's own
# defaults, which are the *small*, fast, local-smoke-test-sized values, not
# these paper-scale ones. So don't run this script with no arguments if you
# only want a quick check -- use smoke_test.py directly for that (this
# script also runs it automatically as a first step, see below).
#
# Known deviations from the paper (documented, not accidental):
#   - Optimizer is plain Adam, not their NT-ASGD (non-monotonically
#     triggered averaged SGD) -- NT-ASGD isn't implemented in this
#     codebase. Their own ablation (Table 4) shows this costs ~5 perplexity
#     points on WikiText-2 (68.6->73.3 valid), i.e. real but not the
#     dominant factor.
#   - Fixed sequence length (SEQ_LENGTH) rather than their randomized
#     BPTT-length trick.
#   - No AR/TAR activation regularization (we only use the weight-dropped
#     LSTM / locked dropout / embedding dropout / weight tying pieces).
#
# If your cluster uses a scheduler (SLURM/PBS/etc.), wrap this script in
# whatever submission script your cluster requires (e.g. `sbatch` with a
# script that just calls `bash run_full_pipeline.sh`) -- this file itself
# makes no scheduler assumptions.

set -eo pipefail
cd "$(dirname "$0")"

echo "=== [0/6] Environment setup ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "nvidia-smi not found -- no CUDA GPU visible on this node?"
fi

if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"

# ---- experiment config (paper-scale AWD-LSTM on WikiText-2) ----
# DEVICE is left unset so train.py's select_device() auto-picks CUDA when
# available (falls back to CPU otherwise, e.g. for a dry run on a
# non-GPU login node) -- uncomment to force a specific device:
# export DEVICE=cuda

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-full_run_paper_scale}
export EMBEDDING_DIM=${EMBEDDING_DIM:-400}
export RNN_UNITS=${RNN_UNITS:-1150}
export NLAYERS=${NLAYERS:-3}
export DROPOUT=${DROPOUT:-0.4}
export DROPOUTH=${DROPOUTH:-0.3}
export DROPOUTI=${DROPOUTI:-0.65}   # paper bumps input dropout for WT2's larger vocab
export DROPOUTE=${DROPOUTE:-0.1}
export WDROP=${WDROP:-0.5}
export SEQ_LENGTH=${SEQ_LENGTH:-70}
export BATCH_SIZE=${BATCH_SIZE:-80}
export EPOCHS=${EPOCHS:-750}
export LR=${LR:-1e-3}
export GRAD_CLIP=${GRAD_CLIP:-1.0}
export CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-25}
export SENSITIVITY_INTERVAL=${SENSITIVITY_INTERVAL:-5}
export SENSITIVITY_PROBES=${SENSITIVITY_PROBES:-4}

KEEP_FRACTION=${KEEP_FRACTION:-0.2}
RANK_STABILITY_PROBES=${RANK_STABILITY_PROBES:-8}
PRUNING_PROBES=${PRUNING_PROBES:-8}

echo
echo "Config: EPOCHS=$EPOCHS EMBEDDING_DIM=$EMBEDDING_DIM RNN_UNITS=$RNN_UNITS NLAYERS=$NLAYERS"
echo "        BATCH_SIZE=$BATCH_SIZE SEQ_LENGTH=$SEQ_LENGTH CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL"
echo "        EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo

echo "=== [1/6] Smoke test (fail fast before committing to the full run) ==="
python3 smoke_test.py

echo "=== [2/6] Base training run ==="
python3 train.py

EXPERIMENT_DIR="outputs/$EXPERIMENT_NAME"

echo "=== [3/6] Rank stability analysis ==="
python3 rank_stability.py "$EXPERIMENT_DIR" --probes "$RANK_STABILITY_PROBES"

echo "=== [4/6] Distribution analysis ==="
python3 analyze_distributions.py "$EXPERIMENT_DIR" --probes "$RANK_STABILITY_PROBES"

echo "=== [5/6] Pruning experiment sweep ==="
# 7 branches evenly spaced across training, each rounded to the nearest
# saved checkpoint (a multiple of CHECKPOINT_INTERVAL) so
# pruning_experiment.py can actually load them.
PRUNE_EPOCHS=$(python3 -c "
epochs = $EPOCHS
interval = $CHECKPOINT_INTERVAL
fracs = [0, 1/6, 1/3, 1/2, 2/3, 5/6, 1]
pts = sorted(set(round(epochs * f / interval) * interval for f in fracs))
print(' '.join(str(p) for p in pts))
")
echo "Prune epochs: $PRUNE_EPOCHS"
python3 pruning_experiment.py "$EXPERIMENT_DIR" --keep-fraction "$KEEP_FRACTION" \
    --prune-epochs $PRUNE_EPOCHS --probes "$PRUNING_PROBES"

echo "=== [6/6] Composite plots (branching trajectory + loss-vs-correlation) ==="
PRUNING_DIR=$(find "$EXPERIMENT_DIR" -maxdepth 1 -type d -name 'pruning_keep*' | head -1)
python3 plot_pruning_story.py "$EXPERIMENT_DIR" --pruning-dir "$PRUNING_DIR" --probes "$RANK_STABILITY_PROBES"

echo "=== PIPELINE COMPLETE ==="
echo "Outputs are under: $EXPERIMENT_DIR"
