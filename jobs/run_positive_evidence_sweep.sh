#!/usr/bin/env bash
# Positive-evidence sweep: experiments specifically designed to test whether early
# functional sensitivity predicts which parameters matter in the fully-trained network.
#
# Run with:  bash jobs/run_positive_evidence_sweep.sh
#
# 4 GPUs hardwired (this node). Distinct from jobs/run_cluster_sweep.sh's TIER=1, which
# answers "does the ordering settle" -- this script targets the complementary, more
# forgiving question: not exact rank, but calibration (importance BAND) and AUROC (binary
# top-k separability), which is what pruning actually needs and what the free checks in
# analysis/early_window.py found genuinely positive evidence for (AUROC up to 0.82 at
# sp=0.99 from S_0 alone; AUROC >0.9 by ~5-8% of training).
#
# What this runs:
#   1. Long, keep_scores=all runs across every architecture (so analysis/early_window.py's
#      time x sparsity report -- currently only possible for ResNet-20 and ViT, the two
#      runs that happened to save full trajectories -- can be produced for MLP, GPT, and
#      LSTM too).
#   2. The reproducibility-ceiling pairs (E12), needed to interpret any resemblance number
#      against what is actually knowable rather than against an uninformative target of 1.0.
#   3. The range report itself (analysis.early_window), across every run this produces.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

N_GPUS=4   # hardwired: this node has 4 GPUs. Edit this line directly to change it.
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
OUT="results/POSITIVE_EVIDENCE_REPORT.md"
LOG_DIR="results/_positive_evidence_logs"
mkdir -p "$LOG_DIR"

PY="./venv/bin/python"
[ -x "$PY" ] || PY="python"

echo "=== positive-evidence sweep: $N_GPUS GPUs, seed=$SEED ==="
date

TASKS=()
add_task() { TASKS+=("$1"); }

# ---- 1. keep_scores=all long runs, every architecture --------------------------------
# One seed each is enough for the range report; E1's cluster configs already run 3 seeds
# per architecture for the main stability claims, this only needs ONE with full score
# retention. Reuses experiments/_common.py::base() so steps/schedule match the rest of
# the project (12000 steps, constant LR by default there).
for arch in mlp resnet20 vit gpt lstm; do
  add_task "$PY -c \"
import sys; sys.path.insert(0, '.')
from experiments._common import base
from fsd import config as C
from fsd.run import execute
cfg = base('posev-$arch', '$arch' if '$arch' != 'lstm' else 'gpt', steps=12000, sens_samples=2048)
if '$arch' == 'lstm':
    cfg.model = C.ModelCfg(arch='lstm', width=256, depth=1, block_size=64)
cfg.seed = $SEED
cfg.train.lr_schedule = 'constant'
cfg.n_ckpts = 26
cfg.keep_scores = 'all'
cfg.track_criteria = False
execute(cfg)
\""
done

# ---- 2. reproducibility ceiling (E12), if not already emitted ------------------------
if [ -d jobs/e12 ]; then
  for cfg in jobs/e12/cfg_*.json; do
    add_task "$PY -m fsd.cli --config $cfg"
  done
fi

N="${#TASKS[@]}"
echo "queued $N tasks across $N_GPUS GPU workers"
if [ "$N" -eq 0 ]; then
  echo "nothing to run"; exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "--- DRY_RUN=1: task queue only ---"
  for ((i=0; i<N; i++)); do
    printf "  [gpu %d] %s\n" "$((i % N_GPUS))" "${TASKS[$i]:0:100}..."
  done
  exit 0
fi

pids=()
for ((gpu=0; gpu<N_GPUS; gpu++)); do
  (
    worker_log="$LOG_DIR/gpu${gpu}.log"
    : > "$worker_log"
    for ((i=gpu; i<N; i+=N_GPUS)); do
      echo "[gpu $gpu] task $((i+1))/$N" | tee -a "$worker_log"
      CUDA_VISIBLE_DEVICES="$gpu" bash -c "${TASKS[$i]}" >> "$worker_log" 2>&1 \
        || echo "[gpu $gpu] FAILED (see $worker_log)" | tee -a "$worker_log"
    done
    echo "[gpu $gpu] worker done" | tee -a "$worker_log"
  ) &
  pids+=($!)
done
echo "waiting on ${#pids[@]} workers -- tail -f $LOG_DIR/gpu*.log to watch"
wait "${pids[@]}"
date
echo "=== all workers finished ==="

# ---- 3. the range report itself, across every run this produced ----------------------
RUN_ARGS=""
for d in results/posev-*; do
  [ -d "$d/scores" ] && RUN_ARGS="$RUN_ARGS --run $d"
done
{
  echo "# Positive-evidence report: AUROC / rho / calibration across time x sparsity"
  echo "_generated $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
  echo
  echo '```'
  if [ -n "$RUN_ARGS" ]; then
    $PY -m analysis.early_window $RUN_ARGS --max-frac 0.5 2>&1
  else
    echo "no runs with keep_scores=all completed -- check $LOG_DIR/gpu*.log"
  fi
  echo '```'
  echo
  echo "## Reproducibility ceiling (E12)"
  echo '```'
  $PY -m analysis.ceiling --tag e12 2>&1 || true
  echo '```'
} > "$OUT"
echo "readable report written to $OUT"
