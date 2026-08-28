#!/usr/bin/env bash
# Full experiment sweep on a single multi-GPU node (default: 4 GPUs).
#
# Usage:
#   jobs/run_cluster_sweep.sh                      # 4 GPUs, everything
#   N_GPUS=8 jobs/run_cluster_sweep.sh              # override GPU count
#   ARCHS=resnet20,vit jobs/run_cluster_sweep.sh    # restrict grid architectures
#   TIER=1 jobs/run_cluster_sweep.sh                # only the two things that matter
#
# Design: no SLURM assumed. GPUs are addressed by CUDA_VISIBLE_DEVICES=<index>, which
# `fsd/storage.py::pick_device("auto")` picks up with zero code changes -- setting the env
# var before a process starts makes torch see only that one GPU as cuda:0. If you DO have
# SLURM, `jobs/e*/submit.sh` (already emitted by each experiment's --emit flag) are the
# array-job entry points instead; this script is for a bare node.
#
# Tiers (see TIER env var):
#   1 = the two things that matter: E1 (rank-stability long runs) + E14 (prune grid,
#       FULL 4x4x3 resolution, every architecture) + E12 (reproducibility ceiling) +
#       the matched dense controls. This is what the paper's two headline tables need.
#   2 = everything else already built (E2 t* scaling, E4 laziness, E5 failure modes,
#       E6 same-init, E7/E7b convergence, E8 long-all, E9 coverage, E10 LSTM). Useful for
#       filling in appendix material and robustness checks, not required for the headline.
# TIER=all (default) runs both. TIER=1 restricts to the two headline results.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DRY_RUN="${DRY_RUN:-0}"   # DRY_RUN=1: build and print the task queue, run nothing

N_GPUS=4   # hardwired: this node has 4 GPUs. Edit this line directly to change it.
ARCHS="${ARCHS:-mlp,resnet20,vit}"
SEED="${SEED:-0}"
TIER="${TIER:-all}"
OUT_ROOT="results"
PROBE_DIR="results/_probe_cluster"
LOG_DIR="results/_cluster_logs"
mkdir -p "$PROBE_DIR" "$LOG_DIR"

PY="./venv/bin/python"
[ -x "$PY" ] || PY="python"

echo "=== cluster sweep: $N_GPUS GPUs, tier=$TIER, archs=$ARCHS, seed=$SEED ==="
echo "logs: $LOG_DIR/   readable report: results/REPORT.md (written at the end)"
date

# --------------------------------------------------------------------------------------
# A simple 4-way (or N-way) job queue: every "task" below is a single shell command.
# Tasks are appended to an array; at the end we fan them out round-robin across N_GPUS
# background workers, each pinned to one GPU via CUDA_VISIBLE_DEVICES, and `wait` for all.
# This is deliberately dumb (no work-stealing) -- if task lengths are very uneven, hand-sort
# TASKS so the longest ones go first, since round-robin assignment is by ARRIVAL order.
# --------------------------------------------------------------------------------------
TASKS=()

add_task() { TASKS+=("$1"); }

# ---- TIER 1: the two headline results ----------------------------------------------
if [ "$TIER" = "1" ] || [ "$TIER" = "all" ]; then
  # A) S_t vs S_T resemblance: E1's long-run rank-stability sweep (both LR schedules,
  #    3 architectures, 3 seeds, +CIFAR-100 +GPT -- see experiments/e1_rank_stability.py).
  #    Uses the pre-emitted per-run configs so results/e1-*/metrics.json land exactly
  #    where analysis/figures.py and analysis/claims.py already expect them.
  if [ -d jobs/e1 ]; then
    for cfg in jobs/e1/cfg_*.json; do
      add_task "$PY -m fsd.cli --config $cfg"
    done
  fi

  # B) Reproducibility ceiling (E12): same-init, different-data-order pairs.
  if [ -d jobs/e12 ]; then
    for cfg in jobs/e12/cfg_*.json; do
      add_task "$PY -m fsd.cli --config $cfg"
    done
  else
    add_task "$PY -m experiments.e12_ceiling --run --skip-done"
  fi

  # C) Matched dense controls -- one number per architecture, needed to interpret B's grid.
  add_task "$PY experiments/e11_dense_control.py --archs $ARCHS --seed $SEED"

  # E) Positive-evidence range report: AUROC/rho/calibration across (time x sparsity),
  #    for every run in tier 1 that saves raw per-checkpoint scores. Cheap (reads results
  #    already on disk); scheduled after A-D so their outputs exist first, but since it is
  #    just a report generator it is a single fast task, not a training job.
  add_task "$PY -m analysis.early_window --run results/e3ref-6b2398644d     --run results/e1p-84953d2a11 --max-frac 0.5 > results/POSITIVE_EVIDENCE.txt 2>&1 || true"

  # D) The prune@ x sparsity x method grid (E14), FULL resolution (4x4x3=48/arch) since
  #    cluster compute affords it -- one architecture per task so they run in parallel
  #    across GPUs rather than one architecture hogging a whole worker's queue slot.
  IFS=',' read -ra ARCH_LIST <<< "$ARCHS"
  for arch in "${ARCH_LIST[@]}"; do
    add_task "$PY -m experiments.e14_grid --archs $arch --seed $SEED \
      --prune-fracs 0,0.10,0.25,0.50 --sparsities 0.20,0.50,0.70,0.90 --out $PROBE_DIR"
  done
fi

# ---- TIER 2: everything else already built -------------------------------------------
if [ "$TIER" = "2" ] || [ "$TIER" = "all" ]; then
  for exp in e2 e4 e5 e6 e7 e7b e8 e9 e10; do
    [ -d "jobs/$exp" ] || continue
    for cfg in "jobs/$exp"/cfg_*.json; do
      add_task "$PY -m fsd.cli --config $cfg"
    done
  done
fi

N="${#TASKS[@]}"
echo "queued $N tasks across $N_GPUS GPU workers"
if [ "$N" -eq 0 ]; then
  echo "nothing to run -- did you 'python -m experiments.e<N> --emit jobs/e<N>' first?"
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "--- DRY_RUN=1: task queue only, nothing will execute ---"
  for ((i=0; i<N; i++)); do
    printf "  [gpu %d] %s\n" "$((i % N_GPUS))" "${TASKS[$i]}"
  done
  exit 0
fi

# --------------------------------------------------------------------------------------
# Fan out: N_GPUS background workers, each a sequential shell loop over its own slice of
# TASKS (indices i, i+N_GPUS, i+2*N_GPUS, ...), pinned to GPU $i via CUDA_VISIBLE_DEVICES.
# A failure in one task is logged and does NOT stop that worker's queue (keeps the sweep
# alive; failed cells are simply absent from the final report rather than blocking others).
# --------------------------------------------------------------------------------------
pids=()
for ((gpu=0; gpu<N_GPUS; gpu++)); do
  (
    worker_log="$LOG_DIR/gpu${gpu}.log"
    : > "$worker_log"
    for ((i=gpu; i<N; i+=N_GPUS)); do
      echo "[gpu $gpu] task $((i+1))/$N: ${TASKS[$i]}" | tee -a "$worker_log"
      CUDA_VISIBLE_DEVICES="$gpu" bash -c "${TASKS[$i]}" >> "$worker_log" 2>&1 \
        || echo "[gpu $gpu] FAILED: ${TASKS[$i]}" | tee -a "$worker_log"
    done
    echo "[gpu $gpu] worker done" | tee -a "$worker_log"
  ) &
  pids+=($!)
done

echo "waiting on ${#pids[@]} workers (pids: ${pids[*]}) -- tail -f $LOG_DIR/gpu*.log to watch"
wait "${pids[@]}"
date
echo "=== all workers finished ==="

# --------------------------------------------------------------------------------------
# Readable results. One markdown report combining: per-claim verdicts (analysis.claims),
# the prune grid (analysis.grid_report), the reproducibility ceiling (analysis.ceiling),
# and the mechanism cross-run check (analysis.mechanism) -- everything a reader needs
# without opening a single JSON file by hand.
# --------------------------------------------------------------------------------------
REPORT="results/REPORT.md"
{
  echo "# Experiment sweep report"
  echo "_generated $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
  echo
  echo "## Claims (C1-C5), by run"
  echo '```'
  $PY -m analysis.claims 2>&1 || true
  echo '```'
  echo
  echo "## Prune@ x Sparsity x Method grid"
  $PY -m analysis.grid_report --probe-dirs "$PROBE_DIR,results/_probe,results/_probe_sp05" 2>&1 || true
  echo
  echo "## Reproducibility ceiling (E12)"
  echo '```'
  $PY -m analysis.ceiling --tag e12 2>&1 || true
  echo '```'
  echo
  echo "## C5 mechanism check across runs"
  echo '```'
  $PY -m analysis.mechanism --tag e 2>&1 || true
  echo '```'
  echo
  echo "## Positive-evidence range report (AUROC / rho / calibration, time x sparsity)"
  echo '```'
  cat results/POSITIVE_EVIDENCE.txt 2>/dev/null || echo "(not generated -- see task E above)"
  echo '```'
} > "$REPORT"

echo "readable report written to $REPORT"
echo "figures: run  python -m analysis.figures --all  separately (writes paper/figures/*.png)"
