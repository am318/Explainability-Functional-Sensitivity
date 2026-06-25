#!/usr/bin/env python3
"""Optuna-based hyperparameter search for structured ViT pruning.

This script searches over the pruning-related configurables exposed through
environment variables in Pruning_Sensitivity_ViT_Tiny.py (or a compatible
training script). It uses Optuna's TPE sampler as the surrogate optimizer.

Default objective: maximize final_test_accuracy from the target script's
summary JSON.

Typical usage:

    python optuna_pruning_opt.py \
        --target-script Pruning_Sensitivity_ViT_Tiny.py \
        --trials 40 \
        --initial-random 10 \
        --search-output runs/optuna_pruning

For expensive runs, search with smaller proxy settings by passing overrides,
for example:

    --trial-epochs 40 --trial-train-subset 20000 --trial-sensitivity-samples 4096

Trials are persisted to a SQLite database (optuna_pruning.db inside
--search-output) so runs can be resumed with --resume at any time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import optuna
from optuna.samplers import TPESampler


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
# Kept as a reference for env-var→Optuna mapping and for the final report.

SPACE_SPECS = [
    dict(name="CONNECTIVITY_CLOSURE",       kind="bool",  default=True),
    dict(name="MIN_CONNECTIONS_PER_UNIT",    kind="int",   low=1,    high=8,    default=2),
    dict(name="ITERATIVE_PRUNING_ROUNDS",    kind="int",   low=1,    high=30,   default=10),
    dict(name="GRADUAL_SPARSIFICATION",      kind="bool",  default=False),
    dict(name="LAYERWISE_NORMALIZE_SCORES",  kind="bool",  default=True),
    dict(name="SENSITIVITY_NORMALIZATION",   kind="cat",   choices=("mad", "zscore", "rank", "none"), default="rank"),
    dict(name="SENSITIVITY_CLIP_QUANTILE",   kind="float", low=0.0,  high=0.10, default=0.01),
    dict(name="MIN_EMBED_KEEP_FRACTION",     kind="float", low=0.01, high=0.50, default=0.10),
    dict(name="MIN_HIDDEN_KEEP_FRACTION",    kind="float", low=0.01, high=0.50, default=0.05),
]


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def _to_env_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Map each SPACE_SPEC to an Optuna suggest_* call."""
    params: Dict[str, Any] = {}
    for spec in SPACE_SPECS:
        name = spec["name"]
        kind = spec["kind"]
        if kind == "bool":
            params[name] = trial.suggest_categorical(name, [False, True])
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        elif kind == "cat":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Unknown param kind: {kind!r}")
    return params


def build_trial_env(
    params: Dict[str, Any],
    base_env: Dict[str, str],
    args: argparse.Namespace,
    trial_dir: Path,
    trial_number: int,
) -> Dict[str, str]:
    env = dict(base_env)
    for spec in SPACE_SPECS:
        env[spec["name"]] = _to_env_value(params[spec["name"]])

    env["OUTPUT_DIR"] = str(trial_dir)
    env["SEED"] = str(args.seed + args.trial_seed_offset)
    env["PYTHONUNBUFFERED"] = "1"

    if args.trial_epochs is not None:
        env["EPOCHS"] = str(args.trial_epochs)
    if args.trial_batch_size is not None:
        env["BATCH_SIZE"] = str(args.trial_batch_size)
    if args.trial_train_subset is not None:
        env["TRAIN_SUBSET"] = str(args.trial_train_subset)
    if args.trial_test_subset is not None:
        env["TEST_SUBSET"] = str(args.trial_test_subset)
    if args.trial_sensitivity_samples is not None:
        env["SENSITIVITY_SAMPLES"] = str(args.trial_sensitivity_samples)
    if args.trial_sensitivity_batch_size is not None:
        env["SENSITIVITY_BATCH_SIZE"] = str(args.trial_sensitivity_batch_size)
    if args.trial_analysis_probes is not None:
        env["ANALYSIS_PROBES"] = str(args.trial_analysis_probes)
    if args.trial_checkpoint_interval is not None:
        env["CHECKPOINT_INTERVAL"] = str(args.trial_checkpoint_interval)

    return env


def make_objective(args: argparse.Namespace, base_env: Dict[str, str], search_dir: Path):
    """Return the Optuna objective closure."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        trial_num = trial.number
        trial_dir = search_dir / f"trial_{trial_num:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        env = build_trial_env(params, base_env, args, trial_dir, trial_num)

        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"[trial {trial_num:03d}] {param_str}")

        command = [sys.executable, str(args.target_script)]
        started = time.time()
        proc = subprocess.run(
            command,
            env=env,
            cwd=str(args.workdir),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - started

        (trial_dir / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (trial_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

        # Store metadata on the Optuna trial for later inspection.
        trial.set_user_attr("elapsed_sec", round(elapsed, 2))
        trial.set_user_attr("returncode", proc.returncode)

        if proc.returncode != 0:
            msg = f"target script exited with code {proc.returncode}"
            trial.set_user_attr("error", msg)
            print(f"  -> failed: {msg}")
            raise optuna.exceptions.TrialPruned(msg)

        summary_path = trial_dir / args.summary_name
        if not summary_path.exists():
            msg = f"summary JSON not found at {summary_path}"
            trial.set_user_attr("error", msg)
            print(f"  -> failed: {msg}")
            raise optuna.exceptions.TrialPruned(msg)

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        value = float(summary.get(args.metric, float("nan")))
        if not np.isfinite(value):
            msg = f"metric '{args.metric}' missing or non-finite in summary"
            trial.set_user_attr("error", msg)
            print(f"  -> failed: {msg}")
            raise optuna.exceptions.TrialPruned(msg)

        trial.set_user_attr("summary", summary)
        print(f"  -> objective={value:.6f}  elapsed={elapsed:.1f}s")
        return value

    return objective


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-script", type=Path, default=Path("Pruning_Sensitivity_ViT_Tiny.py"))
    p.add_argument("--workdir", type=Path, default=Path("."))
    p.add_argument("--search-output", type=Path, default=Path("optuna_search_runs"))
    p.add_argument("--summary-name", type=str, default="vit_structured_sensitivity_pruning_summary.json")
    p.add_argument("--metric", type=str, default="final_test_accuracy")
    p.add_argument("--trials", type=int, default=250)
    p.add_argument("--initial-random", type=int, default=40,
                   help="Number of random startup trials before TPE kicks in.")
    p.add_argument("--candidate-pool-size", type=int, default=512,
                   help="n_ei_candidates for TPESampler (EI candidate draws per proposal).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial-seed-offset", type=int, default=1000)
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing study from the SQLite storage.")
    p.add_argument("--study-name", type=str, default="pruning_search",
                   help="Optuna study name (used as the key in storage).")
    p.add_argument("--trial-epochs", type=int, default=None)
    p.add_argument("--trial-batch-size", type=int, default=None)
    p.add_argument("--trial-train-subset", type=int, default=None)
    p.add_argument("--trial-test-subset", type=int, default=None)
    p.add_argument("--trial-sensitivity-samples", type=int, default=None)
    p.add_argument("--trial-sensitivity-batch-size", type=int, default=None)
    p.add_argument("--trial-analysis-probes", type=int, default=None)
    p.add_argument("--trial-checkpoint-interval", type=int, default=None)
    p.add_argument("--timeout-minutes", type=float, default=None)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel workers for study.optimize (1 = sequential).")
    p.add_argument("--set", action="append", default=[],
                   help="Additional NAME=VALUE env overrides.")
    return p.parse_args()


def parse_set_overrides(items: Sequence[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set override: {item!r}. Expected NAME=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --set override: {item!r}")
        overrides[k] = v
    return overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    search_dir = args.search_output
    search_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.update(parse_set_overrides(args.set))

    # Persist the study to SQLite so --resume works across invocations.
    storage_path = search_dir / "optuna_pruning.db"
    storage_url = f"sqlite:///{storage_path}"

    sampler = TPESampler(
        n_startup_trials=args.initial_random,
        n_ei_candidates=args.candidate_pool_size,
        seed=args.seed,
    )

    load_if_exists = args.resume
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=load_if_exists,
    )

    if args.resume:
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Resuming study '{args.study_name}' — {completed} completed trial(s) already in storage.")

    timeout_sec = None if args.timeout_minutes is None else args.timeout_minutes * 60.0

    objective = make_objective(args, base_env, search_dir)

    # suppress Optuna's per-trial INFO logs; our objective prints its own.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=timeout_sec,
        n_jobs=args.n_jobs,
        catch=(Exception,),     # log failures as pruned rather than crashing
    )

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    best_path = search_dir / "best_config.json"
    report_path = search_dir / "best_result.json"

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No successful trials were completed.")
        return 1

    best = study.best_trial
    best_payload = {
        "metric": args.metric,
        "best_objective": best.value,
        "best_params": best.params,
        "best_trial_id": best.number,
        "summary": best.user_attrs.get("summary", {}),
        "elapsed_sec": best.user_attrs.get("elapsed_sec"),
        "trials_evaluated": len(study.trials),
        "successful_trials": len(completed_trials),
        "search_space": SPACE_SPECS,
        "storage": storage_url,
    }
    write_json(best_path, best.params)
    write_json(report_path, best_payload)

    print("\nBest trial")
    print("==========")
    print(json.dumps(
        {k: v for k, v in best_payload.items() if k != "search_space"},
        indent=2,
    ))
    print(f"\nWrote best params  → {best_path}")
    print(f"Wrote full report  → {report_path}")
    print(f"Optuna DB          → {storage_path}")
    print(f"\nTo visualise: optuna-dashboard {storage_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
