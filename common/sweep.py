"""
Learning-rate sweep across optimizers, shared across experiments.

Run in two passes, because the two things being asked for have very
different costs:

  1. Selection. Every (optimizer, learning rate) pair is trained with loss
     tracking only -- no sensitivity scoring, no checkpoints. This is just
     ordinary training, and answers "which learning rate should this
     optimizer get?".
  2. Instrumentation. The winning learning rate for each optimizer is
     re-trained from the same seed with sensitivity tracking and per-epoch
     checkpointing switched on, then handed to run_rank_stability_analysis.

Re-training the winner costs one extra run per optimizer, but it keeps the
expensive scoring off the losing configurations entirely -- for a 3x4 grid
that is 3 instrumented runs instead of 12.

Selection criterion: lowest *minimum* test loss over the run (see
`best_learning_rates`). Minimum rather than final because a run that
overfits after its best epoch has still shown what that learning rate can
do, and because a diverged run's final loss can be inf/NaN, which does not
order meaningfully.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiment import BaseConfig, BuildExperiment, run_training, select_device
from optimizers import OPTIMIZER_LABELS, OPTIMIZERS
from plotting import plot_lr_sweep
from rank_stability import plot_optimizer_comparison
from rank_stability_runner import load_curves


# One grid, every optimizer, every dataset. Half-decade (sqrt(10)) spacing
# centred on 3e-2, which is roughly the geometric mean of where the three
# optimizers landed on two moons when each had its own grid (Adam 3e-3,
# SGD+momentum 1e-1, SGD 3e-1). Five points is the most that stays readable
# as five curves in one panel; covering Adam's usual 1e-3 as well would need
# a sixth, so Adam is expected to sit at or near the low end here.
DEFAULT_LEARNING_RATES: Tuple[float, ...] = (3e-3, 1e-2, 3e-2, 1e-1, 3e-1)


def _run_key(optimizer: str, lr: float) -> str:
    return f"{optimizer}_lr{lr:g}"


def best_learning_rates(
    results: Dict[str, List[Tuple[float, List[Dict[str, float]]]]]
) -> Dict[str, float]:
    """Per optimizer, the learning rate whose run reached the lowest test
    loss at any epoch. Non-finite losses (a diverged run) are excluded
    rather than compared, so divergence can never win by accident."""
    best: Dict[str, float] = {}
    for optimizer, runs in results.items():
        scored = []
        for lr, history in runs:
            losses = [h["val_loss"] for h in history if np.isfinite(h.get("val_loss", np.nan))]
            if losses:
                scored.append((min(losses), lr))
        if scored:
            best[optimizer] = min(scored)[1]
    return best


def run_sweep(
    cfg: BaseConfig,
    build_experiment: BuildExperiment,
    learning_rates: Sequence[float],
    rank_stability_fn: Callable[[Path, torch.device], None],
    optimizers: Sequence[str] = OPTIMIZERS,
    device: Optional[torch.device] = None,
    title: str = "Learning-rate sweep",
    probes: int = 8,
) -> Dict:
    """Sweep, select, re-train instrumented, analyse, and plot.

    `cfg` supplies every setting the sweep does not vary (seed, epochs,
    batch size, sensitivity settings, output_root); optimizer/lr/
    experiment_name/tracking flags are overridden per run.
    `learning_rates` is one grid tried by *every* optimizer -- deliberately
    a flat sequence rather than a per-optimizer mapping, so the comparison
    between optimizers is over identical settings and cannot silently drift
    into each optimizer being tuned on a different grid. The cost is that a
    single grid spanning Adam's usual range and plain SGD's cannot bracket
    both with fine spacing, so an optimizer may select an endpoint; the
    summary records which, and the figures make it visible.
    `rank_stability_fn(experiment_dir, device)` runs the experiment's own
    rank-stability analysis over a finished instrumented run.

    Returns the summary dict that is also written to sweep_summary.json.
    """
    if device is None:
        device = select_device()
    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Pass 1/2: learning-rate selection ({len(optimizers) * len(learning_rates)} runs) ===")
    results: Dict[str, List[Tuple[float, List[Dict[str, float]]]]] = {}
    for optimizer in optimizers:
        results[optimizer] = []
        for lr in learning_rates:
            run_cfg = replace(
                cfg,
                optimizer=optimizer,
                lr=lr,
                experiment_name=f"sweep/{_run_key(optimizer, lr)}",
                track_sensitivity=False,
                checkpoint_interval=0,
            )
            print(f"\n--- {optimizer} lr={lr:g} ---")
            history = run_training(run_cfg, build_experiment, device, progress=False, verbose=False)
            final = history[-1]
            best_val = min((h["val_loss"] for h in history if np.isfinite(h["val_loss"])), default=float("nan"))
            print(f"    best test loss={best_val:.4f} | final train={final['loss']:.4f} "
                  f"test={final['val_loss']:.4f} acc={final['val_acc']:.4f}")
            results[optimizer].append((lr, history))

    best = best_learning_rates(results)
    print("\nSelected learning rates: " + ", ".join(f"{o}={lr:g}" for o, lr in best.items()))

    sweep_plot = out_root / "sweep_learning_rates.png"
    plot_lr_sweep(results, best, sweep_plot, labels=OPTIMIZER_LABELS, title=title)
    print(f"Saved {sweep_plot}")

    print(f"\n=== Pass 2/2: instrumented runs at the selected learning rates ===")
    entries: List[Dict] = []
    for optimizer, lr in best.items():
        run_cfg = replace(
            cfg,
            optimizer=optimizer,
            lr=lr,
            experiment_name=f"best/{_run_key(optimizer, lr)}",
        )
        print(f"\n--- {optimizer} lr={lr:g} (instrumented) ---")
        history = run_training(run_cfg, build_experiment, device, progress=False, verbose=True)
        print(f"--- rank stability: {optimizer} lr={lr:g} ---")
        rank_stability_fn(run_cfg.output_dir, device)
        entry = {"optimizer": optimizer, "lr": lr, "history": history, "dir": str(run_cfg.output_dir)}
        entry.update(load_curves(run_cfg.output_dir, kind="unsigned"))
        entries.append(entry)

    comparison_plot = out_root / "optimizer_comparison.png"
    plot_optimizer_comparison(entries, comparison_plot, kind="unsigned", labels=OPTIMIZER_LABELS)
    print(f"\nSaved {comparison_plot}")

    summary = {
        "title": title,
        "learning_rates": list(learning_rates),
        "optimizers": list(optimizers),
        "best_lr": best,
        "best_lr_at_grid_edge": {
            optimizer: lr in (learning_rates[0], learning_rates[-1]) for optimizer, lr in best.items()
        },
        "runs": [
            {
                "optimizer": optimizer,
                "lr": lr,
                "best_val_loss": float(min(h["val_loss"] for h in history)),
                "final_train_loss": float(history[-1]["loss"]),
                "final_val_loss": float(history[-1]["val_loss"]),
                "final_val_acc": float(history[-1]["val_acc"]),
                "selected": best.get(optimizer) == lr,
            }
            for optimizer, runs in results.items()
            for lr, history in runs
        ],
        "instrumented": [{"optimizer": e["optimizer"], "lr": e["lr"], "dir": e["dir"]} for e in entries],
    }
    summary_path = out_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")
    return summary
