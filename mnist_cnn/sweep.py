"""
The MNIST experiment end to end: sweep the shared learning-rate grid for
Adam, SGD and SGD+momentum, pick the best learning rate for each, re-train
those three fully instrumented, and analyse how the parameter-wise
sensitivity ordering settles over training.

See common/sweep.py for why this runs in two passes (cheap selection, then
instrumented re-training of the winners only), how "best" is defined, and
why the learning-rate grid is shared across every optimizer.

Usage:
    python sweep.py
    python sweep.py --epochs 5 --probes 4      # quicker, coarser

run_sweep.sbatch wraps the same command for a SLURM cluster; it is a
convenience for one particular setup, not a requirement.

Outputs, under --output-root (default outputs/):
    sweep_learning_rates.png   every learning rate's train/test loss
    optimizer_comparison.png   the three winners: loss + rank stability
    sweep_summary.json         selected learning rates and final losses
    sweep/<opt>_lr<lr>/        one directory per swept run (loss only)
    best/<opt>_lr<lr>/         the instrumented runs, with all their plots
"""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from experiment import select_device
from sweep import DEFAULT_LEARNING_RATES, run_sweep
from train import Config, build_experiment

RANK_STABILITY_SCRIPT = Path(__file__).resolve().parent / "rank_stability.py"


def make_rank_stability_fn(probes: int):
    """Run this directory's rank_stability.py as a subprocess rather than
    importing it. It shares a module name with common/rank_stability.py, and
    importing both into one process would shadow the latter -- which
    common/rank_stability_runner.py itself imports. Shelling out also means
    the command printed here is exactly the one to re-run by hand.
    """

    def run(experiment_dir: Path, device: torch.device) -> None:
        cmd = [sys.executable, str(RANK_STABILITY_SCRIPT), str(experiment_dir), "--probes", str(probes)]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "DEVICE": str(device)})

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=None, help="override the configured number of epochs")
    parser.add_argument("--probes", type=int, default=8, help="Hutchinson probes for sensitivity scoring")
    parser.add_argument("--output-root", type=str, default=None, help="where to write all sweep outputs")
    args = parser.parse_args()

    cfg = Config()
    overrides = {"sensitivity_probes": args.probes}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    cfg = replace(cfg, **overrides)

    device = select_device()
    print(f"Using device: {device}", flush=True)
    run_sweep(
        cfg, build_experiment, DEFAULT_LEARNING_RATES, make_rank_stability_fn(args.probes),
        device=device, title="MNIST CNN: learning-rate sweep", probes=args.probes,
    )


if __name__ == "__main__":
    main()
