"""
The one figure that puts every experiment on the same axes: for each
model/optimizer pair (at its selected learning rate), how far through its
loss descent was the run when its parameter-wise sensitivity ordering had
already settled?

Each trajectory runs from epoch 1 (top left: full loss still to come, the
ordering not yet resembling its final form) to the end of training (bottom
right). A trajectory that travels along the *top* of the panel before
dropping has settled its ordering while most of the loss reduction was
still ahead of it -- which is the claim this figure exists to test.

Reads whatever sweeps have already been run: each experiment's
outputs/sweep_summary.json names the instrumented runs, and each of those
directories holds the history.json and rank_stability_curves.npz written by
train.py and rank_stability.py.

Usage:
    python summary_figure.py
    python summary_figure.py --loss val_loss     # normalise test loss instead
    python summary_figure.py --out outputs/summary.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "common"))

from optimizers import OPTIMIZER_LABELS
from rank_stability import plot_normalized_loss_vs_correlation
from rank_stability_runner import load_curves

# (label, outputs directory). Anything without a sweep_summary.json yet is
# skipped with a warning, so this works before every experiment has run.
EXPERIMENTS = [
    ("Two moons MLP", "twomoons_mlp/outputs"),
    ("MNIST CNN", "mnist_cnn/outputs"),
]


def collect(label: str, outputs_dir: Path) -> List[Dict]:
    summary_path = outputs_dir / "sweep_summary.json"
    if not summary_path.exists():
        print(f"Skipping {label}: no {summary_path}")
        return []
    summary = json.loads(summary_path.read_text())

    entries = []
    for run in summary["instrumented"]:
        # Prefer reconstructing the path from outputs_dir; the absolute path
        # recorded at sweep time is only a fallback, since it does not
        # survive the repo being moved or the outputs being copied.
        run_dir = outputs_dir / "best" / f"{run['optimizer']}_lr{run['lr']:g}"
        if not run_dir.exists():
            run_dir = Path(run["dir"])
        if not (run_dir / "rank_stability_curves.npz").exists():
            print(f"Skipping {label}/{run['optimizer']}: no rank_stability_curves.npz in {run_dir}")
            continue
        history = json.loads((run_dir / "history.json").read_text())["history"]
        entry = {
            "dataset": label,
            "optimizer": run["optimizer"],
            "lr": run["lr"],
            "history": history,
        }
        entry.update(load_curves(run_dir, kind="unsigned"))
        entries.append(entry)
        print(f"Loaded {label} / {run['optimizer']} (lr={run['lr']:g}, {len(history)} epochs)")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", type=str, default="loss", choices=["loss", "val_loss"],
                        help="which loss to normalise (default: train)")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"))
    args = parser.parse_args()

    entries: List[Dict] = []
    for label, rel in EXPERIMENTS:
        entries.extend(collect(label, REPO_ROOT / rel))
    if not entries:
        raise SystemExit("No instrumented runs found -- run each experiment's sweep.py first.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Both scalings, because they answer slightly different questions: the
    # linear one asks how much of the loss *value* was still to come, the
    # log one how many further halvings were still to come. Loss falls
    # roughly exponentially, so the linear version reaches ~0 within a few
    # epochs and would overstate the case on its own.
    for log_loss, stem in [(False, "summary_loss_vs_correlation"), (True, "summary_logloss_vs_correlation")]:
        out_path = out_dir / f"{stem}.png"
        plot_normalized_loss_vs_correlation(
            entries, out_path, kind="unsigned", loss_key=args.loss,
            labels=OPTIMIZER_LABELS, log_loss=log_loss,
        )
        print(f"Saved {out_path} ({len(entries)} model/optimizer pairs)")


if __name__ == "__main__":
    main()
