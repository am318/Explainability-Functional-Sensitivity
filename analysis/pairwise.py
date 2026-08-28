"""The freezing diagnostic: agreement between every pair of checkpoints.

Reading the matrix:

  * **Freezing.** A saturated block in the bottom-right -- every pair of checkpoints after
    t* agrees strongly, no matter how far apart. The block's top-left corner *is* t*.
  * **Constant-rate churn.** Agreement falls off with the ratio t'/t and never saturates;
    the matrix looks like smooth diagonal banding all the way to the corner.

This distinguishes the two cases that "agreement with S_final" cannot, because that curve
is pinned to 1.0 at t = T either way.

    python -m analysis.pairwise --tag e7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.claims import load_runs

OUT = Path("paper/figures")


def report(m: dict, sparsity: float) -> None:
    pw = m.get("pairwise")
    if not pw:
        print(f"  {m['run_id']}: no pairwise data (re-run needed)")
        return
    key = f"{sparsity:g}"
    steps = pw["steps"]
    A = np.array(pw[key]["adjusted"])
    n = len(steps)
    c = m["config"]
    print(f"\n{c['model']['arch']}/{c['data']['dataset']} "
          f"{c['train']['lr_schedule']} LR, {c['train']['steps']} steps, sparsity {key}")

    # For each candidate t*, the weakest agreement between ANY two later checkpoints.
    # Freezing means this stays high; churn means it keeps decaying.
    print(f"  {'t*cand':>8} {'min pairwise agreement after it':>34}")
    for i in range(0, n - 1):
        block = A[i:, i:]
        print(f"  {steps[i]:>8} {block.min():>34.3f}")


def figure(runs, sparsity: float) -> None:
    runs = [m for m in runs if m.get("pairwise")]
    if not runs:
        print("no runs carry pairwise data")
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(1.85 * len(runs) + 0.6, 2.0),
                             squeeze=False)
    for ax, m in zip(axes[0], runs):
        pw = m["pairwise"]
        A = np.array(pw[f"{sparsity:g}"]["adjusted"])
        im = ax.imshow(A, vmin=0, vmax=1, cmap="viridis", origin="lower")
        steps = pw["steps"]
        ticks = list(range(0, len(steps), max(1, len(steps) // 5)))
        ax.set_xticks(ticks); ax.set_xticklabels([steps[i] for i in ticks], fontsize=6)
        ax.set_yticks(ticks); ax.set_yticklabels([steps[i] for i in ticks], fontsize=6)
        c = m["config"]
        names = {"mlp": "MLP", "resnet20": "ResNet-20", "vit": "ViT"}
        ax.set_title(names.get(c["model"]["arch"], c["model"]["arch"]), fontsize=8)
        ax.set_xlabel("step"); ax.grid(False)
    axes[0][0].set_ylabel("step")
    fig.colorbar(im, ax=axes[0][-1], fraction=0.046, label="adjusted agreement")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig5_pairwise.pdf")
    fig.savefig(OUT / "fig5_pairwise.png")
    print(f"\nwrote {OUT}/fig5_pairwise.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--schedule", default=None,
                    help="restrict to one LR schedule; the constant-LR panels make the "
                         "cleanest argument because decay cannot flatten them")
    args = ap.parse_args()
    runs = load_runs(args.results, args.tag)
    if args.schedule:
        runs = [m for m in runs
                if m["config"]["train"]["lr_schedule"] == args.schedule]
    if not runs:
        print("no runs")
        return 1
    runs.sort(key=lambda m: m["config"]["model"]["arch"])
    for m in runs:
        report(m, args.sparsity)
    figure(runs, args.sparsity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
