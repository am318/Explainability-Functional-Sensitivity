"""
The "branching trajectory" figure: rank stability (Spearman rho) on top, the
baseline (dense, unpruned) run's own train+test loss trajectory as a trunk
below it, and one small subplot per prune-epoch branch showing that
branch's own loss after pruning (keep 20%) and continuing training to the
same final epoch -- connected back to the trunk by an arrow dropping
straight down from the x-axis at the exact epoch it forked from.

Reuses already-saved artifacts (history.json, results.json, the cached
full-resolution sensitivity scores) wherever possible; the only thing
computed fresh here is the dense trunk's test loss at each checkpoint epoch
(cheap: forward passes only, no training) and epoch 0's sensitivity (for the
Spearman curve, matching rank_stability.py's convention).

Usage:
    python plot_pruning_story.py outputs/full_run_60ep --pruning-dir outputs/full_run_60ep/pruning_keep0.20
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import CharSequenceDataset, build_vocab, encode, load_text
from model import CharLSTM
from rank_stability import full_resolution_correlation_curves
from sensitivity import compute_sensitivity, flatten_scores
from train import Config, build_loaders, select_device, set_seed
from pruning_experiment import evaluate, build_model_at_epoch

# A restrained red/blue palette: train and test loss are the two "primary"
# colours (blue/red) throughout; the rank-stability breakdown echoes them
# (embedding a blue, head a red) with a third neutral purple for lstm so it
# doesn't fight with either. Reference/annotation lines stay in dark gray so
# they read as scaffolding rather than data.
TRAIN_COLOR = "#3B6FA0"
TEST_COLOR = "#B03A2E"
REF_COLOR = "#4D4D4D"
GROUP_COLORS = {"embedding": "#6699CC", "lstm": "#8E6C9E", "head": "#CC6666"}
TOTAL_COLOR = "#222222"


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


def get_trunk_test_loss(experiment_dir: Path, cfg, vocab, probe_loader, device, checkpoint_epochs):
    cache_path = experiment_dir / "trunk_heldout_loss.json"
    if cache_path.exists():
        cached = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
        if set(cached) >= set(checkpoint_epochs):
            return {e: cached[e] for e in checkpoint_epochs}

    criterion = nn.CrossEntropyLoss()
    result = {}
    for e in checkpoint_epochs:
        model = build_model_at_epoch(e, cfg, len(vocab), device, experiment_dir)
        loss, acc = evaluate(model, probe_loader, criterion, device)
        result[e] = {"loss": loss, "acc": acc}
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def get_spearman_curve(experiment_dir: Path, cfg, vocab, probe_loader, device, n_probes: int):
    cache = np.load(experiment_dir / "full_resolution_sensitivity_cache.npz", allow_pickle=True)
    cached_epochs = cache["epochs"].tolist()
    boundaries = [(g, int(s), int(e)) for g, s, e in cache["boundaries"].tolist()]

    set_seed(cfg.seed)
    model0 = CharLSTM(len(vocab), cfg.embedding_dim, cfg.rnn_units).to(device)
    unsigned0, _ = compute_sensitivity(model0, probe_loader, device, n_probes=n_probes, show_progress=True)
    flat0, _ = flatten_scores(model0, unsigned0)

    all_flats = [flat0.numpy()] + list(cache["unsigned"])
    all_epochs = [0] + cached_epochs
    curves = full_resolution_correlation_curves(all_flats, boundaries)["spearman"]
    # Drop the final point: it compares the reference epoch against itself,
    # trivially exactly 1.0 by construction, not a measurement -- left in,
    # it reads as a sharp last-epoch jump that isn't a real training effect
    # (see common/rank_stability.py / notes.md for the full investigation).
    all_epochs = all_epochs[:-1]
    curves = {g: v[:-1] for g, v in curves.items()}
    return all_epochs, curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("--pruning-dir", type=str, required=True)
    parser.add_argument("--probes", type=int, default=8)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    pruning_dir = Path(args.pruning_dir)
    cfg = load_config(experiment_dir)
    device = select_device()
    print(f"Using device: {device}")

    text = load_text(cfg.data_path)
    vocab, char2idx = build_vocab(text)
    text_as_int = encode(text, char2idx)
    dataset = CharSequenceDataset(text_as_int, cfg.seq_length)
    _, probe_loader = build_loaders(cfg, dataset)

    history = json.loads((experiment_dir / "history.json").read_text())["history"]
    pruning_results = json.loads((pruning_dir / "results.json").read_text())
    branches = {r["prune_epoch"]: r for r in pruning_results["results"]}
    branch_epochs = sorted(e for e in branches if e < cfg.epochs)  # exclude the no-retrain epoch==final case
    no_retrain_epoch = cfg.epochs
    no_retrain_loss = branches[no_retrain_epoch]["final_loss"] if no_retrain_epoch in branches else None

    checkpoint_epochs = list(range(0, cfg.epochs + 1, 5))
    print("Evaluating trunk test loss at each checkpoint...")
    trunk_test = get_trunk_test_loss(experiment_dir, cfg, vocab, probe_loader, device, checkpoint_epochs)

    print("Computing Spearman rank-correlation curve...")
    rho_epochs, rho_curves = get_spearman_curve(experiment_dir, cfg, vocab, probe_loader, device, args.probes)

    # ---- figure layout ----
    # Top two panels (rank stability, trunk loss) are narrower than the full
    # width and centered, so the branch row below can spread wider and the
    # connecting arrows fan out left/right to reach the outer branches
    # instead of the whole figure being stretched to branch-row width.
    n_branches = len(branch_epochs)
    fig = plt.figure(figsize=(2.0 * n_branches, 8.5))
    gs = fig.add_gridspec(3, n_branches, height_ratios=[1.0, 1.0, 0.9], hspace=0.6, wspace=0.15)

    margin = max(1, n_branches // 6)
    if n_branches - 2 * margin < 1:
        margin = 0  # too few branches to afford a margin -- span the full width instead
    ax_rho = fig.add_subplot(gs[0, margin:n_branches - margin])
    ax_trunk = fig.add_subplot(gs[1, margin:n_branches - margin], sharex=ax_rho)
    branch_axes = [fig.add_subplot(gs[2, i]) for i in range(n_branches)]

    # ---- rank-stability panel ----
    groups = [g for g in rho_curves if g != "total"]
    ax_rho.plot(rho_epochs, rho_curves["total"], "o-", color=TOTAL_COLOR, markersize=4, label="total")
    for g in groups:
        ax_rho.plot(rho_epochs, rho_curves[g], "o--", color=GROUP_COLORS.get(g, "gray"), markersize=3,
                    label=g)
    ax_rho.axhline(1.0, color=REF_COLOR, linewidth=0.5)
    ax_rho.set_ylabel(r"Spearman $\rho$" "\n(vs. final epoch)")
    ax_rho.set_title("Rank stability of the dense baseline's own sensitivity, over its own training")
    ax_rho.legend(fontsize=7, loc="lower right", ncol=4)
    ax_rho.set_ylim(-0.05, 1.05)
    ax_rho.xaxis.set_major_locator(MultipleLocator(10))

    # ---- trunk panel ----
    train_epochs = [h["epoch"] for h in history]
    train_losses = [h["loss"] for h in history]
    test_epochs = sorted(trunk_test)
    test_losses = [trunk_test[e]["loss"] for e in test_epochs]

    ax_trunk.plot(train_epochs, train_losses, "-", color=TRAIN_COLOR, linewidth=1.5, label="baseline train loss")
    ax_trunk.plot(test_epochs, test_losses, "o-", color=TEST_COLOR, linewidth=1.5, markersize=4,
                  label="baseline test loss")
    best_test_epoch = min(trunk_test, key=lambda e: trunk_test[e]["loss"])
    ax_trunk.axhline(trunk_test[best_test_epoch]["loss"], color=TEST_COLOR, linestyle=":", linewidth=1,
                      label=f"baseline's own best test loss (epoch {best_test_epoch})")
    ax_trunk.set_ylabel("Loss")
    ax_trunk.set_title("Baseline (dense, unpruned) loss over training")
    ax_trunk.set_yscale("log")
    ax_trunk.legend(fontsize=7, loc="upper right")

    # ---- global y-range shared by trunk + every branch subplot ----
    all_vals = list(train_losses) + list(test_losses)
    for e in branch_epochs:
        r = branches[e]
        all_vals += r["retrain_epoch_losses"]
        all_vals += [r["immediate_loss"], r["final_loss"]]
    if no_retrain_loss is not None:
        all_vals.append(no_retrain_loss)
    y_lo, y_hi = min(all_vals) * 0.85, max(all_vals) * 1.15
    ax_trunk.set_ylim(y_lo, y_hi)
    ax_trunk.xaxis.set_major_locator(MultipleLocator(10))

    # ---- branch subplots ----
    for i, (e, ax) in enumerate(zip(branch_epochs, branch_axes)):
        r = branches[e]
        retrain_losses = r["retrain_epoch_losses"]
        retrain_epochs = list(range(e + 1, cfg.epochs + 1))
        test_losses = r.get("retrain_test_losses") or []

        ax.plot(retrain_epochs, retrain_losses, "-", color=TRAIN_COLOR, linewidth=1.3, label="train")
        if test_losses:
            # Full per-epoch test trajectory (immediate post-prune point,
            # then one evaluation per retrain epoch).
            test_epochs = [e] + retrain_epochs
            test_curve = [r["immediate_loss"]] + test_losses
            ax.plot(test_epochs, test_curve, "-", color=TEST_COLOR, linewidth=1.3, label="test")
            ax.plot([e, cfg.epochs], [r["immediate_loss"], r["final_loss"]], "o", color=TEST_COLOR, markersize=4)
        else:
            # Fallback for older results without a full test trajectory:
            # just the two endpoints, connected by a straight line.
            ax.plot([e, cfg.epochs], [r["immediate_loss"], r["final_loss"]], "o-", color=TEST_COLOR,
                    markersize=4, linewidth=1.3, label="test")
        if no_retrain_loss is not None:
            ax.axhline(no_retrain_loss, color=REF_COLOR, linestyle=":", linewidth=1,
                        label="pruning after training\n(no retrain) baseline")

        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(e, cfg.epochs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5, steps=[1, 2, 5, 10]))
        ax.set_title(f"prune @ epoch {e}", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if i == 0:
            ax.set_ylabel("Loss", fontsize=8)
            ax.legend(fontsize=6, loc="best")
        else:
            ax.tick_params(labelleft=False)

    # ---- arrows from the trunk's x-axis down to each branch subplot ----
    for e, ax in zip(branch_epochs, branch_axes):
        con = ConnectionPatch(
            xyA=(e, y_lo), coordsA=ax_trunk.transData,
            xyB=(0.5, 1.22), coordsB=ax.transAxes,
            arrowstyle="-|>", color=REF_COLOR, linewidth=1.1, mutation_scale=12,
            shrinkA=0, shrinkB=0,
        )
        fig.add_artist(con)

    fig.suptitle("Pruning (keep 20%) at different points in training, then retraining to the same final epoch",
                  fontsize=13)

    out_path = pruning_dir / "pruning_story.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
