"""
Two figures built from the same expensive setup (baseline trunk test loss,
full-resolution rank-stability curves):

1. The "branching trajectory" figure (pruning_story.png): rank stability
   (Spearman rho) on top, the baseline (dense, unpruned) run's own train+test
   loss trajectory as a trunk below it, and one small subplot per prune-epoch
   branch showing that branch's own loss after pruning (keep 20%) and
   continuing training to the same final epoch -- connected back to the
   trunk by an arrow dropping straight down from the x-axis at the exact
   epoch it forked from. Requires --pruning-dir (a pruning_experiment.py
   output directory).

2. Loss-vs-rank-stability (loss_vs_correlation.png): the same rank-stability
   statistics (Pearson/Spearman/Kendall, unsigned sensitivity, vs. final
   epoch), but plotted against the loss achieved at that point in training
   instead of against epoch -- one line for train loss, one for test loss.
   This reframes "does the ranking stabilize early" as "does the ranking
   stabilize while the model is still far from its final loss, or only once
   it's nearly converged" -- doesn't need --pruning-dir.

Reuses already-saved artifacts (history.json, results.json, the cached
full-resolution sensitivity scores) wherever possible; the only things
computed fresh here are the dense trunk's test loss at each checkpoint epoch
(cheap: forward passes only, no training) and epoch 0's sensitivity (for the
correlation curves, matching rank_stability.py's convention).

Usage:
    python plot_pruning_story.py outputs/full_run_40ep --pruning-dir outputs/full_run_40ep/pruning_keep0.20
    python plot_pruning_story.py outputs/full_run_40ep  # loss_vs_correlation.png only
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
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from rank_stability import full_resolution_correlation_curves, METHOD_LABELS, METHODS
from sensitivity import compute_sensitivity, flatten_scores
from train import Config, build_loaders, build_model, select_device, set_seed
from pruning_experiment import evaluate, build_model_at_epoch

# A restrained red/blue palette: train and test loss are the two "primary"
# colours (blue/red) throughout; the rank-stability breakdown echoes them
# (embedding a blue, head a red) with a third neutral purple for the rnn
# stack so it doesn't fight with either. Reference/annotation lines stay in
# dark gray so they read as scaffolding rather than data.
TRAIN_COLOR = "#3B6FA0"
TEST_COLOR = "#B03A2E"
REF_COLOR = "#4D4D4D"
GROUP_COLORS = {"embedding": "#6699CC", "rnns": "#8E6C9E", "head": "#CC6666"}
TOTAL_COLOR = "#222222"
# Display label for the "rnns" parameter group (see model.py: the AWDLSTM's
# named_parameters() groups every recurrent-layer weight under "rnns"). At
# small legend font sizes "rnns" is easy to misread as "mns" (the "rn"
# bigram visually merges into what looks like an "m"), so use a clearer
# label here rather than the raw attribute name.
GROUP_DISPLAY_NAMES = {"rnns": "LSTM stack"}


def load_config(experiment_dir: Path) -> Config:
    history = json.loads((experiment_dir / "history.json").read_text())
    cfg_dict = {k: v for k, v in history["config"].items() if k != "output_dir"}
    return Config(**cfg_dict)


def get_trunk_test_loss(experiment_dir: Path, cfg, vocab_size, probe_loader, device, checkpoint_epochs):
    cache_path = experiment_dir / "trunk_heldout_loss.json"
    if cache_path.exists():
        cached = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
        if set(cached) >= set(checkpoint_epochs):
            return {e: cached[e] for e in checkpoint_epochs}

    criterion = nn.CrossEntropyLoss()
    result = {}
    for e in checkpoint_epochs:
        model = build_model_at_epoch(e, cfg, vocab_size, device, experiment_dir)
        loss, acc = evaluate(model, probe_loader, criterion, device)
        result[e] = {"loss": loss, "acc": acc}
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def get_trunk_train_eval_loss(experiment_dir: Path, cfg, vocab_size, device) -> float:
    """Loss of the freshly-initialized (epoch 0) model, evaluated (eval
    mode, no dropout) on a fixed sample of *training* data. history.json has
    no epoch-0 entry -- it only records loss from completed training epochs,
    and nothing has been trained on yet at epoch 0 -- so the train-loss
    curve is otherwise missing its starting point entirely, unlike the test
    curve (get_trunk_test_loss), whose epoch-0 point is well-defined (just
    evaluate the reconstructed init model on the held-out probe set).

    This closes that gap the same way: evaluate on a fixed sample of
    *training* data instead. Note it is not literally the same quantity as
    the recorded per-epoch training losses for epoch >= 1 (those are a
    train-mode, dropout-active running average accumulated *during*
    training); this is a clean eval-mode pass, chosen so it is directly
    comparable to the test curve's own epoch-0 point rather than noisier."""
    cache_path = experiment_dir / "trunk_train_eval_loss_epoch0.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())["loss"]

    set_seed(cfg.seed)
    model0 = build_model(cfg, vocab_size, device)
    train_loader, _, _ = build_loaders(cfg)
    n = min(cfg.sensitivity_chunks, len(train_loader.dataset))
    eval_loader = DataLoader(
        Subset(train_loader.dataset, list(range(n))), batch_size=cfg.sensitivity_batch_size, shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    loss, _ = evaluate(model0, eval_loader, criterion, device)
    cache_path.write_text(json.dumps({"loss": loss}))
    return loss


def get_correlation_curves(experiment_dir: Path, cfg, vocab_size, probe_loader, device, n_probes: int):
    """Unsigned-sensitivity rank-stability curves (vs. final epoch) for
    every method in common/rank_stability.py's METHODS, at full parameter
    resolution, spanning epoch 0 (reconstructed from seed) through the last
    checkpoint. The trivial final self-comparison point (== 1.0 by
    construction for every method/group) is dropped -- see
    common/rank_stability.py's module docstring / rank_stability.py's usage
    for why."""
    cache = np.load(experiment_dir / "full_resolution_sensitivity_cache.npz", allow_pickle=True)
    cached_epochs = cache["epochs"].tolist()
    boundaries = [(g, int(s), int(e)) for g, s, e in cache["boundaries"].tolist()]

    set_seed(cfg.seed)
    model0 = build_model(cfg, vocab_size, device)
    unsigned0, _ = compute_sensitivity(model0, probe_loader, device, n_probes=n_probes, show_progress=True)
    flat0, _ = flatten_scores(model0, unsigned0)

    all_flats = [flat0.numpy()] + list(cache["unsigned"])
    all_epochs = [0] + cached_epochs
    curves = full_resolution_correlation_curves(all_flats, boundaries)
    all_epochs = all_epochs[:-1]
    curves = {m: {g: v[:-1] for g, v in groups_dict.items()} for m, groups_dict in curves.items()}
    return all_epochs, curves


def plot_pruning_story(
    experiment_dir: Path, pruning_dir: Path, cfg, history, trunk_test, rho_epochs, rho_curves, train_loss_epoch0,
) -> Path:
    pruning_results = json.loads((pruning_dir / "results.json").read_text())
    branches = {r["prune_epoch"]: r for r in pruning_results["results"]}
    branch_epochs = sorted(e for e in branches if e < cfg.epochs)  # exclude the no-retrain epoch==final case
    no_retrain_epoch = cfg.epochs
    no_retrain_loss = branches[no_retrain_epoch]["final_loss"] if no_retrain_epoch in branches else None

    n_branches = len(branch_epochs)
    fig = plt.figure(figsize=(2.0 * n_branches, 8.5))
    gs = fig.add_gridspec(3, n_branches, height_ratios=[1.0, 1.0, 0.9], hspace=0.6, wspace=0.15)

    margin = max(1, n_branches // 6)
    if n_branches - 2 * margin < 1:
        margin = 0
    ax_rho = fig.add_subplot(gs[0, margin:n_branches - margin])
    ax_trunk = fig.add_subplot(gs[1, margin:n_branches - margin], sharex=ax_rho)
    branch_axes = [fig.add_subplot(gs[2, i]) for i in range(n_branches)]

    # ---- rank-stability panel ----
    groups = [g for g in rho_curves if g != "total"]
    ax_rho.plot(rho_epochs, rho_curves["total"], "o-", color=TOTAL_COLOR, markersize=4, label="total")
    for g in groups:
        ax_rho.plot(rho_epochs, rho_curves[g], "o--", color=GROUP_COLORS.get(g, "gray"), markersize=3,
                    label=GROUP_DISPLAY_NAMES.get(g, g))
    ax_rho.axhline(1.0, color=REF_COLOR, linewidth=0.5)
    ax_rho.set_ylabel(r"Spearman $\rho$" "\n(vs. final epoch)")
    ax_rho.set_title("Rank stability of the dense baseline's own sensitivity, over its own training")
    ax_rho.legend(fontsize=8, loc="lower right", ncol=4, handletextpad=0.5, columnspacing=1.0)
    ax_rho.set_ylim(-0.05, 1.05)
    ax_rho.xaxis.set_major_locator(MultipleLocator(10))

    # ---- trunk panel ----
    train_epochs = [0] + [h["epoch"] for h in history]
    train_losses = [train_loss_epoch0] + [h["loss"] for h in history]
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

    for i, (e, ax) in enumerate(zip(branch_epochs, branch_axes)):
        r = branches[e]
        retrain_losses = r["retrain_epoch_losses"]
        retrain_epochs = list(range(e + 1, cfg.epochs + 1))
        branch_test_losses = r.get("retrain_test_losses") or []

        ax.plot(retrain_epochs, retrain_losses, "-", color=TRAIN_COLOR, linewidth=1.3, label="train")
        if branch_test_losses:
            test_epochs_b = [e] + retrain_epochs
            test_curve = [r["immediate_loss"]] + branch_test_losses
            ax.plot(test_epochs_b, test_curve, "-", color=TEST_COLOR, linewidth=1.3, label="test")
            ax.plot([e, cfg.epochs], [r["immediate_loss"], r["final_loss"]], "o", color=TEST_COLOR, markersize=4)
        else:
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
            # which="both": tick_params defaults to affecting only major
            # ticks, but a log-scale axis spanning less than one decade also
            # draws labeled minor ticks (e.g. "6x10^0") that would otherwise
            # stay visible and overlap the neighbouring subplot.
            ax.tick_params(labelleft=False, which="both")

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
    return out_path


def plot_loss_vs_correlation(
    experiment_dir: Path, history, trunk_test, corr_epochs, corr_curves, train_loss_epoch0,
) -> Path:
    """corr_curves: {method: {'total': array, group: array}} from
    get_correlation_curves, aligned with corr_epochs. One subplot per
    method, x = loss achieved at that epoch (train and, separately, test),
    y = that epoch's rank-stability ('total', vs. final epoch)."""
    train_loss_by_epoch = {0: train_loss_epoch0, **{h["epoch"]: h["loss"] for h in history}}

    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.0 * len(METHODS), 4.8), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        corr_by_epoch = dict(zip(corr_epochs, corr_curves[method]["total"]))

        train_e = sorted(e for e in corr_by_epoch if e in train_loss_by_epoch)
        tx = [train_loss_by_epoch[e] for e in train_e]
        ty = [corr_by_epoch[e] for e in train_e]
        test_e = sorted(e for e in corr_by_epoch if e in trunk_test)
        sx = [trunk_test[e]["loss"] for e in test_e]
        sy = [corr_by_epoch[e] for e in test_e]

        ax.plot(tx, ty, "o-", color=TRAIN_COLOR, markersize=5, linewidth=1.3, label="train loss")
        ax.plot(sx, sy, "s--", color=TEST_COLOR, markersize=5, linewidth=1.3, label="test loss")
        # Only label each trajectory's starting epoch: later epochs bunch up
        # tightly near convergence (loss drops fast, correlation saturates
        # near 1), so labeling every endpoint there just overlaps -- the
        # starting point is the one epoch that's actually spread out, and
        # the trajectory's direction (toward lower loss, higher correlation)
        # is otherwise clear from the line itself.
        for xs, ys, es, color in [(tx, ty, train_e, TRAIN_COLOR), (sx, sy, test_e, TEST_COLOR)]:
            if not es:
                continue
            ax.annotate(f"e={es[0]}", (xs[0], ys[0]), fontsize=7, color=color,
                        xytext=(4, 4), textcoords="offset points")

        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Loss")
        ax.set_ylabel(METHOD_LABELS[method])
        ax.set_title(f"{METHOD_LABELS[method]} vs. loss")
        ax.grid(True, alpha=0.3, linewidth=0.4)
        # Loss decreases over training, so a plain ascending x-axis reads
        # backwards in time (higher loss/earlier epochs on the right).
        # Inverting it makes left-to-right match training's actual
        # direction: high loss (early) on the left, low loss (late, more
        # converged) on the right, alongside correlation rising toward 1.
        ax.invert_xaxis()

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Rank stability of unsigned sensitivity (vs. final epoch) as a function of the loss\n"
        "achieved at that point in training, rather than epoch number",
        fontsize=11,
    )

    out_path = experiment_dir / "loss_vs_correlation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("--pruning-dir", type=str, default=None,
                         help="If given, also produce pruning_story.png (needs pruning_experiment.py output)")
    parser.add_argument("--probes", type=int, default=8)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    pruning_dir = Path(args.pruning_dir) if args.pruning_dir else None
    cfg = load_config(experiment_dir)
    device = select_device()
    print(f"Using device: {device}")

    _train_loader, probe_loader, vocab_size = build_loaders(cfg)

    history = json.loads((experiment_dir / "history.json").read_text())["history"]

    checkpoint_epochs = list(range(0, cfg.epochs + 1, cfg.checkpoint_interval))
    print("Evaluating trunk test loss at each checkpoint...")
    trunk_test = get_trunk_test_loss(experiment_dir, cfg, vocab_size, probe_loader, device, checkpoint_epochs)

    print("Evaluating trunk train loss at epoch 0 (initialization)...")
    train_loss_epoch0 = get_trunk_train_eval_loss(experiment_dir, cfg, vocab_size, device)

    print("Computing full-resolution rank-stability curves (Pearson/Spearman/Kendall)...")
    corr_epochs, corr_curves = get_correlation_curves(experiment_dir, cfg, vocab_size, probe_loader, device, args.probes)

    loss_corr_path = plot_loss_vs_correlation(experiment_dir, history, trunk_test, corr_epochs, corr_curves, train_loss_epoch0)
    print(f"Saved {loss_corr_path}")

    if pruning_dir is not None:
        story_path = plot_pruning_story(
            experiment_dir, pruning_dir, cfg, history, trunk_test, corr_epochs, corr_curves["spearman"],
            train_loss_epoch0,
        )
        print(f"Saved {story_path}")


if __name__ == "__main__":
    main()
