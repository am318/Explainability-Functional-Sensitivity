"""Loss / sensitivity plots, shared across experiments. Group names (e.g.
embedding/lstm/head for CharLSTM, input/hidden/output for MLP) are derived
from the data rather than hardcoded, so this works for any model.

Which sensitivity scores are drawn is likewise data-driven: every function
taking a `kinds` argument (or an optional signed matrix) will draw only the
unsigned score if that is all the experiment tracked, and both if it tracked
both -- see BaseConfig.track_signed."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    _MATPLOTLIB_AVAILABLE = False

_PALETTE = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

TRAIN_COLOR = "black"
VAL_COLOR = "tab:red"

# Per-score-kind axis styling, so every figure that draws a sensitivity
# panel labels and scales it the same way. Unsigned S_i is non-negative and
# spans orders of magnitude (log); signed Sbar_i straddles zero (linear,
# with a zero line for reference).
KIND_STYLE = {
    "unsigned": {
        "ylabel": r"$\sum_i S_i(\theta)$ (unsigned)",
        "title": "Unsigned sensitivity: " r"$S_i = \mathbb{E}_x[\|\partial F(x)/\partial\theta_i\|_2^2]$",
        "log": True,
        "zero_line": False,
    },
    "signed": {
        "ylabel": r"$\sum_i \bar S_i(\theta)$ (signed)",
        "title": r"Signed sensitivity: $\bar S_i = \mathbb{E}_x[\sum_y \partial F(x)_y/\partial\theta_i]$",
        "log": False,
        "zero_line": True,
    },
}


def plot_loss_panel(ax, history: List[Dict[str, float]], legend: bool = True) -> None:
    """Training loss (and validation loss, if the run recorded any) against
    epoch, on a log y-axis. Shared by every figure that puts a loss panel
    above something else, so the loss always looks the same.

    Runs predating validation tracking (e.g. shakespeare_lstm) simply have no
    `val_loss` key and get the single-curve version they had before."""
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["loss"] for h in history], color=TRAIN_COLOR, label="train")
    val_rows = [h for h in history if "val_loss" in h]
    if val_rows:
        ax.plot([h["epoch"] for h in val_rows], [h["val_loss"] for h in val_rows],
                color=VAL_COLOR, label="test")
        ax.set_title("Train vs. test loss")
        if legend:
            ax.legend(fontsize=8)
    else:
        ax.set_title("Training loss")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")


def _groups_from_history(rows: List[Dict[str, float]], prefix: str) -> List[str]:
    groups: List[str] = []
    for h in rows:
        for key in h:
            if key.startswith(prefix) and key != f"{prefix}total":
                group = key[len(prefix):]
                if group not in groups:
                    groups.append(group)
    return groups


def _group_colors(groups: List[str]) -> Dict[str, str]:
    return {group: _PALETTE[i % len(_PALETTE)] for i, group in enumerate(groups)}


def plot_training_history(
    history: List[Dict[str, float]],
    out_path: Path,
    kinds: Sequence[str] = ("unsigned", "signed"),
) -> None:
    """Loss on top, then one summed-sensitivity panel per tracked kind, all
    sharing an epoch axis. `kinds` defaults to both scores; an experiment
    that only tracked S_i passes ("unsigned",) and gets a two-panel figure."""
    if not _MATPLOTLIB_AVAILABLE or not history:
        return

    rows = [h for h in history if "unsigned_total" in h]
    if not rows:
        return

    kinds = [k for k in kinds if any(f"{k}_total" in h for h in rows)]
    if not kinds:
        return

    groups = _groups_from_history(rows, "unsigned_")
    colors = _group_colors(groups)
    sens_epochs = [h["epoch"] for h in rows]

    fig, axes = plt.subplots(
        1 + len(kinds), 1, figsize=(8.0, 3.3 * (1 + len(kinds))), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    plot_loss_panel(axes[0], history)

    for ax, kind in zip(axes[1:], kinds):
        style = KIND_STYLE[kind]
        ax.plot(sens_epochs, [h[f"{kind}_total"] for h in rows], color="black", label="total")
        for group in groups:
            key = f"{kind}_{group}"
            if any(key in h for h in rows):
                ax.plot(
                    sens_epochs, [h.get(key, float("nan")) for h in rows],
                    color=colors[group], linestyle="--", label=group,
                )
        if style["zero_line"]:
            ax.axhline(0.0, color="gray", linewidth=0.8)
        if style["log"]:
            ax.set_yscale("log")
        ax.set_ylabel(style["ylabel"])
        ax.set_title(style["title"])
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Epoch")
    fig.savefig(out_path)
    plt.close(fig)


def _apply_group_yticks(ax, boundaries: List[Tuple[str, int]], n_rows: int) -> None:
    tick_rows = [row for _, row in boundaries] + [n_rows]
    label_rows = [(a + b) / 2 for a, b in zip(tick_rows[:-1], tick_rows[1:])]
    labels = [group for group, _ in boundaries]
    ax.set_yticks(label_rows)
    ax.set_yticklabels(labels)
    for row in tick_rows[1:-1]:
        ax.axhline(row, color="white", linewidth=1.0)


def _epoch_bin_edges(epochs: List[int]) -> np.ndarray:
    """Cell edges along the epoch axis for pcolormesh, placed at the true
    epoch values (not just column index) so a heatmap column sits at its
    real epoch and lines up with a line plot sharing the same x-axis."""
    epochs = np.asarray(epochs, dtype=float)
    if epochs.size == 1:
        return np.array([epochs[0] - 0.5, epochs[0] + 0.5])
    mid = (epochs[:-1] + epochs[1:]) / 2.0
    first = epochs[0] - (mid[0] - epochs[0])
    last = epochs[-1] + (epochs[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def plot_sensitivity_heatmap(
    history: List[Dict[str, float]],
    unsigned_matrix: np.ndarray,
    signed_matrix: Optional[np.ndarray],
    heatmap_epochs: List[int],
    boundaries: List[Tuple[str, int]],
    n_rows: int,
    out_path: Path,
) -> None:
    """Parameter-wise sensitivity over training: x=epoch, y=parameter index
    (pooled/binned and ordered by module), colour=sensitivity value -- with
    the loss plotted above on the *same* (real-valued) epoch axis, so the two
    can be compared directly rather than just index-aligned.

    `boundaries` gives the (module_name, first_row) of each module's block
    in the pooled row ordering, used to label the y-axis by module.
    `signed_matrix` may be None, for runs that only tracked S_i.
    """
    if not _MATPLOTLIB_AVAILABLE or unsigned_matrix.size == 0 or not history:
        return

    x_edges = _epoch_bin_edges(heatmap_epochs)
    y_edges = np.arange(n_rows + 1)
    n_maps = 1 if signed_matrix is None else 2

    fig, axes = plt.subplots(
        1 + n_maps, 1, figsize=(10.0, 4.0 + 4.0 * n_maps), sharex=True,
        gridspec_kw={"height_ratios": [1.0] + [2.2] * n_maps},
        constrained_layout=True,
    )
    ax_loss, ax_u = axes[0], axes[1]

    plot_loss_panel(ax_loss, history)
    ax_loss.set_xlim(x_edges[0], x_edges[-1])

    positive = unsigned_matrix[unsigned_matrix > 0]
    vmin = float(positive.min()) if positive.size else 1e-12
    vmax = float(unsigned_matrix.max()) if unsigned_matrix.size else 1.0
    im_u = ax_u.pcolormesh(
        x_edges, y_edges, unsigned_matrix,
        cmap="viridis", norm=LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-12)),
    )
    ax_u.invert_yaxis()
    ax_u.set_title(r"Unsigned sensitivity $S_i(\theta)$")
    _apply_group_yticks(ax_u, boundaries, n_rows)
    fig.colorbar(im_u, ax=ax_u, pad=0.02).set_label(r"$S_i(\theta)$")

    if signed_matrix is not None:
        ax_s = axes[2]
        signed_absmax = float(np.abs(signed_matrix).max()) if signed_matrix.size else 1.0
        im_s = ax_s.pcolormesh(
            x_edges, y_edges, signed_matrix,
            cmap="RdBu_r",
            norm=SymLogNorm(linthresh=max(signed_absmax * 1e-3, 1e-12), vmin=-signed_absmax, vmax=signed_absmax),
        )
        ax_s.invert_yaxis()
        ax_s.set_title(r"Signed sensitivity $\bar S_i(\theta)$")
        _apply_group_yticks(ax_s, boundaries, n_rows)
        fig.colorbar(im_s, ax=ax_s, pad=0.02).set_label(r"$\bar S_i(\theta)$")

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Loss vs. parameter-wise sensitivity over training (rows pooled by module)")
    fig.savefig(out_path)
    plt.close(fig)


def _finite(values: List[float]) -> np.ndarray:
    """Replace inf/NaN with NaN so matplotlib breaks the line rather than
    blowing up the axis limits -- diverging learning rates are expected in a
    sweep and should not squash every other curve into a flat line."""
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, np.nan)


def plot_lr_sweep(
    sweep_runs: Dict[str, List[Tuple[float, List[Dict[str, float]]]]],
    best: Dict[str, float],
    out_path: Path,
    labels: Optional[Dict[str, str]] = None,
    title: str = "Learning-rate sweep",
) -> None:
    """One panel per optimizer, one colour per learning rate, dashed=train /
    solid=test. The selected learning rate is drawn thicker.

    The y-limit is set from the *first* epoch's losses rather than from the
    data's true maximum, so a learning rate that diverges runs off the top of
    the panel instead of compressing every other curve into a flat line.
    """
    if not _MATPLOTLIB_AVAILABLE or not sweep_runs:
        return
    labels = labels or {}
    optimizers = list(sweep_runs)

    fig, axes = plt.subplots(
        1, len(optimizers), figsize=(5.0 * len(optimizers), 4.6), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    lo, hi = np.inf, -np.inf
    for _, runs in sweep_runs.items():
        for _, history in runs:
            values = _finite([h["loss"] for h in history] + [h.get("val_loss", np.nan) for h in history])
            first = _finite([history[0]["loss"], history[0].get("val_loss", np.nan)])
            if np.any(np.isfinite(values)):
                lo = min(lo, float(np.nanmin(values)))
            if np.any(np.isfinite(first)):
                hi = max(hi, float(np.nanmax(first)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = None, None

    cmap = plt.get_cmap("viridis")
    for ax, optimizer in zip(axes, optimizers):
        runs = sorted(sweep_runs[optimizer], key=lambda item: item[0])
        n_lr = max(1, len(runs) - 1)
        for i, (lr, history) in enumerate(runs):
            color = cmap(0.85 * i / n_lr)
            is_best = best.get(optimizer) == lr
            width = 2.4 if is_best else 1.1
            epochs = [h["epoch"] for h in history]
            ax.plot(epochs, _finite([h["loss"] for h in history]),
                    color=color, linewidth=width, linestyle="--", alpha=0.75)
            ax.plot(epochs, _finite([h.get("val_loss", np.nan) for h in history]),
                    color=color, linewidth=width,
                    label=f"lr={lr:g}" + (" (best)" if is_best else ""))
        ax.set_yscale("log")
        if lo is not None:
            ax.set_ylim(lo * 0.8, hi * 1.5)
        ax.set_xlabel("Epoch")
        ax.set_title(labels.get(optimizer, optimizer))
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Loss")
    fig.suptitle(f"{title} (dashed = train, solid = test)")
    fig.savefig(out_path)
    plt.close(fig)


BEFORE_COLOR = "tab:gray"
AFTER_COLOR = "tab:orange"


def group_labels(boundaries: List[Tuple[str, int, int]], total_len: int) -> np.ndarray:
    """Expand (group, start, end) boundaries into a per-element group-name array."""
    labels = np.empty(total_len, dtype=object)
    for group, start, end in boundaries:
        labels[start:end] = group
    return labels


def plot_sensitivity_distributions_all(before_u, after_u, before_s, after_s, out_path: Path) -> None:
    """Before-vs-after histograms of unsigned and signed per-parameter
    sensitivity, pooling every parameter in the model together."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    fig, (ax_u, ax_s) = plt.subplots(1, 2, figsize=(13.0, 5.5), constrained_layout=True)

    pos = np.concatenate([before_u[before_u > 0], after_u[after_u > 0]])
    if pos.size:
        bins_u = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 80)
        ax_u.hist(before_u, bins=bins_u, alpha=0.55, density=True, label="before", color=BEFORE_COLOR)
        ax_u.hist(after_u, bins=bins_u, alpha=0.55, density=True, label="after", color=AFTER_COLOR)
        ax_u.set_xscale("log")
    ax_u.set_yscale("log")
    ax_u.set_xlabel(r"$S_i(\theta)$")
    ax_u.set_ylabel("density")
    ax_u.set_title("Unsigned sensitivity (all parameters)")
    ax_u.legend()

    combined = np.concatenate([before_s, after_s])
    lo, hi = np.percentile(combined, [0.5, 99.5])
    if hi > lo:
        bins_s = np.linspace(lo, hi, 100)
        ax_s.hist(before_s, bins=bins_s, alpha=0.55, density=True, label="before", color=BEFORE_COLOR)
        ax_s.hist(after_s, bins=bins_s, alpha=0.55, density=True, label="after", color=AFTER_COLOR)
    ax_s.axvline(0.0, color="black", linewidth=0.8)
    ax_s.set_yscale("log")
    ax_s.set_xlabel(r"$\bar S_i(\theta)$")
    ax_s.set_title("Signed sensitivity (all parameters, 0.5th-99.5th pct)")
    ax_s.legend()

    fig.suptitle("Sensitivity distribution before vs. after training")
    fig.savefig(out_path)
    plt.close(fig)


def plot_sensitivity_distributions_by_module(
    before_u, after_u, before_s, after_s, labels: np.ndarray, groups: List[str], out_path: Path
) -> None:
    """Same before-vs-after comparison as plot_sensitivity_distributions_all,
    faceted into one row per parameter group (module)."""
    if not _MATPLOTLIB_AVAILABLE or not groups:
        return
    fig, axes = plt.subplots(len(groups), 2, figsize=(13.0, 4.0 * len(groups)), constrained_layout=True)
    if len(groups) == 1:
        axes = axes.reshape(1, 2)

    for row, group in enumerate(groups):
        mask = labels == group
        bu, au = before_u[mask], after_u[mask]
        bs, as_ = before_s[mask], after_s[mask]

        ax_u = axes[row, 0]
        pos = np.concatenate([bu[bu > 0], au[au > 0]])
        if pos.size:
            bins_u = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
            ax_u.hist(bu, bins=bins_u, alpha=0.55, density=True, label="before", color=BEFORE_COLOR)
            ax_u.hist(au, bins=bins_u, alpha=0.55, density=True, label="after", color=AFTER_COLOR)
            ax_u.set_xscale("log")
            ax_u.set_yscale("log")
        ax_u.set_ylabel(group, fontsize=11, fontweight="bold")
        if row == 0:
            ax_u.set_title(r"Unsigned $S_i(\theta)$")
        if row == len(groups) - 1:
            ax_u.set_xlabel(r"$S_i(\theta)$")
        ax_u.legend(fontsize=8)

        ax_s = axes[row, 1]
        combined = np.concatenate([bs, as_])
        if combined.size:
            lo, hi = np.percentile(combined, [0.5, 99.5])
            if hi > lo:
                bins_s = np.linspace(lo, hi, 60)
                ax_s.hist(bs, bins=bins_s, alpha=0.55, density=True, label="before", color=BEFORE_COLOR)
                ax_s.hist(as_, bins=bins_s, alpha=0.55, density=True, label="after", color=AFTER_COLOR)
        ax_s.axvline(0.0, color="black", linewidth=0.8)
        ax_s.set_yscale("log")
        if row == 0:
            ax_s.set_title(r"Signed $\bar S_i(\theta)$ (0.5th-99.5th pct)")
        if row == len(groups) - 1:
            ax_s.set_xlabel(r"$\bar S_i(\theta)$")
        ax_s.legend(fontsize=8)

    fig.suptitle("Sensitivity distribution before vs. after training, by module")
    fig.savefig(out_path)
    plt.close(fig)
