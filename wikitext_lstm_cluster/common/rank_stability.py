"""
How much does the *ordering* (and, via Pearson, the raw linear relationship)
of parameter-wise sensitivity change over training? Measured by comparing
sensitivity at epoch e against sensitivity at the final epoch, as a function
of e, under three complementary statistics:

  - Pearson r: linear correlation of the raw values. Sensitive to outliers
    and to the *magnitude* of agreement, not just ordering.
  - Spearman rho: Pearson r computed on ranks. Robust to outliers and
    monotonic (nonlinear) relationships; only cares about ordering.
  - Kendall tau: fraction of concordant minus discordant pairs. Also
    rank-based like Spearman but more robust to small perturbations
    (a single badly-placed point moves it less than it moves Spearman), at
    the cost of being a less familiar scale.

All three near 1 means "epoch e's sensitivity already looks like the final
epoch's, however you slice it"; disagreement between them (e.g. high
Spearman/Kendall but low Pearson) points at *where* epoch e differs -- e.g.
same ordering but different scale.

Per-module breakdowns need full-resolution (unpooled) data to be
trustworthy: the heatmap's pooled rows are allocated proportionally to each
module's parameter count, so a small module (e.g. an embedding layer that's
<1% of the parameters) can collapse to just a handful of pooled rows or even
zero -- far too few points for any of these statistics to mean anything. The
pooled heatmap data is only reliable for a whole-model ("total") curve,
since that always pools over every parameter regardless of module size;
per-module curves here are always computed at full resolution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    _MATPLOTLIB_AVAILABLE = False

_PALETTE = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

METHODS = ("pearson", "spearman", "kendall")
_METHOD_FN = {
    "pearson": lambda a, b: pearsonr(a, b)[0],
    "spearman": lambda a, b: spearmanr(a, b)[0],
    "kendall": lambda a, b: kendalltau(a, b)[0],
}
METHOD_LABELS = {
    "pearson": "Pearson $r$",
    "spearman": "Spearman $\\rho$",
    "kendall": "Kendall $\\tau$",
}


def correlation_to_reference(matrix: np.ndarray, reference_col: int = -1, method: str = "spearman") -> np.ndarray:
    """`method`-correlation between each column of `matrix` (rows=parameters
    /bins, cols=epochs) and the reference column, elementwise over rows."""
    fn = _METHOD_FN[method]
    ref = matrix[:, reference_col]
    n_cols = matrix.shape[1]
    ref_idx = reference_col % n_cols
    out = np.full(n_cols, np.nan)
    for e in range(n_cols):
        out[e] = 1.0 if e == ref_idx else fn(matrix[:, e], ref)
    return out


def pooled_total_correlation_curves(matrix: np.ndarray, reference_col: int = -1) -> Dict[str, np.ndarray]:
    """The one pooled-data curve that's always trustworthy regardless of
    module sizes: whole pooled parameter vector at each epoch vs. the final
    epoch, under every method."""
    return {m: correlation_to_reference(matrix, reference_col, method=m) for m in METHODS}


def full_resolution_correlation_curves(
    per_checkpoint_flat: List[np.ndarray],
    boundaries: List[Tuple[str, int, int]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """per_checkpoint_flat: list of full-resolution flattened score vectors,
    one per checkpoint, in chronological order (last = final/reference).
    boundaries: (group, start, end) index ranges into those flat vectors
    (same for every checkpoint, since architecture doesn't change).
    Returns {method: {'total': ..., group: ...}} correlation-to-final curves.
    """
    final = per_checkpoint_flat[-1]
    groups = sorted({g for g, _, _ in boundaries})
    group_idx = {
        group: np.concatenate([np.arange(s, e) for g, s, e in boundaries if g == group])
        for group in groups
    }
    result: Dict[str, Dict[str, List[float]]] = {m: {"total": [], **{g: [] for g in groups}} for m in METHODS}
    for flat in per_checkpoint_flat:
        is_final = flat is final
        for method in METHODS:
            fn = _METHOD_FN[method]
            result[method]["total"].append(1.0 if is_final else fn(flat, final))
            for group, idx in group_idx.items():
                result[method][group].append(1.0 if is_final else fn(flat[idx], final[idx]))
    return {m: {k: np.array(v) for k, v in groups_dict.items()} for m, groups_dict in result.items()}


def correlation_consecutive(matrix: np.ndarray, method: str = "spearman") -> np.ndarray:
    """corr(column[e], column[e-1]) for e=1..n_cols-1 -- step-to-step change,
    as opposed to correlation_to_reference's change-relative-to-the-end.
    This is what actually tests "did something change a lot in this specific
    window", including the last one: unlike the vs-final curve, the last
    entry here is a real comparison (final vs. second-to-last), not a
    trivial self-comparison, so it's included rather than dropped.
    out[0] is undefined (no previous column) and left as NaN.
    """
    fn = _METHOD_FN[method]
    n_cols = matrix.shape[1]
    out = np.full(n_cols, np.nan)
    for e in range(1, n_cols):
        out[e] = fn(matrix[:, e], matrix[:, e - 1])
    return out


def pooled_total_consecutive_curves(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    return {m: correlation_consecutive(matrix, method=m) for m in METHODS}


def full_resolution_consecutive_curves(
    per_checkpoint_flat: List[np.ndarray],
    boundaries: List[Tuple[str, int, int]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Same idea as full_resolution_correlation_curves, but each checkpoint
    is compared against the *previous* checkpoint rather than the final one.
    Index 0 (no previous checkpoint) is NaN, matching correlation_consecutive
    / pooled_total_consecutive_curves's convention -- same length as
    per_checkpoint_flat, so it slices consistently with the pooled curves.
    """
    groups = sorted({g for g, _, _ in boundaries})
    group_idx = {
        group: np.concatenate([np.arange(s, e) for g, s, e in boundaries if g == group])
        for group in groups
    }
    result: Dict[str, Dict[str, List[float]]] = {
        m: {"total": [np.nan], **{g: [np.nan] for g in groups}} for m in METHODS
    }
    for prev, curr in zip(per_checkpoint_flat[:-1], per_checkpoint_flat[1:]):
        for method in METHODS:
            fn = _METHOD_FN[method]
            result[method]["total"].append(fn(curr, prev))
            for group, idx in group_idx.items():
                result[method][group].append(fn(curr[idx], prev[idx]))
    return {m: {k: np.array(v) for k, v in groups_dict.items()} for m, groups_dict in result.items()}


def _plot_stability_one(
    kind: str,
    history: List[Dict[str, float]],
    pooled_epochs: List[int],
    pooled: Dict[str, np.ndarray],
    checkpoint_epochs: List[int],
    checkpoint: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    *,
    reference_label: str,
    title_word: str,
    drop_last: bool,
    drop_first: bool,
) -> None:
    groups = [g for g in checkpoint[METHODS[0]] if g != "total"]
    colors = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(groups)}
    symbol = "S" if kind == "unsigned" else r"\bar S"

    lo = 1 if drop_first else 0
    hi = -1 if drop_last else None
    sl = slice(lo, hi)
    pooled_epochs_plot = pooled_epochs[sl]
    pooled_plot = {m: pooled[m][sl] for m in pooled}
    checkpoint_epochs_plot = checkpoint_epochs[sl]
    checkpoint_plot = {m: {k: v[sl] for k, v in groups_dict.items()} for m, groups_dict in checkpoint.items()}

    fig, axes = plt.subplots(
        len(METHODS) + 1, 1, figsize=(9.0, 3.0 * (len(METHODS) + 1)), sharex=True,
        gridspec_kw={"height_ratios": [0.8] + [1.0] * len(METHODS)},
        constrained_layout=True,
    )

    ax_loss = axes[0]
    ax_loss.plot([h["epoch"] for h in history], [h["loss"] for h in history], color="black")
    ax_loss.set_ylabel("Train loss")
    ax_loss.set_title("Training loss")
    ax_loss.set_yscale("log")

    for row, method in enumerate(METHODS, start=1):
        ax = axes[row]
        ax.plot(pooled_epochs_plot, pooled_plot[method], color="black", alpha=0.35, linewidth=1,
                label="total (pooled, every epoch)")
        ax.plot(checkpoint_epochs_plot, checkpoint_plot[method]["total"], "o-", color="black", markersize=5,
                label="total (full resolution)")
        for g in groups:
            ax.plot(checkpoint_epochs_plot, checkpoint_plot[method][g], "o-", color=colors[g], markersize=4, label=g)
        ax.axhline(1.0, color="gray", linewidth=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(METHOD_LABELS[method])
        ax.set_title(f"{METHOD_LABELS[method]}: " rf"corr$({symbol}_i^{{(e)}}, {symbol}_i^{{{reference_label}}})$",
                     fontsize=10)
        if row == 1:
            ax.legend(fontsize=8, loc="lower right")

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(
        f"Loss vs. {title_word} of {kind} parameter-wise sensitivity over training\n"
        "(per-module curves full-resolution)",
        fontsize=10,
    )
    fig.savefig(out_path)
    plt.close(fig)


def plot_rank_stability(
    history: List[Dict[str, float]],
    pooled_epochs: List[int],
    pooled_unsigned: Dict[str, np.ndarray],
    pooled_signed: Dict[str, np.ndarray],
    checkpoint_epochs: List[int],
    checkpoint_unsigned: Dict[str, Dict[str, np.ndarray]],
    checkpoint_signed: Dict[str, Dict[str, np.ndarray]],
    out_path_unsigned: Path,
    out_path_signed: Path,
) -> None:
    """Two separate figures (unsigned, signed), each with training loss on
    top and Pearson/Spearman/Kendall correlation-vs-*final*-epoch below, all
    sharing the same epoch x-axis. The final point is dropped: it compares
    the reference epoch against itself, so it is trivially exactly 1.0 for
    every method and every group by construction, not a measurement -- left
    in, it reads as a sharp late jump that has nothing to do with training
    dynamics. Note this means the last real transition (second-to-last
    checkpoint -> final) is invisible here; use plot_rank_stability_consecutive
    to see whether anything unusual happens in that specific window.

    pooled_* : {method: array} dense (every tracked epoch), whole-model-only
    curves from the pooled heatmap data.
    checkpoint_* : {method: {'total':..., group:...}} authoritative
    full-resolution curves at a sparser set of checkpoint epochs.
    """
    if not _MATPLOTLIB_AVAILABLE:
        return
    for kind, pooled, checkpoint, out_path in [
        ("unsigned", pooled_unsigned, checkpoint_unsigned, out_path_unsigned),
        ("signed", pooled_signed, checkpoint_signed, out_path_signed),
    ]:
        _plot_stability_one(
            kind, history, pooled_epochs, pooled, checkpoint_epochs, checkpoint, out_path,
            reference_label=r"^{(\mathrm{final})}", title_word="rank stability vs. final epoch",
            drop_last=True, drop_first=False,
        )


def plot_rank_stability_consecutive(
    history: List[Dict[str, float]],
    pooled_epochs: List[int],
    pooled_unsigned: Dict[str, np.ndarray],
    pooled_signed: Dict[str, np.ndarray],
    checkpoint_epochs: List[int],
    checkpoint_unsigned: Dict[str, Dict[str, np.ndarray]],
    checkpoint_signed: Dict[str, Dict[str, np.ndarray]],
    out_path_unsigned: Path,
    out_path_signed: Path,
) -> None:
    """Same layout as plot_rank_stability, but each point compares a
    checkpoint against the *previous* checkpoint/epoch rather than the
    final one -- i.e. step-to-step change, not change-relative-to-the-end.
    This directly answers "did the ranking change a lot in this specific
    window", including the last window, since there is no trivial
    self-comparison to exclude here (the point at the final epoch is a real
    comparison: final vs. second-to-last).

    Inputs are the *_consecutive curves from pooled_total_consecutive_curves
    / full_resolution_consecutive_curves (index 0 undefined, dropped here).
    """
    if not _MATPLOTLIB_AVAILABLE:
        return
    for kind, pooled, checkpoint, out_path in [
        ("unsigned", pooled_unsigned, checkpoint_unsigned, out_path_unsigned),
        ("signed", pooled_signed, checkpoint_signed, out_path_signed),
    ]:
        _plot_stability_one(
            kind, history, pooled_epochs, pooled, checkpoint_epochs, checkpoint, out_path,
            reference_label=r"^{(e-1)}", title_word="step-to-step rank stability",
            drop_last=False, drop_first=True,
        )
