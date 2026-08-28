"""Paper figures, one function per figure, each reading results/ directly.

Figures are the argument, so each one is built to show its own controls rather than a
flattering curve: Fig. 1 draws the noise-floor ceiling and the chance level on the same
axes as the signal, and Fig. 3 draws the theory's failure margin rather than only its fit.

    python -m analysis.figures --all
    python -m analysis.figures --fig 1 --tag e1
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.claims import load_runs

OUT = Path("paper/figures")
PALETTE = ["#1b6ca8", "#c1440e", "#2a9d8f", "#8d5524", "#6a4c93"]
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 7,
})


def _x(steps: List[int]) -> np.ndarray:
    """Steps on a log axis, with step 0 placed at half the first nonzero step."""
    s = np.array(steps, dtype=float)
    first = s[s > 0].min() if (s > 0).any() else 1.0
    s[s == 0] = first / 2
    return s


def _group(runs: List[dict], key) -> Dict[str, List[dict]]:
    out = defaultdict(list)
    for m in runs:
        out[key(m)].append(m)
    return dict(out)


def _arch(m: dict) -> str:
    c = m["config"]
    return f"{c['model']['arch']}/{c['data']['dataset']}"


# ---------------------------------------------------------------------------

def fig1_stability(runs: List[dict], out: Path) -> None:
    """C1/C2/C3: the core curve, with every control drawn on the same axes."""
    groups = _group(runs, _arch)
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.4), squeeze=False, sharey=True)
    for ax, (name, ms) in zip(axes[0], sorted(groups.items())):
        m0 = ms[0]
        steps = m0["steps"]
        x = _x(steps)
        sps = m0["sparsities"]
        for i, sp in enumerate([s for s in sps if s in (0.5, 0.9, 0.99)]):
            key = f"{sp:g}"
            curves = np.array([[r["topk"][key]["adjusted"] for r in m["vs_final"]]
                               for m in ms if m["steps"] == steps])
            mean, sd = curves.mean(0), curves.std(0)
            ax.plot(x, mean, color=PALETTE[i], lw=1.4, label=f"top-k, sparsity {sp:g}")
            if len(curves) > 1:
                ax.fill_between(x, mean - sd, mean + sd, color=PALETTE[i], alpha=0.18, lw=0)
        # within-layer Spearman: the layerwise confound stripped entirely
        wl = np.array([[r["within_layer"]["weighted"] for r in m["vs_final"]]
                       for m in ms if m["steps"] == steps]).mean(0)
        ax.plot(x, wl, color="0.35", lw=1.0, ls="--", label="within-layer $\\rho$")
        # measurement ceiling from disjoint data folds
        ceil = m0["tstar"].get(f"sp0.9_ceiling")
        if ceil:
            ax.axhline(ceil, color="0.2", lw=0.8, ls=":", zorder=0)
            ax.text(x[-1], ceil + 0.015, "measurement ceiling", ha="right", fontsize=6,
                    color="0.2")
        t = m0["tstar"].get("sp0.9_crossing")
        if t:
            ax.axvline(max(t, x[0]), color="#c1440e", lw=0.8, alpha=0.7)
            ax.text(max(t, x[0]) * 1.15, 0.05, f"$t^*$={t}", fontsize=6, color="#c1440e")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("training step")
    axes[0][0].set_ylabel("adjusted overlap with $S_{\\rm final}$")
    axes[0][-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig1_stability.pdf")
    fig.savefig(out / "fig1_stability.png")
    print(f"  fig1 <- {len(runs)} runs, {len(groups)} panels")


def fig1b_controls(runs: List[dict], out: Path) -> None:
    """The same run shown with raw vs chance-corrected metrics side by side.

    This is the panel that answers 'your ranking is just a layer budget': the raw curve
    and the chance level sit on top of each other, and only the adjusted curve carries
    information."""
    m = runs[0]
    steps, x = m["steps"], _x(m["steps"])
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    key = "0.9"
    raw = [r["topk"][key]["overlap"] for r in m["vs_final"]]
    chance = [r["topk"][key]["chance_layer"] for r in m["vs_final"]]
    gchance = [r["topk"][key]["chance_global"] for r in m["vs_final"]]
    adj = [r["topk"][key]["adjusted"] for r in m["vs_final"]]
    rho = [r["spearman"] for r in m["vs_final"]]
    ax.plot(x, rho, color="0.6", lw=1.0, ls="-.", label="global Spearman $\\rho$")
    ax.plot(x, raw, color=PALETTE[0], lw=1.4, label="raw top-k overlap")
    ax.plot(x, chance, color=PALETTE[1], lw=1.0, ls="--", label="chance (layer-budget matched)")
    ax.plot(x, gchance, color="0.75", lw=0.8, ls=":", label="chance (uniform)")
    ax.plot(x, adj, color=PALETTE[2], lw=1.6, label="adjusted overlap")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"agreement with $S_{{\\rm final}}$ (sparsity {key})")
    ax.set_title(_arch(m), fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig1b_controls.pdf")
    fig.savefig(out / "fig1b_controls.png")
    print("  fig1b <- controls panel")


def fig2_tstar(runs: List[dict], out: Path) -> None:
    """C4: does t* move systematically, and does it track kernel velocity?"""
    factors = [("model.width", "width"), ("model.depth", "depth"),
               ("train.lr", "learning rate"), ("train.batch_size", "batch size")]
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.2))
    for ax, (path, label) in zip(axes, factors):
        sec, field = path.split(".")
        pts = defaultdict(list)
        for m in runs:
            t = m["tstar"].get("sp0.9_crossing")
            if t is None:
                continue
            pts[_arch(m)].append((m["config"][sec][field], t))
        for i, (name, ps) in enumerate(sorted(pts.items())):
            if len({p[0] for p in ps}) < 2:
                continue
            ps.sort()
            ax.plot([p[0] for p in ps], [p[1] for p in ps], "o-", ms=3, lw=1.2,
                    color=PALETTE[i % len(PALETTE)], label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(label)
    axes[0].set_ylabel("$t^*$ (steps)")
    axes[0].legend(loc="best")

    ax = axes[4]
    for i, (name, ms) in enumerate(sorted(_group(runs, _arch).items())):
        xs, ys = [], []
        for m in ms:
            t = m["tstar"].get("sp0.9_crossing")
            kv = [r.get("kernel_velocity_step") for r in m["laziness"]
                  if r.get("kernel_velocity_step") is not None]
            if t is not None and kv:
                xs.append(float(np.mean(kv))); ys.append(t)
        if xs:
            ax.plot(xs, ys, "o", ms=4, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_xlabel("mean kernel velocity per checkpoint")
    ax.set_ylabel("$t^*$")
    ax.set_yscale("log")
    ax.set_title("t* vs departure from lazy", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_tstar.pdf")
    fig.savefig(out / "fig2_tstar.png")
    print(f"  fig2 <- {len(runs)} runs")


def fig3_theory(runs: List[dict], out: Path) -> None:
    """C5: the drift model's point prediction, the counting bound, and the spectrum."""
    m = runs[0]
    x = _x(m["steps"])
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.3))

    ax = axes[0]
    for i, sp in enumerate([s for s in m["sparsities"] if s in (0.5, 0.9, 0.99)]):
        key = f"{sp:g}"
        obs = [r[f"obs_overlap_{key}"] for r in m["theory"]]
        pred = [r[f"pred_overlap_{key}"] for r in m["theory"]]
        ax.plot(x, obs, "-", color=PALETTE[i], lw=1.4, label=f"measured (sp {key})")
        ax.plot(x, pred, "--", color=PALETTE[i], lw=1.0, alpha=0.8,
                label=f"predicted (sp {key})")
    ax.set_xscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("top-k overlap")
    ax.set_title("drift model: one scalar per step", fontsize=8)
    ax.legend(loc="lower right", ncol=2, fontsize=5.5)

    ax = axes[1]
    for i, sp in enumerate([s for s in m["sparsities"] if s in (0.9, 0.99)]):
        key = f"{sp:g}"
        obs = [r[f"obs_overlap_{key}"] for r in m["theory"]]
        bound = [r[f"bound_overlap_{key}"] for r in m["theory"]]
        ax.plot(x, obs, "-", color=PALETTE[i], lw=1.4, label=f"measured (sp {key})")
        ax.plot(x, bound, ":", color=PALETTE[i], lw=1.2, label=f"bound (sp {key})")
        ax.fill_between(x, bound, obs, color=PALETTE[i], alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xlabel("training step")
    ax.set_title("counting bound (must lie below)", fontsize=8)
    ax.legend(loc="lower right", fontsize=6)

    ax = axes[2]
    alpha = [r["hill_alpha"] for r in m["theory"]]
    sigma = [r["drift_sigma"] for r in m["theory"]]
    ax.plot(x, sigma, color=PALETTE[0], lw=1.4, label="drift scale $\\sigma_t$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("$\\sigma_t$", color=PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(x, alpha, color=PALETTE[1], lw=1.2, ls="--")
    ax2.set_ylabel("Hill tail index $\\alpha$", color=PALETTE[1])
    ax2.grid(False)
    ax.set_title("drift scale and tail index", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_theory.pdf")
    fig.savefig(out / "fig3_theory.png")
    print("  fig3 <- theory panel")


def fig4_prune(panels: List[Path], out: Path) -> None:
    """C6: accuracy against prune time, with t* marked."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.4),
                             squeeze=False)
    for ax, path in zip(axes[0], panels):
        rows = json.load(open(path))
        metrics = json.load(open(path.parent / "metrics.json"))
        by = defaultdict(list)
        for r in rows:
            by[(r["criterion"], r["sparsity"])].append((r["prune_step"], r["test_acc"]))
        for i, ((crit, sp), pts) in enumerate(sorted(by.items())):
            pts.sort()
            xs = _x([p[0] for p in pts])
            ax.plot(xs, [p[1] for p in pts], "o-", ms=3, lw=1.2,
                    color=PALETTE[i % len(PALETTE)], label=f"{crit}, sp {sp:g}")
        t = metrics["tstar"].get("sp0.9_crossing")
        if t:
            ax.axvline(max(t, 1), color="0.3", lw=0.8, ls="--")
            ax.text(max(t, 1) * 1.1, ax.get_ylim()[0] + 0.01, "$t^*$", fontsize=6)
        dense = metrics.get("final_eval", {}).get("test_acc")
        if dense:
            ax.axhline(dense, color="0.5", lw=0.8, ls=":")
            ax.text(xs[0], dense + 0.004, "dense", fontsize=6, color="0.4")
        ax.set_xscale("log")
        ax.set_xlabel("prune step $t$")
        ax.set_ylabel("final test accuracy")
        ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig4_prune.pdf")
    fig.savefig(out / "fig4_prune.png")
    print(f"  fig4 <- {len(panels)} panel(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--fig", type=int, action="append", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    want = set(args.fig) or ({1, 2, 3, 4} if args.all else {1})
    print(f"figures -> {OUT}/")

    if 1 in want:
        runs = load_runs(args.results, args.tag or "e1")
        if runs:
            fig1_stability(runs, OUT)
            fig1b_controls(runs, OUT)
        else:
            print("  fig1: no e1 runs yet")
    if 2 in want:
        runs = load_runs(args.results, args.tag or "e2")
        if runs:
            fig2_tstar(runs, OUT)
        else:
            print("  fig2: no e2 runs yet")
    if 3 in want:
        runs = load_runs(args.results, args.tag or "e1")
        if runs:
            fig3_theory(runs, OUT)
        else:
            print("  fig3: no runs yet")
    if 4 in want:
        panels = sorted(Path(args.results).glob("*/prune_panel.json"))
        if panels:
            fig4_prune(panels, OUT)
        else:
            print("  fig4: no prune panels yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
