"""Does the C3 mechanism hold across runs?

The heavy-tail corollary of Prop. 1 says

    overlap >= 1 - 2 * Delta * rho_k * p / k,

so a *lower* spectral density at the top-k cut (equivalently, wider gaps, equivalently a
heavier-tailed sensitivity spectrum) buys a more stable boundary. That is a prediction
relating two independently measured quantities, made before either was looked at:

    tail index alpha (Hill)  -->  boundary advantage (adj@high-sparsity - adj@low-sparsity)

The pilot (n=3) ordered perfectly: ViT alpha=1.40 (+0.160), ResNet alpha=1.52 (+0.045),
MLP alpha=1.69 (-0.048), with the intermediate variable gap/spread ordering the same way.
Three points is suggestive, not conclusive -- E1 and E2 supply ~40 more across widths,
depths, datasets and architectures, and this module scores them automatically.

    python -m analysis.mechanism --tag e
"""
from __future__ import annotations

import argparse
import math
from typing import List

from analysis.claims import load_runs


def boundary_advantage(m: dict) -> float:
    """Mean (adjusted overlap at the highest sparsity) - (at the lowest), over training.

    Positive means the top-k boundary a pruner uses is more stable than the bulk.
    """
    sps = [f"{s:g}" for s in m["sparsities"]]
    rows = m["vs_final"][:-1]          # drop t=T, where every curve is 1 by construction
    if not rows:
        return float("nan")
    return sum(r["topk"][sps[-1]]["adjusted"] - r["topk"][sps[0]]["adjusted"]
               for r in rows) / len(rows)


def gap_ratio(m: dict) -> float:
    """gap/spread at the extreme cut, the mechanism's intermediate variable."""
    sps = [f"{s:g}" for s in m["sparsities"]]
    try:
        return m["theory"][-1][f"gap_{sps[-1]}"]["gap_over_spread"]
    except (KeyError, IndexError):
        return float("nan")


def _pearson(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return cov / den if den > 0 else float("nan")


def _spearman(xs: List[float], ys: List[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
    if len(pairs) < 3:
        return float("nan")
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    return _pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    runs = load_runs(args.results, args.tag)
    if not runs:
        print("no runs")
        return 1

    alphas, advs, gaps, labels = [], [], [], []
    for m in runs:
        c = m["config"]
        alphas.append(m.get("spectrum_final", {}).get("hill_alpha", float("nan")))
        advs.append(boundary_advantage(m))
        gaps.append(gap_ratio(m))
        labels.append(f"{c['model']['arch']}/{c['data']['dataset']} w{c['model']['width']}"
                      f" d{c['model']['depth']} s{c['seed']}")

    print(f"{'run':38}{'alpha':>9}{'gap/spread':>12}{'boundary adv':>14}")
    for lab, a, g, adv in sorted(zip(labels, alphas, gaps, advs), key=lambda t: t[1]):
        print(f"{lab:38}{a:>9.3f}{g:>12.4f}{adv:>14.3f}")

    print(f"\nn = {len(runs)}")
    print(f"  corr(alpha, boundary advantage)      pearson {_pearson(alphas, advs):+.3f}"
          f"   spearman {_spearman(alphas, advs):+.3f}   [predicted NEGATIVE]")
    print(f"  corr(gap/spread, boundary advantage) pearson {_pearson(gaps, advs):+.3f}"
          f"   spearman {_spearman(gaps, advs):+.3f}   [predicted POSITIVE]")
    print(f"  corr(alpha, gap/spread)              pearson {_pearson(alphas, gaps):+.3f}"
          f"   spearman {_spearman(alphas, gaps):+.3f}   [predicted NEGATIVE]")
    print("\nThe mechanism predicts all three signs. Getting the sign right on the "
          "intermediate\nvariable (gap/spread) as well as the endpoints is what "
          "distinguishes a mechanism\nfrom a correlation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
