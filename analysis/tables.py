"""LaTeX tables for the paper, generated from results/ so they cannot drift from the runs.

Table 1 is the headline: t* per setting, as an absolute step count and as a fraction of
training, with the measurement ceiling that defines it and the step-0 value that rules out
"it was fixed at initialisation".

Table 2 reports the budget/placement decomposition -- separately, when the network settles
*how many* weights each layer deserves and *which* weights inside a layer deserve them.
These need not happen together, and if they do not, the asymmetry constrains what an early
pruner can safely do: a settled budget alone licenses layerwise sparsity allocation long
before it licenses a specific mask.

    python -m analysis.tables --tag e1 > paper/tables.tex
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import List

from analysis.claims import load_runs, _sp_key, _series


def _label(m: dict) -> str:
    c = m["config"]
    arch = {"vit": "ViT", "resnet20": "ResNet-20", "mlp": "MLP", "gpt": "GPT"}.get(
        c["model"]["arch"], c["model"]["arch"])
    ds = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "text": "WikiText-2 (char)"}.get(
        c["data"]["dataset"], c["data"]["dataset"])
    return f"{arch} / {ds}"


def _agg(vals: List[float]) -> str:
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.0f}"
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"{mean:.0f}\\,$\\pm$\\,{sd:.0f}"


def table1(runs: List[dict]) -> str:
    groups = defaultdict(list)
    for m in runs:
        groups[_label(m)].append(m)
    rows = []
    for label, ms in sorted(groups.items()):
        key = _sp_key(ms[0])
        tstars = [m["tstar"].get(f"sp{key}_crossing") for m in ms]
        tstars = [t for t in tstars if t is not None]
        steps = ms[0]["config"]["train"]["steps"]
        fracs = [100.0 * t / steps for t in tstars]
        ceil = [m["tstar"].get(f"sp{key}_ceiling") for m in ms]
        ceil = [c for c in ceil if c is not None and c == c]
        init = [_series(m, key)[0] for m in ms]
        plateau = [max(_series(m, key)[:-1]) for m in ms if len(_series(m, key)) > 1]
        rows.append(
            f"{label} & {len(ms)} & {steps} & {_agg(tstars)} & "
            f"{_agg(fracs)}\\% & "
            f"{sum(ceil)/len(ceil):.3f} & "
            f"{sum(init)/len(init):.3f} & "
            f"{sum(plateau)/len(plateau):.3f} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Stabilisation of the sensitivity ordering. $t^*$ is the first step after which
further training perturbs the top-$k$ ordering less than resampling the estimation data
does; the ceiling column is that resampling agreement (Spearman--Brown corrected), which
defines $t^*$ without a free threshold. The step-0 column shows the ordering is \\emph{{not}}
already determined at initialisation. Mean\\,$\\pm$\\,s.d.\\ over seeds.}}
\\label{{tab:tstar}}
\\small
\\begin{{tabular}}{{lrrrrrrr}}
\\toprule
Setting & seeds & steps & $t^*$ & $t^*/T$ & ceiling & adj.\\ at $t{{=}}0$ & plateau \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def table2(runs: List[dict]) -> str:
    """Budget settles when? Placement settles when?"""
    groups = defaultdict(list)
    for m in runs:
        groups[_label(m)].append(m)
    rows = []
    for label, ms in sorted(groups.items()):
        m = ms[0]
        key = _sp_key(m)
        steps = m["steps"]
        # budget: first step where the layer budget is within 5% TV of final
        bud = [r["topk"][key]["budget_distance"] for r in m["vs_final"]]
        t_bud = next((s for s, v in zip(steps, bud) if v <= 0.05), None)
        # placement: first step where within-layer rho reaches 90% of its final value
        wl = [r["within_layer"]["weighted"] for r in m["vs_final"]]
        target = 0.9 * max(wl[:-1]) if len(wl) > 1 else float("nan")
        t_place = next((s for s, v in zip(steps, wl) if v >= target), None)
        rows.append(f"{label} & {t_bud if t_bud is not None else '--'} & "
                    f"{t_place if t_place is not None else '--'} & "
                    f"{m['config']['train']['steps']} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Budget versus placement. The per-layer \\emph{{budget}} (how many weights each layer
keeps) and the within-layer \\emph{{placement}} (which weights) settle at different times.
A settled budget licenses layerwise sparsity allocation; only settled placement licenses a
specific mask.}}
\\label{{tab:decomposition}}
\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Setting & budget settles & placement settles & total steps \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="e1")
    args = ap.parse_args()
    runs = load_runs(args.results, args.tag)
    if not runs:
        print(f"% no runs matching '{args.tag}'")
        return 1
    print(table1(runs))
    print()
    print(table2(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
