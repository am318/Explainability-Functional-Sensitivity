"""Human-readable report for the prune@ x sparsity x method grid (E11/E13/E14).

Reads every `results/_probe*/e14_*.json` (and, as a fallback, `probe_*.json` /
`e13_*.json` from the earlier ad-hoc runs) and the dense control, and renders one markdown
table per architecture: rows are prune time (as % of the fine-tune budget), columns are
sparsity, cells show accuracy for each of the three criteria plus the gap to dense.

    python -m analysis.grid_report --out results/REPORT_grid.md
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

FT = {"mlp": 8000, "resnet20": 16000, "vit": 8000}


def load_cells(probe_dirs: List[str]) -> Dict[str, List[dict]]:
    """arch -> list of result rows, deduplicated by (prune_step, sparsity, criterion),
    later files win (so a cluster run's full grid overrides a local partial one)."""
    by_arch: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for d in probe_dirs:
        for path in sorted(glob.glob(f"{d}/*.json")):
            name = Path(path).name
            if "dense_control" in name:
                continue
            try:
                rows = json.load(open(path))
            except Exception:
                continue
            if not isinstance(rows, list) or not rows:
                continue
            for r in rows:
                if "arch" not in r or "prune_step" not in r:
                    continue
                key = (r["prune_step"], r["sparsity"], r["criterion"])
                by_arch[r["arch"]][key] = r
    return {arch: list(cells.values()) for arch, cells in by_arch.items()}


def load_dense(probe_dirs: List[str]) -> Dict[str, float]:
    dense = {}
    for d in probe_dirs:
        p = Path(d) / "dense_control.json"
        if p.exists():
            for arch, r in json.load(open(p)).items():
                dense[arch] = r["test_acc"]
    return dense


def render_arch(arch: str, rows: List[dict], dense: float | None) -> str:
    ft = FT.get(arch, max((r.get("finetune_steps") or 1 for r in rows), default=1))
    by_step: Dict[int, Dict[float, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_step[r["prune_step"]][r["sparsity"]][r["criterion"]] = r["test_acc"]

    steps = sorted(by_step)
    sparsities = sorted({sp for s in by_step.values() for sp in s})
    criteria = ["sensitivity", "random", "magnitude"]

    out = [f"## {arch}"]
    if dense is not None:
        out.append(f"\nDense control (same fine-tune budget, no pruning): "
                   f"**{dense:.4f}**\n")
    else:
        out.append("\n_(no matched dense control found -- run "
                   "`experiments/e11_dense_control.py` for this arch)_\n")

    header = "| prune@ (%T) | sparsity | " + " | ".join(criteria) + " | best gap to dense |"
    sep = "|---" * (3 + len(criteria)) + "|"
    out += [header, sep]
    for step in steps:
        pct = f"{100*step/ft:.0f}%" if ft else str(step)
        for sp in sparsities:
            cell = by_step[step].get(sp, {})
            if not cell:
                continue
            vals = [cell.get(c) for c in criteria]
            cellstrs = [f"{v:.4f}" if v is not None else "--" for v in vals]
            best = max((v for v in vals if v is not None), default=None)
            gap = f"{dense - best:+.4f}" if (dense is not None and best is not None) else "--"
            out.append(f"| {pct} | {sp:g} | " + " | ".join(cellstrs) + f" | {gap} |")
    n_cells = sum(len(v) for s in by_step.values() for v in s.values())
    out.append(f"\n_{n_cells} cells reported ({len(steps)} prune points x "
               f"{len(sparsities)} sparsities x up to {len(criteria)} criteria)._\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dirs", default="results/_probe,results/_probe_sp05")
    ap.add_argument("--out", default=None, help="write markdown here instead of stdout")
    args = ap.parse_args()

    dirs = [d for d in args.probe_dirs.split(",") if Path(d).exists()]
    cells = load_cells(dirs)
    dense = load_dense(dirs)

    if not cells:
        print(f"no grid results found in {dirs}")
        return 1

    sections = ["# Prune@ x Sparsity x Method grid\n"]
    for arch in sorted(cells):
        sections.append(render_arch(arch, cells[arch], dense.get(arch)))
    report = "\n\n".join(sections)

    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
