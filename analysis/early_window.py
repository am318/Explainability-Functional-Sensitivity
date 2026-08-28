"""Range report: AUROC / rho / calibration across training time AND sparsity.

Every number in this project so far has either fixed the sparsity (stability curves) or
fixed the checkpoint (the single-snapshot AUROC/calibration tests). This renders the full
2D range -- checkpoints from t=0 to ~50% of training, crossed with the sparsities pruning
actually uses -- as one table, so "does early sensitivity predict final importance" has an
answer that shows its variation across both axes rather than one favourable point.

Two data sources, used transparently:
  * raw per-checkpoint score vectors (results/<run>/scores/step_*.npz, from runs with
    keep_scores="all") -- used for runs predating the AUROC/calibration wiring in
    fsd/run.py, recomputing the metrics directly from the saved scores.
  * metrics.json's vs_final[*]['auroc'/'calibration'] -- used for any run executed after
    that wiring landed, which already carries these fields for free.

    python -m analysis.early_window --run results/e3ref-6b2398644d
    python -m analysis.early_window --run results/e1p-84953d2a11 --max-frac 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from fsd import config as C, models, rank_metrics as R, sensitivity as S


def from_raw_scores(run_dir: Path, sparsities: List[float], max_frac: float) -> None:
    cfg = C.load(f"{run_dir}/config.json")
    torch.manual_seed(cfg.seed)
    model = models.build(cfg.model, cfg.data)
    names = S.param_names(model)
    prun = S.flatten(S.prunable_mask(model, cfg.sens), names)

    files = sorted(glob.glob(f"{run_dir}/scores/step_*.npz"),
                   key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    if not files:
        print(f"  no raw score checkpoints in {run_dir}/scores/ "
              f"(need keep_scores='all'); trying metrics.json instead")
        return from_metrics_json(run_dir, sparsities, max_frac)

    steps = [int(re.search(r"step_(\d+)", f).group(1)) for f in files]
    data = [torch.from_numpy(np.load(f)["scores"].astype(np.float32))[prun] for f in files]
    T = steps[-1]
    final = data[-1]

    cutoff = max_frac * T
    rows = [(s, v) for s, v in zip(steps, data) if s <= cutoff or s == steps[-1]]

    print(f"\n=== {run_dir.name}  ({cfg.model.arch}/{cfg.data.dataset}, T={T} steps) ===")
    print(f"showing checkpoints up to {100*max_frac:.0f}% of training "
          f"({len(rows)} of {len(steps)} checkpoints)\n")

    print("AUROC(S_t predicts top-k(S_final))  --  0.5=random, 1.0=perfect")
    header = f"{'step':>7}{'%T':>6}" + "".join(f"  sp={sp:g}" for sp in sparsities)
    print(header)
    for s, v in rows:
        cells = "".join(f"  {R.auroc_topk(v, final, sp):>6.3f}" for sp in sparsities)
        print(f"{s:>7}{100*s/T:>5.1f}%{cells}")

    print("\nSpearman rho(S_t, S_final)  --  for reference, same checkpoints")
    for s, v in rows:
        print(f"  step {s:>6} ({100*s/T:>4.1f}%): rho={R.spearman(v, final):.4f}")

    print("\nCalibration: monotonic decile staircase? spearman-of-bin-means")
    for s, v in rows:
        c = R.calibration_deciles(v, final)
        print(f"  step {s:>6} ({100*s/T:>4.1f}%): monotonic={str(c['monotonic']):5} "
              f"rho_of_means={c['spearman_of_bin_means']:.3f}")


def from_metrics_json(run_dir: Path, sparsities: List[float], max_frac: float) -> None:
    m = json.load(open(run_dir / "metrics.json"))
    T = m["config"]["train"]["steps"]
    rows = [r for r in m["vs_final"] if r["step"] <= max_frac * T or r["step"] == T]
    if not rows or "auroc" not in rows[0]:
        print(f"  {run_dir}: no AUROC data (run predates the wiring and has no raw scores "
              f"-- re-run with keep_scores='all', or on the current codebase)")
        return
    c = m["config"]
    print(f"\n=== {run_dir.name}  ({c['model']['arch']}/{c['data']['dataset']}, T={T}) ===")
    print(f"showing checkpoints up to {100*max_frac:.0f}% of training "
         f"({len(rows)} of {len(m['vs_final'])} checkpoints)\n")
    print("AUROC(S_t predicts top-k(S_final))")
    header = f"{'step':>7}{'%T':>6}" + "".join(f"  sp={sp:g}" for sp in sparsities)
    print(header)
    for r in rows:
        cells = "".join(f"  {r['auroc'].get(f'{sp:g}', float('nan')):>6.3f}"
                        for sp in sparsities)
        print(f"{r['step']:>7}{100*r['step']/T:>5.1f}%{cells}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="path to a results/<run_id> directory; repeatable")
    ap.add_argument("--sparsities", default="0.5,0.7,0.9,0.95,0.99")
    ap.add_argument("--max-frac", type=float, default=0.5,
                    help="show checkpoints up to this fraction of total training (+final)")
    args = ap.parse_args()
    sps = [float(x) for x in args.sparsities.split(",")]
    for run in args.run:
        from_raw_scores(Path(run), sps, args.max_frac)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
