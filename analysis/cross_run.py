"""Compare sensitivity orderings *across* runs that share an initialisation.

Only valid for runs with identical `seed` (initialisation) and differing `data_seed`:
parameter indices are comparable only when the networks start from the same weights.

    python -m analysis.cross_run --tag e6
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from fsd import rank_metrics as R


def _scores(run_dir: Path, step: int) -> torch.Tensor:
    return torch.from_numpy(np.load(run_dir / "scores" / f"step_{step}.npz")["scores"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="e6")
    ap.add_argument("--sparsity", type=float, default=0.9)
    args = ap.parse_args()

    runs: Dict[tuple, List[dict]] = {}
    for path in sorted(glob.glob(f"{args.results}/{args.tag}*/metrics.json")):
        m = json.load(open(path))
        c = m["config"]
        key = (c["model"]["arch"], c["data"]["dataset"], c["seed"])
        m["_dir"] = Path(path).parent
        runs.setdefault(key, []).append(m)

    if not runs:
        print(f"no runs matching '{args.tag}'")
        return 1

    print(f"sparsity {args.sparsity:g}; runs share an initialisation and differ only in "
          f"batch order\n")
    print(f"{'setting':22} {'step':>6} {'within-run':>11} {'predictive':>11} {'across-run':>11}")
    for (arch, ds, seed), ms in sorted(runs.items()):
        if len(ms) < 2:
            print(f"{arch}/{ds} seed{seed}: need 2 runs, have {len(ms)}")
            continue
        a, b = ms[0], ms[1]
        steps = [s for s in a["steps"] if s in b["steps"]]
        final = steps[-1]
        try:
            fa, fb = _scores(a["_dir"], final), _scores(b["_dir"], final)
        except FileNotFoundError:
            print(f"{arch}/{ds}: raw scores missing (need keep_scores='all')")
            continue
        mask_fa, mask_fb = R.topk_mask(fa, args.sparsity), R.topk_mask(fb, args.sparsity)
        across = R.overlap(mask_fa, mask_fb)
        for step in steps:
            sa = _scores(a["_dir"], step)
            within = R.overlap(R.topk_mask(sa, args.sparsity), mask_fa)
            predictive = R.overlap(R.topk_mask(sa, args.sparsity), mask_fb)
            print(f"{arch+'/'+ds:22} {step:>6} {within:>11.3f} {predictive:>11.3f} "
                  f"{across:>11.3f}")
        print()
    print("within-run : S_t^(a) vs S_T^(a)   predictive : S_t^(a) vs S_T^(b)   "
          "across-run : S_T^(a) vs S_T^(b)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
