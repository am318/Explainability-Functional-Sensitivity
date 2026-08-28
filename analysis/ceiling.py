"""Normalise early-vs-final agreement by what is reproducible at all.

Reads a pair of runs sharing an initialisation and differing only in batch order, and
reports, for each checkpoint t:

    raw      = rho(S_t^A, S_T^A)                  agreement with its own final ordering
    ceiling  = rho(S_T^A, S_T^B)                  agreement between two converged runs
    frac     = raw / ceiling                      fraction of the KNOWABLE structure at t

`frac` is the quantity the "do early sensitivities resemble trained ones" question actually
asks about. A raw value of 0.16 against a ceiling of 0.20 (frac 0.8) and the same raw value
against a ceiling of 0.95 (frac 0.17) are opposite conclusions from identical raw numbers.

All statistics are computed on prunable parameters only and within-layer, since the
cross-layer scale structure is shared by every quantity here and would inflate all of them
alike -- including the ceiling, which would make `frac` look better than it is.

    python -m analysis.ceiling --tag e12
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from fsd import config as C, models, rank_metrics as R, sensitivity as S


def _load(run_dir: Path):
    files = sorted(glob.glob(f"{run_dir}/scores/step_*.npz"),
                   key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    steps = [int(re.search(r"step_(\d+)", f).group(1)) for f in files]
    return steps, [torch.from_numpy(np.load(f)["scores"].astype(np.float32)) for f in files]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="e12")
    args = ap.parse_args()

    groups: Dict[str, List[Path]] = {}
    for p in sorted(glob.glob(f"{args.results}/{args.tag}*/metrics.json")):
        m = json.load(open(p))
        c = m["config"]
        key = f"{c['model']['arch']}/{c['data']['dataset']}"
        groups.setdefault(key, []).append(Path(p).parent)

    if not groups:
        print(f"no runs matching '{args.tag}'")
        return 1

    for key, dirs in sorted(groups.items()):
        if len(dirs) < 2:
            print(f"\n{key}: need 2 runs (same init, different data_seed), have {len(dirs)}"
                  f" -- still running?")
            continue
        a, b = dirs[0], dirs[1]
        cfg = C.load(f"{a}/config.json")
        torch.manual_seed(cfg.seed)
        model = models.build(cfg.model, cfg.data)
        names = S.param_names(model)
        lids = S.layer_index(model, names)
        nl = len(names)
        prun = S.flatten(S.prunable_mask(model, cfg.sens), names)

        steps_a, data_a = _load(a)
        steps_b, data_b = _load(b)
        fa, fb = data_a[-1], data_b[-1]

        ceil_wl = R.within_layer_spearman(fa[prun], fb[prun], lids[prun], nl)["weighted"]
        ceil_g = R.spearman(fa[prun], fb[prun])
        print(f"\n=== {key} ===")
        print(f"  CEILING rho(S_T^A, S_T^B): within-layer={ceil_wl:.4f}  global={ceil_g:.4f}")
        print(f"  (two runs, identical init, batch order differs only)\n")
        print(f"  {'step':>7} {'%T':>5} {'raw_wl':>9} {'frac_of_ceiling':>17}"
              f" {'raw_top10%':>12} {'ceil_top10%':>12} {'frac':>7}")
        T = steps_a[-1]
        ceil_top = R.overlap(R.topk_mask(fa[prun], 0.9), R.topk_mask(fb[prun], 0.9))
        for s, v in zip(steps_a, data_a):
            wl = R.within_layer_spearman(v[prun], fa[prun], lids[prun], nl)["weighted"]
            ov = R.overlap(R.topk_mask(v[prun], 0.9), R.topk_mask(fa[prun], 0.9))
            f_wl = wl / ceil_wl if ceil_wl > 0 else float("nan")
            f_ov = ov / ceil_top if ceil_top > 0 else float("nan")
            print(f"  {s:>7} {100*s/T:>4.0f}% {wl:>9.4f} {f_wl:>17.3f}"
                  f" {ov:>12.4f} {ceil_top:>12.4f} {f_ov:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
