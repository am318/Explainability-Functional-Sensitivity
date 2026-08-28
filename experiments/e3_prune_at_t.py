"""E3 -- the feasibility panel. Claim C6, and the seed of the follow-up paper.

Two phases:
  1. a reference run that saves weights and sensitivity scores at every checkpoint;
  2. for a handful of prune times t, prune to a target sparsity and train to completion.

The claim under test is narrow on purpose: *t\\* predicts the usable prune time*. If final
accuracy is flat for t >= t* and degrades below it, then the measurement in E1 has
operational content and the follow-up paper has a foundation. If accuracy keeps climbing
well past t*, C6 fails and we say so -- the dynamics result in E1 survives either way,
which is exactly why the measurement paper comes first.

The random baseline is layer-budget-matched, so a win over it is a win about *which*
weights, not about *how many per layer* -- the distinction that sank a generation of
prune-at-init methods.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from fsd import config as C, storage
from fsd.prune import prune_and_continue
from fsd.run import execute
from fsd.schedule import log_checkpoints
from experiments._common import base


def reference_config(arch: str, steps: int, seed: int = 0) -> C.RunCfg:
    cfg = base("e3ref", arch, steps=steps)
    cfg.seed = seed
    cfg.keep_scores = "all"
    cfg.save_state_at = [-1]     # every checkpoint, so any prune time is reachable
    cfg.n_ckpts = 14
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="resnet20")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sparsities", default="0.9,0.99")
    ap.add_argument("--n-prune-times", type=int, default=6)
    ap.add_argument("--criteria", default="sensitivity,magnitude,random")
    ap.add_argument("--reference-only", action="store_true")
    args = ap.parse_args()

    cfg = reference_config(args.arch, args.steps, args.seed)
    run = storage.Run(cfg.out_dir, cfg.run_id())
    if not run.exists("metrics.json"):
        print(f"=== phase 1: reference run {cfg.run_id()} ===")
        execute(cfg)
    else:
        print(f"=== phase 1: reusing {cfg.run_id()} ===")
    metrics = json.load(open(run.dir / "metrics.json"))
    if args.reference_only:
        print(json.dumps(metrics["tstar"], indent=2))
        return 0

    ckpts = metrics["steps"]
    # log-spaced subset of the available checkpoints, always including 0 and the last
    idx = sorted(set(np.round(np.linspace(0, len(ckpts) - 1, args.n_prune_times)).astype(int)))
    prune_times = [ckpts[i] for i in idx]
    sparsities = [float(s) for s in args.sparsities.split(",")]
    criteria = args.criteria.split(",")

    print(f"=== phase 2: prune at {prune_times} x sparsity {sparsities} "
          f"x {criteria} ===")
    print(f"    t* (crossing) = "
          f"{ {k: v for k, v in metrics['tstar'].items() if k.endswith('crossing')} }")

    results = []
    for t in prune_times:
        state_path = run.dir / "state" / f"step_{t}.pt"
        score_path = run.dir / "scores" / f"step_{t}.npz"
        if not state_path.exists() or not score_path.exists():
            print(f"  missing artefacts for step {t}, skipping")
            continue
        state = torch.load(state_path, map_location="cpu")
        scores = torch.from_numpy(np.load(score_path)["scores"])
        for sp in sparsities:
            for crit in criteria:
                r = prune_and_continue(cfg, t, sp, crit, state, flat_scores=scores)
                results.append(r)
                json.dump(results, open(run.dir / "prune_panel.json", "w"), indent=2)
    print(f"\nwrote {run.dir}/prune_panel.json  ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
