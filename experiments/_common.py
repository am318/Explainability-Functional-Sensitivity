"""Shared driver for every experiment.

Each experiment module exposes `configs()` -> List[RunCfg]. This driver either runs them
here or writes them out as a job array. Nothing about an experiment changes between the
laptop and the cluster except which of those two you pick.

    python -m experiments.e1_rank_stability --list
    python -m experiments.e1_rank_stability --run
    python -m experiments.e1_rank_stability --emit jobs/e1     # + jobs/e1/submit.sh
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsd import config as C
from fsd.run import execute

SLURM = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --array=0-{last}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time={walltime}
#SBATCH --output={dir}/logs/%A_%a.out
mkdir -p {dir}/logs
CONFIGS=({dir}/cfg_*.json)
python -m fsd.cli --config "${{CONFIGS[$SLURM_ARRAY_TASK_ID]}}"
"""


def driver(name: str, configs_fn: Callable[[], List[C.RunCfg]], doc: str = "") -> int:
    ap = argparse.ArgumentParser(prog=name, description=doc)
    ap.add_argument("--list", action="store_true", help="show the runs and exit")
    ap.add_argument("--run", action="store_true", help="run sequentially here")
    ap.add_argument("--emit", metavar="DIR", help="write configs + a slurm array script")
    ap.add_argument("--only", type=int, default=None, help="run just this index")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip runs whose metrics.json already exists")
    args = ap.parse_args()

    cfgs = configs_fn()
    if args.only is not None:
        cfgs = [cfgs[args.only]]

    if args.list or not (args.run or args.emit):
        for i, c in enumerate(cfgs):
            print(f"[{i:3d}] {c.run_id():28s} {c.model.arch:9s} {c.data.dataset:10s} "
                  f"w{c.model.width} d{c.model.depth} lr{c.train.lr:g} "
                  f"bs{c.train.batch_size} steps{c.train.steps} seed{c.seed}")
        print(f"{len(cfgs)} runs")
        return 0

    if args.emit:
        out = Path(args.emit)
        out.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(cfgs):
            C.dump(c, str(out / f"cfg_{i:04d}.json"))
        # Walltime from the longest run in the array, with generous headroom: an array
        # killed at the wall loses every job in it, and asking for a few extra hours costs
        # only queue priority.
        max_steps = max(c.train.steps for c in cfgs)
        hours = max(4, int(2 + max_steps / 2500))
        (out / "submit.sh").write_text(
            SLURM.format(name=name, last=len(cfgs) - 1, dir=str(out),
                         walltime=f"{hours:02d}:00:00"))
        print(f"wrote {len(cfgs)} configs to {out}/  ->  sbatch {out}/submit.sh")
        return 0

    t0 = time.time()
    for i, c in enumerate(cfgs):
        done = Path(c.out_dir) / c.run_id() / "metrics.json"
        if args.skip_done and done.exists():
            print(f"[{i+1}/{len(cfgs)}] skip {c.run_id()} (done)")
            continue
        print(f"\n[{i+1}/{len(cfgs)}] {c.run_id()}")
        try:
            execute(c)
        except Exception as exc:  # keep the sweep alive; a dead run is data too
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    print(f"\ntotal {time.time()-t0:.0f}s")
    return 0


def base(tag: str, arch: str, dataset: str = "cifar10", steps: int = 12000,
         sens_samples: int = 2048, **train_kw) -> C.RunCfg:
    """Default run configuration.

    Two numbers here were set by measurement rather than convention.

    `sens_samples=2048` comes from the noise-floor calibration
    (tests/calibrate_samples.py): it puts the measurement ceiling at ~0.96 adjusted
    overlap, leaving room for a real signal to be visible beneath it.

    `steps=12000` (~30 epochs of CIFAR-10) comes from the E1 pilot, which ran 4000 steps
    and produced a monotonically rising overlap curve with no plateau. The training loss
    was still descending at the end, so "agreement with S_final" was largely measuring
    convergence: the reference itself had not settled. A stability claim needs the
    reference to be taken from a converged model, so every run must outlast convergence
    by a margin. E7 tests this explicitly.
    """
    cfg = C.RunCfg(tag=tag, out_dir="results")
    cfg.data = C.DataCfg(dataset=dataset, data_dir="./data", image_size=32,
                         augment=True, workers=4, test_subset=5000)
    cfg.train = C.TrainCfg(steps=steps, batch_size=128, lr=1e-3, warmup_steps=200,
                           **train_kw)
    cfg.sens = C.SensCfg(n_samples=sens_samples, batch_size=32, folds=2, ntk_examples=48,
                         n_probes=8)
    cfg.n_ckpts = 22
    cfg.sparsities = [0.5, 0.8, 0.9, 0.95, 0.99]
    cfg.keep_scores = "none"
    if arch == "vit":
        cfg.model = C.ModelCfg(arch="vit", width=192, depth=6, heads=3, patch_size=4)
    elif arch == "resnet20":
        cfg.model = C.ModelCfg(arch="resnet20", width=16, depth=20)
    elif arch == "mlp":
        cfg.model = C.ModelCfg(arch="mlp", width=512, depth=4)
    elif arch == "gpt":
        # Sized against measured cost, not aesthetics. The GPT output dimension is
        # block_size x vocab (~6k), so the exact estimator is impossible and every probe
        # costs a full backward pass. Extrapolating tests/smoke_text.py, a 2.7M-param GPT
        # at 512 samples x 16 probes would cost ~18 min *per checkpoint*. This
        # configuration lands near 90s instead, at the price of a lower measurement
        # ceiling -- which the runs report rather than hide.
        cfg.model = C.ModelCfg(arch="gpt", width=128, depth=4, heads=4, block_size=64)
        cfg.data = C.DataCfg(dataset="text", data_dir="./data", workers=0,
                             text_file="data/wikitext-2/train.txt")
        cfg.sens = C.SensCfg(n_samples=min(256, sens_samples), batch_size=8, folds=2,
                             ntk_examples=24, n_probes=8, estimator="hutchinson")
        cfg.train.batch_size = 32
    else:
        raise ValueError(arch)
    return cfg
