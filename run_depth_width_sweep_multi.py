#!/usr/bin/env python3
"""Run depth/width sweeps for the provided training scripts.

This launcher avoids shell scripting and uses subprocess directly.
It assigns jobs to GPU slots using CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Sequence, Tuple


DEFAULT_SCRIPTS = [
    "/mnt/data/initial_experiment_2d_morse_sweep.py",
    "/mnt/data/initial_experiment_2d_exp_test_sweep.py",
]


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def build_jobs(scripts: Sequence[str], depths: Sequence[int], widths: Sequence[int], seeds: Sequence[int]) -> List[Tuple[str, int, int, int]]:
    return list(itertools.product(scripts, depths, widths, seeds))


def launch_job(script: str, depth: int, width: int, seed: int, gpu: str, out_dir: Path, epochs: int | None) -> Tuple[subprocess.Popen, object]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["N_HIDDEN"] = str(depth)
    env["HIDDEN_WIDTH"] = str(width)
    env["SEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"
    if epochs is not None:
        env["EPOCHS"] = str(epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / (
        f"{Path(script).stem}_depth{depth}_width{width}_seed{seed}_gpu{gpu}.log"
    )
    log_file = open(log_path, "w", buffering=1)

    cmd = [sys.executable, script]
    print(f"Launching: {Path(script).name} depth={depth} width={width} seed={seed} gpu={gpu}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path(script).resolve().parent),
    )
    return proc, log_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Run depth/width sweeps across multiple GPUs.")
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=DEFAULT_SCRIPTS,
        help="Python training scripts to run.",
    )
    parser.add_argument(
        "--depths",
        default="1,2,3,4",
        help="Comma-separated hidden-layer counts.",
    )
    parser.add_argument(
        "--widths",
        default="16,32,64,128",
        help="Comma-separated hidden widths.",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU ids to cycle through.",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/data/sweep_logs",
        help="Directory for per-run logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for all runs.",
    )
    args = parser.parse_args()

    scripts = [str(Path(s).resolve()) for s in args.scripts]
    depths = parse_int_list(args.depths)
    widths = parse_int_list(args.widths)
    seeds = parse_int_list(args.seeds)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("Need at least one GPU id")

    for script in scripts:
        if not Path(script).exists():
            raise FileNotFoundError(script)

    jobs = deque(build_jobs(scripts, depths, widths, seeds))
    if not jobs:
        print("No jobs to run.")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    free_gpus: Deque[str] = deque(gpus)
    running: List[Tuple[subprocess.Popen, object, str]] = []
    completed = 0
    total = len(jobs)

    while jobs or running:
        while jobs and free_gpus:
            gpu = free_gpus.popleft()
            script, depth, width, seed = jobs.popleft()
            proc, log_file = launch_job(script, depth, width, seed, gpu, out_dir, args.epochs)
            running.append((proc, log_file, gpu))

        still_running: List[Tuple[subprocess.Popen, object, str]] = []
        for proc, log_file, gpu in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, log_file, gpu))
                continue

            log_file.close()
            free_gpus.append(gpu)
            completed += 1
            status = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{completed}/{total}] gpu={gpu} finished with {status}")

        running = still_running

        if jobs and not free_gpus:
            time.sleep(5)
        elif running:
            time.sleep(1)

    print("All jobs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
