"""Validate every emitted job config before a cluster array commits hours to it.

Each config is built end to end -- model, task, data, sensitivity estimator, one optimiser
step, one measurement -- so a typo in a sweep only costs seconds here instead of a whole
array's walltime.
"""
import argparse, subprocess, sys, glob, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu",
                help="force a device; cpu keeps validation off the GPU so it can run "
                     "alongside a real training job")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--pattern", default="jobs/*/cfg_*.json")
args = ap.parse_args()

cfgs = sorted(glob.glob(args.pattern))
if not cfgs:
    print("no job configs found; run --emit first")
    raise SystemExit(1)

print(f"validating {len(cfgs)} configs on {args.device} "
      f"({args.workers} workers)\n", flush=True)
failures = []
t0 = time.time()
done = [0]


def run_one(c):
    r = subprocess.run([sys.executable, "-m", "fsd.cli", "--config", c, "--dry-run",
                        "--set", f"device={args.device}"],
                       capture_output=True, text=True)
    done[0] += 1
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        failures.append((c, tail))
        print(f"[{done[0]}/{len(cfgs)}] FAIL {c}\n         {tail}", flush=True)
    else:
        print(f"[{done[0]}/{len(cfgs)}] {r.stdout.strip()}", flush=True)


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    list(ex.map(run_one, cfgs))

print(f"\n{len(cfgs)-len(failures)}/{len(cfgs)} valid in {time.time()-t0:.0f}s")
if failures:
    print("\nFAILURES:")
    for c, msg in failures:
        print(f"  {c}: {msg}")
    raise SystemExit(1)
