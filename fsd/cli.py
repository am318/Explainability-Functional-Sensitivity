"""Single-run CLI. One invocation == one config == one results directory.

    python -m fsd.cli --tag pilot --set model.arch=vit --set train.steps=2000

Designed for cluster job arrays: no shared state, no coordination, and `--config` accepts a
JSON file so a sweep is just a directory of configs.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config as C
from .run import execute


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file (overrides defaults)")
    ap.add_argument("--set", action="append", default=[], metavar="path=value",
                    help="dotted override, e.g. train.lr=3e-4 (repeatable)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="if results/<run_id>/metrics.json already exists, skip this run")
    ap.add_argument("--print-id", action="store_true", help="print run id and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="build everything and take one step + one measurement, then exit; "
                         "validates a config before a job array commits hours to it")
    args = ap.parse_args(argv)

    cfg = C.load(args.config) if args.config else C.RunCfg()
    for item in args.set:
        if "=" not in item:
            ap.error(f"--set expects path=value, got '{item}'")
        path, value = item.split("=", 1)
        cfg = C.override(cfg, path, value)
    if args.tag:
        cfg = C.override(cfg, "tag", args.tag)
    if args.out_dir:
        cfg = C.override(cfg, "out_dir", args.out_dir)

    if args.print_id:
        print(cfg.run_id())
        return 0

    if args.dry_run:
        import copy
        probe = copy.deepcopy(cfg)
        probe.tag = f"dryrun-{cfg.tag}"
        probe.out_dir = "results/_dryrun"
        # Deliberately minimal. The job of a dry run is to catch config errors -- a bad
        # arch name, a missing corpus, a head count that does not divide the width, an
        # estimator that cannot handle the output dimension. It is not to benchmark, so
        # everything is shrunk to the smallest size that still exercises each code path.
        probe.train.steps = 2
        probe.n_ckpts = 2
        probe.sens.n_samples = 4
        probe.sens.batch_size = 2
        probe.sens.n_probes = min(probe.sens.n_probes, 2)
        probe.sens.ntk_examples = 4
        probe.sens.folds = 2
        probe.train.batch_size = min(probe.train.batch_size, 8)
        probe.data.test_subset = 64
        probe.data.workers = 0
        probe.track_criteria = False
        probe.track_structured = False
        probe.keep_scores = "none"
        probe.save_state_at = []
        m = execute(probe, verbose=False)
        print(f"OK  {cfg.run_id()}  {cfg.model.arch}/{cfg.data.dataset} "
              f"steps={cfg.train.steps} params={m['n_params']/1e6:.2f}M "
              f"estimator={m['vs_final'][0].get('_est', probe.sens.estimator)}")
        return 0

    metrics = execute(cfg, verbose=not args.quiet, skip_done=args.skip_done)
    print(json.dumps({"run_id": metrics["run_id"], "tstar": metrics["tstar"],
                      "final_eval": metrics["final_eval"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
