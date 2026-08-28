"""E1 pilot -- one seed per architecture, run locally to get real curves fast.

Exists so the claim checks and figures meet real data before the full sweep is committed
to. Everything it produces is a strict subset of e1_rank_stability.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch in ("mlp", "resnet20", "vit"):     # cheapest first, so results arrive early
        c = base("e1p", arch, "cifar10", steps=4000)
        c.seed = 0
        c.keep_scores = "all" if arch == "vit" else "none"
        cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e1_pilot", configs, __doc__))
