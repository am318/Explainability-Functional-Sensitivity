"""E6 -- is the ordering a property of the initialisation, the data, or the trajectory?

Cross-*seed* comparison of parameterwise rankings is meaningless: two networks with
different initialisations are related by a hidden permutation of hidden units, so their
parameter indices do not correspond. The well-posed version fixes the initialisation and
varies only the batch order. Then index i means the same thing in both runs.

Three quantities, all measured at sparsity levels a pruner would use:

  within-run   S_t^{(a)} vs S_T^{(a)}   -- the E1 curve, for reference
  across-run   S_T^{(a)} vs S_T^{(b)}   -- how much of the final ordering is determined by
                                           the initialisation rather than the data order
  predictive   S_t^{(a)} vs S_T^{(b)}   -- can an early ordering from one run predict a
                                           *different* run's final ordering?

The third is the one the follow-up pruning paper needs: it asks whether an early mask
transfers, which is a strictly stronger requirement than the mask being stable within its
own run.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch in ("mlp", "resnet20", "vit"):
        for data_seed in (0, 1):
            c = base("e6", arch, "cifar10", steps=4000)
            c.seed = 0                 # SAME initialisation ...
            c.data_seed = data_seed    # ... different batch order
            c.keep_scores = "all"      # cross-run comparison needs the raw vectors
            c.track_criteria = False   # not needed here; halves the measurement cost
            cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e6_same_init", configs, __doc__))
