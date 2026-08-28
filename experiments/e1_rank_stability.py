"""E1 -- the core measurement. Claims C1, C2, C3.

Trains each setting once with log-spaced sensitivity checkpoints and records, at every
checkpoint: agreement with the final ordering, the between-fold noise floor at that same
checkpoint, the layerwise controls, and the kernel/drift diagnostics.

This is the experiment Figure 1 is made of. Everything else in the paper either controls
it (E4, E5), scales it (E2), or spends it (E3).

What would falsify C1/C2/C3 here:
  * adjusted overlap rising only near the end of training            -> C1 false
  * adjusted overlap indistinguishable from 0                        -> C2b false (layer budget only)
  * step-0 overlap already at the plateau                            -> C2c false (fixed at init)
  * top-k overlap tracking global Spearman rather than exceeding it  -> C3 false
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    # main panel: three architectures x three seeds x both LR schedules.
    #
    # The constant-LR arm is not a robustness check tucked in an appendix -- cosine decay
    # drives the LR to ~0 near the end, which flattens *any* "agreement with final" curve
    # and could manufacture the plateau this paper claims to observe. Both schedules run
    # in the main panel so the headline cannot be a schedule artefact.
    for arch in ("vit", "resnet20", "mlp"):
        for seed in (0, 1, 2):
          for schedule in ("cosine", "constant"):
            c = base("e1", arch, "cifar10")
            c.seed = seed
            c.train.lr_schedule = schedule
            # keep the raw score vectors for one reference run -- the spectrum and
            # drift-distribution figures need them, and re-running to recover them later
            # would cost more than the disk does.
            if arch == "vit" and seed == 0 and schedule == "cosine":
                c.keep_scores = "all"
            cfgs.append(c)
    # second dataset: does task difficulty move t*?
    for seed in (0, 1):
        c = base("e1", "vit", "cifar100")
        c.seed = seed
        cfgs.append(c)
    # non-vision: the generality check that makes this a claim about training
    for seed in (0, 1):
        c = base("e1", "gpt")
        c.seed = seed
        cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e1_rank_stability", configs, __doc__))
