"""E5 -- where the ordering does NOT freeze.

A stability claim with no known boundary is a claim nobody can use. This probes the two
regimes where the mechanism predicts freezing should fail:

  * **No warmup, large LR.** The lazy argument says rank conservation needs small Jacobian
    motion. Removing warmup and raising the LR maximises early Jacobian motion, so t*
    should grow sharply or vanish entirely.
  * **Constant LR.** Cosine decay freezes *everything* late in training, so part of any
    plateau could be an artefact of the schedule rather than a property of the ordering.
    A constant-LR run removes that explanation.
  * **Long training.** If the plateau is real, extending training 4x should not move t*.
    If t* scales with the run length, the "early" in our title is doing no work.

These are reported whether or not they are flattering; the constant-LR control in
particular is a check on our own headline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch in ("vit", "resnet20"):
        c = base("e5-nowarm", arch, steps=2000, sens_samples=1024)
        c.train.warmup_steps = 0
        c.train.lr = 1e-2
        cfgs.append(c)

        c = base("e5-const", arch, steps=2000, sens_samples=1024)
        c.train.lr_schedule = "constant"
        cfgs.append(c)

    c = base("e5-long", "vit", steps=16000, sens_samples=1024)
    c.n_ckpts = 26
    cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e5_failure_modes", configs, __doc__))
