"""E7 -- is there a plateau that a short run simply cuts off?

The E1 pilot showed adjusted overlap rising monotonically to 1.0 at t=T with no plateau.
That shape is exactly what you would see for a network still training: agreement with
S_final then measures *convergence*, not *freezing*, because the reference itself is still
moving. The MLP pilot confirms the worry -- training loss was still descending at step
4000 (10.2 epochs of CIFAR-10).

This experiment removes both confounds at once:

  * **Long enough that T is well past convergence**, so S_final is a stable reference and
    a plateau, if one exists, has room to appear and persist.
  * **Constant learning rate**, so the late flattening of any "agreement with final" curve
    cannot be manufactured by cosine decay driving the LR to zero.

Read it as follows. If adjusted overlap plateaus at some t* << T and stays flat, C1 is
supported and the pilot was simply truncated. If it keeps climbing all the way to T even
after the loss has flattened, C1 is false and the ordering tracks training indefinitely.
Both outcomes are reportable; only one of them is the paper we set out to write.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch, steps in (("mlp", 20000), ("resnet20", 20000)):
        for schedule in ("constant", "cosine"):
            c = base(f"e7-{schedule}", arch, "cifar10", steps=steps, sens_samples=2048)
            c.train.lr_schedule = schedule
            c.train.warmup_steps = 200
            c.n_ckpts = 26
            c.track_criteria = False    # not the question here; halves measurement cost
            c.keep_scores = "none"
            cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e7_convergence", configs, __doc__))
