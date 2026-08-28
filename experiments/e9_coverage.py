"""E9 -- the two coverage gaps, at converged length.

Everything measured so far is CIFAR-10. Before committing to a framing built on the gap
between Spearman rho (which settles early) and top-k mask overlap (which does not), that
gap should be shown to survive a change of dataset and a change of task type.

  ViT / CIFAR-100   same architecture, harder task, 100 classes rather than 10
  GPT / WikiText-2  next-token prediction; S is label-free so it transfers unchanged,
                    and the output dimension forces the Hutchinson estimator

Constant learning rate throughout, so no schedule decay can flatten the curves.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    c = base("e9-c100", "vit", "cifar100", steps=20000, sens_samples=2048)
    c.train.lr_schedule = "constant"; c.n_ckpts = 26
    c.track_criteria = False; c.keep_scores = "none"
    cfgs.append(c)

    c = base("e9-gpt", "gpt", steps=20000, sens_samples=256)
    c.train.lr_schedule = "constant"; c.n_ckpts = 26
    c.track_criteria = False; c.keep_scores = "none"
    cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e9_coverage", configs, __doc__))
