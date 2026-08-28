"""E8 -- the convergence test at full coverage, for the cluster.

E7/E7b answer the plateau question on one machine for MLP, ResNet-20 and ViT. This is the
same test across every architecture, both schedules, three seeds and both datasets, sized
for a GPU cluster rather than a laptop. It is the experiment the paper's central figure
should be built from, because it is the only one where the reference S_final is taken from
a model that has actually stopped improving.

30000 steps is ~77 epochs of CIFAR-10, comfortably past the point where these models stop
gaining accuracy, so a plateau in the ordering has room to appear and persist well before T.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch in ("vit", "resnet20", "mlp"):
        for schedule in ("constant", "cosine"):
            for seed in (0, 1, 2):
                c = base(f"e8-{schedule}", arch, "cifar10", steps=30000)
                c.seed = seed
                c.train.lr_schedule = schedule
                c.n_ckpts = 28
                c.keep_scores = "all" if (arch == "vit" and seed == 0) else "none"
                cfgs.append(c)
    for seed in (0, 1):
        c = base("e8-constant", "gpt", steps=30000)
        c.seed = seed
        c.train.lr_schedule = "constant"
        c.n_ckpts = 28
        cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e8_long_all", configs, __doc__))
