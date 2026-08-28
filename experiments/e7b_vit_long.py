"""E7b -- the same convergence test on the architecture most likely to freeze.

E7 covers MLP and ResNet-20. The ViT is the strongest candidate for C1 and was missing:
it has the heaviest sensitivity tail of the three (Hill alpha 1.399 vs 1.519 and 1.687),
and it was the only architecture where the boundary-stability prediction of C3 held at
every checkpoint. By our own mechanism, if any of these settings freezes, this is the one.

Running it separately rather than folding it into E7 keeps the queue ordered by
informativeness: the ViT arm matters more than the ResNet cosine arm it would otherwise
sit behind.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for schedule in ("constant", "cosine"):
        c = base(f"e7b-{schedule}", "vit", "cifar10", steps=20000, sens_samples=2048)
        c.train.lr_schedule = schedule
        c.train.warmup_steps = 200
        c.n_ckpts = 26
        c.track_criteria = False
        c.keep_scores = "none"
        cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e7b_vit_long", configs, __doc__))
