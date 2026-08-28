"""E2 -- is t* predictable? Claim C4.

A stabilisation time that is just "early, roughly" is an observation. A stabilisation time
that moves systematically with width, depth, learning rate and batch size -- and tracks the
kernel velocity measured in the same runs -- is a predictive principle, which is what this
venue is asking for.

Sweeps are one-factor-at-a-time around a fixed reference point, because a 4-page paper can
show a trend line but not a full factorial, and because interactions are not the claim.

Falsifies C4 if: t* scatters without structure, or is uncorrelated with kernel velocity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch in ("vit", "mlp"):
        for width in (64, 128, 192, 384) if arch == "vit" else (128, 256, 512, 1024):
            c = base("e2w", arch, steps=2000, sens_samples=1024)
            c.model.width = width
            if arch == "vit":
                c.model.heads = max(1, width // 64)
            cfgs.append(c)
    for depth in (2, 4, 6, 10):
        c = base("e2d", "vit", steps=2000, sens_samples=1024)
        c.model.depth = depth
        cfgs.append(c)
    for lr in (3e-4, 1e-3, 3e-3, 1e-2):
        c = base("e2lr", "vit", steps=2000, sens_samples=1024)
        c.train.lr = lr
        cfgs.append(c)
    for bs in (32, 64, 128, 256):
        c = base("e2bs", "vit", steps=2000, sens_samples=1024)
        c.train.batch_size = bs
        cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e2_tstar_scaling", configs, __doc__))
