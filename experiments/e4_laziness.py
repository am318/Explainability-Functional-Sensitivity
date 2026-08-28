"""E4 -- laziness as an intervention, not a correlation. Claim C5.1.

Every run in E1 reports kernel velocity alongside rank churn, but co-movement of two
measured quantities does not establish that one causes the other. Here laziness is a dial:
the output is rescaled by alpha with the learning rate scaled by 1/alpha^2, at fixed
architecture and fixed initialisation.

Prediction, if the lazy-conservation argument is right:
  as alpha grows, the Jacobian stops moving, kernel velocity -> 0, and the sensitivity
  ordering is conserved from step 0, so t* -> 0.

Falsifies C5.1 if: t* is unmoved by alpha while kernel velocity collapses. That would say
rank churn is driven by something other than Jacobian motion, and the theory section would
have to be rewritten around whatever that is.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch in ("vit", "mlp"):
        for alpha in (1.0, 3.0, 10.0, 30.0):
            c = base(f"e4-a{alpha:g}", arch, steps=2000, sens_samples=1024)
            c.model.lazy_alpha = alpha
            cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e4_laziness", configs, __doc__))
