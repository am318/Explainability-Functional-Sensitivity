"""E12 -- the reproducibility ceiling: how much of S_T is knowable *at all*?

Every stability number in this project so far compares S_t against S_T with an implicit
target of 1.0. That is the wrong target. S_T is the endpoint of a stochastic trajectory,
and some of it is irreducibly run-specific -- determined by batch order, not by anything
an early predictor could ever access. Measuring against 1.0 therefore systematically
understates how much early sensitivity knows.

This experiment measures the achievable ceiling directly: two runs from the SAME
initialisation differing only in batch order. Their agreement at convergence,
rho(S_T^A, S_T^B), bounds what any predictor computed before the data order was seen could
possibly achieve. The normalised quantity

    rho(S_0^A, S_T^A) / rho(S_T^A, S_T^B)

is the fraction of the *knowable* structure already present at initialisation.

Same init, different data order is the only well-posed version of this comparison:
parameterwise rankings from DIFFERENT initialisations are related by a hidden permutation
of hidden units, so their parameter indices do not correspond and the correlation is
meaningless. Fixing the init and varying only `data_seed` keeps index i referring to the
same weight in both runs.

Interpretation is pre-specified, so this is a test rather than a search:
  * ceiling near 1.0  -> S_T is highly reproducible; a low rho(S_0, S_T) genuinely means
                         early sensitivity does not resemble the trained network.
  * ceiling comparable to rho(S_0, S_T) -> most of what is knowable IS known at init, and
                         the earlier "does not settle" results were measuring irreducible
                         run-to-run noise rather than a failure of early prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments._common import base, driver


def configs():
    cfgs = []
    for arch, steps in (("mlp", 12000), ("resnet20", 12000)):
        for data_seed in (0, 1):
            c = base(f"e12-{arch}", arch, "cifar10", steps=steps, sens_samples=2048)
            c.seed = 0                  # SAME initialisation for both
            c.data_seed = data_seed     # different batch order only
            c.train.lr_schedule = "constant"
            c.n_ckpts = 16
            c.keep_scores = "all"       # cross-run comparison needs the raw vectors
            c.track_criteria = False
            cfgs.append(c)
    return cfgs


if __name__ == "__main__":
    raise SystemExit(driver("e12_ceiling", configs, __doc__))
