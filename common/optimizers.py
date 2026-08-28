"""
The optimizers compared across experiments, behind one name -> instance
function so that an experiment's config only has to carry a string (and so
that sweeps can iterate over OPTIMIZERS without knowing anything about
torch.optim's differing constructor signatures).

"sgd" and "sgd_momentum" are both torch.optim.SGD; they are kept as separate
names because plain SGD and SGD+momentum behave differently enough (and want
different learning rates) that treating them as one optimizer with a
hyperparameter would defeat the point of the comparison. Momentum is fixed
at SGD_MOMENTUM rather than swept -- the sweep is over learning rate only.
"""

from typing import Dict, Iterable, Tuple

import torch

SGD_MOMENTUM = 0.9

OPTIMIZERS: Tuple[str, ...] = ("adam", "sgd", "sgd_momentum")

OPTIMIZER_LABELS: Dict[str, str] = {
    "adam": "Adam",
    "sgd": "SGD",
    "sgd_momentum": f"SGD (momentum={SGD_MOMENTUM})",
}


def build_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=SGD_MOMENTUM)
    raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")
