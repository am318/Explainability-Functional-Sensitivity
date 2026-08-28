"""
Two moons: the standard two-class toy problem (scikit-learn's make_moons),
used here because it is the smallest dataset on which a network has to learn
a genuinely nonlinear decision boundary -- enough structure for the
sensitivity ordering to have something to say, small enough that a full
optimizer x learning-rate sweep with per-epoch checkpointing runs in
seconds on CPU.

Train and test sets are drawn from the same generator with *different*
random states, so the test set is a fresh sample from the same
distribution rather than a partition of one sample.

Inputs are standardised using training-set statistics only. make_moons
returns raw coordinates spanning roughly [-1.2, 2.2] x [-0.6, 1.1], whose
per-axis scales differ by about 2x; standardising makes a single learning
rate mean the same thing along both axes, which matters when the point of
the exercise is to compare learning rates across optimizers.
"""

from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


def build_moons_datasets(
    n_train: int, n_test: int, noise: float, seed: int
) -> Tuple[TensorDataset, TensorDataset]:
    x_train, y_train = make_moons(n_samples=n_train, noise=noise, random_state=seed)
    x_test, y_test = make_moons(n_samples=n_test, noise=noise, random_state=seed + 1)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    def to_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(y).long(),
        )

    return to_dataset(x_train, y_train), to_dataset(x_test, y_test)
