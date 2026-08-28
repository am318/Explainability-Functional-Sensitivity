"""
Train the two-moons MLP, tracking parameter-wise functional sensitivity
(see common/sensitivity.py) over training. No pruning is applied -- this
only records how sensitivity evolves alongside the train and test loss.

The loop itself lives in common/experiment.py, shared with mnist_cnn/; this
file supplies only what is specific to two moons: the config defaults, and a
build_experiment that constructs the data, the loaders and the model.

Use environment variables to change settings without editing the file:

    EPOCHS=10 OPTIMIZER=sgd LR=0.1 python train.py

For the optimizer x learning-rate sweep this file is a building block of,
see sweep.py.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import build_moons_datasets
from experiment import (
    BaseConfig,
    Experiment,
    build_train_probe_loaders,
    env_float,
    env_int,
    env_str,
    run_training,
    select_device,
)
from model import MoonsMLP


@dataclass
class Config(BaseConfig):
    output_root: str = env_str("OUTPUT_ROOT", str(Path(__file__).resolve().parent / "outputs"))

    epochs: int = env_int("EPOCHS", 100)
    batch_size: int = env_int("BATCH_SIZE", 64)

    # n_train is the training *pool*: probe_samples of it are held out for
    # sensitivity probing, and the rest is what the model actually trains on.
    n_train: int = env_int("N_TRAIN", 2048)
    n_test: int = env_int("N_TEST", 1024)
    noise: float = env_float("NOISE", 0.2)
    hidden_dim: int = env_int("HIDDEN_DIM", 32)


def build_experiment(cfg: Config, device) -> Experiment:
    """Data -> loaders -> model, in that order. rank_stability.py calls this
    again under the same seed to reconstruct the epoch-0 model, so the order
    in which the global RNG is consumed here must not change."""
    train_pool, test_set = build_moons_datasets(cfg.n_train, cfg.n_test, cfg.noise, cfg.seed)
    train_loader, probe_loader = build_train_probe_loaders(cfg, train_pool)
    val_loader = DataLoader(test_set, batch_size=cfg.probe_batch_size, shuffle=False)
    model = MoonsMLP(hidden_dim=cfg.hidden_dim).to(device)
    return Experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        probe_loader=probe_loader,
        criterion=nn.CrossEntropyLoss(),
    )


def main() -> None:
    cfg = Config()
    run_training(cfg, build_experiment, select_device())


if __name__ == "__main__":
    main()
