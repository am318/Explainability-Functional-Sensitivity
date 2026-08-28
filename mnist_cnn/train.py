"""
Train the small MNIST CNN, tracking parameter-wise functional sensitivity
(see common/sensitivity.py) over training. No pruning is applied -- this
only records how sensitivity evolves alongside the train and test loss.

The loop itself lives in common/experiment.py, shared with twomoons_mlp/;
this file supplies only what is specific to MNIST: the config defaults, and
a build_experiment that constructs the data, the loaders and the model.

Use environment variables to change settings without editing the file:

    EPOCHS=2 OPTIMIZER=sgd LR=0.1 python train.py

For the optimizer x learning-rate sweep this file is a building block of,
see sweep.py (and run_sweep.sbatch to run that on the cluster).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from dataset import build_mnist_datasets
from experiment import (
    BaseConfig,
    Experiment,
    build_train_probe_loaders,
    env_int,
    env_str,
    run_training,
    select_device,
)
from model import MnistCNN


@dataclass
class Config(BaseConfig):
    output_root: str = env_str("OUTPUT_ROOT", str(Path(__file__).resolve().parent / "outputs"))
    data_dir: str = env_str("DATA_DIR", str(Path(__file__).resolve().parent / "data"))

    epochs: int = env_int("EPOCHS", 30)
    batch_size: int = env_int("BATCH_SIZE", 128)
    # 512 held-out training images, the same probe budget as two moons: the
    # sensitivity estimate is an average over the probe set, and 512 images
    # is already far more than needed for the ordering to settle.
    probe_samples: int = env_int("PROBE_SAMPLES", 512)
    probe_batch_size: int = env_int("PROBE_BATCH_SIZE", 128)
    eval_batch_size: int = env_int("EVAL_BATCH_SIZE", 1000)

    channels1: int = env_int("CHANNELS1", 8)
    channels2: int = env_int("CHANNELS2", 16)


def build_experiment(cfg: Config, device) -> Experiment:
    """Data -> loaders -> model, in that order. rank_stability.py calls this
    again under the same seed to reconstruct the epoch-0 model, so the order
    in which the global RNG is consumed here must not change."""
    train_pool, test_set = build_mnist_datasets(Path(cfg.data_dir))
    train_loader, probe_loader = build_train_probe_loaders(cfg, train_pool)
    val_loader = DataLoader(test_set, batch_size=cfg.eval_batch_size, shuffle=False)
    model = MnistCNN(channels1=cfg.channels1, channels2=cfg.channels2).to(device)
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
