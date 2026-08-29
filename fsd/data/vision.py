"""CIFAR-10/100 loaders.

The *sensitivity* stream is separate from the training stream and never augmented:
S(theta) is an expectation over the data distribution, and random crops would inject
variance that looks exactly like rank instability. It is also label-free by construction --
the sensitivity loaders yield inputs only.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "synthetic": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}


class Synthetic(torch.utils.data.Dataset):
    """Structured random data: a smoke-test target that is learnable but needs no download.
    Used by tests/ to exercise the whole pipeline offline."""

    def __init__(self, n: int, image_size: int, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.proto = torch.randn(num_classes, 3, image_size, image_size, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
        self.x = self.proto[self.y] + 0.6 * torch.randn(
            n, 3, image_size, image_size, generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def _tfms(cfg, train: bool):
    mean, std = STATS[cfg.dataset.lower()]
    ops = []
    if train and cfg.augment:
        ops += [transforms.RandomCrop(cfg.image_size, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip()]
    elif cfg.image_size != 32:
        ops += [transforms.Resize(cfg.image_size)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def _subset(ds, n: int, seed: int):
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


def build(cfg, seed: int):
    if cfg.dataset.lower() == "synthetic":
        train = Synthetic(2048, cfg.image_size, seed=seed)
        return train, train, Synthetic(512, cfg.image_size, seed=seed + 99)
    cls = datasets.CIFAR10 if cfg.dataset.lower() == "cifar10" else datasets.CIFAR100
    train = cls(cfg.data_dir, train=True, transform=_tfms(cfg, True), download=cfg.download)
    clean = cls(cfg.data_dir, train=True, transform=_tfms(cfg, False), download=cfg.download)
    test = cls(cfg.data_dir, train=False, transform=_tfms(cfg, False), download=cfg.download)
    return (_subset(train, cfg.train_subset, seed),
            clean,
            _subset(test, cfg.test_subset, seed + 2))


def sensitivity_batches(clean_set, n_samples: int, batch_size: int, seed: int,
                        folds: int) -> Tuple[List[torch.Tensor], List[int]]:
    """A *fixed* set of inputs, split into disjoint folds.

    Fixed across checkpoints so that rank changes reflect the network moving, not the
    evaluation set moving. Disjoint folds give the C2a noise floor: agreement between
    folds at one checkpoint bounds any agreement claimable across checkpoints.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(clean_set), generator=g)[:n_samples].tolist()
    xs = torch.stack([clean_set[i][0] for i in idx])
    batches, fold_ids = [], []
    per_fold = len(idx) // max(1, folds)
    for f in range(folds):
        chunk = xs[f * per_fold:(f + 1) * per_fold]
        for s in range(0, len(chunk), batch_size):
            batches.append(chunk[s:s + batch_size])
            fold_ids.append(f)
    return batches, fold_ids
