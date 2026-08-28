"""CIFAR ResNet-20 and a plain MLP.

Two non-attention controls for C1/C4: if the freezing effect were a transformer quirk it
would not show up here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.GroupNorm(8, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x))


class ResNetCifar(nn.Module):
    """ResNet-20 style. GroupNorm rather than BatchNorm: sensitivity is a per-example
    functional quantity, and BatchNorm makes f(x) depend on the rest of the batch."""

    def __init__(self, num_classes: int = 10, width: int = 16, depth: int = 20):
        super().__init__()
        n = max(1, (depth - 2) // 6)
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.GroupNorm(8, width)
        layers = []
        in_ch = width
        for stage, mult in enumerate((1, 2, 4)):
            for blk in range(n):
                stride = 2 if (stage > 0 and blk == 0) else 1
                layers.append(BasicBlock(in_ch, width * mult, stride))
                in_ch = width * mult
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10, width: int = 512, depth: int = 4):
        super().__init__()
        dims = [in_dim] + [width] * max(1, depth - 1)
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.flatten(1)))
