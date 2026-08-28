"""
A small MNIST CNN: two 3x3 convolutions (8 then 16 channels), each followed
by ReLU and 2x2 max-pooling, then a single linear head. 9,098 parameters --
about 7.5x the two-moons MLP, still small enough to score every parameter's
sensitivity at full resolution at every epoch, and enough to reach ~99% test
accuracy in a handful of epochs.

The three parameterised modules are named `conv1`, `conv2` and `head`
because sensitivity.parameter_group derives its reporting groups from the
top-level attribute name. Note how lopsided the split is -- 80 / 1,168 /
7,850 parameters -- which is exactly the situation the pooled heatmap
handles badly and the full-resolution rank-stability curves handle
correctly.
"""

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self, channels1: int = 8, channels2: int = 16, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=3, padding=1)
        self.head = nn.Linear(channels2 * 7 * 7, n_classes)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.activation(self.conv1(x)))
        h = self.pool(self.activation(self.conv2(h)))
        return self.head(h.flatten(1))
