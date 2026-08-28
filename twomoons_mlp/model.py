"""
A deliberately tiny MLP for two moons: 2 -> 32 -> 32 -> 2, ReLU, 1,218
parameters. Small enough that every rank-stability statistic can be computed
at full parameter resolution at every epoch (no pooling approximation
anywhere), while still being deep enough to have a hidden layer whose
sensitivity behaves differently from the input and output layers.

The three Linear layers are named `input`, `hidden` and `output` because
sensitivity.parameter_group derives its reporting groups from the top-level
attribute name -- so those names are what appear as the module breakdown on
every plot.
"""

import torch
import torch.nn as nn


class MoonsMLP(nn.Module):
    def __init__(self, in_features: int = 2, hidden_dim: int = 32, n_classes: int = 2):
        super().__init__()
        self.input = nn.Linear(in_features, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_classes)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input(x))
        h = self.activation(self.hidden(h))
        return self.output(h)
