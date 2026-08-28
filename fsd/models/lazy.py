"""A laziness knob.

Chizat-Oyallon-Bach: training the rescaled model

    f_alpha(x; theta) = alpha * ( F(x; theta) - F(x; theta_0) )

with the learning rate scaled by 1/alpha^2 interpolates between feature learning
(alpha small) and the linearised/lazy regime (alpha large), *at fixed architecture and
fixed initialisation*.

This matters for C5.1. Kernel velocity measured along an ordinary training run tells us
that rank churn and departure-from-lazy move together, but that is a correlation. Here
laziness is a dial we turn: the lazy argument predicts that as alpha grows the Jacobian
stops moving, so the sensitivity ordering must be conserved from step 0 and t* -> 0. If
t* does not collapse as alpha grows, the mechanism is wrong, and no amount of correlational
evidence rescues it.

Implementation note: the frozen reference copy holds its weights as *buffers*, not
parameters. An earlier version kept them as parameters and hid them by overriding
`named_parameters`, which silently desynchronised those names from the module tree --
`torch.func.functional_call` then tried to resolve inner names against the wrapper and
failed. Buffers get this right structurally: the reference is part of the *function*, not
of theta, so it should never have been a parameter in the first place.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _freeze_to_buffers(module: nn.Module) -> None:
    """Turn every parameter in `module` into a non-persistent buffer of the same name."""
    for m in module.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            delattr(m, name)
            m.register_buffer(name, p.detach().clone(), persistent=False)


class LazyScaled(nn.Module):
    def __init__(self, model: nn.Module, alpha: float):
        super().__init__()
        self.model = model
        self.alpha = float(alpha)
        self.frozen = copy.deepcopy(model)
        _freeze_to_buffers(self.frozen)
        self.frozen.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.frozen(x)
        return self.alpha * (self.model(x) - base)


def wrap(model: nn.Module, alpha: float) -> nn.Module:
    return model if alpha == 1.0 else LazyScaled(model, alpha)


def strip_wrapper_prefix(name: str) -> str:
    """Parameter names gain a 'model.' prefix under the wrapper; name-based rules
    (which tensors are prunable) must see the same string either way."""
    return name[len("model."):] if name.startswith("model.") else name
