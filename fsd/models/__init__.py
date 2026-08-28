"""Model registry. `build(cfg_model, cfg_data)` is the only entry point runners use."""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .convnets import MLP, ResNetCifar
from .gpt import TinyGPT
from .lstm import CharLSTM
from .vit import ViT

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "synthetic": 10}


def build(model_cfg, data_cfg) -> nn.Module:
    net = _build_backbone(model_cfg, data_cfg)
    alpha = float(getattr(model_cfg, "lazy_alpha", 1.0))
    if alpha != 1.0:
        from .lazy import wrap
        net = wrap(net, alpha)
    return net


def _build_backbone(model_cfg, data_cfg) -> nn.Module:
    arch = model_cfg.arch.lower()
    if arch == "lstm":
        return CharLSTM(vocab_size=model_cfg.vocab_size,
                        embedding_dim=max(16, model_cfg.width // 4),
                        rnn_units=model_cfg.width,
                        num_layers=model_cfg.depth)
    if arch == "gpt":
        return TinyGPT(
            vocab_size=model_cfg.vocab_size,
            block_size=model_cfg.block_size,
            width=model_cfg.width,
            depth=model_cfg.depth,
            heads=model_cfg.heads,
            mlp_ratio=model_cfg.mlp_ratio,
        )

    num_classes = NUM_CLASSES[data_cfg.dataset.lower()]
    if arch == "vit":
        return ViT(
            image_size=data_cfg.image_size,
            patch_size=model_cfg.patch_size,
            num_classes=num_classes,
            embed_dim=model_cfg.width,
            depth=model_cfg.depth,
            num_heads=model_cfg.heads,
            mlp_ratio=model_cfg.mlp_ratio,
        )
    if arch == "resnet20":
        return ResNetCifar(num_classes=num_classes, width=model_cfg.width, depth=model_cfg.depth)
    if arch == "mlp":
        in_dim = 3 * data_cfg.image_size * data_cfg.image_size
        return MLP(in_dim, num_classes=num_classes, width=model_cfg.width, depth=model_cfg.depth)
    raise ValueError(f"unknown arch '{model_cfg.arch}'")


def output_dim(model: nn.Module, sample_batch) -> int:
    """Flat output dimension per example — decides exact vs. Hutchinson."""
    import torch
    with torch.no_grad():
        out = model(sample_batch[:1])
    return int(out[0].numel())
