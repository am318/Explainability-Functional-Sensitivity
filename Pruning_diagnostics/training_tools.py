import torch
import torch.nn as nn
import math
from typing import Dict, Iterable, List, Optional, Tuple

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bsz = images.shape[0]
        loss_sum += float(loss.item()) * bsz
        acc_sum += accuracy(logits, targets) * bsz
        n += bsz
    return {"loss": loss_sum / max(1, n), "accuracy": acc_sum / max(1, n)}


def compute_epoch_lr(epoch: int, cfg) -> float:
    if cfg.epochs <= 0:
        return cfg.min_lr
    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        start = cfg.lr * 1e-3
        progress = (epoch - 1) / max(1, warmup_epochs - 1)
        return start + (cfg.lr - start) * progress
    decay_epochs = max(1, cfg.epochs - warmup_epochs)
    decay_progress = (epoch - warmup_epochs) / decay_epochs
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_progress)))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine_factor