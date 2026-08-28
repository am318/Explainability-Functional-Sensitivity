"""
Global top-k magnitude pruning by (unsigned) functional sensitivity.

Keeps the `keep_fraction` of parameters -- pooled across *every* tensor in
the model, not per-layer -- with the highest S_i, and zeroes the rest. Used
to study how the effect of pruning depends on when during training the
sensitivity ranking used to decide the mask was computed.
"""

from typing import Dict

import torch
import torch.nn as nn


def compute_topk_mask(
    model: nn.Module, unsigned_scores: Dict[str, torch.Tensor], keep_fraction: float
) -> Dict[str, torch.Tensor]:
    """Boolean mask per parameter tensor (True = keep), selected by a single
    global threshold on unsigned sensitivity across the whole model."""
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")

    names = [name for name, _ in model.named_parameters() if name in unsigned_scores]
    parts = [unsigned_scores[name].detach().reshape(-1).float() for name in names]
    sizes = [p.numel() for p in parts]
    flat = torch.cat(parts)
    n_total = flat.numel()
    k = max(1, min(n_total, int(round(keep_fraction * n_total))))

    topk_idx = torch.topk(flat, k, largest=True, sorted=False).indices
    keep_flat = torch.zeros(n_total, dtype=torch.bool)
    keep_flat[topk_idx] = True

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, size in zip(names, sizes):
        shape = unsigned_scores[name].shape
        masks[name] = keep_flat[offset:offset + size].reshape(shape).clone()
        offset += size
    return masks


@torch.no_grad()
def apply_mask_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Zero out masked-off parameters in place."""
    for name, p in model.named_parameters():
        if name in masks:
            p.mul_(masks[name].to(device=p.device, dtype=p.dtype))


def zero_grad_for_mask_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Zero the gradient of masked-off parameters in place (call after
    backward(), before optimizer.step(), so pruned weights never move and
    Adam's per-parameter moment estimates for them stay at zero)."""
    for name, p in model.named_parameters():
        if name in masks and p.grad is not None:
            p.grad.mul_(masks[name].to(device=p.grad.device, dtype=p.grad.dtype))


def mask_stats(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    total = sum(m.numel() for m in masks.values())
    kept = sum(int(m.sum().item()) for m in masks.values())
    return {"total": total, "kept": kept, "kept_fraction": kept / max(1, total)}
