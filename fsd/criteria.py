"""Competing importance criteria, measured at the same checkpoints as S.

The point is comparative. "Some ranking stabilises early" is a weak claim; the interesting
version is whether the *functional* ranking behaves differently from *loss-based* ones. The
companion draft argues that Fisher information is the same Sobolev-type seminorm weighted
by the task loss, and that removing the weighting yields an architecture-centric measure.
That predicts something testable here: the unweighted quantity should be the more stable
one, because the loss weighting keeps moving while the function's local geometry settles.

  functional  S_i = E_x || dF/dtheta_i ||^2          label-free  (this paper)
  fisher      F_i = E_{x,y} [ (dL/dtheta_i)^2 ]      the same seminorm, loss-weighted
  snip        |theta_i * dL/dtheta_i|                Lee et al. 2019
  synflow     |theta_i * dR/dtheta_i|, R from |theta| on a ones-input   Tanaka et al. 2020
  magnitude   |theta_i|

All are computed on the same fixed inputs, so differences between their stability curves
are differences between the criteria and not between evaluation sets.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .sensitivity import Tensors, _maybe_empty_cache

NAMES = ("functional", "fisher", "snip", "synflow", "magnitude")


def magnitude(model: nn.Module) -> Tensors:
    return {n: p.detach().abs().float().cpu()
            for n, p in model.named_parameters() if p.requires_grad}


def loss_based(model: nn.Module, batches: Sequence, device, task,
               per_example: bool = True) -> Dict[str, Tensors]:
    """Fisher (per-example squared loss gradient) and SNIP (|theta * mean gradient|).

    Fisher is deliberately the *per-example* second moment E[(dL/dtheta)^2], not the square
    of the averaged gradient -- the same distinction that matters for S, and the one that
    makes it the loss-weighted counterpart of the same seminorm.
    """
    model.eval()
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    names = list(params)

    def single_loss(prm, x, y):
        out = functional_call(model, (prm, buffers), (x.unsqueeze(0),))
        return task.loss(out, y.unsqueeze(0))

    gfn = grad(single_loss)
    sq = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    mean_g = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    count = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        try:
            g = vmap(gfn, in_dims=(None, 0, 0))(params, x, y)
        except Exception:  # noqa: BLE001
            outs = [gfn(params, x[i], y[i]) for i in range(x.shape[0])]
            g = {n: torch.stack([o[n] for o in outs]) for n in outs[0]}
        with torch.no_grad():
            for n in names:
                sq[n] += g[n].float().pow(2).sum(0)
                mean_g[n] += g[n].float().sum(0)
        count += x.shape[0]
        del g
        _maybe_empty_cache(device)

    c = max(1, count)
    fisher = {n: (v / c).cpu() for n, v in sq.items()}
    snip = {n: (params[n].detach().float() * (mean_g[n] / c)).abs().cpu() for n in names}
    return {"fisher": fisher, "snip": snip}


def synflow(model: nn.Module, input_shape, device, is_text: bool = False) -> Tensors:
    """|theta * dR/dtheta| with R the all-ones-input output sum under |theta|.

    Data-free by construction, so it is the natural "does this need data at all?" control.
    Signs are restored afterwards; the model is left exactly as it was found.
    """
    model.eval()
    signs = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            signs[n] = p.sign()
            p.abs_()
    try:
        if is_text:
            x = torch.zeros(1, *input_shape, dtype=torch.long, device=device)
        else:
            x = torch.ones(1, *input_shape, device=device)
        out = model(x)
        model.zero_grad(set_to_none=True)
        out.sum().backward()
        scores = {n: (p.detach() * p.grad.detach()).abs().float().cpu()
                  if p.grad is not None else torch.zeros_like(p, dtype=torch.float32).cpu()
                  for n, p in model.named_parameters() if p.requires_grad}
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.mul_(signs[n])
        model.zero_grad(set_to_none=True)
    return scores


def all_criteria(model: nn.Module, sens_scores: Tensors, labelled_batches, device, task,
                 input_shape, is_text: bool = False) -> Dict[str, Tensors]:
    out = {"functional": sens_scores, "magnitude": magnitude(model)}
    out.update(loss_based(model, labelled_batches, device, task))
    try:
        out["synflow"] = synflow(model, input_shape, device, is_text)
    except Exception as exc:  # noqa: BLE001 - synflow overflows on some architectures
        print(f"    synflow skipped: {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# structured aggregation
# ---------------------------------------------------------------------------

def structured_groups(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Map each parameter tensor to a per-output-unit group id.

    Structured sparsity (whole heads, channels, neurons) is what yields actual speedups,
    so the follow-up paper needs to know whether the *structured* ordering freezes on the
    same timescale as the unstructured one. Grouping by output unit -- rows of a Linear,
    filters of a Conv -- is the coarsest useful version and needs no extra measurement:
    the group score is just the sum of S over the group.
    """
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad or p.ndim < 2:
            continue
        rows = p.shape[0]
        groups[n] = torch.arange(rows).repeat_interleave(p[0].numel())
    return groups


def aggregate_structured(scores: Tensors, groups: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum S within each output unit, concatenated across tensors."""
    out = []
    for n, gid in groups.items():
        flat = scores[n].reshape(-1).float()
        n_groups = int(gid.max()) + 1
        out.append(torch.zeros(n_groups).index_add_(0, gid, flat))
    return torch.cat(out) if out else torch.zeros(0)
