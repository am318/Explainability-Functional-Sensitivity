"""
pruning_baselines.py
====================
Training-free pruning baselines (SNIP, GraSP, SynFlow) for Vision
Transformers (ViT), written as fair comparison baselines for a NeurIPS paper
against a method that can prune *any* component.

Parameter-set policy
--------------------
The competing method operates over all parameters, so the baselines must
match that search space as closely as their theoretical foundations allow.

SNIP  – scores and masks **all** ``requires_grad`` parameters, including
        biases, LayerNorm affine parameters, positional embeddings, and the
        CLS token.  SNIP's criterion |g·w| is defined for any differentiable
        parameter, so no exclusions are needed for a fair comparison.

GraSP – same full parameter set as SNIP.  The Hessian-gradient product is
        well-defined for every differentiable leaf.

SynFlow – excludes **only** ``pos_embed``, ``cls_token``, and ``dist_token``
        from the scored set.  Those tensors are additive positional /
        classification biases, not multiplicative synapses; including them
        breaks the synaptic-flow conservation identity on which SynFlow's
        layer-collapse-free guarantee rests.  LayerNorm affine parameters
        and bias vectors are multiplicative or additive in a way that *is*
        captured by the hook-based scalar, so they are included.
        This narrow exclusion is the only one that is theoretically
        motivated; it should be disclosed in the paper.

All three methods
-----------------
* Tied / shared parameters are deduplicated by ``id(p)`` so the flat score
  vector is never double-counted.
* Parameters without ``requires_grad`` are silently skipped and receive
  all-True masks.
* ``_global_mask_from_scores`` applies the sparsity threshold only over the
  scored set; every unscored parameter gets an all-True (keep-all) mask so
  downstream ``mask * weight`` logic is uniform.

References
----------
* SNIP    – Lee et al., ICLR 2019
* GraSP   – Wang et al., ICLR 2020
* SynFlow – Tanaka et al., NeurIPS 2020
"""

from __future__ import annotations

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Callable, Dict, Tuple

import copy

from sensitivity_pruning import *

from build_sparse_model import *

# ---------------------------------------------------------------------------
# Parameter enumeration helpers
# ---------------------------------------------------------------------------

_SYNFLOW_SKIP = ("pos_embed", "cls_token", "dist_token")


def _all_learnable_params(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Return every ``requires_grad`` parameter, deduplicated by identity.

    Used by SNIP and GraSP, which impose no theoretical restrictions on
    which parameter types may be scored.
    """
    seen: set = set()
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        out[name] = p
    return out


def _synflow_params(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Return learnable parameters eligible for SynFlow scoring.

    Excludes ``pos_embed``, ``cls_token``, and ``dist_token`` because those
    tensors are additive positional / classification offsets rather than
    multiplicative synapses.  Including them violates the synaptic-flow
    conservation identity and breaks SynFlow's layer-collapse-free guarantee.
    All other parameters — including LayerNorm affine weights, biases, and
    projection matrices — are included.
    """
    seen: set = set()
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        if any(tok in name.lower() for tok in _SYNFLOW_SKIP):
            continue
        seen.add(id(p))
        out[name] = p
    return out


# ---------------------------------------------------------------------------
# Global threshold + mask construction
# ---------------------------------------------------------------------------

def _global_mask_from_scores(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    target_sparsity: float,
) -> Dict[str, torch.Tensor]:
    """Apply a global magnitude threshold to *scores* and return boolean masks.

    The threshold is computed over the scored parameters only.  Every
    ``requires_grad`` parameter absent from *scores* receives an all-True
    mask (i.e. it is left unpruned).

    Parameters
    ----------
    model:
        The network whose ``named_parameters()`` defines the full mask dict.
    scores:
        ``{name: score_tensor}`` for every parameter that should compete in
        the global threshold.  Shapes must match the corresponding parameters.
    target_sparsity:
        Fraction of *scored* parameters to zero out, in ``[0, 1)``.

    Returns
    -------
    masks : dict[str, BoolTensor]
        One boolean mask per ``requires_grad`` parameter.
    """
    if not (0.0 <= target_sparsity < 1.0):
        raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")

    flat = torch.cat([scores[n].reshape(-1) for n in scores])
    n_keep = max(1, int(round((1.0 - target_sparsity) * flat.numel())))

    keep_idx = torch.topk(flat, n_keep, largest=True, sorted=False).indices
    flat_mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    flat_mask[keep_idx] = True

    # Slice flat mask back into per-parameter tensors
    scored_masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name in scores:
        n = scores[name].numel()
        scored_masks[name] = flat_mask[offset : offset + n].view_as(scores[name])
        offset += n

    # Build full mask dict; unscored params kept intact
    masks: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        masks[name] = scored_masks.get(name, torch.ones_like(p, dtype=torch.bool))
    return masks


# ---------------------------------------------------------------------------
# SNIP
# ---------------------------------------------------------------------------

def build_snip_masks(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_sparsity: float,
    *,
    n_batches: int = 4,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """SNIP – Single-shot Network Pruning (Lee et al., ICLR 2019).

    Scores **all** ``requires_grad`` parameters via ``|g·w| / (|w| + ε)``.

    Baseline policy
    ---------------
    No parameter types are excluded.  This matches the search space of the
    competing method and makes the comparison fair.

    ViT-specific adjustments
    ------------------------
    * Gradients are accumulated over ``n_batches`` equal micro-batches to
      reduce the variance introduced by stochastic attention.
    * The score denominator is stabilised by ``eps`` so that near-zero
      weights do not dominate the global ranking relative to large, well-
      trained weights.

    Parameters
    ----------
    model:
        ViT model.  Should be in training mode so dropout is active; gradients
        must be enabled.
    x, y:
        A representative input batch split into ``n_batches`` micro-batches
        for gradient accumulation.
    loss_fn:
        Task loss, e.g. ``nn.CrossEntropyLoss()``.
    target_sparsity:
        Fraction of all learnable parameters to remove.
    n_batches:
        Number of gradient-accumulation steps.
    eps:
        Denominator stabiliser.
    """
    params = _all_learnable_params(model)
    model.zero_grad(set_to_none=True)

    chunk_size = max(1, x.size(0) // n_batches)
    n_chunks = 0
    for start in range(0, x.size(0), chunk_size):
        loss = loss_fn(model(x[start : start + chunk_size]),
                       y[start : start + chunk_size]) / n_batches
        loss.backward()
        n_chunks += 1

    scores: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in params.items():
            g = p.grad
            if g is None:
                scores[name] = torch.zeros_like(p)
            else:
                scores[name] = (g * p).abs() / (p.abs() + eps)

    model.zero_grad(set_to_none=True)
    return _global_mask_from_scores(model, scores, target_sparsity)


# ---------------------------------------------------------------------------
# GraSP
# ---------------------------------------------------------------------------

def build_grasp_masks(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_sparsity: float,
) -> Dict[str, torch.Tensor]:
    """GraSP – Gradient Signal Preservation (Wang et al., ICLR 2020).

    Scores **all** ``requires_grad`` parameters via ``-w · Hg``, clipped to
    ``[0, ∞)`` (negative values are not informative for pruning decisions).

    Baseline policy
    ---------------
    No parameter types are excluded, matching the competing method's search
    space.  The Hessian-gradient product is well-defined for any differentiable
    leaf tensor.

    ViT-specific adjustments
    ------------------------
    * ``allow_unused=True`` in both ``autograd.grad`` calls prevents crashes
      on parameters that are technically in the graph but receive no gradient
      in a particular forward pass (e.g. unused expert parameters in MoE ViTs).
    * ``None`` gradients are replaced with zero tensors so that the score
      tensor dict is always fully populated.

    Parameters
    ----------
    model, x, y, loss_fn, target_sparsity:
        Same semantics as :func:`build_snip_masks`.
    """
    params = _all_learnable_params(model)
    param_list = list(params.values())

    # First-order gradients (retain graph for second-order pass)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(
        loss, param_list, create_graph=True, allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grads, param_list)]

    # Hessian-gradient product via double backprop on ||g||²
    gnorm = sum((g * g).sum() for g in grads)
    hg = torch.autograd.grad(gnorm, param_list, allow_unused=True)
    hg = [h if h is not None else torch.zeros_like(p)
          for h, p in zip(hg, param_list)]

    scores: Dict[str, torch.Tensor] = {}
    for (name, p), h in zip(params.items(), hg):
        scores[name] = torch.clamp(-(p * h).detach(), min=0.0)

    model.zero_grad(set_to_none=True)
    return _global_mask_from_scores(model, scores, target_sparsity)


# ---------------------------------------------------------------------------
# SynFlow
# ---------------------------------------------------------------------------

def build_synflow_masks(
    model: nn.Module,
    /,
    *args,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """SynFlow – Synaptic Flow Pruning (Tanaka et al., NeurIPS 2020).

    Scores all learnable parameters **except** ``pos_embed``, ``cls_token``,
    and ``dist_token``.

    Baseline policy and theoretical justification for the narrow exclusion
    -----------------------------------------------------------------------
    SynFlow's layer-collapse-free guarantee rests on a synaptic-flow
    conservation identity: the score of each parameter equals the fraction of
    the network's total synaptic flow that passes through it.  This identity
    holds only for *multiplicative* synapses (weights and biases of linear /
    convolutional layers, LayerNorm affine parameters).

    Positional embeddings and CLS / distillation tokens are *additive* offsets
    injected after the patch projection; they do not participate in the
    multiplicative weight-activation product that defines synaptic flow.
    Including them would break the conservation identity and could cause the
    method to assign arbitrarily high scores to those tensors, distorting the
    global threshold.

    LayerNorm ``weight`` and ``bias``, attention biases, and projection biases
    *are* included because they enter the computation graph multiplicatively or
    additively in a way that the hook-based scalar captures correctly.

    This exclusion is the only theoretically motivated deviation from a fully
    unconstrained parameter set and should be disclosed as a footnote in the
    paper.

    ViT-specific adjustments
    ------------------------
    * The dummy input is built from the raw **image** shape ``(B,C,H,W)`` so
      the patch-embedding Conv2d is always part of the dataflow.
    * The backward scalar is the sum of **absolute pre-activation values**
      accumulated by forward hooks on ``nn.Linear``, ``nn.Conv2d``, and
      ``nn.LayerNorm`` modules — not the sum of final logits.  This preserves
      the conservation identity through the attention softmax.
    * Weights are temporarily replaced by their absolute values; ``sign(0)``
      is mapped to ``+1`` to prevent permanent annihilation of zero-valued
      entries.  Originals are restored via ``p.data.copy_()`` rather than
      ``p.mul_(sign)`` to avoid the same bug.

    Accepted call signatures
    ------------------------
    Canonical::

        build_synflow_masks(model, target_sparsity, image_shape)

    SNIP-compatible drop-in (extra data arguments are ignored)::

        build_synflow_masks(model, x, y, loss_fn, target_sparsity)

    Parameters
    ----------
    model:
        ViT model.
    target_sparsity:
        Fraction of scored parameters to remove.
    image_shape:
        Full image tensor shape, e.g. ``(1, 3, 224, 224)``.  In SNIP-
        compatible mode this is inferred from the leading tensor argument
        (which must be 4-D).
    """
    target_sparsity, image_shape = _parse_synflow_args(args, kwargs)
    device = next(model.parameters()).device
    params = _synflow_params(model)

    # Save originals; set all weights to |w|, mapping sign(0) → +1
    saved: Dict[str, torch.Tensor] = {}
    for name, p in params.items():
        saved[name] = p.data.clone()
        s = torch.sign(p.data)
        s[s == 0] = 1
        p.data.copy_(p.data.abs())

    # Forward hooks accumulate sum(|activation|) over Linear/Conv2d/LayerNorm
    _accum = [torch.tensor(0.0, device=device)]
    hooks = []

    def _hook(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            _accum[0] = _accum[0] + t.abs().sum()

    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            hooks.append(mod.register_forward_hook(_hook))

    x_ones = torch.ones(image_shape, device=device)
    model.zero_grad(set_to_none=True)
    try:
        model(x_ones)
        _accum[0].backward()
    finally:
        for h in hooks:
            h.remove()

    scores: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in params.items():
            g = p.grad
            scores[name] = (p * g).abs().detach() if g is not None \
                           else torch.zeros_like(p)

    # Restore original weights
    with torch.no_grad():
        for name, p in params.items():
            p.data.copy_(saved[name])

    model.zero_grad(set_to_none=True)
    return _global_mask_from_scores(model, scores, target_sparsity)


def _parse_synflow_args(
    args: tuple,
    kwargs: dict,
) -> Tuple[float, Tuple[int, ...]]:
    """Resolve the overloaded positional signature of :func:`build_synflow_masks`."""
    target_sparsity = kwargs.pop("target_sparsity", None)
    image_shape = kwargs.pop("image_shape", None) or kwargs.pop("input_shape", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(kwargs))}")

    if len(args) == 2:
        # (model, target_sparsity, image_shape)
        if target_sparsity is not None or image_shape is not None:
            raise TypeError(
                "Do not mix positional and keyword target_sparsity / image_shape"
            )
        target_sparsity, image_shape = args

    elif len(args) == 4:
        # (model, x, y, loss_fn, target_sparsity)  — SNIP-compatible
        if target_sparsity is not None:
            raise TypeError("target_sparsity supplied twice")
        x, _y, _loss_fn, target_sparsity = args
        if image_shape is None:
            if not torch.is_tensor(x):
                raise TypeError(
                    "In SNIP-compatible mode the first extra argument must be a tensor"
                )
            if x.dim() != 4:
                raise ValueError(
                    "Cannot infer image_shape from a non-4-D tensor. "
                    "Pass image_shape explicitly."
                )
            image_shape = (1,) + tuple(x.shape[1:])

    else:
        raise TypeError(
            "build_synflow_masks expected (model, target_sparsity, image_shape) "
            f"or (model, x, y, loss_fn, target_sparsity); got {len(args)} extra args."
        )

    if target_sparsity is None:
        raise TypeError("target_sparsity was not provided.")
    if image_shape is None:
        raise TypeError("image_shape could not be inferred; pass it explicitly.")

    return float(target_sparsity), tuple(image_shape)

def build_calibrated_baseline_masks(
    base_model: nn.Module,
    prune_images: torch.Tensor,
    prune_targets: torch.Tensor,
    target_actual_prune_fraction: float,
    cfg,
    device: torch.device,
    PRUNING_METHOD: str,
    ref_model,
    max_search_rounds: int = 30,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], float]:
    """Search for an effective prune_fraction that best matches the target eligible prune rate."""

    criterion = nn.CrossEntropyLoss()
    search_low = 0.0
    search_high = 0.9999
    best_masks = None
    best_stats = None
    best_effective_fraction = None
    best_actual_fraction = None
    best_abs_error = float("inf")

    for _ in range(max_search_rounds):
        candidate_fraction = 0.5 * (search_low + search_high)

        if PRUNING_METHOD == "snip":
            candidate_masks = build_snip_masks(
                base_model,
                prune_images,
                prune_targets,
                criterion,
                candidate_fraction,
            )
        elif PRUNING_METHOD == "synflow":
            candidate_masks = build_synflow_masks(
                base_model,
                prune_images,
                prune_targets,
                criterion,
                candidate_fraction,
            )
        else:
            raise ValueError(f"Unsupported PRUNING_METHOD={PRUNING_METHOD!r}")

        candidate_model = copy.deepcopy(base_model)
        apply_masks_(candidate_model, candidate_masks)
        candidate_sparse_model = build_sparse_model_from_masks(candidate_model, candidate_masks, cfg, device)
        candidate_stats = compute_eligible_pruning_stats(ref_model, candidate_sparse_model, candidate_masks, cfg)
        actual_fraction = float(candidate_stats["actual_prune_fraction_eligible_baseline"])
        abs_error = abs(actual_fraction - target_actual_prune_fraction)

        if abs_error < best_abs_error:
            best_abs_error = abs_error
            best_masks = candidate_masks
            best_stats = candidate_stats
            best_effective_fraction = candidate_fraction
            best_actual_fraction = actual_fraction

        # Assuming monotonicity
        if actual_fraction < target_actual_prune_fraction:
            search_low = candidate_fraction
        else:
            search_high = candidate_fraction

    assert best_masks is not None and best_stats is not None and best_effective_fraction is not None and best_actual_fraction is not None
    best_stats = dict(best_stats)
    best_stats["baseline_effective_prune_fraction"] = float(best_effective_fraction)
    best_stats["baseline_target_actual_prune_fraction"] = float(target_actual_prune_fraction)
    best_stats["baseline_actual_prune_fraction_error"] = float(best_actual_fraction - target_actual_prune_fraction)
    return best_masks, best_stats, float(best_effective_fraction)