"""Parameterwise functional sensitivity.

    S(theta) = E_x [ (1/d) * sum_{i=1}^{d} ( d f_i(x) / d theta )^2 ]

Label-free and loss-free: it depends only on the function the network computes. This is
the diagonal of the Gauss-Newton / NTK operator in parameter space, which is what lets the
lazy-training argument (C5.1) attach to it.

Two estimators, both unbiased for the same quantity:

* ``exact``      -- average the squared gradient over every output coordinate. Costs d
                    backward passes per batch, so it is used when d is small (CIFAR: d=10).
                    Has *no probe noise at all*, which removes an entire class of reviewer
                    objection from the vision results.
* ``hutchinson`` -- Rademacher probes r with E[r r^T] = I/d, so E[(d(f.r)/dtheta)^2]
                    equals the target. Used when d is large (GPT: d = T*V).

Both compute *per-example* gradients via ``torch.func.vmap``, then square. That matters:
squaring a batch-averaged gradient estimates (E g)^2, not E[g^2], and the two differ by
exactly the gradient covariance. The per-example form is the definition we want and has
far lower variance for the same compute.

The same pass yields the empirical **trace NTK Gram** on a fixed probe batch, which drives
the kernel-velocity measurement behind C4/C5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

Tensors = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# which parameters take part in a ranking
# ---------------------------------------------------------------------------

def is_prunable(name: str, p: torch.nn.Parameter, cfg) -> bool:
    """Eligibility for the *ranking*, mirroring what a pruner would actually touch.

    Scores are always computed for every trainable parameter; this only decides which ones
    enter the rank statistics, because including 1-D norm/bias parameters (whose
    sensitivity is orders of magnitude larger) would manufacture stability for free.
    """
    if not p.requires_grad:
        return False
    if cfg.include == "all":
        return True
    from .models.lazy import strip_wrapper_prefix
    name = strip_wrapper_prefix(name)
    if p.ndim == 1 and not cfg.prune_bias:
        return False
    lname = name.lower()
    if ("norm" in lname or lname.endswith(".bn1.weight")) and not cfg.prune_norm:
        return False
    if any(k in name for k in ("patch_embed", "pos_embed", "cls_token", "tok_emb", "pos_emb")) \
            and not cfg.prune_embeddings:
        return False
    if name.startswith("head") and not cfg.prune_head:
        return False
    return True


def prunable_mask(model: nn.Module, cfg) -> Tensors:
    return {n: torch.full(p.shape, is_prunable(n, p, cfg), dtype=torch.bool)
            for n, p in model.named_parameters() if p.requires_grad}


def param_names(model: nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def flatten(tensors: Tensors, names: Sequence[str]) -> torch.Tensor:
    return torch.cat([tensors[n].reshape(-1) for n in names])


def layer_index(model: nn.Module, names: Sequence[str]) -> torch.Tensor:
    """Per-element integer id of the owning parameter tensor.

    This is the grouping used by every layerwise control (C2b): within-layer rank
    correlation and the layer-budget-matched chance baseline both key off it.
    """
    sizes = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    return torch.cat([torch.full((sizes[n],), i, dtype=torch.int32)
                      for i, n in enumerate(names)])


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class SensResult:
    scores: Tensors                       # mean over all folds
    fold_scores: List[Tensors]            # disjoint folds -> C2a noise floor
    ntk: Optional[torch.Tensor] = None    # (m, m) trace-NTK Gram
    meta: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------

def _zeros_like_params(params: Tensors) -> Tensors:
    return {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}


def _directional_grad_fn(model: nn.Module, buffers: Tensors):
    """d/dtheta of <f(x), v> for a single example, as a function of (params, x, v)."""

    def scalar(params: Tensors, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return (out.reshape(-1) * v).sum()

    return grad(scalar)


def _rademacher(shape, d: int, generator, device, dtype) -> torch.Tensor:
    r = torch.randint(0, 2, shape, generator=generator, device="cpu").to(dtype) * 2 - 1
    return (r / math.sqrt(d)).to(device)


def _directions(estimator: str, out_dim: int, n_probes: int, batch: int,
                generator, device, dtype) -> List[Tuple[torch.Tensor, bool]]:
    """The list of output-space directions to differentiate along.

    Returns (v, per_example) pairs. `per_example` says whether v has a leading batch dim.
    Exact mode walks the d one-hot directions and rescales by 1/sqrt(d) so that the two
    estimators target *numerically the same* quantity and can be compared directly.
    """
    if estimator == "exact":
        scale = 1.0 / math.sqrt(out_dim)
        eye = torch.eye(out_dim, device=device, dtype=dtype) * scale
        return [(eye[i], False) for i in range(out_dim)]
    return [(_rademacher((batch, out_dim), out_dim, generator, device, dtype), True)
            for _ in range(n_probes)]


def compute_sensitivity(
    model: nn.Module,
    batches: Sequence[torch.Tensor],
    cfg,
    device: torch.device,
    ntk_batch: Optional[torch.Tensor] = None,
    fold_of_batch: Optional[Sequence[int]] = None,
) -> SensResult:
    """Estimate S(theta) over `batches` (input tensors only -- labels are never used).

    Args:
        batches: input-only batches. Labels are deliberately not accepted: functional
            sensitivity must not be able to see them.
        fold_of_batch: fold id per batch, giving disjoint estimates for the noise floor.
    """
    model.eval()
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    names = list(params)

    with torch.no_grad():
        probe_out = functional_call(model, (params, buffers), (batches[0][:1].to(device),))
    out_dim = int(probe_out.reshape(-1).numel())
    dtype = probe_out.dtype

    estimator = cfg.estimator
    if estimator == "auto":
        estimator = "exact" if out_dim <= cfg.exact_max_outputs else "hutchinson"
    if estimator == "exact" and out_dim > cfg.exact_max_outputs:
        raise ValueError(
            f"exact estimator requested with output dim {out_dim} > "
            f"exact_max_outputs={cfg.exact_max_outputs}")

    n_folds = max(1, int(cfg.folds))
    if fold_of_batch is None:
        fold_of_batch = [i % n_folds for i in range(len(batches))]

    grad_fn = _directional_grad_fn(model, buffers)
    use_vmap = cfg.impl != "loop"

    fold_sums = [_zeros_like_params(params) for _ in range(n_folds)]
    fold_counts = [0 for _ in range(n_folds)]
    generator = torch.Generator().manual_seed(int(cfg.seed))

    for batch, fold in zip(batches, fold_of_batch):
        x = batch.to(device, non_blocking=True)
        bsz = x.shape[0]
        dirs = _directions(estimator, out_dim, cfg.n_probes, bsz, generator, device, dtype)
        n_dirs = len(dirs)
        for v, per_example in dirs:
            g = _grads_for_direction(grad_fn, params, x, v, per_example, use_vmap)
            with torch.no_grad():
                for n in names:
                    # sum over the batch of squared per-example gradients
                    fold_sums[fold][n] += g[n].float().pow(2).sum(dim=0)
            del g
        fold_counts[fold] += bsz
        _maybe_empty_cache(device)

    # normalise: exact sums over d directions (already scaled by 1/sqrt(d) each, so the
    # squares carry 1/d and summing over i gives the average); hutchinson averages probes.
    divisor_extra = float(cfg.n_probes) if estimator == "hutchinson" else 1.0
    # Both estimators internally carry a 1/d_y from the probe scaling. Multiply it back
    # out so the stored quantity is exactly the draft's
    #     S_i(theta) = E_x || dF_theta(x) / dtheta_i ||_2^2,
    # and the identity tr(K) = tr(Q) = sum_i S_i holds without a hidden constant.
    fold_scores: List[Tensors] = []
    for sums, count in zip(fold_sums, fold_counts):
        denom = max(1, count) * divisor_extra / float(out_dim)
        fold_scores.append({n: (t / denom).cpu() for n, t in sums.items()})

    total_count = sum(fold_counts)
    scores = {
        n: sum(fs[n] * (c / max(1, total_count)) for fs, c in zip(fold_scores, fold_counts))
        for n in names
    }

    ntk = None
    if ntk_batch is not None and ntk_batch.numel():
        ntk = compute_trace_ntk(model, ntk_batch, cfg, device, estimator, out_dim, dtype)

    meta = {
        "estimator": estimator,
        "out_dim": out_dim,
        "n_examples": int(total_count),
        "fold_counts": list(fold_counts),
        "n_probes": int(cfg.n_probes) if estimator == "hutchinson" else out_dim,
        "impl": "vmap" if use_vmap else "loop",
    }
    return SensResult(scores=scores, fold_scores=fold_scores, ntk=ntk, meta=meta)


def _grads_for_direction(grad_fn, params: Tensors, x: torch.Tensor, v: torch.Tensor,
                         per_example: bool, use_vmap: bool) -> Tensors:
    """Per-example gradients, shaped [B, *param_shape] for each parameter."""
    if use_vmap:
        in_dims = (None, 0, 0 if per_example else None)
        try:
            return vmap(grad_fn, in_dims=in_dims)(params, x, v)
        except Exception:  # noqa: BLE001 - fall back to the slow path on any vmap issue
            use_vmap = False
    outs: List[Tensors] = []
    for b in range(x.shape[0]):
        vb = v[b] if per_example else v
        outs.append(grad_fn(params, x[b], vb))
    return {n: torch.stack([o[n] for o in outs], dim=0) for n in outs[0]}


def compute_trace_ntk(model: nn.Module, batch: torch.Tensor, cfg, device: torch.device,
                      estimator: str, out_dim: int, dtype) -> torch.Tensor:
    """Empirical trace-NTK Gram K_ij = (1/d) sum_k <df_k(x_i)/dtheta, df_k(x_j)/dtheta>.

    Uses a *shared* direction across examples so the inner products are well defined.
    Kernel velocity -- the normalised drift of K across checkpoints -- is our measure of
    departure from lazy training (C4, C5.1).
    """
    model.eval()
    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    grad_fn = _directional_grad_fn(model, buffers)
    x = batch.to(device)
    m = x.shape[0]
    generator = torch.Generator().manual_seed(int(cfg.seed) + 7919)

    if estimator == "exact":
        scale = 1.0 / math.sqrt(out_dim)
        dirs = [torch.eye(out_dim, device=device, dtype=dtype)[i] * scale
                for i in range(out_dim)]
    else:
        dirs = [_rademacher((out_dim,), out_dim, generator, device, dtype)
                for _ in range(cfg.n_probes)]

    gram = torch.zeros(m, m, dtype=torch.float32)
    for v in dirs:
        g = _grads_for_direction(grad_fn, params, x, v, per_example=False,
                                 use_vmap=cfg.impl != "loop")
        with torch.no_grad():
            flat = torch.cat([g[n].reshape(m, -1) for n in g], dim=1).float()
            gram += (flat @ flat.T).cpu()
        del g, flat
        _maybe_empty_cache(device)
    # Exact mode *sums* over the d one-hot directions (each already carrying 1/sqrt(d));
    # Hutchinson *averages* over probes. Both then equal
    #     K_ij = (1/d) sum_k <df_k(x_i)/dtheta, df_k(x_j)/dtheta>,
    # so K_ii == sum_theta S(theta; x_i) and the vision (exact) and GPT (Hutchinson)
    # kernel-velocity numbers live on the same scale.
    gram = gram if estimator == "exact" else gram / len(dirs)
    return gram * float(out_dim)   # same 1/d_y rescale as the sensitivity scores


def _maybe_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
