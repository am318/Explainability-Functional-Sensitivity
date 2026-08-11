import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


def parameter_count(model: nn.Module) -> int:
    """Count all parameters in a model, regardless of ``requires_grad``.

    Args:
        model: The model whose parameters will be counted.

    Returns:
        Total number of scalar parameter elements in the model.
    """
    return sum(p.numel() for p in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    """Count only trainable (``requires_grad=True``) parameters in a model.

    Args:
        model: The model whose trainable parameters will be counted.

    Returns:
        Total number of scalar elements across trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_prunable_parameter(name: str, p: torch.nn.Parameter, cfg) -> bool:
    """Decide whether a named parameter is eligible for pruning under ``cfg``.

    Args:
        name: Fully qualified parameter name (e.g. ``"blocks.0.attn.qkv.weight"``),
            as produced by ``model.named_parameters()``.
        p: The parameter tensor itself. Must have ``requires_grad=True`` to be
            considered prunable; its ``ndim`` is used to detect bias vectors.
        cfg: Configuration object exposing boolean flags ``prune_bias``,
            ``prune_norm``, ``prune_embeddings``, and ``prune_head`` that control
            which parameter categories are excluded from pruning.

    Returns:
        True if the parameter is eligible for pruning, False otherwise.
    """
    if not p.requires_grad:
        return False
    if p.ndim == 1 and not cfg.prune_bias:
        return False
    if ("norm" in name.lower()) and not cfg.prune_norm:
        return False
    if (("patch_embed" in name) or ("pos_embed" in name) or ("cls_token" in name)) and not cfg.prune_embeddings:
        return False
    if name.startswith("head") and not cfg.prune_head:
        return False
    return True


def init_score_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Allocate a zero-initialized CPU float32 buffer per trainable parameter.

    Args:
        model: The model whose trainable parameters define the buffer shapes.
            Buffers are keyed by name (as from ``model.named_parameters()``)
            and shaped to match each corresponding parameter.

    Returns:
        Dict mapping parameter name to a zero tensor of matching shape, on CPU,
        dtype float32.
    """
    return {
        name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def _make_probe(logits: torch.Tensor) -> torch.Tensor:
    """Generate a Rademacher (+-1) random probe, scaled to unit expected norm.

    Args:
        logits: Reference tensor whose shape and last-dimension size determine
            the probe's shape and scaling. Only ``logits.shape`` is used, not
            its values.

    Returns:
        A tensor of the same shape as ``logits``, containing independent +-1
        entries scaled by ``1 / sqrt(logits.shape[-1])``.
    """
    probe = torch.empty_like(logits).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    probe = probe / math.sqrt(max(1, logits.shape[-1]))
    return probe


def flatten_like_model(
    model: nn.Module,
    tensors: Dict[str, torch.Tensor],
    only_active: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Flatten and concatenate per-parameter tensors in model parameter order.

    Args:
        model: The model whose ``named_parameters()`` order and names determine
            which entries of ``tensors`` are included and in what order.
        tensors: Dict mapping parameter name to a tensor of matching shape
            (e.g. scores or magnitudes per parameter). Entries whose name is
            not a parameter of ``model`` are skipped.
        only_active: Optional dict mapping parameter name to a boolean mask of
            matching shape. When provided, only the masked (True) entries of
            each tensor in ``tensors`` are kept before concatenation.

    Returns:
        A 1D float32 tensor formed by flattening and concatenating each
        included tensor, in the order parameters appear in ``model``. Empty
        if no entries matched.
    """
    parts = []
    for name, p in model.named_parameters():
        if name not in tensors:
            continue
        t = torch.as_tensor(tensors[name]).detach().cpu().reshape(-1)
        if only_active is not None:
            m = torch.as_tensor(only_active[name], dtype=torch.bool).detach().cpu().reshape(-1)
            t = t[m]
        parts.append(t.float())
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts)


def flatten_param_magnitudes(
    model: nn.Module,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Flatten and concatenate absolute parameter values in model order.

    Args:
        model: The model whose parameters (in ``named_parameters()`` order)
            supply the magnitude values.
        masks: Optional dict mapping parameter name to a boolean mask of
            matching shape. When provided, only the masked (True) entries of
            each parameter's magnitudes are kept before concatenation. Must
            contain an entry for every parameter in ``model`` if provided.

    Returns:
        A 1D float32 tensor of concatenated absolute parameter values, in the
        order parameters appear in ``model``. Empty if the model has no
        parameters.
    """
    parts = []
    for name, p in model.named_parameters():
        t = p.detach().abs().cpu().reshape(-1)
        if masks is not None:
            m = masks[name].detach().cpu().bool().reshape(-1)
            t = t[m]
        parts.append(t.float())
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Spearman rank correlation between two tensors.

    Args:
        a: First tensor of values. Flattened before comparison; entries that
            are non-finite (paired with their counterpart in ``b``) are
            excluded from the computation.
        b: Second tensor of values, same number of elements as ``a`` after
            flattening. Compared element-wise against ``a``.

    Returns:
        The Spearman rank correlation coefficient as a float in [-1, 1], or
        ``nan`` if fewer than 2 finite paired values remain or the ranks have
        zero variance.
    """
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.numel() < 2:
        return float("nan")
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = torch.linalg.norm(ar) * torch.linalg.norm(br)
    if float(denom) == 0.0:
        return float("nan")
    return float((ar @ br / denom).item())


def topk_indices(values: torch.Tensor, frac: float) -> torch.Tensor:
    """Return the indices of the top fraction of values by magnitude/value.

    Args:
        values: Tensor of values to rank. Flattened before ranking.
        frac: Fraction of elements to select, in (0, 1]. Rounded up to at
            least 1 element and capped at ``values.numel()``.

    Returns:
        A 1D long tensor of indices (into the flattened ``values``) of the
        top ``k`` largest entries, on CPU, in unspecified order. Empty if
        ``values`` has no elements.
    """
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    k = max(1, int(math.ceil(float(frac) * values.numel())))
    k = min(k, values.numel())
    return torch.topk(values, k=k, largest=True, sorted=False).indices.cpu()


def mass_on_indices(values: torch.Tensor, indices: torch.Tensor) -> float:
    """Compute the fraction of total (clamped non-negative) mass at given indices.

    Args:
        values: Tensor of non-negative-interpretable values (negative entries
            are clamped to 0). Flattened before indexing.
        indices: 1D tensor of indices into the flattened ``values``, whose
            selected mass is measured against the total. Indices are clamped
            into valid range.

    Returns:
        The fraction, in [0, 1], of the total clamped mass of ``values`` that
        falls on the selected ``indices``. Returns ``nan`` if ``values`` or
        ``indices`` is empty, or if the total mass is non-positive.
    """
    values = values.detach().float().cpu().reshape(-1).clamp_min(0)
    if values.numel() == 0 or indices.numel() == 0:
        return float("nan")
    denom = float(values.sum().item())
    if denom <= 0.0:
        return float("nan")
    idx = indices.to(dtype=torch.long).clamp(0, values.numel() - 1)
    return float(values[idx].sum().item() / denom)


def effective_rank(eigvals: torch.Tensor) -> float:
    """Compute the exponential-entropy effective rank of a spectrum.

    Args:
        eigvals: Tensor of eigenvalues (or any non-negative-interpretable
            spectrum; negative entries are clamped to 0). Flattening/shape is
            not otherwise significant.

    Returns:
        The effective rank, computed as ``exp(entropy(p))`` where ``p`` is the
        spectrum normalized to sum to 1. Returns 0.0 if the total mass is
        non-positive.
    """
    vals = eigvals.detach().float().cpu().clamp_min(0)
    total = vals.sum()
    if float(total) <= 0.0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def probe_covariance_eigvals(probe_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Compute sorted eigenvalues of the (mean-centered) probe covariance.

    Args:
        probe_matrix: Optional 2D tensor of shape ``(n_rows, n_features)``
            where each row is one probe/gradient observation. Rows are
            mean-centered per feature before forming the covariance (Gram)
            matrix. If ``None`` or empty, no computation is performed.

    Returns:
        A 1D CPU tensor of eigenvalues sorted in descending order and clamped
        to be non-negative, or ``None`` if ``probe_matrix`` is ``None`` or
        empty.
    """
    if probe_matrix is None or probe_matrix.numel() == 0:
        return None
    X = probe_matrix.float()
    X = X - X.mean(dim=0, keepdim=True)
    gram = (X @ X.T) / max(1, X.shape[0])
    return torch.linalg.eigvalsh(gram).flip(0).clamp_min(0).detach().cpu()


def probe_spectrum_confidence(eigvals: Optional[torch.Tensor]) -> float:
    """Confidence derived only from probe-spectrum geometry.

    Higher when the probe spectrum is less dominated by a single direction and
    has more effective rank; lower when the signal is very peaky.

    Args:
        eigvals: Optional 1D tensor of eigenvalues (as returned by
            ``probe_covariance_eigvals``), assumed sorted in descending order.
            If ``None`` or empty, no signal is available.

    Returns:
        A confidence score in [0.25, 1.0]. Returns 1.0 if ``eigvals`` is
        ``None``, empty, or has non-positive total mass.
    """
    if eigvals is None or eigvals.numel() == 0:
        return 1.0

    vals = eigvals.detach().float().cpu().clamp_min(0)
    total = float(vals.sum().item())
    if total <= 0.0:
        return 1.0

    top_mass = float(vals[0].item() / total) if vals.numel() > 0 else 1.0
    er = effective_rank(vals)
    er_norm = er / max(1.0, float(vals.numel()))

    confidence = 0.5 * er_norm + 0.5 * (1.0 - top_mass)
    return max(0.25, min(1.0, confidence))


def parameter_gradient_row(output: torch.Tensor, params: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flatten d(output)/d(params) into one 1D row, zero-filling unused parameters.

    Args:
        output: A scalar (0-dim) tensor with an autograd graph depending on
            (some or all of) ``params``.
        params: Tuple of parameter tensors to differentiate ``output`` with
            respect to. Parameters not connected to ``output``'s graph receive
            a zero-filled gradient in the result (via ``allow_unused=True``).

    Returns:
        A 1D tensor formed by flattening and concatenating each parameter's
        gradient (or zeros, for unused parameters) in the order given by
        ``params``.
    """
    grads = torch.autograd.grad(output, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(params, grads)
        ]
    )


def _unflatten_into_scores(
    flat: torch.Tensor,
    names: List[str],
    shapes: List[torch.Size],
    scores: Dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> None:
    """Scatter a flat 1D tensor back into per-parameter score buffers.

    Args:
        flat: 1D tensor whose contiguous segments correspond, in order, to the
            flattened parameters named in ``names`` with shapes in ``shapes``.
            Its total length must equal the sum of the shapes' element counts.
        names: Parameter names, in the same order used to build ``flat`` and
            aligned with ``shapes``; used as keys into ``scores``.
        shapes: Target shapes for each segment of ``flat``, aligned by
            position with ``names``.
        scores: Dict mapping parameter name to a destination tensor of
            matching shape. Updated in place by accumulating each reshaped
            segment of ``flat``, scaled by ``alpha``.
        alpha: Scalar weight applied to each segment before accumulating into
            ``scores`` (i.e. ``scores[name] += alpha * chunk``).

    Returns:
        None. ``scores`` is modified in place.
    """
    offset = 0
    for name, shape in zip(names, shapes):
        n = 1
        for d in shape:
            n *= d
        chunk = flat[offset:offset + n].reshape(shape)
        scores[name].add_(chunk, alpha=alpha)
        offset += n


def compute_sensitivity_scores(
    model: nn.Module,
    loader,
    device: torch.device,
    exact: bool = False,
    n_probes: int = 8,
    collect_probe_matrix: bool = False,
    probe_matrix_rows: int = 64,
    max_probe_matrix_elements: int = 10_000_000,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Dict[str, object]]:
    """Sensitivity scores: Hutchinson/Rademacher estimate (default) or exact
    sum-of-squared-gradients over every output dimension (exact=True).

    In expectation the two agree: for a random probe r with E[r_i r_j] = delta_ij/d,
        E[(d(logits . r)/dtheta)^2] = (1/d) * sum_i (d logit_i/dtheta)^2.
    Exact mode is O(num_classes) backwards per example, so it's meant for small
    output dimensions / validation, not routine large-scale use.

    Args:
        model: The model to score. Must be callable on batches from ``loader``
            and expose trainable parameters via ``named_parameters()``.
        loader: Iterable of ``(images, targets)`` batches. ``targets`` is
            unused; only ``images`` is fed to ``model``. Each batch's images
            are moved to ``device`` before the forward pass.
        device: Device to run the forward/backward passes on. ``images`` are
            moved here; the model is expected to already reside on this
            device.
        exact: If False (default), estimate sensitivity via ``n_probes``
            random Rademacher probes per batch (Hutchinson estimator). If
            True, compute exact per-example sensitivity by backpropagating
            each output dimension separately and summing squared gradients;
            ``n_probes`` and ``collect_probe_matrix``'s row budget still apply
            to what's collected, but the estimator loop itself is skipped in
            favor of an exact per-example loop.
        n_probes: Number of random probe directions drawn per batch when
            ``exact=False``. Ignored when ``exact=True``. Higher values reduce
            estimator variance at the cost of more backward passes.
        collect_probe_matrix: If True, additionally collect a matrix of
            per-probe (or, in exact mode, per-example mean) gradient rows for
            spectral diagnostics, subject to ``probe_matrix_rows`` and
            ``max_probe_matrix_elements``. If False, no matrix is collected
            and the second and third return values related to it are ``None``
            for the matrix (metadata is still populated with defaults).
        probe_matrix_rows: Maximum number of rows to collect for the probe
            matrix when ``collect_probe_matrix=True``. Collection stops once
            this many rows have been gathered.
        max_probe_matrix_elements: Safety cap on the total number of elements
            (``active_count * probe_matrix_rows``) the probe matrix is allowed
            to occupy. If the projected size exceeds this, probe-matrix
            collection is disabled even if ``collect_probe_matrix=True``.
        masks: Optional dict mapping parameter name to a boolean tensor of
            matching shape, restricting which parameter entries are included
            when building probe-matrix rows (and, in exact mode, which
            entries of the mean gradient row are kept). Does not affect the
            per-parameter ``scores`` buffers, which always cover all
            trainable parameters.

    Returns:
        A 3-tuple:
            - ``scores``: Dict mapping parameter name to a CPU float32 tensor
              (matching that parameter's shape) of mean squared-gradient
              sensitivity, averaged over all accumulated samples.
            - ``probe_matrix``: A 2D CPU float32 tensor of collected rows (see
              ``collect_probe_matrix``), or ``None`` if none were collected.
            - ``meta``: Dict of diagnostics, including per-parameter
              ``parameter_sensitivity_variance`` and ``parameter_confidence``
              tensors, probe-spectrum eigenvalues (``probe_eigvals``),
              ``probe_effective_rank``, ``probe_top_mass``,
              ``probe_confidence``, and the boolean ``exact`` flag used.

    Raises:
        RuntimeError: If ``loader`` yields no samples.
    """
    model.eval()
    scores = init_score_buffers(model)
    score_second_moment = init_score_buffers(model)
    n_accum = 0
    probe_rows: List[torch.Tensor] = []

    active_count = (
        int(sum(int(m.detach().cpu().bool().sum().item()) for m in masks.values()))
        if masks is not None
        else int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    can_collect = collect_probe_matrix and (active_count * probe_matrix_rows <= max_probe_matrix_elements)

    if not exact:
        for images, _targets in loader:
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]

            for _ in range(n_probes):
                model.zero_grad(set_to_none=True)
                logits = model(images)
                probe = _make_probe(logits)
                ((logits * probe).sum() / batch_size).backward()

                with torch.no_grad():
                    grad_parts = []
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach().float()
                        sens = g.cpu().pow(2)
                        scores[name].add_(sens, alpha=batch_size)
                        score_second_moment[name].add_(sens.pow(2), alpha=batch_size)

                        if can_collect and len(probe_rows) < probe_matrix_rows:
                            row = g if masks is None else g[masks[name].to(device=p.device, dtype=torch.bool)]
                            grad_parts.append(row.detach().cpu().reshape(-1))

                    if can_collect and grad_parts and len(probe_rows) < probe_matrix_rows:
                        probe_rows.append(torch.cat(grad_parts))

                n_accum += batch_size
    else:
        param_items = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        names = [name for name, _ in param_items]
        params = tuple(p for _, p in param_items)
        shapes = [p.shape for p in params]
        flat_mask = (
            torch.cat([masks[n].detach().cpu().bool().reshape(-1) for n in names])
            if masks is not None else None
        )

        for images, _targets in loader:
            images = images.to(device, non_blocking=True)

            for b in range(images.shape[0]):
                model.zero_grad(set_to_none=True)
                logits = model(images[b:b + 1]).reshape(-1)
                num_outputs = logits.shape[0]

                example_sq_sum = torch.zeros(sum(p.numel() for p in params), dtype=torch.float32)
                rows: List[torch.Tensor] = []
                for i in range(num_outputs):
                    row = parameter_gradient_row(logits[i], params)
                    example_sq_sum += row.pow(2)
                    if can_collect and len(probe_rows) < probe_matrix_rows:
                        rows.append(row)
                example_sq_sum /= max(1, num_outputs)

                with torch.no_grad():
                    _unflatten_into_scores(example_sq_sum, names, shapes, scores)
                    _unflatten_into_scores(example_sq_sum.pow(2), names, shapes, score_second_moment)

                if can_collect and rows and len(probe_rows) < probe_matrix_rows:
                    mean_row = torch.stack(rows, dim=0).mean(dim=0)
                    probe_rows.append(mean_row if flat_mask is None else mean_row[flat_mask])

                n_accum += 1

    if n_accum == 0:
        raise RuntimeError("No samples were available for sensitivity scoring.")

    meta: Dict[str, object] = {
        "probe_eigvals": None,
        "probe_effective_rank": 0.0,
        "probe_top_mass": 0.0,
        "probe_confidence": 1.0,
        "parameter_confidence": {},
        "parameter_sensitivity_variance": {},
        "exact": exact,
    }

    for name in scores:
        scores[name].div_(n_accum)
        score_second_moment[name].div_(n_accum)

        variance = (score_second_moment[name] - scores[name].pow(2)).clamp_min(0.0)
        meta["parameter_sensitivity_variance"][name] = variance.cpu()

        rel_noise = variance.sqrt() / scores[name].abs().clamp_min(1e-12)
        meta["parameter_confidence"][name] = (1.0 / (1.0 + rel_noise)).clamp(0.25, 1.0).cpu()

    probe_matrix = torch.stack(probe_rows, dim=0) if probe_rows else None
    eigvals = probe_covariance_eigvals(probe_matrix)
    meta["probe_eigvals"] = eigvals
    if eigvals is not None:
        meta["probe_effective_rank"] = float(effective_rank(eigvals))
        total = float(eigvals.sum().item())
        meta["probe_top_mass"] = float(eigvals[0].item() / max(1e-12, total))
        meta["probe_confidence"] = float(probe_spectrum_confidence(eigvals))

    return scores, probe_matrix, meta