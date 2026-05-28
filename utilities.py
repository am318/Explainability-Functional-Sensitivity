import torch
import torch.nn as nn
import numpy as np
from torch.func import jacrev, functional_call, vmap
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def _capture_state(
    model: nn.Module,
) -> tuple[list[str], tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """
    Snapshot parameters/buffers once.

    Using a stable name order keeps flattening deterministic and avoids
    repeated named_parameter traversal inside the hot path.
    """
    param_items = list(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    param_names = [k for k, _ in param_items]
    params = tuple(v for _, v in param_items)

    return param_names, params, buffer_dict


def _flat_row_from_param_grads(grads: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Flatten a pytree of parameter gradients into one 1D row."""
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def _make_single_sample_jacobian_fn(
    model: nn.Module,
    param_names: list[str],
    buffer_dict: dict[str, torch.Tensor],
):
    """
    Returns a function jac_row(xi, params) -> [P] for one sample.
    """

    def forward_on_params(
        params: tuple[torch.Tensor, ...],
        xi: torch.Tensor,
    ) -> torch.Tensor:
        param_dict = dict(zip(param_names, params))
        out = functional_call(
            model,
            {**param_dict, **buffer_dict},
            (xi.unsqueeze(0),),
        )
        return out.squeeze()

    jacobian_of_forward = jacrev(forward_on_params, argnums=0)

    def jac_row(xi: torch.Tensor, params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        grads = jacobian_of_forward(params, xi)
        return _flat_row_from_param_grads(grads)

    return jac_row


def compute_parameter_jacobian(
    model: nn.Module,
    x: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Returns J with shape [N_data, N_parameters].

    This version:
    - snapshots params/buffers once
    - uses vmap(jacrev(...)) for per-sample Jacobians
    - computes in chunks to bound peak memory
    - avoids rebuilding parameter dictionaries inside the inner loop
    """
    device = next(model.parameters()).device
    x = x.to(device)

    param_names, params, buffer_dict = _capture_state(model)
    jac_row = _make_single_sample_jacobian_fn(model, param_names, buffer_dict)

    batched_jacobian = vmap(jac_row, in_dims=(0, None), randomness="error")

    rows: list[torch.Tensor] = []
    for start in range(0, x.shape[0], chunk_size):
        x_chunk = x[start : start + chunk_size]
        rows.append(batched_jacobian(x_chunk, params))

    return torch.cat(rows, dim=0)


def iter_jacobian_chunks(
    model: nn.Module,
    x: torch.Tensor,
    chunk_size: int = 64,
):
    """
    Yield Jacobian chunks without materializing the full J.

    Each yielded tensor has shape [chunk, P].
    Useful when the downstream goal is covariance, sensitivity scores,
    or other reductions over J.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    param_names, params, buffer_dict = _capture_state(model)
    jac_row = _make_single_sample_jacobian_fn(model, param_names, buffer_dict)

    batched_jacobian = vmap(jac_row, in_dims=(0, None), randomness="error")

    for start in range(0, x.shape[0], chunk_size):
        x_chunk = x[start : start + chunk_size]
        yield batched_jacobian(x_chunk, params)


def compute_covariance_from_model(
    model: nn.Module,
    x: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Compute C = (1/N) J^T J without ever storing full J.

    This is the preferred path if you only need the covariance matrix.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    # Infer P from one Jacobian chunk, then accumulate into C.
    jac_chunks = iter_jacobian_chunks(model, x, chunk_size=chunk_size)

    first_chunk = next(jac_chunks)
    _, P = first_chunk.shape

    C = torch.zeros((P, P), dtype=first_chunk.dtype, device=first_chunk.device)
    N = 0

    C.add_(first_chunk.T @ first_chunk)
    N += first_chunk.shape[0]

    for Jc in jac_chunks:
        C.add_(Jc.T @ Jc)
        N += Jc.shape[0]

    return C / N


def compute_covariance(J: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """
    C = (1/N) J^T J

    Retained for the case where J is already available.
    """
    N, P = J.shape
    C = torch.zeros((P, P), dtype=J.dtype, device=J.device)

    for start in range(0, N, chunk_size):
        Jc = J[start : start + chunk_size]
        C.add_(Jc.T @ Jc)

    return C / N


def sensitivity_scores_from_model(
    model: nn.Module,
    x: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    S_i = E[(df/dtheta_i)^2]

    Computes the mean of squared Jacobian entries directly from chunks.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    total_sq: torch.Tensor | None = None
    N = 0

    for Jc in iter_jacobian_chunks(model, x, chunk_size=chunk_size):
        sq = (Jc * Jc).sum(dim=0)
        total_sq = sq if total_sq is None else total_sq + sq
        N += Jc.shape[0]

    if total_sq is None:
        raise ValueError("Input x is empty; cannot compute sensitivity scores.")

    return total_sq / N


def sensitivity_scores(J: torch.Tensor) -> torch.Tensor:
    """
    S_i = E[(df/dtheta_i)^2]

    Kept for the case where J is already materialized.
    """
    return torch.mean(J * J, dim=0)


def eigvals_from_covariance(C: torch.Tensor) -> torch.Tensor:
    """
    Eigenvalues of symmetric PSD covariance matrix.

    Uses eigvalsh, which exploits symmetry.
    """
    eig = torch.linalg.eigvalsh(C.cpu())
    eig = torch.clamp(eig, min=0.0)
    return torch.flip(eig, dims=[0]).to(dtype=torch.float32)


def topk_indices(scores: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
    k = max(1, int(frac * scores.numel()))
    return torch.topk(scores, k).indices


def mass_on_indices(scores: torch.Tensor, indices: torch.Tensor) -> float:
    return (scores[indices].sum() / (scores.sum() + 1e-12)).item()


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Rank correlation without scipy.

    Optimisations vs the original:
    - Rank computation is done entirely on the MPS device with
      torch.argsort(torch.argsort(...)) before the final corrcoef,
      avoiding two round-trips to CPU numpy for the sorting step.
    - Only the scalar corrcoef call drops to numpy (it's a 2×N dot
      product — negligible cost, and numpy's corrcoef is fine here).
    """
    # Keep on device for the argsort passes
    a_t = a.detach().float()
    b_t = b.detach().float()

    ra = torch.argsort(torch.argsort(a_t)).float()
    rb = torch.argsort(torch.argsort(b_t)).float()

    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0

    # corrcoef is cheap; move once after the heavy sort work is done
    ra_np = ra.cpu().numpy()
    rb_np = rb.cpu().numpy()
    return float(np.corrcoef(ra_np, rb_np)[0, 1])

def build_trainable_masks(model, scores, frac=0.1, reduction="sum"):
    """
    Build boolean masks for each parameter tensor.

    True  -> keep active
    False -> zero after compare_epoch

    For the 2D Van der Pol case, `scores` may contain one block per output
    component, so we first collapse it to one score per scalar parameter.
    """
    scores = scores.detach().flatten()

    param_count = sum(p.numel() for p in model.parameters())

    # Case 1: already one score per parameter
    if scores.numel() == param_count:
        param_scores = scores

    # Case 2: vector-output sensitivity, e.g. 2D Van der Pol
    elif scores.numel() % param_count == 0:
        out_dim = scores.numel() // param_count
        scores = scores.view(out_dim, param_count)

        if reduction == "sum":
            param_scores = scores.sum(dim=0)
        elif reduction == "mean":
            param_scores = scores.mean(dim=0)
        else:
            raise ValueError("reduction must be 'sum' or 'mean'")

    else:
        raise RuntimeError(
            f"Cannot align scores of length {scores.numel()} with parameter count {param_count}."
        )

    keep_idx = topk_indices(param_scores, frac=frac)
    global_mask = torch.zeros_like(param_scores, dtype=torch.bool)
    global_mask[keep_idx] = True

    masks = []
    offset = 0
    for p in model.parameters():
        n = p.numel()
        masks.append(global_mask[offset: offset + n].view_as(p).detach().clone())
        offset += n

    if offset != param_count:
        raise RuntimeError(
            f"Mask construction failed: consumed {offset} parameters, expected {param_count}."
        )

    return masks


def apply_masks_to_weights_and_state(model, optimizer, masks):
    """
    Clamp low-sensitivity parameters to zero and clear their Adam state.
    This prevents post-compare updates and momentum carry-over.
    """
    for p, mask in zip(model.parameters(), masks):
        p.data.mul_(mask)

        if p.grad is not None:
            p.grad.mul_(mask)

        state = optimizer.state.get(p, None)
        if not state:
            continue

        if "exp_avg" in state:
            state["exp_avg"].mul_(mask)
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask)

def apply_masks_to_state(model, optimizer, masks):
    """
    Clear the Adam state of low-sensitivity parameters.
    This prevents post-compare updates and momentum carry-over.
    """
    for p, mask in zip(model.parameters(), masks):

        if p.grad is not None:
            p.grad.mul_(mask)

        state = optimizer.state.get(p, None)
        if not state:
            continue

        if "exp_avg" in state:
            state["exp_avg"].mul_(mask)
        if "exp_avg_sq" in state:
            state["exp_avg_sq"].mul_(mask)

def effective_rank(eigs, eps=1e-30):
    x = eigs.detach().float().cpu().numpy()
    x = np.clip(x, eps, None)
    p = x / x.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def reduce_jacobian_to_parameter_vector(J: torch.Tensor, mode: str = "abs") -> torch.Tensor:
    """
    Reduce a Jacobian/sensitivity tensor to one scalar per parameter.

    Expected J shapes:
      - [n_samples, n_params]
      - [n_samples, n_outputs, n_params]

    Returns:
      - [n_params]
    """
    J = J.detach()

    if J.ndim == 2:
        if mode == "abs":
            return J.abs().mean(dim=0)
        elif mode == "signed":
            return J.mean(dim=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if J.ndim == 3:
        if mode == "abs":
            return J.abs().mean(dim=(0, 1))
        elif mode == "signed":
            return J.mean(dim=(0, 1))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    raise ValueError(f"Unsupported Jacobian shape: {tuple(J.shape)}")

def flatten_param_magnitudes(mod: nn.Module):
    """Flatten all trainable parameters into one 1D vector of absolute values."""
    chunks = []
    for p in mod.parameters():
        if p.requires_grad:
            chunks.append(p.detach().abs().reshape(-1))
    return torch.cat(chunks, dim=0)


def reduce_sensitivity_to_parameter_level(mod: nn.Module, sens: torch.Tensor):
    """
    Reduce sensitivities to one value per scalar parameter.

    This handles the common case where sensitivity_scores(J) returns
    one sensitivity per parameter per output dimension, e.g. length = 2 * n_params
    for a 2D output field.
    """
    sens_flat = sens.detach().reshape(-1)

    param_count = sum(p.numel() for p in mod.parameters() if p.requires_grad)

    # Already aligned.
    if sens_flat.numel() == param_count:
        return sens_flat

    # Common case: output_dim x n_params or n_params x output_dim flattened.
    if sens_flat.numel() % param_count == 0:
        out_dim = sens_flat.numel() // param_count
        sens_reduced = sens_flat.view(out_dim, param_count).norm(dim=0)
        return sens_reduced

    raise ValueError(
        f"Cannot reduce sensitivity vector of length {sens_flat.numel()} "
        f"to parameter count {param_count}."
    )

def prettify_axes(ax, *, grid: bool = False):
    ax.tick_params(which="major", direction="in", top=True, right=True, length=5, width=0.8)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    if grid:
        ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def beautify_legend(ax, **kwargs):
    defaults = dict(frameon=True, framealpha=0.92, edgecolor="0.8")
    defaults.update(kwargs)
    leg = ax.legend(**defaults)
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)
    return leg


def save_pub_figure(fig, path):
    fig.savefig(path)
    plt.close(fig)


def reduce_jacobian_to_output_parameter_matrix(J):
    """
    Return per-output mean absolute sensitivities over the sample axis.

    Expected common layouts:
      J.shape == [n_samples, n_outputs, n_params]
      J.shape == [n_samples, n_outputs, ...parameter_tensor_shape...]
    """
    J_abs = J.detach().abs()

    if J_abs.ndim == 3:
        # [n_samples, n_outputs, n_params]
        return J_abs.mean(dim=0)

    if J_abs.ndim >= 4:
        # [n_samples, n_outputs, ...parameter_tensor_shape...]
        return J_abs.mean(dim=0).flatten(start_dim=1)

    raise ValueError(f"Unexpected Jacobian shape: {tuple(J_abs.shape)}")

def mean_abs_sensitivity_by_output(model, x):
    """
    Returns a tensor of shape [n_outputs, n_params] containing
    mean_x |d f_k / d theta_i| for each output k and parameter i.
    """
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    param_names = list(params.keys())

    def model_with_params(params_, x_):
        return torch.func.functional_call(model, (params_, buffers), (x_,))

    def per_sample(xi):
        def f(params_):
            # Shape: [1, n_outputs] -> [n_outputs]
            return model_with_params(params_, xi.unsqueeze(0)).squeeze(0)

        # jac[name] has shape [n_outputs, *param.shape]
        jac = torch.func.jacrev(f)(params)

        per_output = []
        n_outputs = next(iter(jac.values())).shape[0]
        for k in range(n_outputs):
            chunks = [jac[name][k].reshape(-1).abs() for name in param_names]
            per_output.append(torch.cat(chunks))
        return torch.stack(per_output, dim=0)  # [n_outputs, n_params]

    return torch.vmap(per_sample)(x).mean(dim=0)

def mean_abs_sensitivity_from_jacobian(J: torch.Tensor) -> torch.Tensor:
    """
    Compute the same quantity used by mean_abs_sensitivity_by_output(model, x_sens),
    but reuse an already computed Jacobian tensor.

    Expected common layout:
        J.shape == [n_samples, n_outputs, n_params]
    Returns:
        [n_outputs, n_params]
    """
    J = J.detach()
    if J.ndim == 3:
        return J.abs().mean(dim=0)
    if J.ndim == 2:
        # Fallback for degenerate/single-output layouts.
        return J.abs().mean(dim=0, keepdim=True)
    raise ValueError(f"Unexpected Jacobian shape: {tuple(J.shape)}")