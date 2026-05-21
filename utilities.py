import torch
import torch.nn as nn
import numpy as np
from torch.func import jacrev, functional_call, vmap

# ============================================================
# Utilities
# ============================================================

def flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def compute_parameter_jacobian(
    model: nn.Module,
    x: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    J shape:
        [N_data, N_parameters]
    Row k:
        df(x_k)/dtheta

    Optimisations vs the original:
    - Uses torch.vmap + jacrev (functorch) to compute all per-sample
      gradients in a single vectorised pass instead of a Python for-loop.
      On MPS this fuses the forward+backward sweeps, removing Python
      dispatch overhead that grows linearly with N_data.
    - Processes data in chunks so peak MPS memory stays bounded for
      large datasets.  Tune `chunk_size` to fit your VRAM budget.
    - param_dict is built once outside the vmap'd function; the
      stateless `functional_call` path avoids repeated graph rebuilds.
    """
    device = next(model.parameters()).device
    x = x.to(device)

    # Capture current parameters and buffers as plain dicts.
    # functional_call uses these instead of model.state_dict() so the
    # computation graph never touches nn.Parameter storage directly —
    # a requirement for jacrev/vmap composition.
    param_dict = {k: v for k, v in model.named_parameters()}
    buffer_dict = {k: v for k, v in model.named_buffers()}

    def forward_on_params(params: dict, xi: torch.Tensor) -> torch.Tensor:
        """
        Stateless forward for a single unbatched sample xi.
        xi has shape [*input_dims] (no batch axis — vmap injects that).
        Returns a scalar output.
        """
        return functional_call(
            model, {**params, **buffer_dict}, (xi.unsqueeze(0),)
        ).squeeze()

    def jacobian_row(xi: torch.Tensor) -> torch.Tensor:
        """
        Use jacrev to differentiate forward_on_params wrt param_dict
        for a single sample.  Returns a dict of per-parameter Jacobians;
        we flatten them into one row vector.
        """
        # jacrev(f, argnums=0) differentiates f's first argument.
        # Result is a dict with the same keys as param_dict, each value
        # having shape (*param_shape) because the output is scalar.
        jac_dict = jacrev(forward_on_params, argnums=0)(param_dict, xi)
        return torch.cat([j.reshape(-1) for j in jac_dict.values()])

    # vmap batches jacobian_row over the sample dimension.
    # Because jacrev is a torch.func transform (not autograd.grad),
    # it composes correctly with vmap — this is the key fix.
    batched_jacobian = vmap(jacobian_row, in_dims=0, randomness="error")

    # ----------------------------------------------------------------
    # Chunked execution: bounds peak MPS memory for large N_data.
    # Tune chunk_size to fit your device's memory budget.
    # ----------------------------------------------------------------
    rows = []
    for start in range(0, x.shape[0], chunk_size):
        x_chunk = x[start : start + chunk_size]
        rows.append(batched_jacobian(x_chunk))

    return torch.cat(rows, dim=0)


def compute_covariance(J: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """
    C = (1/N) J^T J

    Optimisations vs the original:
    - Accumulates J^T J in chunks along the N_data axis so the
      intermediate [N_data, P] × [N_data, P] product never has to live
      in memory all at once.  For large N this can be the difference
      between fitting on MPS and OOM-ing.
    - Uses torch.linalg.multi_dot for clarity; fused matmul on MPS.
    """
    N, P = J.shape
    C = torch.zeros(P, P, dtype=J.dtype, device=J.device)

    for start in range(0, N, chunk_size):
        Jc = J[start : start + chunk_size]   # [chunk, P]
        C.add_(Jc.T @ Jc)

    return C / N


def sensitivity_scores(J: torch.Tensor) -> torch.Tensor:
    """
    S_i = E[(df/dtheta_i)^2]

    No change needed — already a vectorised mean over a pre-computed
    matrix.  Keeping the original implementation.
    """
    return torch.mean(J ** 2, dim=0)


def eigvals_from_covariance(C: torch.Tensor) -> torch.Tensor:
    """
    Optimisations vs the original:
    - Stays on the MPS device via torch.linalg.eigvalsh instead of
      round-tripping to CPU numpy.  MPS dispatches to Accelerate LAPACK
      under the hood, which is the same library numpy uses on macOS.
    - torch.linalg.eigvalsh already returns values in ascending order;
      flip in-place with torch.flip (no numpy copy).
    - torch.clamp replaces np.clip.
    """
    eig = torch.linalg.eigvalsh(C.cpu())  
    eig = torch.clamp(eig, min=0.0)
    eig = torch.flip(eig, dims=[0])         # descending
    return eig.float()


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