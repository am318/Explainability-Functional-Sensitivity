import torch
import torch.nn as nn


def flatten_sensitivity(S):
    S = torch.as_tensor(S)
    if S.ndim > 1:
        S = S.sum(dim=0)
    return S.reshape(-1)


def unflatten_param_sensitivity(model, S_flat):
    """Map a flat parameter-sensitivity vector back to each named parameter."""
    out = {}
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        out[name] = S_flat[offset : offset + n].view_as(p).detach().clone()
        offset += n
    if offset != S_flat.numel():
        raise ValueError(f"Sensitivity length mismatch: consumed {offset}, got {S_flat.numel()}")
    return out


def _register_mask_hook(param, mask):
    mask = mask.to(device=param.device, dtype=param.dtype)

    def hook(grad):
        return grad * mask

    return param.register_hook(hook)


def _threshold_sensitivity(sens, threshold, threshold_mode):
    """Return an effective sensitivity tensor used for thresholding."""
    threshold_mode = threshold_mode.lower()
    sens = sens.detach()

    if threshold_mode == "raw":
        return sens

    if threshold_mode == "max_normalized":
        max_val = sens.max()
        denom = max_val.clamp_min(torch.finfo(sens.dtype).eps)
        return sens / denom

    if threshold_mode == "quantile":
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError("For threshold_mode='quantile', threshold must be in [0, 1].")
        q = torch.quantile(sens, float(threshold))
        return sens, q

    raise ValueError("threshold_mode must be one of {'raw', 'max_normalized', 'quantile'}")


def prune_sensitive_subnetwork_mlp(
    model,
    S_param_flat,
    threshold,
    mode="freeze",
    threshold_mode="raw",
):
    """
    Prune a SmallMLP-style Sequential:
        Linear -> LayerNorm -> SiLU -> ... -> Linear

    Parameters
    ----------
    model : nn.Module
        Model to prune in-place.
    S_param_flat : torch.Tensor or array-like
        Flat parameter sensitivity scores, aligned with model.named_parameters().
    threshold : float
        Threshold used according to `threshold_mode`.

        - raw: keep values with sensitivity >= threshold
        - max_normalized: keep values with sensitivity/max(sensitivity) >= threshold
        - quantile: keep values with sensitivity >= quantile(sensitivity, threshold)
          in which case `threshold` must lie in [0, 1]
    mode : str
        'freeze' -> keep values, but mask gradients
        'zero'   -> zero masked parameters and mask gradients
    threshold_mode : str
        'raw', 'max_normalized', or 'quantile'

    Returns
    -------
    param_masks : dict[str, torch.Tensor]
        Boolean masks for each parameter tensor.
    unit_masks : list[torch.Tensor]
        Boolean masks for neuron/unit retention at each linear layer boundary.
    hooks : list
        Gradient hooks; keep this list alive if training continues.
    """
    if mode not in {"freeze", "zero"}:
        raise ValueError("mode must be 'freeze' or 'zero'")

    threshold_mode = threshold_mode.lower()
    if threshold_mode not in {"raw", "max_normalized", "quantile"}:
        raise ValueError("threshold_mode must be one of {'raw', 'max_normalized', 'quantile'}")

    S_flat = flatten_sensitivity(S_param_flat)
    sens = unflatten_param_sensitivity(model, S_flat)

    modules = list(model.net)
    linear_positions = [(i, m) for i, m in enumerate(modules) if isinstance(m, nn.Linear)]
    layernorm_positions = [(i, m) for i, m in enumerate(modules) if isinstance(m, nn.LayerNorm)]

    if not linear_positions:
        raise ValueError("No Linear layers found in model.net")

    n_linear = len(linear_positions)
    device = next(model.parameters()).device

    # Helper lookup: module index -> linear layer ordinal.
    linear_ord = {module_idx: ord_idx for ord_idx, (module_idx, _) in enumerate(linear_positions)}

    unit_masks = [None] * (n_linear + 1)
    unit_masks[-1] = torch.ones(
        linear_positions[-1][1].out_features,
        dtype=torch.bool,
        device=device,
    )

    edge_required = {}  # key: linear module name -> bool matrix [out, in]

    # Backward closure over the linear stack.
    for layer_idx in reversed(range(n_linear)):
        module_idx, lin = linear_positions[layer_idx]
        w_name = f"net.{module_idx}.weight"
        b_name = f"net.{module_idx}.bias"

        W_sens = sens[w_name].detach()
        b_sens = sens[b_name].detach() if b_name in sens else torch.zeros(lin.out_features, device=device)

        # Neuron sensitivity criterion.
        if threshold_mode == "quantile":
            _, thr = _threshold_sensitivity(W_sens.abs().sum(dim=1) + b_sens.abs(), threshold, threshold_mode)
            neuron_keep_metric = W_sens.abs().sum(dim=1) + b_sens.abs()
            out_keep = neuron_keep_metric >= thr
        else:
            neuron_keep_metric = W_sens.abs().sum(dim=1) + b_sens.abs()
            effective_neuron_metric = _threshold_sensitivity(neuron_keep_metric, threshold, threshold_mode)
            out_keep = effective_neuron_metric >= threshold

        # Ensure at least one unit survives if the whole layer is below threshold.
        if not torch.any(out_keep):
            out_keep[torch.argmax(neuron_keep_metric)] = True

        in_keep = torch.zeros(lin.in_features, dtype=torch.bool, device=device)
        required = torch.zeros_like(W_sens, dtype=torch.bool)

        # For each retained output neuron, preserve upstream information flow.
        for j in torch.nonzero(out_keep, as_tuple=False).flatten().tolist():
            parent_scores = W_sens[j].abs()

            if threshold_mode == "quantile":
                parent_thr = torch.quantile(parent_scores, float(threshold))
                parent_keep = parent_scores >= parent_thr
            else:
                effective_parent_scores = _threshold_sensitivity(parent_scores, threshold, threshold_mode)
                parent_keep = effective_parent_scores >= threshold

            if torch.any(parent_keep):
                in_keep |= parent_keep
                required[j, parent_keep] = True
            else:
                # Connectivity fallback: retain the single most sensitive incoming edge.
                i_star = torch.argmax(parent_scores).item()
                in_keep[i_star] = True
                required[j, i_star] = True

        unit_masks[layer_idx] = in_keep
        edge_required[w_name] = required

    # Assemble parameter masks.
    param_masks = {}

    for module_idx, lin in linear_positions:
        ord_idx = linear_ord[module_idx]
        w_name = f"net.{module_idx}.weight"
        b_name = f"net.{module_idx}.bias"

        out_mask = unit_masks[ord_idx + 1]
        in_mask = unit_masks[ord_idx]

        W_sens = sens[w_name].detach()
        if threshold_mode == "quantile":
            base_thresh = torch.quantile(W_sens.abs(), float(threshold))
            base_w_mask = W_sens.abs() >= base_thresh
        else:
            effective_w_scores = _threshold_sensitivity(W_sens.abs(), threshold, threshold_mode)
            base_w_mask = effective_w_scores >= threshold

        w_mask = (base_w_mask & out_mask[:, None] & in_mask[None, :]) | edge_required[w_name]
        param_masks[w_name] = w_mask

        if b_name in sens:
            param_masks[b_name] = out_mask.clone()

    # LayerNorm masks follow the hidden unit masks.
    for module_idx, ln in layernorm_positions:
        w_name = f"net.{module_idx}.weight"
        b_name = f"net.{module_idx}.bias"

        prev_linear_idx = None
        for li in range(len(linear_positions) - 1):
            if linear_positions[li][0] < module_idx < linear_positions[li + 1][0]:
                prev_linear_idx = li
                break
        if prev_linear_idx is None:
            prev_linear_idx = 0

        hidden_mask = unit_masks[prev_linear_idx + 1]
        param_masks[w_name] = hidden_mask.clone()
        param_masks[b_name] = hidden_mask.clone()

    hooks = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            mask = param_masks.get(name, None)
            if mask is None:
                continue

            if mode == "zero":
                p.mul_(mask.to(dtype=p.dtype, device=p.device))

            hooks.append(_register_mask_hook(p, mask))

    return param_masks, unit_masks, hooks