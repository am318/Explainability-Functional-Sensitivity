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
        out[name] = S_flat[offset:offset + n].view_as(p).detach().clone()
        offset += n
    if offset != S_flat.numel():
        raise ValueError(f"Sensitivity length mismatch: consumed {offset}, got {S_flat.numel()}")
    return out

def _register_mask_hook(param, mask):
    mask = mask.to(device=param.device, dtype=param.dtype)

    def hook(grad):
        return grad * mask

    return param.register_hook(hook)

def prune_sensitive_subnetwork_mlp(model, S_param_flat, threshold, mode="freeze"):
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
        Keep parameters with sensitivity >= threshold, unless needed for connectivity.
    mode : str
        'freeze' -> keep values, but mask gradients
        'zero'    -> zero masked parameters and mask gradients

    Returns
    -------
    param_masks : dict[str, torch.Tensor]
        Boolean masks for each parameter tensor.
    unit_masks : list[torch.Tensor]
        Boolean masks for neuron/unit retention at each linear layer boundary.
    hooks : list
        Gradient hooks; keep this list alive if mode='freeze' or 'zero' and training continues.
    """
    if mode not in {"freeze", "zero"}:
        raise ValueError("mode must be 'freeze' or 'zero'")

    S_flat = flatten_sensitivity(S_param_flat)
    sens = unflatten_param_sensitivity(model, S_flat)

    modules = list(model.net)
    linear_positions = [(i, m) for i, m in enumerate(modules) if isinstance(m, nn.Linear)]
    layernorm_positions = [(i, m) for i, m in enumerate(modules) if isinstance(m, nn.LayerNorm)]

    if not linear_positions:
        raise ValueError("No Linear layers found in model.net")

    # Unit masks are defined on the neuron sets at layer boundaries:
    # unit_masks[0] = input features
    # unit_masks[k] = output neurons of linear layer k-1
    n_linear = len(linear_positions)
    device = next(model.parameters()).device

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

        # Seed by sensitivity of this layer's output units.
        # A unit is "sensitive" if its incoming weights or bias are above threshold.
        neuron_sens = W_sens.abs().sum(dim=1) + b_sens.abs()
        out_keep = neuron_sens >= threshold

        # Ensure at least one unit survives if the whole layer is below threshold.
        if not torch.any(out_keep):
            out_keep[torch.argmax(neuron_sens)] = True

        in_keep = torch.zeros(lin.in_features, dtype=torch.bool, device=device)
        required = torch.zeros_like(W_sens, dtype=torch.bool)

        # For each retained output neuron, preserve upstream information flow.
        for j in torch.nonzero(out_keep, as_tuple=False).flatten().tolist():
            parent_scores = W_sens[j].abs()
            parent_keep = parent_scores >= threshold

            if torch.any(parent_keep):
                # Normal case: retain all sensitive parents.
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
        w_name = f"net.{module_idx}.weight"
        b_name = f"net.{module_idx}.bias"

        out_mask = unit_masks[
            [idx for idx, (pos, _) in enumerate(linear_positions) if pos == module_idx][0] + 1
        ]
        in_mask = unit_masks[
            [idx for idx, (pos, _) in enumerate(linear_positions) if pos == module_idx][0]
        ]

        base_w_mask = (sens[w_name].detach() >= threshold)
        w_mask = (base_w_mask & out_mask[:, None] & in_mask[None, :]) | edge_required[w_name]

        param_masks[w_name] = w_mask
        if b_name in sens:
            param_masks[b_name] = out_mask.clone()

    # LayerNorm masks follow the hidden unit masks.
    for module_idx, ln in layernorm_positions:
        w_name = f"net.{module_idx}.weight"
        b_name = f"net.{module_idx}.bias"

        # Find the linear layer immediately preceding this LayerNorm.
        prev_linear_idx = None
        for li in range(len(linear_positions) - 1):
            if linear_positions[li][0] < module_idx < linear_positions[li + 1][0]:
                prev_linear_idx = li
                break
        if prev_linear_idx is None:
            # First LayerNorm should correspond to the first linear layer.
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

            # Always mask gradients so pruned entries stay pruned during further training.
            hooks.append(_register_mask_hook(p, mask))

    return param_masks, unit_masks, hooks





# USAGE!

# # After you compute S_final
# S_param = reduce_sensitivity_to_parameter_level(model, S_final)
# S_param = flatten_sensitivity(S_param)

# threshold = 1e-4
# param_masks, unit_masks, hooks = prune_sensitive_subnetwork_mlp(
#     model,
#     S_param,
#     threshold=threshold,
#     mode="freeze",   # or "zero"
# )