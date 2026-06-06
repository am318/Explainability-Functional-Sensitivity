
import torch
import torch.nn as nn

def _global_mask_from_scores(model, scores, target_sparsity):
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_keep = max(1, int(round((1.0 - target_sparsity) * n_total)))
    flat = torch.cat([scores[n].reshape(-1) for n, _ in model.named_parameters()])
    keep_idx = torch.topk(flat, n_keep).indices
    global_mask = torch.zeros_like(flat, dtype=torch.bool)
    global_mask[keep_idx] = True

    masks = {}
    offset = 0
    for name, p in model.named_parameters():
        n = p.numel()
        masks[name] = global_mask[offset:offset+n].view_as(p)
        offset += n
    return masks

def build_snip_masks(model, x, y, loss_fn, target_sparsity):
    model.zero_grad(set_to_none=True)
    loss_fn(model(x), y).backward()

    scores = {}
    for name, p in model.named_parameters():
        scores[name] = (p.grad * p).abs().detach()
    return _global_mask_from_scores(model, scores, target_sparsity)

def build_grasp_masks(model, x, y, loss_fn, target_sparsity):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                create_graph=True)

    gnorm = sum((g**2).sum() for g in grads)
    hg = torch.autograd.grad(gnorm, [p for p in model.parameters() if p.requires_grad])

    scores = {}
    idx = 0
    for name, p in model.named_parameters():
        scores[name] = (-(p * hg[idx])).detach()
        idx += 1
    return _global_mask_from_scores(model, scores, target_sparsity)

def build_synflow_masks(model, *args, **kwargs):
    """Build SynFlow masks with backward-compatible call signatures.

    Supported call patterns:
      - build_synflow_masks(model, target_sparsity, input_shape)
      - build_synflow_masks(model, x, y, loss_fn, target_sparsity)

    The extra SNIP-style arguments are ignored; they are accepted only so the
    SynFlow helper can be swapped into the same call site without raising a
    TypeError.
    """
    target_sparsity = kwargs.pop("target_sparsity", None)
    input_shape = kwargs.pop("input_shape", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if len(args) == 2:
        if target_sparsity is not None or input_shape is not None:
            raise TypeError("Do not mix positional and keyword target_sparsity/input_shape")
        target_sparsity, input_shape = args
    elif len(args) == 4:
        if target_sparsity is not None:
            raise TypeError("target_sparsity was provided twice")
        x, _y, _loss_fn, target_sparsity = args
        if input_shape is None:
            if not torch.is_tensor(x):
                raise TypeError("When called in SNIP-style form, the first extra argument must be a tensor")
            input_shape = tuple(x[:1].shape)
    else:
        raise TypeError(
            "build_synflow_masks expected either (model, target_sparsity, input_shape) "
            "or (model, x, y, loss_fn, target_sparsity)"
        )

    if input_shape is None:
        raise TypeError("input_shape could not be inferred for SynFlow masking")

    signs = {}
    for name, p in model.named_parameters():
        signs[name] = torch.sign(p.data)
        p.data.abs_()

    x = torch.ones(input_shape, device=next(model.parameters()).device)
    model.zero_grad(set_to_none=True)
    torch.sum(model(x)).backward()

    scores = {}
    for name, p in model.named_parameters():
        scores[name] = (p.grad * p).abs().detach()

    with torch.no_grad():
        for name, p in model.named_parameters():
            p.mul_(signs[name])

    return _global_mask_from_scores(model, scores, target_sparsity)
