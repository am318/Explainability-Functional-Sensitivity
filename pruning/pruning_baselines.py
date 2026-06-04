
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

def build_synflow_masks(model, target_sparsity, input_shape):
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
