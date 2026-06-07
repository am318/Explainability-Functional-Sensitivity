import torch
import torch.nn as nn
from typing import Dict, Iterable, List, Optional, Tuple
import math


def min_keep_count(total: int, min_keep_fraction: float) -> int:
    return max(1, min(total, int(math.ceil(float(min_keep_fraction) * total))))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_prunable_parameter(name: str, p: torch.nn.Parameter, cfg) -> bool:
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
    return {
        name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def _make_probe(logits: torch.Tensor) -> torch.Tensor:
    probe = torch.empty_like(logits).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    probe = probe / math.sqrt(max(1, logits.shape[-1]))
    return probe


def flatten_like_model(model: nn.Module, tensors: Dict[str, torch.Tensor], only_active: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
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


def flatten_param_magnitudes(model: nn.Module, masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    parts = []
    for name, p in model.named_parameters():
        t = p.detach().abs().cpu().reshape(-1)
        if masks is not None:
            m = masks[name].detach().cpu().bool().reshape(-1)
            t = t[m]
        parts.append(t.float())
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
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
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    k = max(1, int(math.ceil(float(frac) * values.numel())))
    k = min(k, values.numel())
    return torch.topk(values, k=k, largest=True, sorted=False).indices.cpu()


def mass_on_indices(values: torch.Tensor, indices: torch.Tensor) -> float:
    values = values.detach().float().cpu().reshape(-1).clamp_min(0)
    if values.numel() == 0 or indices.numel() == 0:
        return float("nan")
    denom = float(values.sum().item())
    if denom <= 0.0:
        return float("nan")
    idx = indices.to(dtype=torch.long).clamp(0, values.numel() - 1)
    return float(values[idx].sum().item() / denom)


def effective_rank(eigvals: torch.Tensor) -> float:
    vals = eigvals.detach().float().cpu().clamp_min(0)
    total = vals.sum()
    if float(total) <= 0.0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def compute_sensitivity_scores(
    model: nn.Module,
    loader,
    cfg,
    device: torch.device,
    probes: Optional[int] = None,
    collect_probe_matrix: bool = False,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Hutchinson/Rademacher estimator for parameter sensitivity.

    If collect_probe_matrix is true, this also stores a bounded matrix of active
    probe gradients. Its Gram spectrum is used as a scalable analogue of the
    sweep script's Jacobian covariance eigenspectrum.
    """
    model.eval()
    n_probes = int(probes if probes is not None else cfg.sensitivity_probes)
    scores = init_score_buffers(model)
    n_accum = 0
    probe_rows: List[torch.Tensor] = []
    if masks is not None:
        active_count = int(sum(int(m.detach().cpu().bool().sum().item()) for m in masks.values()))
    else:
        active_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    can_collect = collect_probe_matrix
    if can_collect and active_count * cfg.analysis_probe_matrix_rows > cfg.max_probe_matrix_elements:
        can_collect = False

    for images, _targets in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        for _ in range(n_probes):
            model.zero_grad(set_to_none=True)
            logits = model(images)
            probe = _make_probe(logits)
            scalar = (logits * probe).sum() / batch_size
            scalar.backward()
            with torch.no_grad():
                grad_parts = []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    scores[name].add_(g.cpu().pow(2), alpha=batch_size)
                    if can_collect and len(probe_rows) < cfg.analysis_probe_matrix_rows:
                        if masks is None:
                            grad_parts.append(g.detach().cpu().reshape(-1))
                        else:
                            m = masks[name].to(device=p.device, dtype=torch.bool)
                            grad_parts.append(g[m].detach().cpu().reshape(-1))
                if can_collect and grad_parts and len(probe_rows) < cfg.analysis_probe_matrix_rows:
                    probe_rows.append(torch.cat(grad_parts))
            n_accum += batch_size

    if n_accum == 0:
        raise RuntimeError("No samples were available for sensitivity scoring.")
    for name in scores:
        scores[name].div_(n_accum)
    probe_matrix = torch.stack(probe_rows, dim=0) if probe_rows else None
    return scores, probe_matrix


def _restore_topk_in_vector(mask_vec: torch.Tensor, score_vec: torch.Tensor, k: int) -> int:
    if mask_vec.numel() == 0 or int(mask_vec.sum().item()) >= k:
        return 0
    k = min(k, mask_vec.numel())
    restore_idx = torch.topk(score_vec.float(), k=k, largest=True, sorted=False).indices
    before = int(mask_vec.sum().item())
    mask_vec[restore_idx] = True
    return int(mask_vec.sum().item()) - before


def connectivity_close_dense_weight(mask: torch.Tensor, scores: torch.Tensor, min_conn: int) -> int:
    """Restore incident edges so dense/conv weights have no empty row/column groups."""
    restored = 0
    if mask.ndim == 2:
        for row in range(mask.shape[0]):
            restored += _restore_topk_in_vector(mask[row, :], scores[row, :], min_conn)
        for col in range(mask.shape[1]):
            restored += _restore_topk_in_vector(mask[:, col], scores[:, col], min_conn)
    elif mask.ndim == 4:
        # Conv2d: flatten spatial kernels when checking output/input channels.
        out_ch, in_ch = mask.shape[:2]
        flat_mask_out = mask.reshape(out_ch, -1)
        flat_scores_out = scores.reshape(out_ch, -1)
        for row in range(out_ch):
            restored += _restore_topk_in_vector(flat_mask_out[row], flat_scores_out[row], min_conn)
        flat_mask_in = mask.permute(1, 0, 2, 3).reshape(in_ch, -1)
        flat_scores_in = scores.permute(1, 0, 2, 3).reshape(in_ch, -1)
        for col in range(in_ch):
            restored += _restore_topk_in_vector(flat_mask_in[col], flat_scores_in[col], min_conn)
    return restored


def make_threshold_connectivity_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    masks: Dict[str, torch.Tensor] = {}
    eligible_total = 0
    raw_retained = 0
    restored_total = 0

    for name, p in model.named_parameters():
        score = scores[name].to(device=p.device, dtype=torch.float32)
        if is_prunable_parameter(name, p, cfg):
            eligible_total += p.numel()
            mask = score >= float(cfg.prune_threshold)
            floor_keep = 1
            if p.ndim == 4 and "patch_embed" in name:
                floor_keep = min_keep_count(p.numel(), cfg.min_embed_keep_fraction)
            elif p.ndim == 2 and ("mlp.fc1" in name or "mlp.fc2" in name or "attn" in name):
                floor_keep = min_keep_count(p.numel(), cfg.min_hidden_keep_fraction)
            if int(mask.sum().item()) < floor_keep:
                top_idx = torch.topk(score.reshape(-1), k=floor_keep, largest=True, sorted=False).indices
                mask = torch.zeros_like(score, dtype=torch.bool).reshape(-1)
                mask[top_idx] = True
                mask = mask.view_as(score)
            if cfg.connectivity_closure and p.ndim in (2, 4):
                restored_total += connectivity_close_dense_weight(mask, score, max(1, cfg.min_connections_per_unit))
            # Never allow a prunable tensor to become completely empty.
            if int(mask.sum().item()) == 0:
                idx = torch.argmax(score.reshape(-1))
                mask.reshape(-1)[idx] = True
                restored_total += 1
            raw_retained += int(mask.sum().item())
        else:
            mask = torch.ones_like(p, dtype=torch.bool, device=p.device)
        masks[name] = mask

    total_trainable = trainable_parameter_count(model)
    retained_eligible = sum(
        int(masks[name].sum().item())
        for name, p in model.named_parameters()
        if is_prunable_parameter(name, p, cfg)
    )
    pruned_eligible = max(0, eligible_total - retained_eligible)
    stats = {
        "eligible_parameter_count": float(eligible_total),
        "threshold": float(cfg.prune_threshold),
        "retained_eligible_parameter_count": float(retained_eligible),
        "pruned_eligible_parameter_count": float(pruned_eligible),
        "actual_prune_fraction_eligible": float(pruned_eligible / max(1, eligible_total)),
        "actual_prune_fraction_all_trainable": float(pruned_eligible / max(1, total_trainable)),
        "connectivity_restored_coordinate_count": float(restored_total),
        "connectivity_closure": bool(cfg.connectivity_closure),
        "min_connections_per_unit": int(cfg.min_connections_per_unit),
    }
    return masks, stats


def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.mul_(masks[name].to(device=p.device, dtype=p.dtype))



def zero_masked_optimizer_state_(optimizer: torch.optim.Optimizer, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        state = optimizer.state.get(p)
        if not state:
            continue
        mask = masks[name].to(device=p.device, dtype=p.dtype)
        for value in state.values():
            if torch.is_tensor(value) and value.shape == p.shape:
                value.mul_(mask)