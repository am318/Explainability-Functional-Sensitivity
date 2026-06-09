import math
import torch 
from typing import Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import torch.nn as nn
import re
from sensitivity_metrics import *
import json


def _restore_topk_in_vector(mask_vec: torch.Tensor, score_vec: torch.Tensor, k: int) -> int:
    if mask_vec.numel() == 0 or int(mask_vec.sum().item()) >= k:
        return 0
    k = min(k, mask_vec.numel())
    restore_idx = torch.topk(score_vec.float(), k=k, largest=True, sorted=False).indices
    before = int(mask_vec.sum().item())
    mask_vec[restore_idx] = True
    return int(mask_vec.sum().item()) - before

def _keep_count(total: int, prune_fraction: float) -> int:
    return max(1, min(total, int(math.ceil((1.0 - prune_fraction) * total))))

def _iterative_prune_fraction(cfg, round_idx: int) -> float:
    """Return the prune fraction to use on a given iterative round."""
    target = float(cfg.prune_fraction)
    if not cfg.gradual_sparsification:
        return target
    total_rounds = max(1, int(cfg.iterative_pruning_rounds))
    # Linearly ramp from a mild first-round prune to the requested final sparsity.
    return target * float(round_idx + 1) / float(total_rounds)


def _cfg_value(cfg, name: str, default):
    return getattr(cfg, name, default)


def _block_index_from_name(name: str) -> Optional[int]:
    match = re.search(r"blocks\.(\d+)", name)
    return int(match.group(1)) if match else None


def _vector_similarity(current: torch.Tensor, previous: torch.Tensor) -> Optional[float]:
    """Return a correlation-style similarity in [-1, 1] for two score vectors."""
    current = current.detach().float().reshape(-1)
    previous = previous.detach().float().reshape(-1)
    if current.numel() == 0 or previous.numel() == 0 or current.numel() != previous.numel():
        return None
    if current.numel() == 1:
        return float(1.0 if torch.isclose(current, previous).item() else 0.0)

    current = current - current.mean()
    previous = previous - previous.mean()
    denom = current.norm() * previous.norm()
    if denom <= 0:
        return None
    sim = torch.dot(current, previous) / denom
    return float(torch.clamp(sim, -1.0, 1.0).item())


def _score_confidence_weight(scores: torch.Tensor) -> float:
    """Downweight vectors whose internal spread is very noisy."""
    scores = scores.detach().float().reshape(-1)
    if scores.numel() <= 1:
        return 1.0
    mean_abs = scores.abs().mean().clamp_min(1e-12)
    spread = scores.std(unbiased=False) / mean_abs
    weight = float((1.0 / (1.0 + spread)).item())
    return max(0.25, min(1.0, weight))

def _as_confidence_scalar(value, default: float = 1.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    try:
        return float(value)
    except Exception:
        return default


def _expand_head_mask(head_sel: torch.Tensor, embed_dim: int, num_heads: int) -> torch.Tensor:
    if head_sel.numel() == 0:
        return torch.ones(embed_dim, dtype=torch.bool)
    if num_heads <= 0:
        return torch.ones(embed_dim, dtype=torch.bool)
    head_dim = embed_dim // num_heads
    if head_dim > 0 and embed_dim % num_heads == 0:
        return head_sel.repeat_interleave(head_dim)[:embed_dim].clone()
    repeat = max(1, math.ceil(embed_dim / max(1, head_sel.numel())))
    return head_sel.repeat_interleave(repeat)[:embed_dim].clone()


def _allocate_keep_budget(
    importance: List[float],
    sizes: List[int],
    total_keep: int,
    floors: List[int],
) -> List[int]:
    """Allocate a global keep budget across blocks while honoring local floors."""
    if not sizes:
        return []

    total_keep = max(0, min(total_keep, sum(sizes)))
    floors = [max(0, min(int(f), int(s))) for f, s in zip(floors, sizes)]

    if sum(floors) > total_keep:
        # If the budget is too small to honor every floor, keep one unit in the most
        # important blocks until the budget is exhausted.
        order = sorted(range(len(sizes)), key=lambda i: float(importance[i]), reverse=True)
        keep = [0 for _ in sizes]
        for idx in order[:total_keep]:
            keep[idx] = 1
        return keep

    keep = floors[:]
    remaining = total_keep - sum(keep)
    if remaining <= 0:
        return keep

    weights = torch.tensor([max(0.0, float(x)) for x in importance], dtype=torch.float32)
    if float(weights.sum().item()) <= 0:
        weights = torch.tensor([float(s) for s in sizes], dtype=torch.float32)
    if float(weights.sum().item()) <= 0:
        weights = torch.ones(len(sizes), dtype=torch.float32)

    capacity = torch.tensor([max(0, s - k) for s, k in zip(sizes, keep)], dtype=torch.int64)
    if int(capacity.sum().item()) <= 0:
        return keep

    raw = weights / weights.sum() * float(remaining)
    add = torch.floor(raw).to(torch.int64)
    add = torch.minimum(add, capacity)
    keep = [int(k + a) for k, a in zip(keep, add.tolist())]
    leftover = total_keep - sum(keep)

    if leftover <= 0:
        return keep

    fractional = (raw - add.float()).tolist()
    order = sorted(range(len(sizes)), key=lambda i: (fractional[i], float(weights[i])), reverse=True)
    while leftover > 0:
        progressed = False
        for idx in order:
            if keep[idx] < sizes[idx]:
                keep[idx] += 1
                leftover -= 1
                progressed = True
                if leftover <= 0:
                    break
        if not progressed:
            break
    return keep


def _normalize_unit_scores(unit_scores: torch.Tensor, cfg) -> torch.Tensor:
    """Normalize scores within a tensor while preserving sensitivity ordering.

    The default mode is robust min-max scaling after quantile clipping. This keeps
    the metric sensitivity-based, but prevents a single tensor or a single probe
    from dominating the final ranking.
    """
    unit_scores = unit_scores.detach().float().cpu().reshape(-1)
    if unit_scores.numel() == 0:
        return unit_scores
    if not cfg.layerwise_normalize_scores:
        return unit_scores

    mode = cfg.sensitivity_normalization.lower()
    eps = 1e-12

    if mode == "none":
        return unit_scores
    if mode == "rank":
        if unit_scores.numel() == 1:
            return torch.ones_like(unit_scores)
        ranks = torch.argsort(torch.argsort(unit_scores)).float()
        return ranks / max(1.0, float(unit_scores.numel() - 1))

    if mode == "zscore":
        center = unit_scores.mean()
        scale = unit_scores.std(unbiased=False).clamp_min(eps)
        norm = (unit_scores - center) / scale
    else:
        # Robust scaling: clip extremes, then rescale to [0, 1].
        q = float(cfg.sensitivity_clip_quantile)
        lo = torch.quantile(unit_scores, q)
        hi = torch.quantile(unit_scores, 1.0 - q)
        clipped = unit_scores.clamp(lo, hi)
        denom = (clipped.max() - clipped.min()).clamp_min(eps)
        norm = (clipped - clipped.min()) / denom
        return norm

    # Keep the z-score mode bounded and monotone for downstream aggregation.
    norm = norm.clamp(min=-6.0, max=6.0)
    norm = norm - norm.min()
    denom = (norm.max() - norm.min()).clamp_min(eps)
    return norm / denom


def _tensor_group_scores(name: str, score: torch.Tensor, p: torch.Tensor, cfg) -> Optional[torch.Tensor]:
    score = score.detach().float().cpu()
    if score.numel() == 0:
        return None

    # Embedding/channel-level tensors.
    if name in {"cls_token", "pos_embed"}:
        return score.abs().mean(dim=tuple(range(score.ndim - 1)))
    if "patch_embed.proj.weight" in name and p.ndim == 4:
        return score.abs().mean(dim=(1, 2, 3))
    if "patch_embed.proj.bias" in name and p.ndim == 1:
        return score.abs()
    if "norm" in name.lower() and p.ndim == 1:
        return score.abs()
    if "attn.in_proj_weight" in name and p.ndim == 2:
        return score.abs().mean(dim=1)
    if "attn.in_proj_bias" in name and p.ndim == 1:
        return score.abs()
    if "attn.out_proj.weight" in name and p.ndim == 2:
        return score.abs().mean(dim=0 if cfg.preserve_attention_heads else 1)
    if "attn.out_proj.bias" in name and p.ndim == 1:
        return score.abs()
    if "mlp.fc2.weight" in name and p.ndim == 2:
        return score.abs().mean(dim=0)
    if "mlp.fc2.bias" in name and p.ndim == 1:
        return score.abs()
    if "head.weight" in name and p.ndim == 2:
        # Select classifier input channels, not class rows.
        return score.abs().mean(dim=0)
    if "head.bias" in name and p.ndim == 1:
        return score.abs()
    return None


def _hidden_unit_scores(name: str, score: torch.Tensor, p: torch.Tensor) -> Optional[torch.Tensor]:
    score = score.detach().float().cpu()
    if "mlp.fc1.weight" in name and p.ndim == 2:
        return score.abs().mean(dim=1)
    if "mlp.fc1.bias" in name and p.ndim == 1:
        return score.abs()
    return None


def _attention_head_scores(name: str, score: torch.Tensor, p: torch.Tensor, cfg) -> Optional[torch.Tensor]:
    """Return a head-level score vector for MHSA parameters when possible."""
    score = score.detach().float().cpu()
    embed_dim = int(_cfg_value(cfg, "embed_dim", score.shape[-1] if score.ndim > 0 else 0))
    num_heads = max(1, int(_cfg_value(cfg, "num_heads", 1)))
    head_dim = embed_dim // num_heads if num_heads > 0 else 0
    if head_dim <= 0 or embed_dim % num_heads != 0:
        return None

    if "attn.in_proj_weight" in name and p.ndim == 2 and score.shape[0] == 3 * embed_dim:
        s = score.abs().reshape(3, num_heads, head_dim, embed_dim)
        q = s[0].mean(dim=(1, 2))
        k = s[1].mean(dim=(1, 2))
        v = s[2].mean(dim=(1, 2))
        coupling = torch.sqrt((q.clamp_min(0.0) + 1e-12) * (k.clamp_min(0.0) + 1e-12))
        return (0.30 * q) + (0.30 * k) + (0.25 * v) + (0.15 * coupling)

    if "attn.in_proj_bias" in name and p.ndim == 1 and score.numel() == 3 * embed_dim:
        s = score.abs().reshape(3, num_heads, head_dim)
        return s.mean(dim=2).mean(dim=0).reshape(-1)

    if "attn.out_proj.weight" in name and p.ndim == 2 and score.shape == (embed_dim, embed_dim):
        rows = score.abs().reshape(num_heads, head_dim, embed_dim).mean(dim=(1, 2))
        cols = score.abs().reshape(embed_dim, num_heads, head_dim).mean(dim=(0, 2))
        return 0.60 * rows + 0.40 * cols

    if "attn.out_proj.bias" in name and p.ndim == 1 and score.numel() == embed_dim:
        return score.abs().reshape(num_heads, head_dim).mean(dim=1)

    return None


def _build_structured_selection_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg,
    prune_fraction: Optional[float] = None,
    stability_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor, Dict[int, torch.Tensor]]:
    """Budgeted structured selection over embedding channels, attention heads, and MLP hidden units."""
    effective_prune_fraction = float(cfg.prune_fraction if prune_fraction is None else prune_fraction)
    stability_weights = stability_weights or {}
    embed_dim = int(cfg.embed_dim)
    num_heads = max(1, int(cfg.num_heads))
    head_dim = max(1, embed_dim // num_heads)
    head_aware_attention = bool(cfg.preserve_attention_heads and embed_dim % num_heads == 0)
    if head_aware_attention:
        group_size = head_dim
        n_embed_groups = num_heads
    else:
        group_size = 1
        n_embed_groups = embed_dim

    embed_group_scores = torch.zeros(n_embed_groups, dtype=torch.float32)
    embed_group_counts = torch.zeros(n_embed_groups, dtype=torch.float32)

    hidden_scores: Dict[int, torch.Tensor] = {}
    hidden_counts: Dict[int, torch.Tensor] = {}
    hidden_block_importance: Dict[int, float] = {}

    attention_scores: Dict[int, torch.Tensor] = {}
    attention_counts: Dict[int, torch.Tensor] = {}
    attention_block_importance: Dict[int, float] = {}

    raw_unit_counts = {"embed": 0, "hidden": 0, "attention": 0}

    for name, p in model.named_parameters():
        if name not in scores:
            continue
        s = scores[name]
        vector_weight = float(stability_weights.get(name, 1.0))

        attn_head = _attention_head_scores(name, s, p, cfg)
        if attn_head is not None:
            attn_conf = _score_confidence_weight(attn_head)
            attn_head = _normalize_unit_scores(attn_head, cfg) * vector_weight * attn_conf
            block_idx = _block_index_from_name(name)
            if block_idx is not None:
                if block_idx not in attention_scores:
                    attention_scores[block_idx] = torch.zeros_like(attn_head)
                    attention_counts[block_idx] = torch.zeros_like(attn_head, dtype=torch.float32)
                attention_scores[block_idx].add_(attn_head.float())
                attention_counts[block_idx].add_(torch.ones_like(attn_head, dtype=torch.float32))
                attention_block_importance[block_idx] = attention_block_importance.get(block_idx, 0.0) + float(attn_head.mean().item())
                raw_unit_counts["attention"] += 1

        # Separate MHSA scoring regime: when attention heads can be represented
        # explicitly, keep attention out of the shared embedding-channel budget.
        if not (head_aware_attention and "attn." in name and attn_head is not None):
            emb = _tensor_group_scores(name, s, p, cfg)
            if emb is not None:
                emb_conf = _score_confidence_weight(emb)
                emb = _normalize_unit_scores(emb, cfg) * vector_weight * emb_conf
                if "patch_embed.proj.weight" in name and p.ndim == 4:
                    if emb.numel() == embed_dim:
                        emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif ("cls_token" in name or "pos_embed" in name) and emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif "attn.in_proj_weight" in name and emb.numel() == 3 * embed_dim:
                    emb = emb[: 3 * n_embed_groups * group_size].reshape(3, n_embed_groups, group_size).mean(dim=(0, 2))
                elif "attn.in_proj_bias" in name and emb.numel() == 3 * embed_dim:
                    emb = emb[: 3 * n_embed_groups * group_size].reshape(3, n_embed_groups, group_size).mean(dim=(0, 2))
                elif "attn.out_proj.weight" in name and p.ndim == 2:
                    if emb.numel() == embed_dim:
                        emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif "attn.out_proj.bias" in name and emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif "mlp.fc2.weight" in name and p.ndim == 2 and emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif "mlp.fc2.bias" in name and emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif "head.weight" in name and emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                elif "head.bias" in name and emb.numel() == cfg.num_classes:
                    # Class logits are not pruned; ignore.
                    emb = None
                elif "norm" in name.lower() and emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)

                if emb is not None and emb.numel() == n_embed_groups:
                    embed_group_scores.add_(emb.float())
                    embed_group_counts.add_(torch.ones_like(emb, dtype=torch.float32))
                    raw_unit_counts["embed"] += 1

        # Extra embed-channel contributions from MLP projections.
        if "mlp.fc1.weight" in name and p.ndim == 2:
            embed_contrib = _normalize_unit_scores(s.abs().mean(dim=0), cfg)
            embed_contrib = embed_contrib * vector_weight * _score_confidence_weight(embed_contrib)
            if embed_contrib.numel() == embed_dim:
                if embed_contrib.numel() == n_embed_groups * group_size:
                    embed_contrib = embed_contrib.reshape(n_embed_groups, group_size).mean(dim=1)
                elif embed_contrib.numel() != n_embed_groups:
                    embed_contrib = embed_contrib[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                embed_group_scores.add_(embed_contrib.float())
                embed_group_counts.add_(torch.ones_like(embed_contrib, dtype=torch.float32))
                raw_unit_counts["embed"] += 1

        if "mlp.fc2.weight" in name and p.ndim == 2:
            embed_contrib = _normalize_unit_scores(s.abs().mean(dim=1), cfg)
            embed_contrib = embed_contrib * vector_weight * _score_confidence_weight(embed_contrib)
            if embed_contrib.numel() == embed_dim:
                if embed_contrib.numel() == n_embed_groups * group_size:
                    embed_contrib = embed_contrib.reshape(n_embed_groups, group_size).mean(dim=1)
                elif embed_contrib.numel() != n_embed_groups:
                    embed_contrib = embed_contrib[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                embed_group_scores.add_(embed_contrib.float())
                embed_group_counts.add_(torch.ones_like(embed_contrib, dtype=torch.float32))
                raw_unit_counts["embed"] += 1

        hid = _hidden_unit_scores(name, s, p)
        if hid is not None:
            hid = _normalize_unit_scores(hid, cfg)
            hid = hid * vector_weight * _score_confidence_weight(hid)
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            if block_match:
                idx = int(block_match.group(1))
                if idx not in hidden_scores:
                    hidden_scores[idx] = torch.zeros_like(hid)
                    hidden_counts[idx] = torch.zeros_like(hid, dtype=torch.float32)
                hidden_scores[idx].add_(hid.float())
                hidden_counts[idx].add_(torch.ones_like(hid, dtype=torch.float32))
                hidden_block_importance[idx] = hidden_block_importance.get(idx, 0.0) + float(hid.mean().item())
                raw_unit_counts["hidden"] += 1

    # Average the aggregated contributions so units with more contributing tensors
    # do not dominate purely by count.
    embed_group_scores = embed_group_scores / embed_group_counts.clamp_min(1.0)

    embed_keep = max(
        _keep_count(n_embed_groups, effective_prune_fraction),
        min_keep_count(n_embed_groups, cfg.min_embed_keep_fraction),
    )
    embed_sel = torch.zeros(n_embed_groups, dtype=torch.bool)
    embed_sel[torch.topk(embed_group_scores, k=embed_keep, largest=True, sorted=False).indices] = True

    embed_channel_sel = embed_sel.repeat_interleave(group_size)
    if embed_channel_sel.numel() < embed_dim:
        pad = torch.zeros(embed_dim - embed_channel_sel.numel(), dtype=torch.bool)
        embed_channel_sel = torch.cat([embed_channel_sel, pad], dim=0)
    elif embed_channel_sel.numel() > embed_dim:
        embed_channel_sel = embed_channel_sel[:embed_dim]

    hidden_sel: Dict[int, torch.Tensor] = {}
    hidden_block_indices = sorted(hidden_scores.keys())
    if hidden_block_indices:
        hidden_sizes = [int(hidden_scores[idx].numel()) for idx in hidden_block_indices]
        hidden_total_keep = _keep_count(sum(hidden_sizes), effective_prune_fraction)
        hidden_floors = [max(1, min_keep_count(size, cfg.min_hidden_keep_fraction)) for size in hidden_sizes]
        hidden_importance = [hidden_block_importance.get(idx, 0.0) for idx in hidden_block_indices]
        hidden_keep_counts = _allocate_keep_budget(hidden_importance, hidden_sizes, hidden_total_keep, hidden_floors)
        for idx, keep_count in zip(hidden_block_indices, hidden_keep_counts):
            hs = hidden_scores[idx] / hidden_counts[idx].clamp_min(1.0)
            keep = max(1, min(int(keep_count), hs.numel()))
            sel = torch.zeros_like(hs, dtype=torch.bool)
            sel[torch.topk(hs, k=keep, largest=True, sorted=False).indices] = True
            hidden_sel[idx] = sel

    attention_sel: Dict[int, torch.Tensor] = {}
    attention_block_indices = sorted(attention_scores.keys())
    if head_aware_attention and attention_block_indices:
        attn_sizes = [int(attention_scores[idx].numel()) for idx in attention_block_indices]
        attn_total_keep = _keep_count(sum(attn_sizes), effective_prune_fraction)
        attn_floor_fraction = float(_cfg_value(cfg, "min_attention_keep_fraction", cfg.min_hidden_keep_fraction))
        attn_floors = [max(1, min_keep_count(size, attn_floor_fraction)) for size in attn_sizes]
        attn_importance = [attention_block_importance.get(idx, 0.0) for idx in attention_block_indices]
        attn_keep_counts = _allocate_keep_budget(attn_importance, attn_sizes, attn_total_keep, attn_floors)
        for idx, keep_count in zip(attention_block_indices, attn_keep_counts):
            hs = attention_scores[idx] / attention_counts[idx].clamp_min(1.0)
            keep = max(1, min(int(keep_count), hs.numel()))
            sel = torch.zeros_like(hs, dtype=torch.bool)
            sel[torch.topk(hs, k=keep, largest=True, sorted=False).indices] = True
            attention_sel[idx] = sel

    masks: Dict[str, torch.Tensor] = {}
    restored_total = 0
    eligible_total = 0
    retained_total = 0

    for name, p in model.named_parameters():
        if not is_prunable_parameter(name, p, cfg):
            masks[name] = torch.ones_like(p, dtype=torch.bool)
            continue

        eligible_total += p.numel()

        block_idx = _block_index_from_name(name)
        if name in {"cls_token", "pos_embed"}:
            mask = embed_channel_sel.view(1, 1, -1).expand_as(p).clone()
        elif "patch_embed.proj.weight" in name and p.ndim == 4:
            mask = embed_channel_sel.view(-1, 1, 1, 1).expand_as(p).clone()
        elif "patch_embed.proj.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "attn.in_proj_weight" in name and p.ndim == 2 and head_aware_attention and block_idx in attention_sel:
            head_mask = _expand_head_mask(attention_sel[block_idx], embed_dim, num_heads)
            row_sel = head_mask.repeat(3)
            if row_sel.numel() != p.shape[0]:
                row_sel = row_sel[:p.shape[0]] if row_sel.numel() > p.shape[0] else torch.cat(
                    [row_sel, torch.zeros(p.shape[0] - row_sel.numel(), dtype=torch.bool)],
                    dim=0,
                )
            col_sel = head_mask
            mask = row_sel.view(-1, 1).expand_as(p).clone() & col_sel.view(1, -1).expand_as(p)
        elif "attn.in_proj_bias" in name and p.ndim == 1 and head_aware_attention and block_idx in attention_sel:
            mask = attention_sel[block_idx].repeat_interleave(head_dim).repeat(3)[: p.numel()].clone()
        elif "attn.out_proj.weight" in name and p.ndim == 2 and head_aware_attention and block_idx in attention_sel:
            head_mask = _expand_head_mask(attention_sel[block_idx], embed_dim, num_heads)
            mask = head_mask.view(-1, 1).expand_as(p).clone() & head_mask.view(1, -1).expand_as(p)
        elif "attn.out_proj.bias" in name and p.ndim == 1 and head_aware_attention and block_idx in attention_sel:
            mask = _expand_head_mask(attention_sel[block_idx], embed_dim, num_heads)
        elif "mlp.fc2.weight" in name and p.ndim == 2:
            hmask = hidden_sel.get(block_idx, torch.ones(p.shape[1], dtype=torch.bool))
            mask = embed_channel_sel.view(-1, 1).expand(p.shape[0], p.shape[1]) & hmask.view(1, -1)
        elif "mlp.fc2.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "mlp.fc1.weight" in name and p.ndim == 2:
            hmask = hidden_sel.get(block_idx, torch.ones(p.shape[0], dtype=torch.bool))
            mask = hmask.view(-1, 1).expand_as(p).clone() & embed_channel_sel.view(1, -1).expand_as(p)
        elif "mlp.fc1.bias" in name and p.ndim == 1:
            mask = hidden_sel.get(block_idx, torch.ones_like(p, dtype=torch.bool)).clone()
        elif "norm" in name.lower() and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "head.weight" in name and p.ndim == 2:
            mask = embed_channel_sel.view(1, -1).expand_as(p).clone()
        elif "head.bias" in name and p.ndim == 1:
            mask = torch.ones_like(p, dtype=torch.bool)
        else:
            mask = torch.ones_like(p, dtype=torch.bool)

        if cfg.connectivity_closure and p.ndim in (2, 4):
            restored_total += connectivity_close_dense_weight(
                mask,
                scores[name].to(mask.device),
                max(1, cfg.min_connections_per_unit),
            )
        if int(mask.sum().item()) == 0:
            idx = torch.argmax(scores[name].reshape(-1))
            mask.reshape(-1)[idx] = True
            restored_total += 1

        retained_total += int(mask.sum().item())
        masks[name] = mask

    stats = {
        "eligible_parameter_count": float(eligible_total),
        "retained_eligible_parameter_count": float(retained_total),
        "pruned_eligible_parameter_count": float(max(0, eligible_total - retained_total)),
        "actual_prune_fraction_eligible": float(max(0, eligible_total - retained_total) / max(1, eligible_total)),
        "pruning_strategy": cfg.pruning_strategy.lower(),
        "prune_fraction": float(effective_prune_fraction),
        "layerwise_normalize_scores": bool(cfg.layerwise_normalize_scores),
        "sensitivity_normalization": str(cfg.sensitivity_normalization),
        "sensitivity_clip_quantile": float(cfg.sensitivity_clip_quantile),
        "min_embed_keep_fraction": float(cfg.min_embed_keep_fraction),
        "min_hidden_keep_fraction": float(cfg.min_hidden_keep_fraction),
        "preserve_attention_heads": bool(cfg.preserve_attention_heads),
        "mhsa_scoring_regime": "head-aware-qkv-o" if head_aware_attention else "tensor-averaged",
        "connectivity_restored_coordinate_count": float(restored_total),
        "embed_groups": float(n_embed_groups),
        "embed_groups_retained": float(int(embed_sel.sum().item())),
        "embed_group_keep_fraction": float(int(embed_sel.sum().item()) / max(1, n_embed_groups)),
        "hidden_block_count": float(len(hidden_scores)),
        "attention_block_count": float(len(attention_scores)),
        "raw_unit_counts_embed": float(raw_unit_counts["embed"]),
        "raw_unit_counts_hidden": float(raw_unit_counts["hidden"]),
        "raw_unit_counts_attention": float(raw_unit_counts["attention"]),
    }
    return masks, stats, embed_sel, hidden_sel


def _intersect_and_keep_one(
    candidate: torch.Tensor,
    previous: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Cumulative pruning step:
    - keep only coordinates that survived previous rounds
    - if that would empty the tensor, keep the best surviving coordinate
    """
    candidate = candidate & previous
    if int(candidate.sum().item()) > 0:
        return candidate

    prev_flat = previous.reshape(-1).bool()
    out = torch.zeros_like(candidate, dtype=torch.bool)

    if int(prev_flat.sum().item()) > 0:
        score_flat = score.reshape(-1).detach().float().clone()
        score_flat = score_flat.masked_fill(~prev_flat, float("-inf"))
        idx = torch.argmax(score_flat)
        out.reshape(-1)[idx] = True
    else:
        # Fallback safety: should not normally happen, but keeps the tensor non-empty.
        idx = torch.argmax(score.reshape(-1))
        out.reshape(-1)[idx] = True

    return out


def build_structured_masks_iterative(
    model: nn.Module,
    loader,
    cfg,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor, Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    """Iteratively recompute sensitivities and refine a structured mask."""
    import copy

    def _snapshot_similarity(current: Dict[str, torch.Tensor], previous: Optional[Dict[str, torch.Tensor]]) -> float:
        if not previous:
            return 1.0
        total_sim = 0.0
        total_weight = 0.0
        for name, cur in current.items():
            prev = previous.get(name)
            sim = _vector_similarity(cur, prev) if prev is not None else None
            if sim is None:
                continue
            weight = float(cur.numel())
            total_sim += float(sim) * weight
            total_weight += weight
        if total_weight <= 0:
            return 1.0
        return max(-1.0, min(1.0, total_sim / total_weight))

    def _snapshot_weights(
        current: Dict[str, torch.Tensor],
        previous: Optional[Dict[str, torch.Tensor]],
        confidence_weights: Optional[Dict[str, object]] = None,
    ) -> Dict[str, float]:
        if not previous:
            return {}
        weights: Dict[str, float] = {}
        for name, cur in current.items():
            prev = previous.get(name)
            sim = _vector_similarity(cur, prev) if prev is not None else None
            if sim is None:
                continue
            stability = max(0.0, min(1.0, 0.5 * (sim + 1.0)))
            base = 0.55 + 0.45 * stability
            conf = _as_confidence_scalar(confidence_weights.get(name), 1.0) if confidence_weights else 1.0
            weights[name] = max(0.25, min(1.0, base * conf))
        return weights

    def _cosine_progress(progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    # Keep the original model untouched until the end.
    # Iterative pruning happens on a working copy so early rounds do not permanently
    # destroy state in the caller's model.
    working_model = copy.deepcopy(model)

    params = dict(working_model.named_parameters())
    active_masks: Dict[str, torch.Tensor] = {
        name: torch.ones_like(p, dtype=torch.bool, device=p.device)
        for name, p in params.items()
    }

    final_masks: Dict[str, torch.Tensor] = active_masks
    final_stats: Dict[str, float] = {}
    final_embed_sel = torch.empty(0, dtype=torch.bool)
    final_hidden_sel: Dict[int, torch.Tensor] = {}
    final_scores: Dict[str, torch.Tensor] = {}

    round_fractions: List[float] = []
    round_stabilities: List[float] = []
    round_probe_confidences: List[float] = []
    previous_scores: Optional[Dict[str, torch.Tensor]] = None
    previous_stability: Optional[float] = None
    stable_rounds = 0
    early_stop_triggered = False

    total_rounds = max(1, int(cfg.iterative_pruning_rounds))
    target_prune_fraction = float(cfg.prune_fraction)
    min_rounds = max(1, int(_cfg_value(cfg, 'iterative_min_rounds', 1)))
    stability_stop = float(_cfg_value(cfg, 'iterative_stability_stop_threshold', 0.995))
    stability_delta_stop = float(_cfg_value(cfg, 'iterative_stability_delta_stop', 0.005))

    for round_idx in range(total_rounds):
        scores, _, score_meta = compute_sensitivity_scores(
            working_model,
            loader,
            cfg,
            device,
            probes=cfg.sensitivity_probes,
            collect_probe_matrix=True,
        )
        final_scores = scores

        stability = _snapshot_similarity(scores, previous_scores)
        round_stabilities.append(float(stability))

        probe_confidence = float(score_meta.get("probe_confidence", 1.0))
        round_probe_confidences.append(probe_confidence)

        stability_weights = _snapshot_weights(
            scores,
            previous_scores,
            confidence_weights=score_meta.get("parameter_confidence", {}),
        )

        progress = float(round_idx + 1) / float(total_rounds)

        if cfg.gradual_sparsification:
            schedule = _cosine_progress(progress)
            stability_gate = 0.5 + 0.5 * max(0.0, min(1.0, stability))

            if previous_stability is None:
                plateau_bonus = 1.0
            else:
                plateau_bonus = 1.0 + 0.15 * max(0.0, stability - previous_stability)

            probe_gate = max(0.25, min(1.0, probe_confidence))

            round_prune_fraction = (
                target_prune_fraction
                * schedule
                * stability_gate
                * plateau_bonus
                * probe_gate
            )
        else:
            round_prune_fraction = target_prune_fraction
        if round_idx == total_rounds - 1:
            round_prune_fraction = target_prune_fraction
        round_prune_fraction = max(0.0, min(target_prune_fraction, round_prune_fraction))
        round_fractions.append(round_prune_fraction)

        candidate_masks, stats, embed_sel, hidden_sel = _build_structured_selection_masks(
            working_model,
            scores,
            cfg,
            prune_fraction=round_prune_fraction,
            stability_weights=stability_weights,
        )

        # Make pruning cumulative without hard-mutating the caller's model.
        for name, cand in candidate_masks.items():
            if is_prunable_parameter(name, params[name], cfg):
                candidate_masks[name] = _intersect_and_keep_one(
                    cand.to(device=params[name].device, dtype=torch.bool),
                    active_masks[name],
                    scores[name].to(device=params[name].device),
                )
            else:
                candidate_masks[name] = cand.to(device=params[name].device, dtype=torch.bool)

        final_masks = candidate_masks
        final_stats = stats
        final_embed_sel = embed_sel
        final_hidden_sel = hidden_sel
        active_masks = candidate_masks

        # Apply masks only to the working copy used for the next round's scoring.
        apply_masks_(working_model, candidate_masks)

        delta = None if previous_stability is None else abs(stability - previous_stability)
        if (
            round_idx + 1 >= min_rounds
            and round_idx + 1 < total_rounds
            and previous_stability is not None
            and stability >= stability_stop
            and delta is not None
            and delta <= stability_delta_stop
        ):
            stable_rounds += 1
        else:
            stable_rounds = 0

        # Stop once the score landscape has plateaued for multiple consecutive rounds.
        if round_idx + 1 >= min_rounds and round_idx + 1 < total_rounds and stable_rounds >= 2:
            early_stop_triggered = True
            break

        previous_scores = scores
        previous_stability = stability

    # Apply only the final mask to the original model.
    apply_masks_(model, final_masks)

    final_stats["iterative_rounds_completed"] = float(len(round_fractions))
    final_stats["gradual_sparsification"] = bool(cfg.gradual_sparsification)
    final_stats["round_prune_fraction_schedule"] = [float(x) for x in round_fractions]
    final_stats["round_stability_schedule"] = [float(x) for x in round_stabilities]
    final_stats["final_round_prune_fraction"] = float(round_fractions[-1]) if round_fractions else float(target_prune_fraction)
    final_stats["final_round_stability"] = float(round_stabilities[-1]) if round_stabilities else 1.0
    final_stats["mean_round_stability"] = float(sum(round_stabilities) / max(1, len(round_stabilities)))
    final_stats["cumulative_iterative_pruning"] = True
    final_stats["early_round_schedule"] = "cosine-adaptive" if cfg.gradual_sparsification else "constant"
    final_stats["iterative_early_stop_triggered"] = bool(early_stop_triggered)
    final_stats["iterative_stable_rounds"] = float(stable_rounds)
    final_stats["round_probe_confidence_schedule"] = [float(x) for x in round_probe_confidences]
    final_stats["final_round_probe_confidence"] = float(round_probe_confidences[-1]) if round_probe_confidences else 1.0
    return final_masks, final_stats, final_embed_sel, final_hidden_sel, final_scores


def masked_density_by_module(model: nn.Module, masks: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, int | float]]:
    groups: Dict[str, Dict[str, int]] = {}
    for name, mask in masks.items():
        group = name.split(".")[0]
        groups.setdefault(group, {"retained": 0, "total": 0})
        groups[group]["retained"] += int(mask.sum().item())
        groups[group]["total"] += int(mask.numel())
    return {
        k: {"retained": v["retained"], "total": v["total"], "density": v["retained"] / max(1, v["total"])}
        for k, v in groups.items()
    }


def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def compute_eligible_pruning_stats(
    reference_model: nn.Module,
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    cfg,
) -> Dict[str, float]:
    """
    Compute pruning stats over eligible parameters.

    Args:
        reference_model: Unpruned reference PyTorch module.
        model: Pruned PyTorch module.
        masks: Dict mapping parameter name -> boolean mask tensor.
               True means retained, False means pruned.
        cfg: Config object used by is_prunable_parameter(...).

    Returns:
        Dict with:
          - eligible_parameter_count
          - retained_eligible_parameter_count
          - pruned_eligible_parameter_count
          - actual_prune_fraction_eligible
    """
    eligible_total = 0
    retained_total = 0

    ref_params = dict(reference_model.named_parameters())
    pruned_params = dict(model.named_parameters())

    for name, ref_p in ref_params.items():
        if not is_prunable_parameter(name, ref_p, cfg):
            continue

        eligible_total += ref_p.numel()

        pruned_p = pruned_params.get(name)
        mask = masks.get(name)

        if pruned_p is not None:
            retained_total += pruned_p.numel()
        elif mask is not None:
            retained_total += int(mask.sum().item())
        else:
            retained_total += ref_p.numel()

    pruned_total = max(0, eligible_total - retained_total)

    return {
        "eligible_parameter_count_baseline": float(eligible_total),
        "retained_eligible_parameter_count_baseline": float(retained_total),
        "pruned_eligible_parameter_count_baseline": float(pruned_total),
        "actual_prune_fraction_eligible_baseline": float(pruned_total / max(1, eligible_total)),
    }

def connectivity_close_dense_weight(mask: torch.Tensor, scores: torch.Tensor, min_conn: int) -> int:
    """
    Connectivity repair with a post-repair trim pass.

    Behavior:
      1) Greedily restore the minimum set of missing coordinates needed to satisfy
         row/column connectivity constraints.
      2) Trim any restored coordinates that are redundant after feasibility is reached,
         while preserving the constraints.

    Returns:
      Number of restored coordinates that remain after trimming.
    """
    if min_conn <= 0 or mask.numel() == 0:
        return 0

    mask = mask.bool()
    scores = scores.detach().float()
    restored = 0

    def _repair_and_trim_2d(mask_2d: torch.Tensor, score_2d: torch.Tensor) -> int:
        added = torch.zeros_like(mask_2d, dtype=torch.bool)

        # Phase 1: minimal greedy repair.
        while True:
            row_count = mask_2d.sum(dim=1)
            col_count = mask_2d.sum(dim=0)

            row_deficit = (min_conn - row_count).clamp_min(0)
            col_deficit = (min_conn - col_count).clamp_min(0)

            if int(row_deficit.sum().item()) == 0 and int(col_deficit.sum().item()) == 0:
                break

            candidates = (~mask_2d).nonzero(as_tuple=False)
            if candidates.numel() == 0:
                break

            r = candidates[:, 0]
            c = candidates[:, 1]
            gain = row_deficit[r] + col_deficit[c]

            if int(gain.max().item()) <= 0:
                break

            cand_scores = score_2d[r, c]
            priority = gain.to(torch.float32) * 1e6 + cand_scores
            best = int(torch.argmax(priority).item())

            i = int(r[best].item())
            j = int(c[best].item())

            mask_2d[i, j] = True
            added[i, j] = True

        # Phase 2: trim redundant restored coordinates.
        if int(added.sum().item()) == 0:
            return 0

        row_count = mask_2d.sum(dim=1)
        col_count = mask_2d.sum(dim=0)

        added_idx = added.nonzero(as_tuple=False)
        added_scores = score_2d[added].reshape(-1)
        order = torch.argsort(added_scores, descending=False)

        for idx in order.tolist():
            i = int(added_idx[idx, 0].item())
            j = int(added_idx[idx, 1].item())

            if not mask_2d[i, j]:
                continue

            if row_count[i] > min_conn and col_count[j] > min_conn:
                mask_2d[i, j] = False
                added[i, j] = False
                row_count[i] -= 1
                col_count[j] -= 1

        return int(added.sum().item())

    if mask.ndim == 2:
        restored += _repair_and_trim_2d(mask, scores)

    elif mask.ndim == 4:
        # Conv2d: enforce connectivity over output and input channels, then trim
        # any redundant restored coordinates.
        out_ch, in_ch = mask.shape[:2]
        flat_mask_out = mask.reshape(out_ch, -1)
        flat_scores_out = scores.reshape(out_ch, -1)
        flat_mask_in = mask.permute(1, 0, 2, 3).reshape(in_ch, -1)
        flat_scores_in = scores.permute(1, 0, 2, 3).reshape(in_ch, -1)

        # Repair by output-channel groups.
        added_out = torch.zeros_like(flat_mask_out, dtype=torch.bool)
        while True:
            out_count = flat_mask_out.sum(dim=1)
            in_count = mask.permute(1, 0, 2, 3).reshape(in_ch, -1).sum(dim=1)

            out_deficit = (min_conn - out_count).clamp_min(0)
            in_deficit = (min_conn - in_count).clamp_min(0)

            if int(out_deficit.sum().item()) == 0 and int(in_deficit.sum().item()) == 0:
                break

            candidates = (~flat_mask_out).nonzero(as_tuple=False)
            if candidates.numel() == 0:
                break

            r = candidates[:, 0]
            c = candidates[:, 1]
            gain = out_deficit[r] + in_deficit[c % in_ch]

            if int(gain.max().item()) <= 0:
                break

            cand_scores = flat_scores_out[r, c]
            priority = gain.to(torch.float32) * 1e6 + cand_scores
            best = int(torch.argmax(priority).item())

            o = int(r[best].item())
            flat_idx = int(c[best].item())

            flat_mask_out[o, flat_idx] = True
            added_out[o, flat_idx] = True

        # Sync back to the 4D mask.
        mask = flat_mask_out.view_as(mask)

        # Trim redundant restored coordinates.
        if int(added_out.sum().item()) > 0:
            out_count = mask.reshape(out_ch, -1).sum(dim=1)
            in_count = mask.permute(1, 0, 2, 3).reshape(in_ch, -1).sum(dim=1)

            added_idx = added_out.nonzero(as_tuple=False)
            added_scores = flat_scores_out[added_out].reshape(-1)
            order = torch.argsort(added_scores, descending=False)

            for idx in order.tolist():
                o = int(added_idx[idx, 0].item())
                flat_idx = int(added_idx[idx, 1].item())

                if not flat_mask_out[o, flat_idx]:
                    continue

                i = flat_idx // (mask.shape[2] * mask.shape[3]) if mask.shape[2] * mask.shape[3] > 0 else 0

                if out_count[o] > min_conn and in_count[i] > min_conn:
                    flat_mask_out[o, flat_idx] = False
                    added_out[o, flat_idx] = False
                    out_count[o] -= 1
                    in_count[i] -= 1

            mask = flat_mask_out.view_as(mask)
            restored += int(added_out.sum().item())

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