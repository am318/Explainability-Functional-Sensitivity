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


def _build_structured_selection_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg,
    prune_fraction: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor, Dict[int, torch.Tensor]]:
    """Budgeted structured selection over embedding channels and MLP hidden units."""
    effective_prune_fraction = float(cfg.prune_fraction if prune_fraction is None else prune_fraction)
    embed_dim = cfg.embed_dim
    head_dim = max(1, embed_dim // max(1, cfg.num_heads))
    if cfg.preserve_attention_heads and embed_dim % max(1, cfg.num_heads) == 0:
        group_size = head_dim
        n_embed_groups = cfg.num_heads
    else:
        group_size = 1
        n_embed_groups = embed_dim

    embed_group_scores = torch.zeros(n_embed_groups, dtype=torch.float32)
    embed_group_counts = torch.zeros(n_embed_groups, dtype=torch.float32)

    hidden_scores: Dict[int, torch.Tensor] = {}
    hidden_counts: Dict[int, torch.Tensor] = {}
    raw_unit_counts = {"embed": 0, "hidden": 0}

    for name, p in model.named_parameters():
        if name not in scores:
            continue
        s = scores[name]

        emb = _tensor_group_scores(name, s, p, cfg)
        if emb is not None:
            emb = _normalize_unit_scores(emb, cfg)
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
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            if block_match:
                idx = int(block_match.group(1))
                if idx not in hidden_scores:
                    hidden_scores[idx] = torch.zeros_like(hid)
                    hidden_counts[idx] = torch.zeros_like(hid, dtype=torch.float32)
                hidden_scores[idx].add_(hid.float())
                hidden_counts[idx].add_(torch.ones_like(hid, dtype=torch.float32))
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
    for idx, hs in hidden_scores.items():
        hs = hs / hidden_counts[idx].clamp_min(1.0)
        keep = max(
            _keep_count(hs.numel(), effective_prune_fraction),
            min_keep_count(hs.numel(), cfg.min_hidden_keep_fraction),
        )
        sel = torch.zeros_like(hs, dtype=torch.bool)
        sel[torch.topk(hs, k=keep, largest=True, sorted=False).indices] = True
        hidden_sel[idx] = sel

    masks: Dict[str, torch.Tensor] = {}
    restored_total = 0
    eligible_total = 0
    retained_total = 0

    for name, p in model.named_parameters():
        if not is_prunable_parameter(name, p, cfg):
            masks[name] = torch.ones_like(p, dtype=torch.bool)
            continue

        eligible_total += p.numel()

        if name in {"cls_token", "pos_embed"}:
            mask = embed_channel_sel.view(1, 1, -1).expand_as(p).clone()
        elif "patch_embed.proj.weight" in name and p.ndim == 4:
            mask = embed_channel_sel.view(-1, 1, 1, 1).expand_as(p).clone()
        elif "patch_embed.proj.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "attn.in_proj_weight" in name and p.ndim == 2:
            row_sel = embed_channel_sel.repeat(3)
            if row_sel.numel() != p.shape[0]:
                row_sel = row_sel[:p.shape[0]] if row_sel.numel() > p.shape[0] else torch.cat(
                    [row_sel, torch.zeros(p.shape[0] - row_sel.numel(), dtype=torch.bool)],
                    dim=0,
                )
            col_sel = embed_channel_sel
            mask = row_sel.view(-1, 1).expand_as(p).clone() & col_sel.view(1, -1).expand_as(p)
        elif "attn.in_proj_bias" in name and p.ndim == 1:
            mask = embed_channel_sel.repeat(3).clone()
        elif "attn.out_proj.weight" in name and p.ndim == 2:
            mask = embed_channel_sel.view(-1, 1).expand_as(p).clone() & embed_channel_sel.view(1, -1).expand_as(p)
        elif "attn.out_proj.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "mlp.fc2.weight" in name and p.ndim == 2:
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc2", name)
            block_idx = int(block_match.group(1)) if block_match else -1
            hmask = hidden_sel.get(block_idx, torch.ones(p.shape[1], dtype=torch.bool))
            mask = embed_channel_sel.view(-1, 1).expand(p.shape[0], p.shape[1]) & hmask.view(1, -1)
        elif "mlp.fc2.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "mlp.fc1.weight" in name and p.ndim == 2:
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            block_idx = int(block_match.group(1)) if block_match else -1
            hmask = hidden_sel.get(block_idx, torch.ones(p.shape[0], dtype=torch.bool))
            mask = hmask.view(-1, 1).expand_as(p).clone() & embed_channel_sel.view(1, -1).expand_as(p)
        elif "mlp.fc1.bias" in name and p.ndim == 1:
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            block_idx = int(block_match.group(1)) if block_match else -1
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
        "iterative_pruning_rounds": int(cfg.iterative_pruning_rounds),
        "layerwise_normalize_scores": bool(cfg.layerwise_normalize_scores),
        "sensitivity_normalization": str(cfg.sensitivity_normalization),
        "sensitivity_clip_quantile": float(cfg.sensitivity_clip_quantile),
        "min_embed_keep_fraction": float(cfg.min_embed_keep_fraction),
        "min_hidden_keep_fraction": float(cfg.min_hidden_keep_fraction),
        "preserve_attention_heads": bool(cfg.preserve_attention_heads),
        "connectivity_restored_coordinate_count": float(restored_total),
        "embed_groups": float(n_embed_groups),
        "embed_groups_retained": float(int(embed_sel.sum().item())),
        "embed_group_keep_fraction": float(int(embed_sel.sum().item()) / max(1, n_embed_groups)),
        "hidden_block_count": float(len(hidden_scores)),
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

    total_rounds = max(1, int(cfg.iterative_pruning_rounds))
    target_prune_fraction = float(cfg.prune_fraction)

    for round_idx in range(total_rounds):
        scores, _ = compute_sensitivity_scores(
            working_model,
            loader,
            cfg,
            device,
            probes=cfg.sensitivity_probes,
        )
        final_scores = scores

        # Conservative early pruning schedule:
        # quadratic ramp keeps the first rounds milder than the original linear ramp.
        if cfg.gradual_sparsification:
            progress = float(round_idx + 1) / float(total_rounds)
            round_prune_fraction = target_prune_fraction * (progress * progress)
        else:
            round_prune_fraction = target_prune_fraction
        round_fractions.append(round_prune_fraction)

        candidate_masks, stats, embed_sel, hidden_sel = _build_structured_selection_masks(
            working_model,
            scores,
            cfg,
            prune_fraction=round_prune_fraction,
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

    # Apply only the final mask to the original model.
    apply_masks_(model, final_masks)

    final_stats["iterative_rounds_completed"] = float(total_rounds)
    final_stats["gradual_sparsification"] = bool(cfg.gradual_sparsification)
    final_stats["round_prune_fraction_schedule"] = [float(x) for x in round_fractions]
    final_stats["final_round_prune_fraction"] = float(round_fractions[-1]) if round_fractions else float(target_prune_fraction)
    final_stats["cumulative_iterative_pruning"] = True
    final_stats["early_round_schedule"] = "quadratic" if cfg.gradual_sparsification else "constant"
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


def probe_covariance_eigvals(probe_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if probe_matrix is None or probe_matrix.numel() == 0:
        return None
    X = probe_matrix.float()
    X = X - X.mean(dim=0, keepdim=True)
    gram = (X @ X.T) / max(1, X.shape[0])
    vals = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0).detach().cpu()
    return vals


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