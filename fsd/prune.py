"""C6 feasibility panel: prune at step t, keep training, see what it costs.

Protocol is prune-and-continue (as in early-bird pruning), not prune-and-rewind: the
follow-up paper's motivation is saving the compute of training dense to convergence, so
the operative question is whether a mask taken at step t and applied from there on is as
good as one taken at the end.

Criteria compared:
  sensitivity  -- S_t, the object of this paper
  magnitude    -- |theta_t|, the standard cheap baseline
  random       -- layer-budget-matched to the sensitivity mask, so the comparison isolates
                  *placement* from *budget* (the Su et al. random-ticket sanity check)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from . import criteria as CR, models, rank_metrics as R, sensitivity as S, storage
from .run import build_optimizer
from .schedule import lr_at
from .tasks import build_task


def _unflatten(keep: torch.Tensor, names, shapes) -> Dict[str, torch.Tensor]:
    out, off = {}, 0
    for n, shape in zip(names, shapes):
        size = int(torch.tensor(shape).prod()) if len(shape) else 1
        out[n] = keep[off:off + size].reshape(shape).clone()
        off += size
    return out


def enforce_min_count(keep_sub: torch.Tensor, scores_sub: torch.Tensor,
                      group_ids: torch.Tensor, n_groups: int,
                      min_keep_fraction: float) -> torch.Tensor:
    """Guarantee every group (layer, or output unit within a layer) retains at least
    `min_keep_fraction` of its weights, preserving the total kept count exactly.

    Unconstrained global top-k does not merely empty whole *layers* at high sparsity --
    even with a layer floor in place, it disproportionately empties individual *output
    units* (rows of a weight tensor: neurons, conv filters, attention head slices) well
    before the layer itself runs out of budget. Measured on ResNet-20 at init: sensitivity
    zeroes 18/794 units by sparsity 0.5 (magnitude: 0) and 454/794 (57%) by 0.95
    (magnitude: 0), while also pruning ~2x more of early layers than magnitude at the same
    global sparsity. A unit with every incoming weight pruned is functionally deleted
    regardless of whether its layer survives, so this is the same failure mode as the
    layer-emptying one, one level finer -- and it is present starting at sparsity 0.5, not
    only in the high-sparsity regime. (Su et al.-style random-ticket sanity checks and
    SynFlow's design both target exactly this: naive saliency pruning collapsing
    structure rather than removing individually unimportant weights.)

    Implemented as count allocation, not top-k-then-repair, for the same reason as before:
    a repair pass can re-empty a group whose weights are uniformly low-scoring.
    """
    counts = torch.zeros(n_groups, dtype=torch.long)
    sizes = torch.zeros(n_groups, dtype=torch.long)
    idx = group_ids.long()
    sizes.index_add_(0, idx, torch.ones_like(idx))
    counts.index_add_(0, idx, keep_sub.long())

    budget = int(counts.sum())
    live = sizes > 0
    floors = torch.zeros(n_groups, dtype=torch.long)
    floors[live] = torch.clamp(
        (min_keep_fraction * sizes[live].double()).round().long(), min=1)
    floors = torch.minimum(floors, sizes)

    new = torch.maximum(counts, floors)
    excess = int(new.sum()) - budget
    if excess > 0:
        for _ in range(64):
            head = (new - floors).clamp(min=0)
            total_head = int(head.sum())
            if excess <= 0 or total_head == 0:
                break
            take = torch.minimum(
                head, torch.ceil(head.double() * excess / total_head).long())
            order = torch.argsort(head, descending=True)
            for l in order.tolist():
                if excess <= 0:
                    break
                t = int(min(int(take[l]), excess, int(head[l])))
                if t > 0:
                    new[l] -= t
                    excess -= t
    new = torch.minimum(torch.maximum(new, floors), sizes)

    keep = torch.zeros_like(keep_sub)
    for l in range(n_groups):
        k = int(new[l])
        if k <= 0:
            continue
        sel = torch.nonzero(idx == l, as_tuple=False).flatten()
        top = sel[torch.topk(scores_sub[sel], min(k, sel.numel())).indices]
        keep[top] = True
    return keep


# Back-compat alias: layer-level floor is enforce_min_count with layer ids as the grouping.
enforce_min_keep = enforce_min_count


def unit_group_ids(model_names, shapes, prunable_flat: torch.Tensor,
                   model: nn.Module) -> torch.Tensor:
    """Per-prunable-entry output-unit id, GLOBAL across tensors (unlike
    `criteria.structured_groups`, which restarts numbering per tensor). 1-D tensors
    (biases, norms) -- almost never prunable, but handled for completeness -- get one unit
    each, since there is no "row" to group by.
    """
    groups = CR.structured_groups(model)
    ids = torch.full((prunable_flat.numel(),), -1, dtype=torch.long)
    off = 0
    next_id = 0
    for n, shape in zip(model_names, shapes):
        size = int(torch.tensor(shape).prod()) if len(shape) else 1
        if n in groups:
            gid = groups[n]
            ids[off:off + size] = gid + next_id
            next_id += int(gid.max()) + 1
        else:
            ids[off:off + size] = next_id
            next_id += 1
        off += size
    return ids


def masks_from_scores(flat_scores: torch.Tensor, names, shapes, sparsity: float,
                      prunable: torch.Tensor, layer_ids: Optional[torch.Tensor] = None,
                      n_layers: int = 0,
                      min_keep_fraction: float = 0.0,
                      unit_ids: Optional[torch.Tensor] = None,
                      min_keep_fraction_unit: float = 0.0) -> Dict[str, torch.Tensor]:
    """Global top-k over prunable entries; non-prunable entries are always kept.

    Two independent floors compose, applied layer first then unit: `min_keep_fraction`
    prevents whole layers being emptied, `min_keep_fraction_unit` prevents individual
    output units (neurons/filters) within a surviving layer being emptied. The unit floor
    is the one that matters starting at moderate sparsity (measured onset: 0.5); the layer
    floor alone is not sufficient -- see `enforce_min_count`.
    """
    keep = torch.ones_like(prunable, dtype=torch.bool)
    sub = flat_scores[prunable]
    keep_sub = R.topk_mask(sub, sparsity)
    if min_keep_fraction > 0 and layer_ids is not None:
        keep_sub = enforce_min_count(keep_sub, sub, layer_ids[prunable], n_layers,
                                     min_keep_fraction)
    if min_keep_fraction_unit > 0 and unit_ids is not None:
        n_units = int(unit_ids.max()) + 1
        keep_sub = enforce_min_count(keep_sub, sub, unit_ids, n_units,
                                     min_keep_fraction_unit)
    keep[prunable] = keep_sub
    return _unflatten(keep, names, shapes)


def budget_matched_random(flat_scores: torch.Tensor, layer_ids: torch.Tensor,
                          n_layers: int, sparsity: float, prunable: torch.Tensor,
                          names, shapes, seed: int = 0,
                          min_keep_fraction: float = 0.0,
                          unit_ids: Optional[torch.Tensor] = None,
                          min_keep_fraction_unit: float = 0.0) -> Dict[str, torch.Tensor]:
    """Random keep-set with the *same per-layer counts* as the (floored) sensitivity mask.

    Matching the budget *after* both floors are applied, so the baseline inherits the same
    valid layer AND unit allocation and the comparison isolates within-unit placement.
    """
    g = torch.Generator().manual_seed(seed)
    ref = R.topk_mask(flat_scores[prunable], sparsity)
    if min_keep_fraction > 0:
        ref = enforce_min_count(ref, flat_scores[prunable], layer_ids[prunable], n_layers,
                                min_keep_fraction)
    if min_keep_fraction_unit > 0 and unit_ids is not None:
        n_units = int(unit_ids.max()) + 1
        ref = enforce_min_count(ref, flat_scores[prunable], unit_ids, n_units,
                                min_keep_fraction_unit)
    # Match counts at the finest available granularity: per unit where units are
    # defined (which automatically preserves each unit's *layer* count too, since a unit
    # is wholly contained in one layer), per layer otherwise (1-D tensors with no "row" to
    # group by). Matching only at the layer level, as an earlier version did, lets random
    # placement accidentally empty a unit the floored reference mask does not -- observed:
    # up to 22/784 units dead under layer-only matching, versus 0 once unit counts match.
    keep = torch.ones_like(prunable, dtype=torch.bool)
    keep_sub = torch.zeros_like(ref)
    if unit_ids is not None:
        n_units = int(unit_ids.max()) + 1
        for u in range(n_units):
            sel = torch.nonzero(unit_ids == u, as_tuple=False).flatten()
            if sel.numel() == 0:
                continue
            k = int(ref[sel].sum())
            if k:
                pick = sel[torch.randperm(sel.numel(), generator=g)[:k]]
                keep_sub[pick] = True
        # entries outside any unit grouping (rare: 1-D prunable tensors) fall back to layer
        ungrouped = unit_ids < 0
        if ungrouped.any():
            lids = layer_ids[prunable]
            for l in range(n_layers):
                sel = torch.nonzero((lids == l) & ungrouped, as_tuple=False).flatten()
                if sel.numel() == 0:
                    continue
                k = int(ref[sel].sum())
                if k:
                    pick = sel[torch.randperm(sel.numel(), generator=g)[:k]]
                    keep_sub[pick] = True
    else:
        lids = layer_ids[prunable]
        for l in range(n_layers):
            sel = torch.nonzero(lids == l, as_tuple=False).flatten()
            if sel.numel() == 0:
                continue
            k = int(ref[sel].sum())
            if k:
                pick = sel[torch.randperm(sel.numel(), generator=g)[:k]]
                keep_sub[pick] = True
    keep[prunable] = keep_sub
    return _unflatten(keep, names, shapes)


@torch.no_grad()
def apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for n, p in model.named_parameters():
        if n in masks:
            p.mul_(masks[n].to(p.device, p.dtype))


def prune_and_continue(cfg, prune_step: int, sparsity: float, criterion: str,
                       state: Dict[str, torch.Tensor],
                       flat_scores: Optional[torch.Tensor] = None,
                       verbose: bool = True,
                       finetune_steps: Optional[int] = None,
                       eval_every: Optional[int] = None,
                       min_keep_fraction: float = 0.0,
                       min_keep_fraction_unit: float = 0.0) -> Dict[str, object]:
    """Resume from `state` (weights at `prune_step`), prune, then fine-tune.

    `finetune_steps`: if given, train for exactly this many steps regardless of
    `prune_step`, so every prune time gets the *same* recovery budget and comparisons
    across prune times are not confounded by how much of the schedule is left. Without it,
    training runs to `cfg.train.steps` as before -- which silently gives ZERO fine-tuning
    when `prune_step == cfg.train.steps`; that combination should not be used for
    comparison (see the E3 findings log entry).

    `eval_every`: if given, evaluate every this-many fine-tune steps and return the full
    trajectory in `out["trajectory"]`, so a stalled-vs-still-climbing gap can be told apart
    from a permanently closed one without a second run.
    """
    device = storage.pick_device(cfg.device)
    torch.manual_seed(cfg.seed + 1000 + prune_step)
    task = build_task(cfg, cfg.seed)
    if getattr(task, "is_text", False):
        cfg.model.vocab_size = task.vocab_size

    model = models.build(cfg.model, cfg.data)
    model.load_state_dict(state)
    model = model.to(device)

    names = S.param_names(model)
    shapes = [dict(model.named_parameters())[n].shape for n in names]
    prunable = S.flatten(S.prunable_mask(model, cfg.sens), names)
    layer_ids = S.layer_index(model, names)
    n_layers = len(names)
    unit_ids = (unit_group_ids(names, shapes, prunable, model)[prunable]
                if min_keep_fraction_unit > 0 else None)

    if criterion == "sensitivity":
        if flat_scores is None:
            batches, folds = task.sensitivity_batches()
            res = S.compute_sensitivity(model, batches, cfg.sens, device,
                                        fold_of_batch=folds)
            flat_scores = S.flatten(res.scores, names)
        masks = masks_from_scores(flat_scores, names, shapes, sparsity, prunable,
                                  layer_ids, n_layers, min_keep_fraction,
                                  unit_ids, min_keep_fraction_unit)
    elif criterion == "magnitude":
        mag = torch.cat([p.detach().abs().reshape(-1).cpu()
                         for p in model.parameters() if p.requires_grad])
        masks = masks_from_scores(mag, names, shapes, sparsity, prunable,
                                  layer_ids, n_layers, min_keep_fraction,
                                  unit_ids, min_keep_fraction_unit)
    elif criterion == "random":
        if flat_scores is None:
            raise ValueError("random criterion needs sensitivity scores to match budgets")
        masks = budget_matched_random(flat_scores, layer_ids, n_layers, sparsity,
                                      prunable, names, shapes, seed=cfg.seed,
                                      min_keep_fraction=min_keep_fraction,
                                      unit_ids=unit_ids,
                                      min_keep_fraction_unit=min_keep_fraction_unit)
    else:
        raise ValueError(criterion)

    dev_masks = {n: m.to(device) for n, m in masks.items()}
    apply_masks(model, dev_masks)
    kept = sum(int(m.sum()) for m in masks.values())
    total = sum(int(m.numel()) for m in masks.values())

    end_step = prune_step + finetune_steps if finetune_steps is not None else cfg.train.steps
    if end_step <= prune_step:
        raise ValueError(f"no fine-tuning budget: prune_step={prune_step} >= end_step={end_step}")

    opt = build_optimizer(model, cfg.train)
    trajectory: List[Dict[str, float]] = []
    n_ft = end_step - prune_step
    for step in range(prune_step, end_step):
        model.train()
        # LR schedule is keyed to elapsed fine-tune steps, not absolute step, so every
        # prune time gets the same warmup/decay shape over its own budget rather than
        # inheriting whatever phase of the original schedule it happens to land in.
        for g in opt.param_groups:
            g["lr"] = lr_at(step - prune_step, cfg.train) if finetune_steps is not None                 else lr_at(step, cfg.train)
        x, y = task.train_batch()
        loss = task.loss(model(x.to(device)), y.to(device), cfg.train.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        apply_masks(model, dev_masks)   # keep the mask enforced after the update

        if eval_every and ((step - prune_step + 1) % eval_every == 0):
            ev = task.evaluate(model, device)
            trajectory.append({"finetune_step": step - prune_step + 1, **ev})

    out = task.evaluate(model, device)
    n_dead = sum(1 for n in masks if int(masks[n].numel()) > 0 and int(masks[n].sum()) == 0)
    n_dead_units = 0
    if unit_ids is not None:
        # unit_ids is already restricted to prunable entries by the caller (see
        # `unit_group_ids(...)[prunable]`), but its value range can still include ids that
        # own zero prunable weights (e.g. a bias/head unit excluded from pruning
        # entirely) -- counting over 0..max() would flag those as "dead" despite there
        # being nothing there to prune, which is what produced the spurious
        # dead_units=2 on the MLP run. Restrict to ids that actually occur.
        present = unit_ids.unique()
        flat_keep = S.flatten(masks, names)[prunable]
        kept_per_unit = torch.zeros(int(unit_ids.max()) + 1, dtype=torch.long).index_add_(
            0, unit_ids, flat_keep.long())
        n_dead_units = int((kept_per_unit[present] == 0).sum())
    out.update({"prune_step": prune_step, "sparsity": sparsity, "criterion": criterion,
                "kept_fraction": kept / max(1, total), "finetune_steps": n_ft,
                "dead_layers": n_dead, "dead_units": n_dead_units,
                "min_keep_fraction": min_keep_fraction,
                "min_keep_fraction_unit": min_keep_fraction_unit})
    if trajectory:
        out["trajectory"] = trajectory
    if verbose:
        print(f"    prune@{prune_step:<6} ft={n_ft:<6} {criterion:12s} sp={sparsity:<5} "
              f"acc={out['test_acc']:.4f} dead_layers={n_dead} dead_units={n_dead_units}")
    return out
