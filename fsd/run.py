"""One run: train a network, measure the sensitivity ordering along the way, and emit the
analysis-ready metric table.

The measurement schedule is log-spaced because everything the paper claims happens in the
first few hundred steps. Each checkpoint produces:

  * S_t, estimated on a *fixed* label-free input set split into disjoint folds;
  * the between-fold comparison at that same checkpoint -- the noise floor (C2a);
  * the empirical trace-NTK Gram, for kernel velocity (C4/C5.1).

Fold vectors are compared and discarded immediately; only the pooled S_t is retained, so
peak memory is (n_ckpts x n_params) floats rather than three times that.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from . import config as C
from . import criteria as CR, models, rank_metrics as R, sensitivity as S, storage, theory as T
from .schedule import log_checkpoints, lr_at
from .tasks import build_task


def build_optimizer(model: nn.Module, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))


def execute(cfg: C.RunCfg, verbose: bool = True) -> Dict[str, object]:
    device = storage.pick_device(cfg.device)
    torch.manual_seed(cfg.seed)

    data_seed = cfg.seed if cfg.data_seed < 0 else cfg.data_seed
    task = build_task(cfg, cfg.seed, data_seed)
    if getattr(task, "is_text", False):
        cfg.model.vocab_size = task.vocab_size

    run = storage.Run(cfg.out_dir, cfg.run_id())
    C.dump(cfg, str(run.dir / "config.json"))

    model = models.build(cfg.model, cfg.data).to(device)
    names = S.param_names(model)
    layer_ids = S.layer_index(model, names)
    n_layers = len(names)
    prunable = S.flatten(S.prunable_mask(model, cfg.sens), names)
    n_params = int(prunable.numel())
    n_prunable = int(prunable.sum())

    # Lazy scaling multiplies the function by alpha, so gradients scale by alpha and the
    # effective step by alpha^2. Dividing the LR restores a comparable trajectory length --
    # otherwise the alpha sweep would confound laziness with learning rate.
    if cfg.model.lazy_alpha != 1.0:
        cfg.train.lr = cfg.train.lr / (cfg.model.lazy_alpha ** 2)
        cfg.train.min_lr = cfg.train.min_lr / (cfg.model.lazy_alpha ** 2)

    opt = build_optimizer(model, cfg.train)
    ckpts = log_checkpoints(cfg.train.steps, cfg.n_ckpts, cfg.ckpt_first)
    save_states = set(ckpts) if cfg.save_state_at == [-1] else set(cfg.save_state_at)

    sens_batches, fold_ids = task.sensitivity_batches()
    ntk_batch = task.ntk_batch()

    history: List[torch.Tensor] = []      # pooled S_t, restricted to prunable entries
    full_history: List[torch.Tensor] = [] # pooled S_t over all params (for layer budgets)
    weights: List[torch.Tensor] = []      # theta_t, for the Lipschitz drift bound
    grams: List[Optional[torch.Tensor]] = []
    crit_history: Dict[str, List[torch.Tensor]] = {}   # competing criteria
    struct_history: List[torch.Tensor] = []            # per-output-unit ranking
    struct_groups = CR.structured_groups(model) if cfg.track_structured else {}
    noise_floor: List[Dict[str, object]] = []
    measured_steps: List[int] = []

    if verbose:
        print(f"[{run.run_id}] {cfg.model.arch} on {cfg.data.dataset} | "
              f"{n_params/1e6:.2f}M params ({n_prunable/1e6:.2f}M rankable) | "
              f"{len(ckpts)} checkpoints | device={device}")

    def measure(step: int) -> None:
        t0 = time.time()
        res = S.compute_sensitivity(model, sens_batches, cfg.sens, device,
                                    ntk_batch=ntk_batch, fold_of_batch=fold_ids)
        pooled = S.flatten(res.scores, names)
        full_history.append(pooled)
        history.append(pooled[prunable].clone())
        weights.append(torch.cat([p.detach().reshape(-1).cpu()
                                  for p in model.parameters() if p.requires_grad]))
        grams.append(res.ntk)

        if cfg.track_structured and struct_groups:
            struct_history.append(CR.aggregate_structured(res.scores, struct_groups))

        # Competing criteria on the same inputs at the same checkpoint, so any difference
        # in their stability curves is a property of the criterion, not of the evaluation.
        if cfg.track_criteria:
            lab = [task.train_batch() for _ in range(max(1, cfg.sens.n_samples // 256))]
            shape = tuple(lab[0][0].shape[1:])
            for name, sc in CR.all_criteria(model, res.scores, lab, device, task, shape,
                                            getattr(task, "is_text", False)).items():
                crit_history.setdefault(name, []).append(
                    S.flatten(sc, names)[prunable].clone())

        # C2a: agreement between disjoint data folds at THIS checkpoint upper-bounds any
        # agreement we can claim between different checkpoints.
        if len(res.fold_scores) >= 2:
            fa = S.flatten(res.fold_scores[0], names)[prunable]
            fb = S.flatten(res.fold_scores[1], names)[prunable]
            floor = R.compare(fa, fb, layer_ids[prunable], n_layers, cfg.sparsities)
        else:
            floor = {}
        floor["step"] = step
        noise_floor.append(floor)
        measured_steps.append(step)

        if cfg.keep_scores == "all":
            run.save_scores(step, pooled,
                            [S.flatten(f, names) for f in res.fold_scores], res.ntk)
        if step in save_states:
            run.save_state(step, model)
        if verbose:
            print(f"  step {step:>6}  sensitivity measured in {time.time()-t0:5.1f}s"
                  f"  (estimator={res.meta['estimator']}, d={res.meta['out_dim']})")

    step = 0
    if 0 in ckpts:
        measure(0)
    remaining = [c for c in ckpts if c > 0]

    t_train = time.time()
    while step < cfg.train.steps:
        model.train()
        lr = lr_at(step, cfg.train)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = task.train_batch()
        x, y = x.to(device), y.to(device)
        loss = task.loss(model(x), y, cfg.train.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        step += 1

        if step % max(1, cfg.train.steps // 50) == 0 or step == cfg.train.steps:
            run.append_jsonl("train_log.jsonl",
                             {"step": step, "loss": float(loss.detach()), "lr": lr})
        if remaining and step == remaining[0]:
            remaining.pop(0)
            measure(step)

    train_time = time.time() - t_train
    final_eval = task.evaluate(model, device)
    if verbose:
        print(f"  trained {cfg.train.steps} steps in {train_time:.0f}s | {final_eval}")

    metrics = analyse(cfg, measured_steps, history, full_history, grams, noise_floor,
                      layer_ids[prunable], n_layers, prunable, weights)
    if crit_history:
        metrics["criteria"] = compare_criteria(cfg, measured_steps, crit_history,
                                               layer_ids[prunable], n_layers)
    if struct_history:
        metrics["structured"] = compare_structured(cfg, measured_steps, struct_history)
    metrics["layerwise"] = compare_layerwise(cfg, measured_steps, full_history,
                                             layer_ids, n_layers, names)
    metrics.update({
        "run_id": run.run_id,
        "config": C.to_dict(cfg),
        "final_eval": final_eval,
        "n_params": n_params,
        "n_prunable": n_prunable,
        "train_seconds": train_time,
        "device": str(device),
    })
    run.write_json("metrics.json", metrics)
    return metrics


def analyse(cfg, steps, history, full_history, grams, noise_floor, layer_ids, n_layers,
            prunable, weights=None) -> Dict[str, object]:
    """Everything the figures read. Computed once, at the end, with S_final as reference."""
    final = history[-1]
    gram_final = grams[-1]

    vs_final, consecutive, laziness, theory_rows = [], [], [], []
    for i, (step, s_t) in enumerate(zip(steps, history)):
        cmp_final = R.compare(s_t, final, layer_ids, n_layers, cfg.sparsities,
                              with_kendall=(i % 3 == 0 or i == len(steps) - 1))
        cmp_final["step"] = step
        # AUROC/calibration: the practically-relevant reframing of "does S_t predict
        # S_final" as a binary-classification / importance-band question rather than a
        # ranking one -- see NARRATIVE.md 2026-08-28 "positive-evidence sweep". Computed
        # at every checkpoint (cheap: O(p log p)) so every run gets this for free.
        cmp_final["auroc"] = {f"{sp:g}": R.auroc_topk(s_t, final, sp) for sp in cfg.sparsities}
        cmp_final["calibration"] = R.calibration_deciles(s_t, final)
        vs_final.append(cmp_final)

        if i > 0:
            cmp_prev = R.compare(history[i - 1], s_t, layer_ids, n_layers, cfg.sparsities,
                                 with_kendall=False)
            cmp_prev["step"] = step
            cmp_prev["prev_step"] = steps[i - 1]
            consecutive.append(cmp_prev)

        row = {"step": step}
        if grams[i] is not None and gram_final is not None:
            row["kernel_velocity_to_final"] = T.kernel_velocity(grams[i], gram_final)
            row["kernel_alignment_to_final"] = T.kernel_alignment(grams[i], gram_final)
            if i > 0 and grams[i - 1] is not None:
                row["kernel_velocity_step"] = T.kernel_velocity(grams[i - 1], grams[i])
        laziness.append(row)

        # C5: two predictions of the same measured curve.
        #   (i) point prediction from one scalar (the drift scale), via the MC drift model;
        #   (ii) rigorous lower bound from the Lipschitz/counting argument.
        sigma = T.drift_scale(s_t, final)
        drift = T.amplitude_drift(s_t, final)
        trow = {"step": step, "drift_sigma": sigma,
                "hill_alpha": T.hill_tail_index(s_t),
                "amplitude_drift": drift}
        if weights is not None:
            trow["weight_distance_to_final"] = T.weight_distance(weights[i], weights[-1])
            trow["weight_distance_from_init"] = T.weight_distance(weights[i], weights[0])
        for sp in cfg.sparsities:
            key = f"{sp:g}"
            trow[f"pred_overlap_{key}"] = T.predicted_overlap(final, sigma, sp)
            trow[f"obs_overlap_{key}"] = cmp_final["topk"][key]["overlap"]
            trow[f"bound_overlap_{key}"] = T.counting_bound(
                final, drift["quantile"], sp)["overlap_lower_bound"]
            trow[f"gap_{key}"] = T.boundary_gap(s_t, sp)
        theory_rows.append(trow)

    # t*, defined without a free threshold:
    #
    #   t* = the first step after which *further training* moves the ordering less than
    #        *resampling the estimation data* does.
    #
    # The comparison quantity is the between-fold agreement at the same checkpoint,
    # Spearman-Brown corrected to full sample size (the folds are half-size; the reported
    # S_t pools both, so the raw fold number understates the reliability of what we use).
    # This makes "the ordering is frozen" an operational statement rather than a choice of
    # tau, and it is the trajectory analogue of the estimation-error lemma: the two error
    # sources are put on the same axis and we report where they cross.
    tstar = {}
    for sp in cfg.sparsities:
        key = f"{sp:g}"
        vals = [row["topk"][key]["adjusted"] for row in vs_final]
        ceils = [T.spearman_brown(row["topk"][key]["adjusted"])
                 for row in noise_floor if row.get("topk")]
        ceiling = float(torch.tensor(ceils, dtype=torch.float64).median()) if ceils else None
        tstar[f"sp{key}_crossing"] = R.stabilisation_step(steps, vals, 1.0, ceiling)
        for tau in (0.8, 0.9, 0.95):
            tstar[f"sp{key}_tau{tau}"] = R.stabilisation_step(steps, vals, tau, ceiling)
        tstar[f"sp{key}_ceiling"] = ceiling
        tstar[f"sp{key}_final_adjusted"] = vals[-2] if len(vals) > 1 else float("nan")

    return {
        "steps": steps,
        "sparsities": list(cfg.sparsities),
        "pairwise": pairwise_agreement(cfg, steps, history, layer_ids, n_layers),
        "vs_final": vs_final,
        "consecutive": consecutive,
        "noise_floor": noise_floor,
        "laziness": laziness,
        "theory": theory_rows,
        "tstar": tstar,
        "spectrum_final": {
            "hill_alpha": T.hill_tail_index(final),
            "log_quantiles": [float(x) for x in
                              T.log_scores(final).quantile(
                                  torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999],
                                               dtype=torch.float64))],
        },
    }


def compare_criteria(cfg, steps, crit_history, layer_ids, n_layers) -> Dict[str, object]:
    """Stability curve for each importance criterion, computed identically.

    If the functional ranking is not measurably more stable than the loss-based ones, the
    paper's emphasis on *functional* sensitivity is decoration and we should say so.
    """
    out = {}
    for name, hist in crit_history.items():
        final = hist[-1]
        rows = []
        for step, v in zip(steps, hist):
            cmp = R.compare(v, final, layer_ids, n_layers, cfg.sparsities,
                            with_kendall=False)
            cmp["step"] = step
            rows.append(cmp)
        key = f"{cfg.sparsities[len(cfg.sparsities)//2]:g}"
        vals = [r["topk"][key]["adjusted"] for r in rows]
        out[name] = {
            "vs_final": rows,
            "tstar_tau0.9": R.stabilisation_step(steps, vals, 0.9),
            "final_adjusted": vals[-2] if len(vals) > 1 else float("nan"),
        }
    return out


def compare_structured(cfg, steps, struct_history) -> Dict[str, object]:
    """Same measurement at the level of whole output units (heads, channels, neurons).

    Structured sparsity is what actually produces speedups; the group score is the sum of
    S over the group, so this needs no extra measurement. Chance correction is unnecessary
    here -- groups are few enough that the layer confound does not dominate -- so we report
    plain overlap and Spearman.
    """
    final = struct_history[-1]
    n = final.numel()
    rows = []
    for step, v in zip(steps, struct_history):
        row = {"step": step, "spearman": R.spearman(v, final), "n_units": int(n)}
        for sp in cfg.sparsities:
            row[f"overlap_{sp:g}"] = R.overlap(R.topk_mask(v, sp), R.topk_mask(final, sp))
        rows.append(row)
    key = f"{cfg.sparsities[len(cfg.sparsities)//2]:g}"
    vals = [r[f"overlap_{key}"] for r in rows]
    return {"vs_final": rows, "tstar_tau0.9": R.stabilisation_step(steps, vals, 0.9)}


def pairwise_agreement(cfg, steps, history, layer_ids, n_layers) -> Dict[str, object]:
    """Agreement between *every* pair of checkpoints, not just each against the last.

    Comparing S_t to S_final has a built-in artefact: the curve is pinned to 1.0 at t=T, so
    a monotonic rise is equally consistent with "the ordering froze" and with "the ordering
    is still moving and t is simply getting closer to T". The full matrix separates them.

    If the ordering freezes at t*, agreement between any two checkpoints after t* is high
    regardless of how far apart they are -- the matrix has a large saturated block in its
    bottom-right corner. If the ordering churns at a constant rate in log-time, agreement
    depends only on the *ratio* t'/t and no such block appears, however long training runs.

    Masks are computed once per (checkpoint, sparsity) and reused across all pairs, so the
    whole matrix costs little more than the diagonal did.
    """
    out: Dict[str, object] = {"steps": steps}
    sizes = torch.bincount(layer_ids.long(), minlength=n_layers).double().clamp(min=1)
    for sp in cfg.sparsities:
        key = f"{sp:g}"
        masks = [R.topk_mask(v, sp) for v in history]
        counts = [R.layer_counts(m, layer_ids, n_layers) for m in masks]
        k = float(masks[0].sum())
        n = len(masks)
        obs = [[0.0] * n for _ in range(n)]
        adj = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                o = R.overlap(masks[i], masks[j])
                chance = float((counts[i] * counts[j] / sizes).sum()) / max(1.0, k)
                a = (o - chance) / (1.0 - chance) if 1.0 - chance > 1e-12 else float("nan")
                obs[i][j] = obs[j][i] = o
                adj[i][j] = adj[j][i] = a
        out[key] = {"overlap": obs, "adjusted": adj}
    return out


def compare_layerwise(cfg, steps, full_history, layer_ids, n_layers, names) -> Dict[str, object]:
    """Stability of the *layer* ordering -- the coarsest granularity there is.

    Ranking a few dozen parameter tensors by their total sensitivity is a far easier
    problem than ranking a few million parameters, and it is the granularity at which a
    layerwise sparsity allocator operates. Reporting it alongside the component-level and
    parameter-level orderings gives a granularity ladder: if stability is a matter of how
    coarsely you look, the ladder shows exactly where the transition happens, and a pruning
    method can be positioned on the correct rung rather than assumed onto one.

    Both total and mean sensitivity per tensor are reported: total scales with tensor size
    and is what a global top-k pruner implicitly uses, whereas mean is size-independent and
    is what "which layer matters most" usually means.
    """
    def per_layer(v: torch.Tensor):
        idx = layer_ids.long()
        total = torch.zeros(n_layers, dtype=torch.float64).index_add_(
            0, idx, v.double())
        counts = torch.bincount(idx, minlength=n_layers).double().clamp(min=1)
        return total, total / counts

    finals = per_layer(full_history[-1])
    rows = []
    for step, v in zip(steps, full_history):
        tot, mean = per_layer(v)
        row = {"step": step,
               "spearman_total": R.spearman(tot.float(), finals[0].float()),
               "spearman_mean": R.spearman(mean.float(), finals[1].float())}
        # top-half agreement: which layers are in the more-sensitive half
        for label, (cur, fin) in (("total", (tot, finals[0])), ("mean", (mean, finals[1]))):
            k = max(1, n_layers // 2)
            a = torch.zeros(n_layers, dtype=torch.bool)
            a[torch.topk(cur.float(), k).indices] = True
            b = torch.zeros(n_layers, dtype=torch.bool)
            b[torch.topk(fin.float(), k).indices] = True
            row[f"tophalf_{label}"] = float((a & b).sum()) / k
        rows.append(row)
    vals = [r["spearman_mean"] for r in rows]
    return {"n_layers": n_layers, "vs_final": rows,
            "tstar_rho0.9": R.stabilisation_step(steps, vals, 0.9),
            "tstar_rho0.95": R.stabilisation_step(steps, vals, 0.95)}
