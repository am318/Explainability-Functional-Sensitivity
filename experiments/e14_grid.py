"""The final results grid: prune@ x sparsity x method, for the two questions that matter.

    prune@   in {0%, 10%, 25%, 50%} of the total training budget
    sparsity in {20%, 50%, 70%, 90%}
    method   in {sensitivity, random, magnitude}

16 (prune, sparsity) cells x 3 methods = 48 accuracy numbers per architecture, all against
a dense control trained under the identical budget/schedule/seed.

One continuous training trajectory is used per architecture: the model trains once,
checkpointing (state + sensitivity scores) at each prune fraction along the way, then
branches into all 12 (sparsity, method) arms from that single checkpoint with a FIXED
fine-tune budget equal to the architecture's full step count. A later prune point does not
get more total compute -- it gets the same recovery budget starting from a
slightly-trained network instead of a random one, which is what isolates "how good is the
mask taken at step X" from "how much compute did this arm get".

Masks use both structural floors throughout (layer 1%, unit 5%; see fsd/prune.py and the
Sec 6 methodological finding) -- unconstrained top-k pruning collapses structure at
sparsities well inside this grid's range and would make the comparison meaningless.

Already-computed cells (from E11/E11b/E13) are detected by (prune_step, sparsity,
criterion) and skipped, so re-running this script is free to resume.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fsd import config as C, models, sensitivity as S, storage
from fsd.prune import prune_and_continue, unit_group_ids
from fsd.run import build_optimizer, _ensure_cudnn_usable
from fsd.schedule import lr_at
from fsd.tasks import build_task
from experiments._common import base

# Fine-tune budget per architecture -- unchanged from E11/E13 so existing cells are
# directly reusable without re-deriving them at a different budget.
FT = {"mlp": 8000, "resnet20": 16000, "vit": 8000}

# Full spec is 4x4 (prune x sparsity); ResNet-20's per-cell cost (~19min) makes the full
# 48-cell grid an ~15h commitment, too close to the deadline to risk. Reduced to 3x3=9
# (prune,sparsity) cells x 3 methods = 27, keeping both axes' endpoints and midpoint so
# the grid shape is still legible -- ~8.4h, fits an overnight run with margin.
PRUNE_FRACS = [0.0, 0.2, 0.40, 0.60]
SPARSITIES = [0.20, 0.40, 0.60, 80]
CRITERIA = ["sensitivity", "random", "magnitude"]


def existing_cells(out_path: Path) -> set:
    """(prune_step, sparsity, criterion) tuples already recorded in this grid's output."""
    if not out_path.exists():
        return set()
    rows = json.load(open(out_path))
    return {(r["prune_step"], r["sparsity"], r["criterion"]) for r in rows}


def run(arch: str, seed: int, min_keep: float, out_dir: Path,
       prune_fracs=None, sparsities=None) -> None:
    ft = FT[arch]
    prune_fracs = DEFAULT_PRUNE_FRACS if prune_fracs is None else prune_fracs
    sparsities = DEFAULT_SPARSITIES if sparsities is None else sparsities
    prune_steps = sorted({int(round(f * ft)) for f in prune_fracs})
    out_path = out_dir / f"e14_{arch}_seed{seed}.json"
    rows = json.load(open(out_path)) if out_path.exists() else []
    done = existing_cells(out_path)

    cfg = base("e14", arch, "cifar10", steps=ft, sens_samples=2048)
    cfg.seed = seed
    cfg.train.lr_schedule = "cosine"
    cfg.train.steps = ft

    device = storage.pick_device(cfg.device)
    _ensure_cudnn_usable(device)
    torch.manual_seed(cfg.seed)
    task = build_task(cfg, cfg.seed)
    model = models.build(cfg.model, cfg.data).to(device)
    names = S.param_names(model)
    shapes = [dict(model.named_parameters())[n].shape for n in names]
    prunable = S.flatten(S.prunable_mask(model, cfg.sens), names)
    unit_ids = unit_group_ids(names, shapes, prunable, model)[prunable]

    print(f"\n[{arch}] grid: prune steps {prune_steps} "
          f"({[f'{100*s/ft:.0f}%' for s in prune_steps]} of {ft}) "
          f"x sparsities {sparsities} x {CRITERIA}")
    n_total = len(prune_steps) * len(sparsities) * len(CRITERIA)
    n_skip = sum(1 for ps in prune_steps for sp in sparsities for c in CRITERIA
                 if (ps, sp, c) in done)
    print(f"  {n_skip}/{n_total} cells already computed, resuming the rest")

    opt = build_optimizer(model, cfg.train)
    next_i = 0
    t0 = time.time()
    for step in range(max(prune_steps) + 1):
        if next_i < len(prune_steps) and prune_steps[next_i] == step:
            cells_here = [(sp, c) for sp in sparsities for c in CRITERIA
                          if (step, sp, c) not in done]
            if cells_here:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                batches, folds = task.sensitivity_batches()
                res = S.compute_sensitivity(model, batches, cfg.sens, device,
                                            fold_of_batch=folds)
                flat = S.flatten(res.scores, names)
                print(f"  step {step} ({time.time()-t0:.0f}s elapsed): "
                      f"{len(cells_here)} cells to fill")
                for sp, crit in cells_here:
                    torch.manual_seed(cfg.seed + 1000 + step)
                    r = prune_and_continue(cfg, step, sp, crit, state, flat_scores=flat,
                                           finetune_steps=ft, eval_every=None,
                                           min_keep_fraction=min_keep,
                                           min_keep_fraction_unit=0.05)
                    r["arch"] = arch
                    rows.append(r)
                    json.dump(rows, open(out_path, "w"), indent=2)
            next_i += 1
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train)
        x, y = task.train_batch()
        loss = task.loss(model(x.to(device)), y.to(device), cfg.train.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
    print(f"[{arch}] grid complete: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", default="mlp,resnet20,vit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-keep", type=float, default=0.01)
    ap.add_argument("--out", default="results/_probe")
    ap.add_argument("--prune-fracs", default=None,
                    help="comma-separated fractions of the ft budget, e.g. '0,0.1,0.25,0.5'"
                         " (default: reduced 3-point local grid)")
    ap.add_argument("--sparsities", default=None,
                    help="comma-separated sparsities, e.g. '0.2,0.5,0.7,0.9'"
                         " (default: reduced 3-point local grid)")
    args = ap.parse_args()
    fracs = ([float(x) for x in args.prune_fracs.split(",")]
            if args.prune_fracs else None)
    sps = ([float(x) for x in args.sparsities.split(",")]
          if args.sparsities else None)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for arch in args.archs.split(","):
        run(arch, args.seed, args.min_keep, out, prune_fracs=fracs, sparsities=sps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
