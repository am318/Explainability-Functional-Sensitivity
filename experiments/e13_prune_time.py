"""When is the earliest a mask can be taken without paying an accuracy cost?

E11 only tests prune@0 (initialisation) -- a real gap flagged directly: we don't know
whether pruning is too early, too late, or about right, because nothing at intermediate
prune times has been tested on valid (floor-corrected) masks. E3's step-9/400/8000 arms
used degenerate masks and are retracted.

This adds prune points at LOW FRACTIONS of the training budget (not just t=0), holding the
fine-tune budget FIXED across prune times so the comparison isolates "how good is the mask
taken at step X" from "how much total compute did this arm get" -- a later prune point does
NOT get more total training, it gets the SAME recovery budget starting from a
slightly-trained network instead of a random one.

MLP first (cheapest, and prune@0/sp=0.9 data already exists from E11 for a free
comparison point). Prune fractions chosen close to the "early" regime this whole project's
sensitivity-dynamics results (E1/E7) found most active: 1% and 5% of the fine-tune budget.
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
from fsd.run import build_optimizer
from fsd.schedule import lr_at
from fsd.tasks import build_task
from experiments._common import base
from experiments.e11_prune_probe import PLAN


def run(arch: str, prune_fractions, sparsities, criteria, seed: int, min_keep: float,
       out_dir: Path) -> list:
    plan = PLAN[arch]
    ft = plan["ft"]
    cfg = base("e13", arch, "cifar10", steps=ft, sens_samples=2048)
    cfg.seed = seed
    cfg.train.lr_schedule = "cosine"    # matches e11_prune_probe.py / the dense control
    cfg.train.steps = ft

    device = storage.pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    task = build_task(cfg, cfg.seed)
    model = models.build(cfg.model, cfg.data).to(device)
    names = S.param_names(model)
    shapes = [dict(model.named_parameters())[n].shape for n in names]
    prunable = S.flatten(S.prunable_mask(model, cfg.sens), names)
    unit_ids = unit_group_ids(names, shapes, prunable, model)[prunable]

    prune_steps = sorted({int(round(f * ft)) for f in prune_fractions})
    print(f"  [{arch}] prune points at steps {prune_steps} "
          f"({[f'{100*s/ft:.1f}%' for s in prune_steps]} of the {ft}-step budget)")

    opt = build_optimizer(model, cfg.train)
    rows = []
    next_i = 0
    t0 = time.time()
    for step in range(max(prune_steps) + 1):
        if next_i < len(prune_steps) and prune_steps[next_i] == step:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            batches, folds = task.sensitivity_batches()
            res = S.compute_sensitivity(model, batches, cfg.sens, device,
                                        fold_of_batch=folds)
            flat = S.flatten(res.scores, names)
            print(f"    step {step}: S computed at {time.time()-t0:.0f}s elapsed")
            for sp in sparsities:
                for crit in criteria:
                    torch.manual_seed(cfg.seed + 1000 + step)
                    r = prune_and_continue(cfg, step, sp, crit, state, flat_scores=flat,
                                           finetune_steps=ft, eval_every=None,
                                           min_keep_fraction=min_keep,
                                           min_keep_fraction_unit=0.05)
                    r["arch"] = arch
                    r["prune_frac_of_budget"] = step / ft
                    rows.append(r)
                    json.dump(rows, open(out_dir / f"e13_{arch}_seed{seed}.json", "w"),
                              indent=2)
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
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", default="mlp")
    ap.add_argument("--fractions", default="0.01,0.05")
    ap.add_argument("--sparsities", default="0.9")
    ap.add_argument("--criteria", default="sensitivity,random,magnitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-keep", type=float, default=0.01)
    ap.add_argument("--out", default="results/_probe")
    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",")]
    sps = [float(x) for x in args.sparsities.split(",")]
    crits = args.criteria.split(",")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for arch in args.archs.split(","):
        print(f"\n=== {arch} prune-time sweep (min_keep={args.min_keep}) ===")
        run(arch, fracs, sps, crits, args.seed, args.min_keep, out)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
