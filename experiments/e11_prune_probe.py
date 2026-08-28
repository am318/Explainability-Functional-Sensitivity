"""E11 -- why did sensitivity pruning lose? Three candidate explanations, tested.

E3 found sensitivity pruning losing to both magnitude and a layer-budget-matched random
baseline. Before treating that as a fact about the criterion, three things had to be ruled
out. The first turned out to be real and invalidates E3's high-sparsity rows outright.

  1. DEGENERATE MASKS. Unconstrained global top-k emptied 11-12 of ResNet-20's 21 conv
     layers at 99% sparsity, for every criterion. A disconnected network scores at chance
     regardless of ranking quality, so E3's sp=0.99 comparison measured which disconnection
     pattern the residual paths tolerate. Fixed here by a per-layer floor
     (`min_keep_fraction`), which preserves the global budget exactly.

  2. TOO FEW ITERATIONS. If sensitivity-pruned networks simply recover more slowly, the gap
     would close given more fine-tuning. Every run gets an identical, generous budget and is
     evaluated periodically, so a still-climbing curve is distinguishable from a converged
     one -- which a single end-point number cannot show.

  3. ARCHITECTURE-SPECIFIC. ResNet-20 has residual paths that make a dead conv layer
     harmless (it degrades to identity), which flatters criteria that happen to kill whole
     layers. MLP has no such escape route and ViT has a different one, so running all three
     separates "sensitivity is a weak criterion" from "ResNet's skip connections reward
     depth reduction".

Prune time is fixed at step 0 (initialisation): that is the zero-shot case the companion
pruning work depends on, and the case E3 found most damning.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fsd import config as C, models, sensitivity as S, storage
from fsd.prune import unit_group_ids
from fsd.prune import prune_and_continue
from fsd.tasks import build_task
from experiments._common import base

PLAN = {
    "mlp":      dict(ft=8000,  sparsities=[0.9, 0.99], eval_every=1000),
    "resnet20": dict(ft=16000, sparsities=[0.9, 0.99], eval_every=2000),
    "vit":      dict(ft=8000,  sparsities=[0.9],       eval_every=2000),
}


def probe(arch: str, seed: int, min_keep: float, out_dir: Path,
         sparsities=None) -> list:
    plan = PLAN[arch]
    ft = plan["ft"]
    sparsities = plan["sparsities"] if sparsities is None else sparsities
    cfg = base("e11", arch, "cifar10", steps=ft, sens_samples=2048)
    cfg.seed = seed
    cfg.train.lr_schedule = "cosine"
    cfg.train.steps = ft

    device = storage.pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    task = build_task(cfg, cfg.seed)
    model = models.build(cfg.model, cfg.data).to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    names = S.param_names(model)

    batches, folds = task.sensitivity_batches()
    t0 = time.time()
    res = S.compute_sensitivity(model, batches, cfg.sens, device, fold_of_batch=folds)
    flat = S.flatten(res.scores, names)
    shapes = [dict(model.named_parameters())[n].shape for n in names]
    prunable = S.flatten(S.prunable_mask(model, cfg.sens), names)
    unit_ids = unit_group_ids(names, shapes, prunable, model)[prunable]
    print(f"  [{arch}] S at init in {time.time()-t0:.0f}s; fine-tune budget {ft} steps")

    rows = []
    for sp in sparsities:
        for crit in ("sensitivity", "random", "magnitude"):
            torch.manual_seed(cfg.seed)
            r = prune_and_continue(cfg, 0, sp, crit, init_state, flat_scores=flat,
                                   finetune_steps=ft, eval_every=plan["eval_every"],
                                   min_keep_fraction=min_keep,
                                   min_keep_fraction_unit=0.05, )
            r["arch"] = arch
            r["seed"] = seed
            rows.append(r)
            tag = "_".join(f"{s:g}" for s in sparsities)
            json.dump(rows, open(out_dir / f"probe_{arch}_seed{seed}_sp{tag}.json", "w"),
                      indent=2)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", default="mlp,resnet20,vit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-keep", type=float, default=0.01)
    ap.add_argument("--out", default="results/_probe")
    ap.add_argument("--sparsities", default=None,
                    help="override the per-arch default sparsity list, e.g. '0.5'")
    args = ap.parse_args()

    sps = ([float(x) for x in args.sparsities.split(",")]
          if args.sparsities else None)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for arch in args.archs.split(","):
        print(f"\n=== {arch} (min_keep={args.min_keep}"
              f"{f', sp={sps}' if sps else ''}) ===")
        probe(arch, args.seed, args.min_keep, out, sparsities=sps)
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
