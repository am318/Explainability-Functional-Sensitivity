"""E11's missing piece: a dense (unpruned) control trained under the IDENTICAL budget.

Every accuracy number reported for E11 so far was compared against a "dense baseline" of
0.585 -- but that number came from a DIFFERENT experiment (E7: MLP, 20000 steps, constant
LR), not from training the same architecture for E11's actual budget (8000/16000 steps,
cosine LR, same seed). That mismatch is the likely explanation for magnitude appearing to
"exceed" the dense baseline at sp=0.9 (0.593 vs the wrong reference of 0.585): the
comparison point was wrong, not a real effect of pruning.

This trains the unpruned model under EXACTLY the config each E11 arch/budget uses (same
`base()` call, same seed, same lr_schedule, same step count) and evaluates. This is the only
valid "sp=0" reference point for every table involving E11.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fsd import storage
from fsd.tasks import build_task
from fsd.run import build_optimizer, _ensure_cudnn_usable
from fsd.schedule import lr_at
from fsd import models
from experiments._common import base
from experiments.e11_prune_probe import PLAN


def dense_control(arch: str, seed: int = 0) -> dict:
    plan = PLAN[arch]
    ft = plan["ft"]
    cfg = base("e11dense", arch, "cifar10", steps=ft, sens_samples=2048)
    cfg.seed = seed
    cfg.train.lr_schedule = "cosine"   # matches e11_prune_probe.py exactly
    cfg.train.steps = ft

    device = storage.pick_device(cfg.device)
    _ensure_cudnn_usable(device)
    torch.manual_seed(cfg.seed)
    task = build_task(cfg, cfg.seed)
    model = models.build(cfg.model, cfg.data).to(device)
    opt = build_optimizer(model, cfg.train)

    losses = []
    for step in range(ft):
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
        if step % max(1, ft // 10) == 0 or step == ft - 1:
            losses.append((step, float(loss.detach())))

    out = task.evaluate(model, device)
    out["arch"] = arch
    out["finetune_steps"] = ft
    out["loss_trace"] = losses
    print(f"  [{arch}] DENSE CONTROL (same {ft}-step, cosine-LR budget as E11): "
          f"acc={out['test_acc']:.4f}  loss {losses[0][1]:.3f}->{losses[-1][1]:.3f}")
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="mlp,resnet20,vit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_path = Path("results/_probe/dense_control.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume: keep any arch already recorded, only train the ones that are missing.
    results = json.load(open(out_path)) if out_path.exists() else {}
    for arch in args.archs.split(","):
        if arch in results:
            print(f"  [{arch}] dense control already recorded -- skipping")
            continue
        results[arch] = dense_control(arch, args.seed)
        json.dump(results, open(out_path, "w"), indent=2)  # checkpoint after each arch
