"""How many inputs does S(theta) need?

The between-fold agreement at a single checkpoint is the ceiling on every stability number
we report (C2a). If that ceiling is low, the paper's headline plateau is unmeasurable. This
sweeps the sensitivity-set size to find the smallest budget that puts the ceiling high
enough to leave room for a real signal, and it doubles as an appendix figure.
"""
import json, sys, time
sys.path.insert(0, ".")
import torch
from fsd import config as C, models, rank_metrics as R, sensitivity as S, storage
from fsd.tasks import build_task
from fsd.run import build_optimizer

cfg = C.RunCfg(tag="calib", out_dir="results/_smoke")
cfg.data = C.DataCfg(dataset="synthetic", image_size=32, augment=False, workers=0)
cfg.model = C.ModelCfg(arch="vit", width=192, depth=6, heads=3, patch_size=4)
cfg.train = C.TrainCfg(steps=200, batch_size=64, lr=1e-3, warmup_steps=20)
device = storage.pick_device(cfg.device)
torch.manual_seed(0)

task = build_task(cfg, 0)
model = models.build(cfg.model, cfg.data).to(device)
names = S.param_names(model)
layer_ids = S.layer_index(model, names)
n_layers = len(names)
opt = build_optimizer(model, cfg.train)
for step in range(cfg.train.steps):
    x, y = task.train_batch()
    loss = task.loss(model(x.to(device)), y.to(device))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"trained {cfg.train.steps} steps, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

from fsd.data import vision
_, clean, _ = vision.build(cfg.data, 0)
prunable = S.flatten(S.prunable_mask(model, cfg.sens), names)
lids = layer_ids[prunable]

rows = []
for n_samples in (64, 128, 256, 512, 1024, 2048):
    batches, folds = vision.sensitivity_batches(clean, n_samples, 32, 1234, 2)
    t0 = time.time()
    res = S.compute_sensitivity(model, batches, C.SensCfg(n_samples=n_samples, folds=2),
                                device, fold_of_batch=folds)
    dt = time.time() - t0
    fa = S.flatten(res.fold_scores[0], names)[prunable]
    fb = S.flatten(res.fold_scores[1], names)[prunable]
    cmp = R.compare(fa, fb, lids, n_layers, [0.5, 0.9, 0.99], with_kendall=False)
    row = {"n_samples": n_samples, "seconds": round(dt, 1), "spearman": round(cmp["spearman"], 4),
           "within": round(cmp["within_layer"]["weighted"], 4)}
    for sp in ("0.5", "0.9", "0.99"):
        row[f"adj@{sp}"] = round(cmp["topk"][sp]["adjusted"], 4)
        row[f"ovl@{sp}"] = round(cmp["topk"][sp]["overlap"], 4)
    rows.append(row)
    print(json.dumps(row))
json.dump(rows, open("results/_smoke/calibration.json", "w"), indent=2)
print("done")
