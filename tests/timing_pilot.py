"""Wall-clock budget for one full-scale run, so the sweep sizes are chosen with numbers
rather than optimism."""
import sys, time
sys.path.insert(0, ".")
from experiments._common import base
from fsd.run import execute

cfg = base("timing", "vit", steps=1000)
cfg.data.dataset = "synthetic"; cfg.data.workers = 0; cfg.data.test_subset = 1000
cfg.n_ckpts = 10
t0 = time.time()
m = execute(cfg)
print(f"TOTAL {time.time()-t0:.0f}s for {cfg.train.steps} steps + {cfg.n_ckpts} checkpoints")
