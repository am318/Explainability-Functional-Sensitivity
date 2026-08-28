"""Smoke test for the GPT / text path.

The vision settings use the exact estimator (d_y = 10). The GPT output is
(block_size x vocab), so it exercises the Hutchinson path, the text task, and the
label-free sensitivity definition on a non-classification model -- i.e. the parts of the
pipeline the vision runs never touch.
"""
import sys, time
sys.path.insert(0, ".")
from fsd import config as C
from fsd.run import execute

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "data/wikitext-2/train.txt"

cfg = C.RunCfg(tag="smoke-gpt", out_dir="results/_smoke")
cfg.model = C.ModelCfg(arch="gpt", width=96, depth=3, heads=3, block_size=64)
cfg.data = C.DataCfg(dataset="text", text_file=CORPUS, workers=0)
cfg.train = C.TrainCfg(steps=40, batch_size=16, lr=1e-3, warmup_steps=5)
cfg.sens = C.SensCfg(n_samples=32, batch_size=4, folds=2, ntk_examples=8,
                     n_probes=4, estimator="hutchinson")
cfg.n_ckpts, cfg.sparsities = 5, [0.5, 0.9]

t0 = time.time()
m = execute(cfg)
r = m["vs_final"][2]
print(f"  gpt  step={r['step']} rho={r['spearman']:.3f} "
      f"overlap@.9={r['topk']['0.9']['overlap']:.3f} "
      f"adj@.9={r['topk']['0.9']['adjusted']:.3f} "
      f"floor={m['noise_floor'][-1]['topk']['0.9']['adjusted']:.3f} "
      f"est={m['config']['sens']['estimator']} "
      f"ppl={m['final_eval'].get('test_ppl', float('nan')):.1f} [{time.time()-t0:.0f}s]")
print("OK")
