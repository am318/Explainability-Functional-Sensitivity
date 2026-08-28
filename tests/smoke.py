"""End-to-end smoke test on synthetic data -- no download required.

Exercises every piece the paper depends on: both sensitivity estimators, the fold-based
noise floor, the layerwise controls, kernel velocity, and the drift/gap prediction.
"""
import sys, time
sys.path.insert(0, ".")
from fsd import config as C
from fsd.run import execute


def build(arch, **model_kw):
    cfg = C.RunCfg(tag=f"smoke-{arch}", out_dir="results/_smoke")
    cfg.data = C.DataCfg(dataset="synthetic", image_size=32, augment=False,
                         test_subset=512, workers=0)
    cfg.model = C.ModelCfg(arch=arch, **model_kw)
    cfg.train = C.TrainCfg(steps=60, batch_size=64, lr=1e-3, warmup_steps=10)
    cfg.sens = C.SensCfg(n_samples=64, batch_size=16, folds=2, ntk_examples=16)
    cfg.n_ckpts, cfg.sparsities = 6, [0.5, 0.9, 0.99]
    return cfg


if __name__ == "__main__":
    for arch, kw in [("vit", dict(width=96, depth=3, heads=3, patch_size=8)),
                     ("resnet20", dict(width=16, depth=14)),
                     ("mlp", dict(width=128, depth=3))]:
        t0 = time.time()
        m = execute(build(arch, **kw))
        r = m["vs_final"][2]
        nf = m["noise_floor"][-1]["topk"]["0.9"]["adjusted"]
        th = m["theory"][2]
        print(f"  {arch:9s} step={r['step']:<4} rho={r['spearman']:.3f} "
              f"overlap@.9={r['topk']['0.9']['overlap']:.3f} "
              f"adj@.9={r['topk']['0.9']['adjusted']:.3f} "
              f"chance@.9={r['topk']['0.9']['chance_layer']:.3f} "
              f"floor={nf:.3f} kv={m['laziness'][2].get('kernel_velocity_to_final', float('nan')):.3f} "
              f"sigma={th['drift_sigma']:.3f} pred/obs@.9={th['pred_overlap_0.9']:.3f}/{th['obs_overlap_0.9']:.3f} "
              f"[{time.time()-t0:.0f}s]")
    print("OK")
