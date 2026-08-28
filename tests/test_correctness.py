"""Regression tests for the quantities the paper's claims are computed from.

Each test pins a property that, if it silently broke, would produce plausible-looking but
wrong numbers -- the failure mode that matters most here, since nothing downstream would
raise.

    python tests/test_correctness.py
"""
import sys
sys.path.insert(0, ".")

import torch
from scipy.stats import rankdata, spearmanr

from fsd import config as C, models, rank_metrics as R, sensitivity as S, theory as T

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def _net():
    torch.manual_seed(0)
    cfg = C.RunCfg()
    cfg.model = C.ModelCfg(arch="mlp", width=48, depth=3)
    return models.build(cfg.model, cfg.data)


def test_trace_identity():
    """tr(K_theta) = tr(Q_theta) = sum_i S_i(theta), per example (draft eq. 7)."""
    net, dev = _net(), torch.device("cpu")
    names = S.param_names(net)
    x = torch.randn(4, 3, 32, 32)
    K = S.compute_trace_ntk(net, x, C.SensCfg(estimator="exact"), dev, "exact", 10,
                            torch.float32)
    per_ex = [float(S.flatten(S.compute_sensitivity(
        net, [x[i:i + 1]], C.SensCfg(estimator="exact", folds=1), dev).scores, names).sum())
        for i in range(4)]
    check("NTK trace identity", torch.allclose(K.diag(), torch.tensor(per_ex), rtol=1e-4))


def test_estimators_agree():
    """Exact and Hutchinson target the same quantity on the same scale."""
    net, dev = _net(), torch.device("cpu")
    names = S.param_names(net)
    x = [torch.randn(8, 3, 32, 32)]
    a = S.flatten(S.compute_sensitivity(net, x, C.SensCfg(estimator="exact", folds=1),
                                        dev).scores, names)
    b = S.flatten(S.compute_sensitivity(net, x, C.SensCfg(
        estimator="hutchinson", n_probes=2000, folds=1, exact_max_outputs=0),
        dev).scores, names)
    rel = float((b.sum() - a.sum()).abs() / a.sum())
    check("exact == hutchinson (2000 probes)", rel < 0.02, f"rel err {rel:.4f}")


def test_ranks_match_scipy():
    """Tie handling matters: exact zeros are common at init and arbitrary tie-breaking
    inflates rho."""
    x = torch.tensor([3., 1., 1., 2., 5., 5., 5., 0.])
    y = torch.tensor([1., 2., 2., 9., 4., 4., 1., 7.])
    check("average ranks match scipy",
          bool((R._average_ranks(x).numpy() == rankdata(x.numpy()) - 1).all()))
    check("spearman matches scipy",
          abs(R.spearman(x, y) - spearmanr(x.numpy(), y.numpy()).correlation) < 1e-9)


def test_chance_correction():
    """The control that defends C2b: rankings sharing nothing within a layer must score
    ~0 adjusted, however high their raw overlap and Spearman look."""
    torch.manual_seed(0)
    n, L = 20000, 10
    lids = torch.arange(n) % L
    scale = 10.0 ** (lids.float() - L / 2)
    a = torch.rand(n) * scale
    b = torch.rand(n) * scale          # independent *within* each layer
    ma, mb = R.topk_mask(a, 0.9), R.topk_mask(b, 0.9)
    raw = R.overlap(ma, mb)
    adj = R.adjusted_overlap(ma, mb, lids, L)
    rho = R.spearman(a, b)
    check("raw overlap is misleadingly high", raw > 0.7, f"{raw:.3f}")
    check("global spearman is misleadingly high", rho > 0.9, f"{rho:.3f}")
    check("adjusted overlap is ~0", abs(adj) < 0.05, f"{adj:.4f}")
    check("identical rankings score 1", abs(R.adjusted_overlap(ma, ma, lids, L) - 1) < 1e-9)


def test_counting_bound_holds():
    """Prop. 1 is an upper bound on churn: it must never exceed the measurement."""
    torch.manual_seed(0)
    n = 100000
    s = (torch.distributions.Pareto(1.0, 1.5).sample((n,)) ** 2).float()
    violations = 0
    for sigma in (0.01, 0.05, 0.2, 0.5):
        st = (10 ** (T.log_scores(s) + sigma * torch.randn(n).double())).float()
        d = T.amplitude_drift(st, s, quantile=1.0)     # true max -> the bound must hold
        for sp in (0.5, 0.9, 0.99):
            obs = R.overlap(R.topk_mask(s, sp), R.topk_mask(st, sp))
            lb = T.counting_bound(s, d["max"], sp)["overlap_lower_bound"]
            violations += int(obs < lb - 1e-9)
    check("counting bound never violated", violations == 0, f"{violations} violations")


def test_drift_model_recovers_sigma():
    """The MAD constant must match the assumed drift family, or sigma is biased low and
    the prediction is systematically optimistic."""
    torch.manual_seed(0)
    n = 100000
    s = (torch.distributions.Pareto(1.0, 1.5).sample((n,)) ** 2).float()
    errs = []
    for sigma in (0.02, 0.1, 0.4):
        z = torch.distributions.StudentT(3.).sample((n,)).double() / 3.0 ** 0.5
        st = (10 ** (T.log_scores(s) + sigma * z)).float()
        errs.append(abs(T.drift_scale(st, s) - sigma) / sigma)
    check("drift scale recovered", max(errs) < 0.05, f"max rel err {max(errs):.3f}")


def test_prunable_excludes_norms():
    """Norm/bias parameters have far larger sensitivity; including them would manufacture
    stability for free."""
    net = _net()
    cfg = C.SensCfg()
    m = S.prunable_mask(net, cfg)
    names = S.param_names(net)
    check("biases excluded from ranking",
          all(not m[n].any() for n in names if n.endswith("bias")))
    check("head excluded from ranking", not m["head.weight"].any())
    check("hidden weights included", m["body.0.weight"].all())


if __name__ == "__main__":
    for fn in (test_trace_identity, test_estimators_agree, test_ranks_match_scipy,
               test_chance_correction, test_counting_bound_holds,
               test_drift_model_recovers_sigma, test_prunable_excludes_norms):
        print(f"{fn.__name__}:")
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        raise SystemExit(1)
    print("all correctness tests passed")
