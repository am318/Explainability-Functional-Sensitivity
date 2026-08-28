"""C5: the drift-vs-gap model, and the laziness measurement it rests on.

The argument in two steps.

**1. Lazy conservation.** S(theta) is the diagonal of the Gauss-Newton operator, built from
the Jacobian J = df/dtheta. In the linearised (lazy) regime J is constant along the
trajectory, so *every* sensitivity value -- and therefore the entire ordering -- is exactly
conserved. Rank churn is thus a direct measurement of departure from laziness, which we
quantify with **kernel velocity**: the normalised drift of the empirical trace-NTK Gram.

**2. Heavy-tail gap.** A parameter crosses the top-k boundary only if its drift exceeds the
local gap in the sorted sensitivity spectrum. That spectrum is heavy-tailed, so gaps near
the top are wide while the bulk is dense: swaps concentrate in the bulk and avoid the
boundary. This is the mechanism behind C3 (boundary beats bulk).

Composed, they give a prediction with **no parameters fitted to the overlap curve**: from
the marginal distribution of log S alone plus *one measured scalar per checkpoint* (the
drift scale sigma_t), predict top-k overlap simultaneously at every sparsity. If the
prediction tracks the measurement, the mechanism is doing real work; if it does not, we
report that, and C5 is falsified.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# laziness
# ---------------------------------------------------------------------------

def kernel_velocity(k_a: torch.Tensor, k_b: torch.Tensor) -> float:
    """|| K_a/||K_a|| - K_b/||K_b|| ||_F in [0, 2]. 0 == the kernel has not rotated.

    Scale-normalised on purpose: the NTK's overall magnitude grows during training even in
    the lazy regime, and that growth is not feature learning."""
    na, nb = k_a.norm(), k_b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((k_a / na - k_b / nb).norm())


def kernel_alignment(k_a: torch.Tensor, k_b: torch.Tensor) -> float:
    """Cosine similarity of the two Gram matrices (1 == identical up to scale)."""
    na, nb = k_a.norm(), k_b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((k_a * k_b).sum() / (na * nb))


# ---------------------------------------------------------------------------
# spectrum shape
# ---------------------------------------------------------------------------

def log_scores(s: torch.Tensor, floor_quantile: float = 1e-4) -> torch.Tensor:
    """log10 S with a floor at a low quantile -- exact zeros (dead units) are common."""
    s = s.double()
    pos = s[s > 0]
    floor = float(pos.quantile(floor_quantile)) if pos.numel() else 1e-30
    return torch.log10(s.clamp(min=max(floor, 1e-30)))


def hill_tail_index(s: torch.Tensor, tail_frac: float = 0.01) -> float:
    """Hill estimator of the tail index alpha of the sensitivity distribution.

    Small alpha == heavier tail == wider gaps at the top == a more stable top-k boundary,
    which is the quantitative content of the C3 mechanism."""
    s = s.double()
    s = s[s > 0]
    if s.numel() < 100:
        return float("nan")
    k = max(10, int(s.numel() * tail_frac))
    top = torch.topk(s, k + 1).values
    thresh = top[-1]
    return float(1.0 / (torch.log(top[:-1] / thresh).mean()))


def boundary_gap(s: torch.Tensor, sparsity: float, window: int = 64) -> Dict[str, float]:
    """Local structure of the sorted log-spectrum at the top-k cut.

    `gap_over_spread` is the operative number: the log-gap right at the cut measured in
    units of the local spread. Large means the boundary is well separated and drift has to
    be large to move a parameter across it."""
    n = s.numel()
    k = max(2, int(round(n * (1.0 - sparsity))))
    ls = torch.sort(log_scores(s), descending=True).values
    lo = max(0, k - window)
    hi = min(n, k + window)
    local = ls[lo:hi]
    gap = float(ls[k - 1] - ls[k]) if k < n else float("nan")
    spread = float(local.std()) if local.numel() > 1 else float("nan")
    return {
        "gap": gap,
        "local_spread": spread,
        "gap_over_spread": gap / spread if spread and spread == spread and spread > 0 else float("nan"),
        "cut_value": float(ls[k - 1]),
        "density_at_cut": float(2 * window / max(1e-12, abs(float(ls[lo]) - float(ls[hi - 1])))),
    }


# ---------------------------------------------------------------------------
# the prediction
# ---------------------------------------------------------------------------

# MAD of a unit-variance draw, per noise family. Using the Gaussian constant while
# drawing Student-t drift biases sigma low and makes the prediction systematically
# optimistic -- the two must agree.
_MAD_UNIT = {"normal": 0.6745, "student_t3": 0.7649 / (3.0 ** 0.5)}


def drift_scale(s_t: torch.Tensor, s_ref: torch.Tensor, robust: bool = True,
                family: str = "student_t3") -> float:
    """sigma_t: the scale of log-sensitivity drift between two checkpoints.

    This is the single scalar the prediction is allowed to use. Robust (MAD-based) by
    default so that a handful of parameters swinging wildly cannot set the scale."""
    d = log_scores(s_t) - log_scores(s_ref)
    if robust:
        mad = float((d - d.median()).abs().median())
        return mad / _MAD_UNIT[family]
    return float(d.std())


def predicted_overlap(s_ref: torch.Tensor, sigma: float, sparsity: float,
                      n_mc: int = 400000, seed: int = 0,
                      heavy_tail: bool = True) -> float:
    """Predict top-k overlap from the reference spectrum and the drift scale alone.

    Model: log S_t = log S_ref + eps, with eps i.i.d. of scale sigma. Nothing about the
    observed overlap enters. Evaluated by Monte Carlo on a subsample of the empirical
    spectrum, so the heavy tail is used as measured rather than assumed parametric.

    `heavy_tail=True` draws eps from a Student-t(3), matching the empirically fat drift
    distribution; Gaussian eps systematically under-predicts churn in the bulk.
    """
    g = torch.Generator().manual_seed(seed)
    ls = log_scores(s_ref)
    n = ls.numel()
    m = min(n, n_mc)
    if m < n:
        idx = torch.randperm(n, generator=g)[:m]
        ls = ls[idx]
    if heavy_tail:
        nu = 3.0
        z = torch.distributions.StudentT(nu).sample((m,)).double()
        z = z / (nu / (nu - 2)) ** 0.5           # unit variance
    else:
        z = torch.randn(m, generator=g, dtype=torch.float64)
    perturbed = ls + sigma * z
    k = max(1, int(round(m * (1.0 - sparsity))))
    a = torch.zeros(m, dtype=torch.bool)
    a[torch.topk(ls, k).indices] = True
    b = torch.zeros(m, dtype=torch.bool)
    b[torch.topk(perturbed, k).indices] = True
    return float((a & b).sum()) / k


# ---------------------------------------------------------------------------
# rigorous counting bound (trajectory version of the pruning-perturbation theorem)
# ---------------------------------------------------------------------------
#
# Write the amplitude a_i(theta) := sqrt(S_i(theta)) = || dF_theta/dtheta_i ||_{L2(mu)},
# i.e. the L2 norm of the i-th column of the Jacobian feature operator. Let
#
#     Delta_i(t,T) := || dF_{theta_t}/dtheta_i - dF_{theta_T}/dtheta_i ||_{L2(mu)}.
#
# Two facts follow immediately.
#
# (a) Reverse triangle inequality in L2(mu; R^{d_y}):
#         | a_i(theta_t) - a_i(theta_T) | <= Delta_i(t,T),
#     and under the same local-Lipschitz Jacobian assumption used for the pruning
#     perturbation bound (with the perturbation taken along the trajectory rather than
#     across a mask),
#         sum_i Delta_i(t,T)^2 = (1/n) || G_{theta_t} - G_{theta_T} ||_F^2
#                             <= L^2 || theta_t - theta_T ||_2^2.
#     So the *total* sensitivity drift is controlled by the distance travelled in
#     parameter space -- a directly measurable quantity.
#
# (b) Counting bound. Let A_t, A_T be the top-k sets of a(theta_t), a(theta_T), and let
#     Delta := max_i Delta_i(t,T). The k-th largest value is 1-Lipschitz under a uniform
#     perturbation, so the cut itself moves by at most Delta; a parameter can therefore
#     only leave the top-k if it started within 2*Delta of the cut. Hence
#
#         |A_T \ A_t|  <=  N_T(2 Delta) := #{ i : |a_i(theta_T) - q_k(theta_T)| <= 2 Delta }
#         overlap      >=  1 - N_T(2 Delta) / k.
#
# N_T is exactly the local density of the sorted amplitude spectrum at the cut. Because
# that spectrum is heavy-tailed, the density near the top is low and the bound is
# non-vacuous precisely where pruning operates -- which is the formal content of C3.


def amplitude(s: torch.Tensor) -> torch.Tensor:
    """a_i = sqrt(S_i): the L2 norm of the i-th Jacobian column. Rank-equivalent to S,
    but the metric in which the drift bound is linear."""
    return s.double().clamp(min=0).sqrt()


def amplitude_drift(s_t: torch.Tensor, s_ref: torch.Tensor,
                    quantile: float = 0.99) -> Dict[str, float]:
    """Measurable surrogate for Delta_i: |a_i(t) - a_i(T)| <= Delta_i.

    Reported at a quantile rather than the max: `max_i` over 10^6 parameters is set by a
    single outlier and yields a vacuous bound. Using the q-quantile gives a bound that
    holds for all but a (1-q) fraction of parameters, which we state as such."""
    d = (amplitude(s_t) - amplitude(s_ref)).abs()
    return {
        "mean": float(d.mean()),
        "median": float(d.median()),
        "quantile": float(d.quantile(quantile)),
        "max": float(d.max()),
        "l2": float(d.norm()),
        "quantile_level": quantile,
    }


def counting_bound(s_ref: torch.Tensor, delta: float, sparsity: float) -> Dict[str, float]:
    """Lower bound on top-k overlap from the local spectral density at the cut.

    overlap >= 1 - #{i : |a_i - q_k| <= 2 delta} / k
    """
    a = amplitude(s_ref)
    n = a.numel()
    k = max(1, int(round(n * (1.0 - sparsity))))
    q_k = float(torch.topk(a, k).values[-1])
    n_near = int(((a - q_k).abs() <= 2.0 * delta).sum())
    return {
        "cut_amplitude": q_k,
        "n_within_2delta": n_near,
        "k": k,
        "overlap_lower_bound": max(0.0, 1.0 - n_near / k),
    }


def weight_distance(theta_t: torch.Tensor, theta_ref: torch.Tensor) -> Dict[str, float]:
    """|| theta_t - theta_T ||_2 and its relative version -- the right-hand side of the
    Lipschitz bound, and the elementary measure of how far training has travelled."""
    d = float((theta_t - theta_ref).norm())
    n_ref = float(theta_ref.norm())
    return {"l2": d, "relative": d / max(1e-12, n_ref)}


def spearman_brown(rho_half: float) -> float:
    """Split-half reliability corrected to full length: 2 rho / (1 + rho).

    The noise floor compares two half-size folds, but the reported S_t pools both, so the
    raw fold agreement *understates* the reliability of the quantity actually used. This
    is the standard correction, and it makes the ceiling honest in the tight direction
    rather than the flattering one."""
    if rho_half != rho_half or rho_half >= 1.0:
        return rho_half
    return 2.0 * rho_half / (1.0 + rho_half)
