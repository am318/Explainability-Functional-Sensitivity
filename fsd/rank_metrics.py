"""Rank-agreement statistics, with the controls the claim actually needs.

Three things distinguish this from "compute a Spearman and call it stable":

1. **Top-k, not the bulk.** Pruning only ever asks "is this parameter in the top k?".
   Global Spearman is dominated by the bulk, where churn is harmless. (C3)

2. **Chance correction.** Sensitivity spans orders of magnitude *across layers*, so a
   global top-k set mostly encodes a per-layer budget, and two independent runs would
   share most of it for free. `adjusted_overlap` subtracts the overlap expected from
   matching layer budgets alone, so a value near 0 means "nothing beyond the budget" and
   1 means "identical within every layer". (C2b)

3. **Budget / placement decomposition.** Overlap is factored into how fast the *layer
   budget* settles versus how fast the *within-layer ordering* settles. These have very
   different implications for pruning, and conflating them is the single most common way
   this kind of claim goes wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# rank correlations
# ---------------------------------------------------------------------------

def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks with ties averaged. Ties matter here: exactly-zero sensitivities are common
    at initialisation (dead units), and breaking those ties arbitrarily inflates rho."""
    n = x.numel()
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks = torch.empty(n, dtype=torch.float64)
    if n:  # assign each run of equal values the mean of the positions it occupies
        change = torch.ones(n, dtype=torch.bool)
        change[1:] = sorted_x[1:] != sorted_x[:-1]
        starts = torch.nonzero(change, as_tuple=False).flatten()
        ends = torch.cat([starts[1:], torch.tensor([n])])
        avg = (starts.to(torch.float64) + ends.to(torch.float64) - 1) / 2.0
        run_id = torch.cumsum(change.to(torch.int64), 0) - 1
        ranks_sorted = avg[run_id]
        ranks[order] = ranks_sorted
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    ra, rb = _average_ranks(a.double()), _average_ranks(b.double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def auroc_topk(scores: torch.Tensor, final: torch.Tensor, sparsity: float) -> float:
    """AUROC of `scores` for predicting membership in `final`'s top-k set.

    The practically relevant reframing of "does early sensitivity predict final
    importance": not "does it get the exact rank right" (Spearman) but "can it separate,
    as a binary classifier, the parameters that end up important from the ones that
    don't". Computed via the rank-sum (Mann-Whitney U) identity, so no sklearn dependency:
    AUROC = (sum of positive-class ranks - n1(n1+1)/2) / (n1 * n0).

    0.5 = no better than random; 1.0 = perfect separation. Reported at the sparsity a
    pruner would actually use, since AUROC (unlike Spearman) is well-defined even when
    only the extreme tail is decision-relevant.
    """
    n = final.numel()
    k = max(1, int(round(n * (1.0 - sparsity))))
    label = torch.zeros(n, dtype=torch.bool)
    label[torch.topk(final, k).indices] = True
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = torch.argsort(torch.argsort(scores)).double()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_deciles(scores: torch.Tensor, final: torch.Tensor,
                        n_bins: int = 10) -> dict:
    """Mean of `final` within each decile bin of `scores`, plus the monotonicity check.

    A weaker, more forgiving claim than rank correlation: even if exact rank churns
    (Spearman low), a monotonic staircase here says the early score still tracks the
    IMPORTANCE BAND a parameter falls into -- which is what a layerwise or coarse pruning
    budget actually needs, as opposed to knowing the precise rank.
    """
    order = torch.argsort(scores)
    bins = torch.chunk(order, n_bins)
    means = [float(final[b].mean()) for b in bins]
    monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    rho_of_means = spearman(torch.arange(len(means)).float(), torch.tensor(means))
    return {"bin_means": means, "monotonic": monotonic,
            "spearman_of_bin_means": rho_of_means, "n_bins": n_bins}


def pearson(a: torch.Tensor, b: torch.Tensor, log: bool = False) -> float:
    """Linear correlation of the raw (or log) values.

    Reported beside Spearman because the *disagreement* between them localises what changed.
    High Spearman with low Pearson means the ordering held while the magnitudes moved; the
    reverse means a few large entries dominate a shuffled bulk. On a heavy-tailed spectrum
    raw Pearson is essentially a statement about the largest few values, so the log version
    is usually the more informative of the two.
    """
    a, b = a.double(), b.double()
    if log:
        floor = lambda v: v.clamp(min=max(float(v[v > 0].quantile(1e-4)), 1e-30)) if (v > 0).any() else v
        a, b = torch.log10(floor(a)), torch.log10(floor(b))
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def kendall_tau(a: torch.Tensor, b: torch.Tensor, max_n: int = 20000,
                seed: int = 0) -> float:
    """Kendall tau-b on a random subsample -- O(n^2) exact is hopeless at 10^6 params.

    Reported alongside Spearman because tau is far less forgiving of local reshuffling,
    which is precisely the regime we are claiming something about."""
    n = a.numel()
    if n < 2:
        return float("nan")
    if n > max_n:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:max_n]
        a, b = a[idx], b[idx]
    a, b = a.double(), b.double()
    m = a.numel()
    conc = 0.0
    total = 0.0
    chunk = 2048
    for start in range(0, m, chunk):
        ai = a[start:start + chunk].unsqueeze(1)
        bi = b[start:start + chunk].unsqueeze(1)
        sa = torch.sign(ai - a.unsqueeze(0))
        sb = torch.sign(bi - b.unsqueeze(0))
        conc += float((sa * sb).sum())
        total += float((sa.abs() * sb.abs()).sum())
    return conc / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# top-k sets
# ---------------------------------------------------------------------------

def topk_mask(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean mask of the parameters a pruner at this sparsity would KEEP."""
    n = scores.numel()
    k = max(1, int(round(n * (1.0 - sparsity))))
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.topk(scores, k, largest=True, sorted=False).indices] = True
    return mask


def overlap(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    """|A and B| / k for equal-sized sets (== recall == precision == 1 - error rate)."""
    k = int(mask_a.sum())
    return float((mask_a & mask_b).sum()) / max(1, k)


def jaccard(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    inter = float((mask_a & mask_b).sum())
    union = float((mask_a | mask_b).sum())
    return inter / max(1.0, union)


def layer_counts(mask: torch.Tensor, layer_ids: torch.Tensor, n_layers: int) -> torch.Tensor:
    return torch.bincount(layer_ids[mask].long(), minlength=n_layers).double()


def chance_overlap(mask_a: torch.Tensor, mask_b: torch.Tensor, layer_ids: torch.Tensor,
                   n_layers: int) -> float:
    """Overlap expected if both sets kept their per-layer budgets but placed picks at
    random *within* each layer: sum_l k_l^A k_l^B / n_l, normalised by k.

    This is the layer-budget-matched random baseline (cf. random tickets). Anything the
    ranking knows beyond "how many weights per layer" has to show up above this line."""
    sizes = torch.bincount(layer_ids.long(), minlength=n_layers).double().clamp(min=1)
    ca = layer_counts(mask_a, layer_ids, n_layers)
    cb = layer_counts(mask_b, layer_ids, n_layers)
    k = float(mask_a.sum())
    return float((ca * cb / sizes).sum()) / max(1.0, k)


def adjusted_overlap(mask_a: torch.Tensor, mask_b: torch.Tensor, layer_ids: torch.Tensor,
                     n_layers: int) -> float:
    """(observed - chance) / (1 - chance). 0 == explained entirely by the layer budget."""
    obs = overlap(mask_a, mask_b)
    exp = chance_overlap(mask_a, mask_b, layer_ids, n_layers)
    if 1.0 - exp <= 1e-12:
        return float("nan")
    return (obs - exp) / (1.0 - exp)


def global_chance_overlap(mask_a: torch.Tensor) -> float:
    """Overlap of two uniformly random k-subsets: k/n (i.e. the keep fraction)."""
    return float(mask_a.sum()) / max(1, mask_a.numel())


# ---------------------------------------------------------------------------
# within-layer statistics and the budget / placement decomposition
# ---------------------------------------------------------------------------

def within_layer_spearman(a: torch.Tensor, b: torch.Tensor, layer_ids: torch.Tensor,
                          n_layers: int, min_size: int = 32) -> Dict[str, float]:
    """Spearman computed inside each parameter tensor, then aggregated.

    Strips the cross-layer scale separation entirely: this is the number that survives the
    'you are only recovering a layer budget' objection."""
    vals, weights = [], []
    for l in range(n_layers):
        sel = layer_ids == l
        n = int(sel.sum())
        if n < min_size:
            continue
        rho = spearman(a[sel], b[sel])
        if rho == rho:  # not nan
            vals.append(rho)
            weights.append(float(n))
    if not vals:
        return {"weighted": float("nan"), "median": float("nan"), "n_layers": 0}
    v = torch.tensor(vals, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    return {
        "weighted": float((v * w).sum() / w.sum()),
        "median": float(v.median()),
        "min": float(v.min()),
        "n_layers": len(vals),
    }


def budget_vector(mask: torch.Tensor, layer_ids: torch.Tensor, n_layers: int) -> torch.Tensor:
    """Per-layer keep fraction -- the only thing a 'random ticket' baseline gets to use."""
    sizes = torch.bincount(layer_ids.long(), minlength=n_layers).double().clamp(min=1)
    return layer_counts(mask, layer_ids, n_layers) / sizes


def budget_distance(mask_a: torch.Tensor, mask_b: torch.Tensor, layer_ids: torch.Tensor,
                    n_layers: int) -> float:
    """Total-variation distance between the two layer budgets (0 == identical budgets).

    Tracking this next to `adjusted_overlap` separates 'the network has decided how much
    each layer deserves' from 'the network has decided which weights inside a layer
    deserve it'. They do not have to happen at the same time, and if they don't, that
    asymmetry is a result in its own right."""
    ca = layer_counts(mask_a, layer_ids, n_layers)
    cb = layer_counts(mask_b, layer_ids, n_layers)
    ka, kb = ca.sum().clamp(min=1), cb.sum().clamp(min=1)
    return 0.5 * float((ca / ka - cb / kb).abs().sum())


# ---------------------------------------------------------------------------
# the full comparison used everywhere
# ---------------------------------------------------------------------------

def compare(a: torch.Tensor, b: torch.Tensor, layer_ids: torch.Tensor, n_layers: int,
            sparsities: Sequence[float], with_kendall: bool = True) -> Dict[str, object]:
    """Every rank statistic for one pair of score vectors."""
    out: Dict[str, object] = {
        "spearman": spearman(a, b),
        "pearson": pearson(a, b),
        "pearson_log": pearson(a, b, log=True),
        "within_layer": within_layer_spearman(a, b, layer_ids, n_layers),
    }
    if with_kendall:
        out["kendall"] = kendall_tau(a, b)
    per_sparsity = {}
    for s in sparsities:
        ma, mb = topk_mask(a, s), topk_mask(b, s)
        per_sparsity[f"{s:g}"] = {
            "overlap": overlap(ma, mb),
            "jaccard": jaccard(ma, mb),
            "chance_layer": chance_overlap(ma, mb, layer_ids, n_layers),
            "chance_global": global_chance_overlap(ma),
            "adjusted": adjusted_overlap(ma, mb, layer_ids, n_layers),
            "budget_distance": budget_distance(ma, mb, layer_ids, n_layers),
        }
    out["topk"] = per_sparsity
    return out


def stabilisation_step(steps: Sequence[int], values: Sequence[float], threshold: float,
                       ceiling: Optional[float] = None) -> Optional[int]:
    """t*: the first step whose value stays >= threshold for the rest of the run.

    When a `ceiling` (the noise floor from C2a) is supplied the threshold is taken as a
    fraction *of the ceiling*, because no measurement can exceed it -- demanding 0.9
    absolute when the measurement ceiling is 0.85 would report 'never stabilises' for a
    ranking that is in fact pinned."""
    target = threshold * ceiling if ceiling is not None and ceiling == ceiling else threshold
    ok = [v >= target for v in values]
    for i in range(len(ok)):
        if all(ok[i:]):
            return int(steps[i])
    return None
