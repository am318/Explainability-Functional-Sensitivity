"""
Parameter-wise functional sensitivity, shared across experiments (currently:
shakespeare_lstm/, function_regression/). Not yet used for pruning -- this
module only estimates the scores so they can be tracked over training.

Notation follows notes.tex: for a parametrised map F_theta: X -> R^{d_y}, the
pointwise sensitivity of parameter theta_i at input x is

    s_i(x; theta) := dF_theta(x) / dtheta_i  in R^{d_y}.

The notes define the (unsigned) functional sensitivity score as the squared
L2 norm, in expectation over the input distribution:

    S_i(theta) := E_x[ || s_i(x; theta) ||_2^2 ].                       (unsigned)

In practice we only ever have a minibatch, so this is estimated as an
expectation over minibatches drawn during training rather than the true
population expectation -- consistent with how S_i would be estimated at any
single point in training anyway.

Computing s_i(x; theta) exactly requires one backward pass per output
coordinate, which is intractable whenever d_y is large (e.g. seq_len *
vocab_size for the char-LSTM). Instead we use the Hutchinson/Rademacher
trick: for a random probe z with iid +-1 entries (E[z]=0, E[z z^T]=I, no
extra normalisation), a single backward pass of the scalar <F_theta(x), z>
gives a gradient g_i = z^T s_i(x; theta) with E_z[g_i^2] = ||s_i(x;
theta)||_2^2 for a single sample x. Batching B samples into one backward
pass and squaring gives E[g_i^2] = sum over the batch of ||s_i(x_b;
theta)||_2^2 (cross terms vanish since probe entries are independent across
samples), so accumulating the *raw* squared gradient (no per-batch
reweighting) and dividing once by (n_probes * total_samples) at the end
gives an unbiased, batch-size-independent Monte Carlo estimate of S_i(theta).

This closely follows the Hutchinson estimator in
Sensitivity-Pruning/Pruning_diagnostics/sensitivity_metrics.py, but that
implementation (a) normalises the probe by 1/sqrt(d_y), which rescales every
score by a constant 1/d_y and so no longer matches the literal S_i in
notes.tex (it estimates S_i/d_y instead -- harmless for rankings within one
model, but not the defined quantity and not comparable across models with
different d_y); and (b) divides the probe-projected scalar by batch_size
before backward and then reweights the *squared* gradient by batch_size
again when accumulating -- these do not cancel for a squared quantity, and
the resulting score is (empirically confirmed) proportional to
1/batch_size, i.e. the reported sensitivity depends on an arbitrary batching
hyperparameter. Both are avoided here: no probe normalisation, and the raw
squared gradient (an unbiased estimate of a *sum* over the batch) is
accumulated unweighted and divided by the total sample count once at the
end.

Alongside this we track a *signed* companion score, requested explicitly
because S_i above discards sign information by construction. Rather than
squaring, the signed score sums (does not square) the same pointwise
quantity over the output coordinates:

    Sbar_i(theta) := E_x[ 1^T s_i(x; theta) ] = E_x[ sum_y dF_theta(x)_y / dtheta_i ].  (signed)

Because this uses a fixed direction (all-ones) rather than a random probe,
it needs no Hutchinson estimator: one backward pass of scalar = sum(F_theta(x))
per batch gives sum_y dF_theta(x)_y / dtheta_i directly and exactly for that
batch, and averaging over batches estimates Sbar_i(theta). Sbar_i can be
negative; S_i cannot.

Model requirement: `model(inputs)` may return either the output tensor
directly, or a tuple/list whose first element is the output tensor (e.g. an
RNN returning (output, hidden_state)) -- either form works unchanged.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _model_output(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    out: Union[torch.Tensor, tuple, list] = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def zeros_like_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def _make_probe(output: torch.Tensor) -> torch.Tensor:
    """Rademacher probe (+-1, unnormalised) over the full batched output."""
    return torch.empty_like(output).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def compute_sensitivity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_probes: int = 4,
    show_progress: bool = False,
    include_signed: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """Estimate per-parameter unsigned S_i and signed Sbar_i over `loader`.

    `loader` should yield (inputs, targets) batches; only inputs are used,
    since F_theta(x) does not depend on the targets.

    `include_signed=False` skips the signed score entirely (one fewer
    backward pass per batch) and returns None in its place, for experiments
    that only care about S_i.
    """
    model.eval()
    unsigned = zeros_like_params(model)
    signed = zeros_like_params(model) if include_signed else None
    n_accum = 0

    batches = tqdm(loader, desc="sensitivity", leave=False, disable=not show_progress)
    for inputs, _targets in batches:
        # NOTE: no non_blocking=True here -- on MPS, an async H2D copy of a
        # non-pinned tensor can read stale/reused source memory before the
        # copy completes, silently corrupting the data (confirmed: produced
        # garbage token indices, reproducibly, only inside function scope on
        # MPS -- see pruning_experiment.py investigation). Synchronous .to()
        # is the safe default; the cost is negligible at these tensor sizes.
        inputs = inputs.to(device)
        bsz = inputs.shape[0]

        # Signed: exact gradient of the summed output, no probe needed.
        if signed is not None:
            model.zero_grad(set_to_none=True)
            output = _model_output(model, inputs)
            scalar = output.sum() / bsz
            scalar.backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    signed[name].add_(p.grad.detach().float().cpu(), alpha=bsz)

        # Unsigned: Hutchinson/Rademacher estimate of E[||dF/dtheta_i||^2].
        # No probe normalisation, no batch_size division on the scalar: the
        # raw squared gradient is an unbiased estimate of the *sum* over this
        # batch (see module docstring), so it is accumulated unweighted and
        # divided by the total sample count once at the end.
        for _ in range(n_probes):
            model.zero_grad(set_to_none=True)
            output = _model_output(model, inputs)
            probe = _make_probe(output)
            scalar = (output * probe).sum()
            scalar.backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float().cpu()
                    unsigned[name].add_(g.pow(2))

        n_accum += bsz

    if n_accum == 0:
        raise RuntimeError("No samples were available for sensitivity scoring.")

    for name in unsigned:
        unsigned[name].div_(n_accum * n_probes)
        if signed is not None:
            signed[name].div_(n_accum)

    model.zero_grad(set_to_none=True)
    return unsigned, signed


def parameter_group(name: str) -> str:
    """Coarse group for reporting: the parameter's top-level module name
    (e.g. embedding/lstm/head for CharLSTM, input/hidden/output for MLP)."""
    return name.split(".")[0]


def flatten_scores(
    model: nn.Module, scores: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, List[Tuple[str, int, int]]]:
    """Concatenate per-parameter score tensors in model.named_parameters()
    order (i.e. definition/attribute order), and return the [start, end)
    index range each top-level group occupies in the flattened vector, for
    later use as heatmap/distribution-plot group labels."""
    parts: List[torch.Tensor] = []
    boundaries: List[Tuple[str, int, int]] = []
    offset = 0
    for name, p in model.named_parameters():
        if name not in scores:
            continue
        t = scores[name].detach().reshape(-1).cpu().float()
        parts.append(t)
        group = parameter_group(name)
        start = offset
        offset += t.numel()
        if boundaries and boundaries[-1][0] == group:
            g, s, _ = boundaries[-1]
            boundaries[-1] = (g, s, offset)
        else:
            boundaries.append((group, start, offset))
    flat = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
    return flat, boundaries


def _row_chunk_sizes(total: int, n_rows: int) -> List[int]:
    """Sizes of `n_rows` chunks covering `total` elements as evenly as
    possible (the first `total % n_rows` chunks get one extra element),
    matching numpy.array_split/torch.tensor_split -- unlike a fixed
    ceil(total/n_rows) bin size, this never leaves a chunk empty."""
    q, r = divmod(total, n_rows)
    return [q + 1] * r + [q] * (n_rows - r)


def pool_rows(flat: torch.Tensor, n_rows: int) -> torch.Tensor:
    """Block-mean-pool a flat parameter vector down to `n_rows` for display.

    Parameter counts can vastly exceed displayable heatmap rows, so
    consecutive parameters (within the same module, since bins rarely exceed
    a module's span at reasonable row counts) are averaged into bins.
    """
    total = flat.numel()
    if total == 0:
        return torch.zeros(0)
    n_rows = max(1, min(n_rows, total))
    chunks = torch.split(flat, _row_chunk_sizes(total, n_rows))
    return torch.stack([c.mean() for c in chunks])


def pooled_group_boundaries(
    boundaries: List[Tuple[str, int, int]], total_len: int, n_rows: int
) -> List[Tuple[str, int]]:
    """Map each group's starting index into pooled row-space, for tick labels."""
    if total_len == 0:
        return []
    n_rows = max(1, min(n_rows, total_len))
    sizes = _row_chunk_sizes(total_len, n_rows)
    row_starts = []
    acc = 0
    for size in sizes:
        row_starts.append(acc)
        acc += size

    def row_of(index: int) -> int:
        row = 0
        for r, start in enumerate(row_starts):
            if start <= index:
                row = r
            else:
                break
        return row

    return [(group, row_of(start)) for group, start, _ in boundaries]


def summarize_sensitivity(scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Sum of scores per top-level parameter group, plus a 'total' entry.

    Summing (rather than averaging) matches Eq. (ntk-trace-sensitivity) in
    notes.tex, where sum_i S_i(theta) = tr(Q_theta) = tr(K_theta) is the NTK
    trace; the same reduction is used for the signed companion score for
    consistency.
    """
    totals: Dict[str, float] = {}
    grand_total = 0.0
    for name, tensor in scores.items():
        group = parameter_group(name)
        value = float(tensor.sum().item())
        totals[group] = totals.get(group, 0.0) + value
        grand_total += value
    totals["total"] = grand_total
    return totals
