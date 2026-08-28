"""
Orchestrates a full rank-stability analysis (vs-final and consecutive-
checkpoint, for every tracked score kind, all plots) for any experiment that
can supply two things:

  - a probe DataLoader + a freshly-initialized model (for epoch 0)
  - a way to load a saved checkpoint into a fresh model

Everything else (pooling, correlation curves, caching, plotting) is generic
and lives in rank_stability.py / sensitivity.py. This is the piece that
would otherwise be duplicated per-experiment; factoring it out means a fix
or new statistic only has to happen once.

Which kinds are analysed follows what the run actually tracked: pass
include_signed=False (or run an experiment with track_signed off, whose
heatmap npz then has no `signed` array) and only the unsigned figures are
produced, at a saving of one backward pass per probe batch.

Besides the figures, the computed curves are written to
rank_stability_curves.npz so that cross-run figures (e.g. the per-dataset
optimizer comparison built by sweep.py) do not have to re-derive them.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rank_stability import (
    METHODS,
    full_resolution_consecutive_curves,
    full_resolution_correlation_curves,
    plot_rank_stability,
    plot_rank_stability_consecutive,
    pooled_total_consecutive_curves,
    pooled_total_correlation_curves,
)
from sensitivity import compute_sensitivity, flatten_scores, pool_rows

CURVES_FILENAME = "rank_stability_curves.npz"


def all_checkpoints(experiment_dir: Path) -> List[Path]:
    return sorted(
        experiment_dir.glob("ckpt_epoch*.pt"),
        key=lambda p: int(p.stem.split("epoch")[-1]),
    )


def _flats_by_kind(
    model: nn.Module, unsigned: Dict[str, torch.Tensor], signed: Optional[Dict[str, torch.Tensor]], kinds: Sequence[str]
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, int, int]]]:
    scores = {"unsigned": unsigned, "signed": signed}
    flats: Dict[str, np.ndarray] = {}
    boundaries: List[Tuple[str, int, int]] = []
    for kind in kinds:
        flat, boundaries = flatten_scores(model, scores[kind])
        flats[kind] = flat.numpy()
    return flats, boundaries


def _score_epoch0(
    build_probe_loader_and_init_model: Callable[[], Tuple[DataLoader, nn.Module]],
    device: torch.device,
    n_probes: int,
    kinds: Sequence[str],
):
    """Score the model's initialization (epoch 0), before any training. No
    checkpoint is saved at epoch 0 (checkpointing starts at
    CHECKPOINT_INTERVAL), but init is fully reproducible from the seed --
    see each experiment's build_probe_loader_and_init_model for how."""
    probe_loader, model = build_probe_loader_and_init_model()
    unsigned, signed = compute_sensitivity(
        model, probe_loader, device, n_probes=n_probes, show_progress=True,
        include_signed="signed" in kinds,
    )
    flats, boundaries = _flats_by_kind(model, unsigned, signed, kinds)
    return flats, boundaries, probe_loader


def _score_checkpoints(
    experiment_dir: Path,
    ckpts: List[Path],
    probe_loader: DataLoader,
    load_model_from_checkpoint: Callable[[Path], nn.Module],
    device: torch.device,
    n_probes: int,
    kinds: Sequence[str],
    use_cache: bool,
):
    cache_path = experiment_dir / "full_resolution_sensitivity_cache.npz"
    expected_epochs = [int(p.stem.split("epoch")[-1]) for p in ckpts]

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if (
            cached["epochs"].tolist() == expected_epochs
            and int(cached["n_probes"]) == n_probes
            and all(kind in cached.files for kind in kinds)
        ):
            print(f"Loaded cached full-resolution scores from {cache_path}")
            boundaries = [(g, int(s), int(e)) for g, s, e in cached["boundaries"].tolist()]
            return {kind: list(cached[kind]) for kind in kinds}, boundaries

    print(f"Full-resolution scoring: {len(ckpts)} checkpoints...")
    flats: Dict[str, List[np.ndarray]] = {kind: [] for kind in kinds}
    boundaries: List[Tuple[str, int, int]] = []
    for ckpt_path in ckpts:
        model = load_model_from_checkpoint(ckpt_path)
        unsigned, signed = compute_sensitivity(
            model, probe_loader, device, n_probes=n_probes, show_progress=True,
            include_signed="signed" in kinds,
        )
        per_kind, boundaries = _flats_by_kind(model, unsigned, signed, kinds)
        for kind in kinds:
            flats[kind].append(per_kind[kind])

    np.savez(
        cache_path,
        epochs=np.array(expected_epochs),
        n_probes=n_probes,
        boundaries=np.array(boundaries, dtype=object),
        **{kind: np.stack(values) for kind, values in flats.items()},
    )
    print(f"Cached full-resolution scores to {cache_path}")
    return flats, boundaries


def _save_curves(
    out_path: Path,
    pooled_epochs: List[int],
    checkpoint_epochs: List[int],
    pooled: Dict[str, Dict[str, np.ndarray]],
    checkpoint: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    pooled_consecutive: Dict[str, Dict[str, np.ndarray]],
    checkpoint_consecutive: Dict[str, Dict[str, Dict[str, np.ndarray]]],
) -> None:
    """Flatten the nested curve dicts into one npz. Keys are joined with a
    double underscore so that module names containing single underscores
    still parse back unambiguously (see load_curves)."""
    arrays: Dict[str, np.ndarray] = {
        "pooled_epochs": np.array(pooled_epochs),
        "checkpoint_epochs": np.array(checkpoint_epochs),
    }
    for prefix, source in [("pooled", pooled), ("pooled_consec", pooled_consecutive)]:
        for kind, by_method in source.items():
            for method, values in by_method.items():
                arrays[f"{prefix}__{kind}__{method}"] = values
    for prefix, source in [("ckpt", checkpoint), ("ckpt_consec", checkpoint_consecutive)]:
        for kind, by_method in source.items():
            for method, by_group in by_method.items():
                for group, values in by_group.items():
                    arrays[f"{prefix}__{kind}__{method}__{group}"] = values
    np.savez(out_path, **arrays)


def load_curves(experiment_dir: Path, kind: str = "unsigned") -> Dict:
    """Read back one kind's curves from rank_stability_curves.npz, in the
    shape plot_optimizer_comparison expects."""
    data = np.load(experiment_dir / CURVES_FILENAME)
    groups = sorted({
        key.split("__", 3)[3] for key in data.files
        if key.startswith(f"ckpt__{kind}__{METHODS[0]}__")
    })
    return {
        "pooled_epochs": data["pooled_epochs"],
        "checkpoint_epochs": data["checkpoint_epochs"],
        "pooled": {m: data[f"pooled__{kind}__{m}"] for m in METHODS},
        "checkpoint": {m: {g: data[f"ckpt__{kind}__{m}__{g}"] for g in groups} for m in METHODS},
        "pooled_consecutive": {m: data[f"pooled_consec__{kind}__{m}"] for m in METHODS},
        "checkpoint_consecutive": {
            m: {g: data[f"ckpt_consec__{kind}__{m}__{g}"] for g in groups} for m in METHODS
        },
    }


def run_rank_stability_analysis(
    experiment_dir: Path,
    device: torch.device,
    n_probes: int,
    recompute: bool,
    build_probe_loader_and_init_model: Callable[[], Tuple[DataLoader, nn.Module]],
    load_model_from_checkpoint: Callable[[Path], nn.Module],
    include_signed: bool = True,
) -> None:
    """Produces rank_stability_{kind}.png and
    rank_stability_consecutive_{kind}.png in experiment_dir, spanning epoch 0
    (reconstructed init) through the final checkpoint, plus the underlying
    curves in rank_stability_curves.npz.

    build_probe_loader_and_init_model: builds the (seeded) probe DataLoader
    and a freshly-initialized model in one call, since both must be built
    from the same seeded sequence for the init reconstruction to be exact
    (see e.g. shakespeare_lstm/rank_stability.py for the concrete version).
    load_model_from_checkpoint: builds a fresh model of the right
    architecture and loads a given checkpoint's state dict into it.
    include_signed: analyse the signed companion score alongside the
    unsigned one. Silently downgraded to False if the run did not track it.
    """
    history = json.loads((experiment_dir / "history.json").read_text())["history"]

    heatmap_data = np.load(experiment_dir / "parameter_sensitivity_heatmap_data.npz", allow_pickle=True)
    n_rows = int(heatmap_data["n_rows"])
    kinds = ["unsigned"] + (["signed"] if include_signed and "signed" in heatmap_data.files else [])

    print(f"Scoring epoch 0 (initialization, reconstructed from seed) for: {', '.join(kinds)}...")
    epoch0, boundaries, probe_loader = _score_epoch0(
        build_probe_loader_and_init_model, device, n_probes, kinds
    )

    pooled_epochs = [0] + heatmap_data["epochs"].tolist()
    pooled_matrix = {
        kind: np.concatenate(
            [pool_rows(torch.from_numpy(epoch0[kind]), n_rows).numpy()[:, None], heatmap_data[kind]], axis=1
        )
        for kind in kinds
    }
    pooled = {kind: pooled_total_correlation_curves(matrix) for kind, matrix in pooled_matrix.items()}
    print("Pooled total curves computed (spearman at epoch 0: "
          + ", ".join(f"{kind}={pooled[kind]['spearman'][0]:.3f}" for kind in kinds) + ")")

    ckpts = all_checkpoints(experiment_dir)
    if len(ckpts) < 2:
        raise RuntimeError(f"Need at least 2 checkpoints in {experiment_dir}, found {len(ckpts)}")

    flats, boundaries = _score_checkpoints(
        experiment_dir, ckpts, probe_loader, load_model_from_checkpoint, device, n_probes, kinds,
        use_cache=not recompute,
    )
    flats = {kind: [epoch0[kind]] + list(values) for kind, values in flats.items()}
    checkpoint_epochs = [0] + [int(p.stem.split("epoch")[-1]) for p in ckpts]

    checkpoint = {kind: full_resolution_correlation_curves(values, boundaries) for kind, values in flats.items()}

    out_paths = {kind: experiment_dir / f"rank_stability_{kind}.png" for kind in kinds}
    plot_rank_stability(history, pooled_epochs, pooled, checkpoint_epochs, checkpoint, out_paths)
    for path in out_paths.values():
        print(f"Saved {path}")

    # Consecutive-checkpoint (step-to-step) curves: unlike the vs-final
    # curves above, these directly show whether any specific window --
    # including the last one -- has an unusually large change, since there
    # is no trivial self-comparison to hide it.
    pooled_consecutive = {kind: pooled_total_consecutive_curves(matrix) for kind, matrix in pooled_matrix.items()}
    checkpoint_consecutive = {
        kind: full_resolution_consecutive_curves(values, boundaries) for kind, values in flats.items()
    }

    out_paths_consecutive = {kind: experiment_dir / f"rank_stability_consecutive_{kind}.png" for kind in kinds}
    plot_rank_stability_consecutive(
        history, pooled_epochs, pooled_consecutive, checkpoint_epochs, checkpoint_consecutive, out_paths_consecutive
    )
    for path in out_paths_consecutive.values():
        print(f"Saved {path}")

    curves_path = experiment_dir / CURVES_FILENAME
    _save_curves(
        curves_path, pooled_epochs, checkpoint_epochs,
        pooled, checkpoint, pooled_consecutive, checkpoint_consecutive,
    )
    print(f"Saved {curves_path}")
