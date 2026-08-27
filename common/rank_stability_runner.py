"""
Orchestrates a full rank-stability analysis (vs-final and consecutive-
checkpoint, unsigned and signed, all four plots) for any experiment that can
supply two things:

  - a probe DataLoader + a freshly-initialized model (for epoch 0)
  - a way to load a saved checkpoint into a fresh model

Everything else (pooling, correlation curves, caching, plotting) is generic
and lives in rank_stability.py / sensitivity.py. This is the piece that
would otherwise be duplicated per-experiment; factoring it out means a fix
or new statistic only has to happen once.
"""

import json
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rank_stability import (
    full_resolution_consecutive_curves,
    full_resolution_correlation_curves,
    plot_rank_stability,
    plot_rank_stability_consecutive,
    pooled_total_consecutive_curves,
    pooled_total_correlation_curves,
)
from sensitivity import compute_sensitivity, flatten_scores, pool_rows


def all_checkpoints(experiment_dir: Path) -> List[Path]:
    return sorted(
        experiment_dir.glob("ckpt_epoch*.pt"),
        key=lambda p: int(p.stem.split("epoch")[-1]),
    )


def _score_epoch0(
    build_probe_loader_and_init_model: Callable[[], Tuple[DataLoader, nn.Module]],
    device: torch.device,
    n_probes: int,
):
    """Score the model's initialization (epoch 0), before any training. No
    checkpoint is saved at epoch 0 (checkpointing starts at
    CHECKPOINT_INTERVAL), but init is fully reproducible from the seed --
    see each experiment's build_probe_loader_and_init_model for how."""
    probe_loader, model = build_probe_loader_and_init_model()
    unsigned, signed = compute_sensitivity(model, probe_loader, device, n_probes=n_probes, show_progress=True)
    u_flat, boundaries = flatten_scores(model, unsigned)
    s_flat, _ = flatten_scores(model, signed)
    return u_flat.numpy(), s_flat.numpy(), boundaries, probe_loader


def _score_checkpoints(
    experiment_dir: Path,
    ckpts: List[Path],
    probe_loader: DataLoader,
    load_model_from_checkpoint: Callable[[Path], nn.Module],
    device: torch.device,
    n_probes: int,
    use_cache: bool,
):
    cache_path = experiment_dir / "full_resolution_sensitivity_cache.npz"
    expected_epochs = [int(p.stem.split("epoch")[-1]) for p in ckpts]

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if cached["epochs"].tolist() == expected_epochs and int(cached["n_probes"]) == n_probes:
            print(f"Loaded cached full-resolution scores from {cache_path}")
            boundaries = [(g, int(s), int(e)) for g, s, e in cached["boundaries"].tolist()]
            return list(cached["unsigned"]), list(cached["signed"]), boundaries

    print(f"Full-resolution scoring: {len(ckpts)} checkpoints...")
    unsigned_flats, signed_flats = [], []
    boundaries = None
    for ckpt_path in ckpts:
        model = load_model_from_checkpoint(ckpt_path)
        unsigned, signed = compute_sensitivity(model, probe_loader, device, n_probes=n_probes, show_progress=True)
        u_flat, boundaries = flatten_scores(model, unsigned)
        s_flat, _ = flatten_scores(model, signed)
        unsigned_flats.append(u_flat.numpy())
        signed_flats.append(s_flat.numpy())

    np.savez(
        cache_path,
        epochs=np.array(expected_epochs),
        n_probes=n_probes,
        unsigned=np.stack(unsigned_flats),
        signed=np.stack(signed_flats),
        boundaries=np.array(boundaries, dtype=object),
    )
    print(f"Cached full-resolution scores to {cache_path}")
    return unsigned_flats, signed_flats, boundaries


def run_rank_stability_analysis(
    experiment_dir: Path,
    device: torch.device,
    n_probes: int,
    recompute: bool,
    build_probe_loader_and_init_model: Callable[[], Tuple[DataLoader, nn.Module]],
    load_model_from_checkpoint: Callable[[Path], nn.Module],
) -> None:
    """Produces rank_stability_{unsigned,signed}.png and
    rank_stability_consecutive_{unsigned,signed}.png in experiment_dir,
    spanning epoch 0 (reconstructed init) through the final checkpoint.

    build_probe_loader_and_init_model: builds the (seeded) probe DataLoader
    and a freshly-initialized model in one call, since both must be built
    from the same seeded sequence for the init reconstruction to be exact
    (see e.g. shakespeare_lstm/rank_stability.py for the concrete version).
    load_model_from_checkpoint: builds a fresh model of the right
    architecture and loads a given checkpoint's state dict into it.
    """
    history = json.loads((experiment_dir / "history.json").read_text())["history"]

    heatmap_data = np.load(experiment_dir / "parameter_sensitivity_heatmap_data.npz", allow_pickle=True)
    n_rows = int(heatmap_data["n_rows"])

    print("Scoring epoch 0 (initialization, reconstructed from seed)...")
    epoch0_u, epoch0_s, boundaries, probe_loader = _score_epoch0(
        build_probe_loader_and_init_model, device, n_probes
    )
    epoch0_u_pooled = pool_rows(torch.from_numpy(epoch0_u), n_rows).numpy()
    epoch0_s_pooled = pool_rows(torch.from_numpy(epoch0_s), n_rows).numpy()

    pooled_epochs = [0] + heatmap_data["epochs"].tolist()
    pooled_unsigned_matrix = np.concatenate([epoch0_u_pooled[:, None], heatmap_data["unsigned"]], axis=1)
    pooled_signed_matrix = np.concatenate([epoch0_s_pooled[:, None], heatmap_data["signed"]], axis=1)
    pooled_unsigned = pooled_total_correlation_curves(pooled_unsigned_matrix)
    pooled_signed = pooled_total_correlation_curves(pooled_signed_matrix)
    print(f"Pooled total curves computed (spearman at epoch 0: "
          f"unsigned={pooled_unsigned['spearman'][0]:.3f}, signed={pooled_signed['spearman'][0]:.3f})")

    ckpts = all_checkpoints(experiment_dir)
    if len(ckpts) < 2:
        raise RuntimeError(f"Need at least 2 checkpoints in {experiment_dir}, found {len(ckpts)}")

    unsigned_flats, signed_flats, boundaries = _score_checkpoints(
        experiment_dir, ckpts, probe_loader, load_model_from_checkpoint, device, n_probes, use_cache=not recompute
    )
    unsigned_flats = [epoch0_u] + unsigned_flats
    signed_flats = [epoch0_s] + signed_flats
    checkpoint_epochs = [0] + [int(p.stem.split("epoch")[-1]) for p in ckpts]

    checkpoint_unsigned = full_resolution_correlation_curves(unsigned_flats, boundaries)
    checkpoint_signed = full_resolution_correlation_curves(signed_flats, boundaries)

    out_unsigned = experiment_dir / "rank_stability_unsigned.png"
    out_signed = experiment_dir / "rank_stability_signed.png"
    plot_rank_stability(
        history, pooled_epochs, pooled_unsigned, pooled_signed,
        checkpoint_epochs, checkpoint_unsigned, checkpoint_signed,
        out_unsigned, out_signed,
    )
    print(f"Saved {out_unsigned}")
    print(f"Saved {out_signed}")

    # Consecutive-checkpoint (step-to-step) curves: unlike the vs-final
    # curves above, these directly show whether any specific window --
    # including the last one -- has an unusually large change, since there
    # is no trivial self-comparison to hide it.
    pooled_consec_unsigned = pooled_total_consecutive_curves(pooled_unsigned_matrix)
    pooled_consec_signed = pooled_total_consecutive_curves(pooled_signed_matrix)
    checkpoint_consec_unsigned = full_resolution_consecutive_curves(unsigned_flats, boundaries)
    checkpoint_consec_signed = full_resolution_consecutive_curves(signed_flats, boundaries)

    out_consec_unsigned = experiment_dir / "rank_stability_consecutive_unsigned.png"
    out_consec_signed = experiment_dir / "rank_stability_consecutive_signed.png"
    plot_rank_stability_consecutive(
        history, pooled_epochs, pooled_consec_unsigned, pooled_consec_signed,
        checkpoint_epochs, checkpoint_consec_unsigned, checkpoint_consec_signed,
        out_consec_unsigned, out_consec_signed,
    )
    print(f"Saved {out_consec_unsigned}")
    print(f"Saved {out_consec_signed}")
