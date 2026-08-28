"""
The train-and-track loop, shared across experiments (currently:
twomoons_mlp/, mnist_cnn/). Generalised from shakespeare_lstm/train.py,
which does the same thing inline: train an epoch, periodically score
parameter-wise sensitivity on a held-out probe set, checkpoint, and write
history.json + the loss/sensitivity plot + the pooled sensitivity heatmap
into one experiment directory.

An experiment supplies a `build_experiment(cfg, device) -> Experiment`
callback and nothing else. That callback must construct *everything*
(dataset, loaders, model) in a fixed order, because rank_stability.py
reconstructs the epoch-0 model by calling the same callback after re-seeding
-- see Experiment below.

Two things differ from the shakespeare_lstm version:

  - a validation set is evaluated every epoch, so history rows carry
    `val_loss` (and `val_acc`) alongside the training `loss`. Note `loss` is
    the running average over the epoch's minibatches, as the model updates
    (the usual convention, and what shakespeare_lstm/train.py records),
    whereas `val_loss` is measured once at the end of the epoch in eval
    mode; the training curve therefore lags the validation curve by roughly
    half an epoch early in training, when the loss is moving fast.
  - the signed companion score is optional (cfg.track_signed). With it off,
    compute_sensitivity skips a backward pass per batch and every downstream
    plot drops its signed half.

Settings come from the config dataclass, whose defaults read environment
variables, so a one-off run needs no code edits:

    EPOCHS=5 OPTIMIZER=sgd LR=0.1 python train.py
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from optimizers import build_optimizer
from plotting import plot_sensitivity_heatmap, plot_training_history
from sensitivity import (
    _model_output,
    compute_sensitivity,
    flatten_scores,
    pool_rows,
    pooled_group_boundaries,
    summarize_sensitivity,
)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def default_experiment_name() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class BaseConfig:
    """Settings every experiment shares. Subclass it to add dataset- and
    model-specific fields (and to give output_root a sensible default)."""

    seed: int = env_int("SEED", 0)
    output_root: str = env_str("OUTPUT_ROOT", "outputs")
    # Every run writes to output_root/experiment_name/. Set EXPERIMENT_NAME to
    # give a run a memorable name; otherwise it is timestamped so runs never
    # collide. Sweeps set it programmatically (e.g. "sweep/adam_lr0.001").
    experiment_name: str = env_str("EXPERIMENT_NAME", default_experiment_name())

    optimizer: str = env_str("OPTIMIZER", "adam")
    lr: float = env_float("LR", 1e-3)
    batch_size: int = env_int("BATCH_SIZE", 64)
    epochs: int = env_int("EPOCHS", 100)
    grad_clip: float = env_float("GRAD_CLIP", 0.0)
    # 0 disables checkpointing entirely (used for the cheap first pass of a
    # learning-rate sweep, where only the loss curves are wanted).
    checkpoint_interval: int = env_int("CHECKPOINT_INTERVAL", 1)

    # Sensitivity tracking. The probe set is a fixed subset held out from
    # training so that sensitivity is measured on data the model is not
    # directly fitting to at that step (though it remains drawn from the same
    # distribution as training), matching shakespeare_lstm/train.py.
    track_sensitivity: bool = env_bool("TRACK_SENSITIVITY", True)
    track_signed: bool = env_bool("TRACK_SIGNED", False)
    probe_samples: int = env_int("PROBE_SAMPLES", 512)
    probe_batch_size: int = env_int("PROBE_BATCH_SIZE", 128)
    sensitivity_probes: int = env_int("SENSITIVITY_PROBES", 8)
    sensitivity_interval: int = env_int("SENSITIVITY_INTERVAL", 1)
    # Rows in the parameter-wise sensitivity heatmap; parameters are
    # block-pooled down to this many rows for display (a no-op for models
    # with fewer parameters than rows -- pool_rows clamps).
    heatmap_rows: int = env_int("HEATMAP_ROWS", 300)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_name

    @property
    def kinds(self) -> Tuple[str, ...]:
        return ("unsigned", "signed") if self.track_signed else ("unsigned",)


@dataclass
class Experiment:
    """Everything run_training needs that is specific to one experiment.

    Built by a `build_experiment(cfg, device)` callback that must, in this
    order: seed nothing itself (run_training seeds first), build the data,
    build the loaders, then build the model. That order matters because
    rank_stability.py re-runs the same callback under the same seed to
    reconstruct the epoch-0 model exactly; anything that consumes the global
    RNG in between (e.g. constructing a model before the loaders) would have
    to be reproduced identically there too.

    loss_fn maps (model output, targets) -> scalar loss. It defaults to
    criterion(output, targets), which covers any model whose output lines up
    with its targets directly; a model needing a reshape first (e.g. a
    seq2seq LSTM's (B, T, V) logits against (B, T) targets) passes its own.
    """

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    probe_loader: DataLoader
    criterion: nn.Module
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None

    def __post_init__(self) -> None:
        if self.loss_fn is None:
            self.loss_fn = lambda output, targets: self.criterion(output, targets)


BuildExperiment = Callable[[BaseConfig, torch.device], Experiment]


def build_train_probe_loaders(cfg: BaseConfig, dataset) -> Tuple[DataLoader, DataLoader]:
    """Split a training pool into a training set and a fixed probe set, the
    latter held out so sensitivity is measured on data the model is not
    directly fitting (the convention set by shakespeare_lstm/train.py's
    build_loaders). The split is driven by its own seeded generator, so it
    does not depend on -- or perturb -- the global RNG stream that model
    initialization consumes."""
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(dataset), generator=generator).tolist()
    n_probe = min(cfg.probe_samples, max(0, len(dataset) - 1))
    probe_subset = torch.utils.data.Subset(dataset, perm[:n_probe])
    train_subset = torch.utils.data.Subset(dataset, perm[n_probe:])

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    probe_loader = DataLoader(probe_subset, batch_size=cfg.probe_batch_size, shuffle=False, drop_last=False)
    return train_loader, probe_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    forced = os.environ.get("DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    exp: Experiment,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    progress: bool,
) -> float:
    model = exp.model
    model.train()
    loss_sum = 0.0
    n = 0
    batches = tqdm(exp.train_loader, desc=f"epoch {epoch} [train]", leave=False, disable=not progress)
    for inputs, targets in batches:
        # No non_blocking=True: unsafe on MPS with non-pinned tensors, can
        # silently corrupt data (see common/sensitivity.py's note).
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = _model_output(model, inputs)
        loss = exp.loss_fn(output, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bsz = inputs.shape[0]
        loss_sum += float(loss.item()) * bsz
        n += bsz
        batches.set_postfix(loss=loss.item())
    return loss_sum / max(1, n)


@torch.no_grad()
def evaluate(exp: Experiment, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Mean loss and accuracy over `loader`, in eval mode. Accuracy assumes
    the output's last axis indexes classes and targets hold class indices,
    which holds for every classifier here (and for token-level targets too)."""
    model = exp.model
    model.eval()
    loss_sum = 0.0
    correct = 0
    n_samples = 0
    n_targets = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = _model_output(model, inputs)
        bsz = inputs.shape[0]
        loss_sum += float(exp.loss_fn(output, targets).item()) * bsz
        correct += int((output.argmax(dim=-1) == targets).sum().item())
        n_samples += bsz
        n_targets += targets.numel()
    return loss_sum / max(1, n_samples), correct / max(1, n_targets)


def _save_checkpoint(cfg: BaseConfig, model: nn.Module, epoch: int, out_dir: Path) -> Path:
    ckpt_path = out_dir / f"ckpt_epoch{epoch}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {**asdict(cfg), "output_dir": str(out_dir)},
            "epoch": epoch,
        },
        ckpt_path,
    )
    return ckpt_path


def run_training(
    cfg: BaseConfig,
    build_experiment: BuildExperiment,
    device: Optional[torch.device] = None,
    progress: bool = True,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Train, track, checkpoint and plot one experiment; returns its history.

    Writes into cfg.output_dir: history.json, loss_and_sensitivity.png,
    parameter_sensitivity_heatmap.png + _data.npz, and ckpt_epoch*.pt --
    the same artifacts, under the same names, as shakespeare_lstm/train.py,
    so the analysis scripts in common/ work unchanged on either.
    """
    set_seed(cfg.seed)
    if device is None:
        device = select_device()
    exp = build_experiment(cfg, device)
    optimizer = build_optimizer(cfg.optimizer, exp.model.parameters(), cfg.lr)

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    if verbose:
        print(f"Experiment: {cfg.experiment_name} -> {out_dir}")
        print(f"Device: {device} | optimizer: {cfg.optimizer} | lr: {cfg.lr:g} | parameters: {n_params}")
        print(
            f"Batches/epoch: {len(exp.train_loader)} | val samples: {len(exp.val_loader.dataset)} | "
            f"probe samples: {len(exp.probe_loader.dataset)}"
        )

    track = cfg.track_sensitivity and len(exp.probe_loader.dataset) > 0
    history: List[Dict[str, float]] = []
    heatmap_epochs: List[int] = []
    heatmap_cols: Dict[str, List[torch.Tensor]] = {kind: [] for kind in cfg.kinds}
    pooled_boundaries: List = []

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc=cfg.experiment_name, disable=not progress)
    for epoch in epoch_bar:
        start = time.time()
        loss = train_one_epoch(exp, optimizer, device, cfg.grad_clip, epoch, progress)
        val_loss, val_acc = evaluate(exp, exp.val_loader, device)
        elapsed = time.time() - start

        row: Dict[str, float] = {
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
        }

        should_track = track and (
            (epoch == 1) or (epoch % cfg.sensitivity_interval == 0) or (epoch == cfg.epochs)
        )
        if should_track:
            unsigned, signed = compute_sensitivity(
                exp.model, exp.probe_loader, device,
                n_probes=cfg.sensitivity_probes, show_progress=progress,
                include_signed=cfg.track_signed,
            )
            scores = {"unsigned": unsigned, "signed": signed}
            for kind in cfg.kinds:
                row.update({f"{kind}_{k}": v for k, v in summarize_sensitivity(scores[kind]).items()})
                flat, boundaries = flatten_scores(exp.model, scores[kind])
                if not pooled_boundaries:
                    pooled_boundaries = pooled_group_boundaries(boundaries, flat.numel(), cfg.heatmap_rows)
                heatmap_cols[kind].append(pool_rows(flat, cfg.heatmap_rows))
            heatmap_epochs.append(epoch)

        history.append(row)
        epoch_bar.set_postfix(loss=loss, val_loss=val_loss)
        if verbose:
            msg = f"epoch={epoch:3d} | loss={loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {elapsed:.1f}s"
            if "unsigned_total" in row:
                msg += f" | unsigned_sensitivity_total={row['unsigned_total']:.4e}"
            if "signed_total" in row:
                msg += f" | signed_sensitivity_total={row['signed_total']:.4e}"
            tqdm.write(msg)

        if cfg.checkpoint_interval > 0 and (epoch % cfg.checkpoint_interval == 0 or epoch == cfg.epochs):
            ckpt_path = _save_checkpoint(cfg, exp.model, epoch, out_dir)
            if verbose:
                tqdm.write(f"Saved checkpoint to {ckpt_path}")

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump({"config": {**asdict(cfg), "output_dir": str(out_dir)}, "history": history}, f, indent=2)
    if verbose:
        print(f"Saved training history to {history_path}")

    plot_path = out_dir / "loss_and_sensitivity.png"
    plot_training_history(history, plot_path, kinds=cfg.kinds)
    if verbose and plot_path.exists():
        print(f"Saved loss/sensitivity plot to {plot_path}")

    if heatmap_cols["unsigned"]:
        matrices = {kind: torch.stack(cols, dim=1).numpy() for kind, cols in heatmap_cols.items()}
        heatmap_data_path = out_dir / "parameter_sensitivity_heatmap_data.npz"
        np.savez(
            heatmap_data_path,
            epochs=np.array(heatmap_epochs),
            boundary_groups=np.array([g for g, _ in pooled_boundaries]),
            boundary_rows=np.array([r for _, r in pooled_boundaries]),
            n_rows=cfg.heatmap_rows,
            **matrices,
        )
        if verbose:
            print(f"Saved raw parameter-wise sensitivity matrices to {heatmap_data_path}")

        heatmap_path = out_dir / "parameter_sensitivity_heatmap.png"
        plot_sensitivity_heatmap(
            history, matrices["unsigned"], matrices.get("signed"), heatmap_epochs,
            pooled_boundaries, cfg.heatmap_rows, heatmap_path,
        )
        if verbose and heatmap_path.exists():
            print(f"Saved parameter-wise sensitivity heatmap to {heatmap_path}")

    return history
