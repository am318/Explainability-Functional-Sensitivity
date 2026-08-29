"""
Word-level AWD-LSTM (see model.py) on WikiText-2, instrumented to track
parameter-wise functional sensitivity (see common/sensitivity.py) over
training -- same tracking pipeline, same plots as shakespeare_lstm/, so the
two experiments are directly comparable.

Motivation: the char-LSTM in shakespeare_lstm/ turned out to badly overfit
(train loss << held-out loss). This experiment swaps in an architecture
(Merity et al. AWD-LSTM) whose specific purpose is closing that gap via
DropConnect + variational/locked dropout + embedding dropout + weight tying,
to see whether the sensitivity/pruning results are an artifact of that
overfitting or hold up on a model that generalizes properly.

Runs on CUDA, MPS, or CPU. Use environment variables to change settings
without editing the file, e.g.:

    EPOCHS=1 BATCH_SIZE=32 SENSITIVITY_INTERVAL=1 python train.py
"""

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from dataset import WordSequenceDataset, build_corpus
from model import AWDLSTM
from plotting import plot_sensitivity_heatmap, plot_training_history
from sensitivity import (
    compute_sensitivity,
    flatten_scores,
    pool_rows,
    pooled_group_boundaries,
    summarize_sensitivity,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _default_experiment_name() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class Config:
    seed: int = _env_int("SEED", 0)
    data_dir: str = _env_str(
        "DATA_DIR", str(Path(__file__).resolve().parent / "dataset" / "wikitext-2")
    )
    output_root: str = _env_str("OUTPUT_ROOT", str(Path(__file__).resolve().parent / "outputs"))
    experiment_name: str = _env_str("EXPERIMENT_NAME", _default_experiment_name())

    seq_length: int = _env_int("SEQ_LENGTH", 35)
    embedding_dim: int = _env_int("EMBEDDING_DIM", 256)
    rnn_units: int = _env_int("RNN_UNITS", 256)
    nlayers: int = _env_int("NLAYERS", 2)
    dropout: float = _env_float("DROPOUT", 0.4)
    dropouth: float = _env_float("DROPOUTH", 0.25)
    dropouti: float = _env_float("DROPOUTI", 0.4)
    dropoute: float = _env_float("DROPOUTE", 0.1)
    wdrop: float = _env_float("WDROP", 0.5)

    batch_size: int = _env_int("BATCH_SIZE", 64)
    epochs: int = _env_int("EPOCHS", 40)
    lr: float = _env_float("LR", 1e-3)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    checkpoint_interval: int = _env_int("CHECKPOINT_INTERVAL", 5)

    # Sensitivity tracking. Probe set = a fixed subset of the REAL validation
    # split (see dataset.py), so sensitivity is measured on genuinely
    # held-out data, not a slice of training data.
    sensitivity_chunks: int = _env_int("SENSITIVITY_CHUNKS", 512)
    sensitivity_batch_size: int = _env_int("SENSITIVITY_BATCH_SIZE", 64)
    sensitivity_probes: int = _env_int("SENSITIVITY_PROBES", 4)
    sensitivity_interval: int = _env_int("SENSITIVITY_INTERVAL", 1)
    heatmap_rows: int = _env_int("HEATMAP_ROWS", 300)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.experiment_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tune_backend(device: torch.device) -> None:
    """Cheap, results-neutral throughput knobs for CUDA. TF32 tensor-core
    matmul (the tied (seq*batch, embed) x (embed, vocab) head dominates the
    FLOPs) and cuDNN autotuning are standard "why is my H100 idle" fixes;
    input shapes here are fixed (drop_last=True, fixed SEQ_LENGTH) so
    benchmark=True has a stable set of shapes to tune against. Neither
    changes the training math in any way that matters for loss or
    sensitivity magnitudes. (The WeightDrop recompaction warning from
    torch/nn/modules/rnn.py is unrelated and unavoidable: WeightDrop swaps a
    fresh non-Parameter weight_hh in every forward, so cuDNN cannot keep the
    LSTM weights in one contiguous block -- that cost is intrinsic to
    DropConnect-on-cuDNN and is not what these flags address.)"""
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    forced = os.environ.get("DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: Config, vocab_size: int, device: torch.device) -> AWDLSTM:
    return AWDLSTM(
        vocab_size, cfg.embedding_dim, cfg.rnn_units, cfg.nlayers,
        cfg.dropout, cfg.dropouth, cfg.dropouti, cfg.dropoute, cfg.wdrop,
        tie_weights=True,
    ).to(device)


def build_loaders(cfg: Config, device: Optional[torch.device] = None) -> tuple:
    """Returns (train_loader, probe_loader, vocab_size). Train batches are
    shuffled chunks of the train split; probe batches are a fixed (seeded)
    subset of chunks from the real validation split, capped at
    cfg.sensitivity_chunks."""
    if device is None:
        device = select_device()
    dictionary, train_ids, valid_ids, _test_ids = build_corpus(cfg.data_dir)

    train_dataset = WordSequenceDataset(train_ids, cfg.seq_length)
    valid_dataset = WordSequenceDataset(valid_ids, cfg.seq_length)

    n_probe = min(cfg.sensitivity_chunks, len(valid_dataset))
    generator = torch.Generator().manual_seed(cfg.seed)
    probe_indices = torch.randperm(len(valid_dataset), generator=generator)[:n_probe].tolist()
    probe_subset = Subset(valid_dataset, probe_indices)

    # On CUDA, overlap host->device copies with compute: pinned staging
    # buffers + a couple of loader workers + non_blocking .to() in the loop.
    # NUM_WORKERS is overridable (0 keeps the old fully-synchronous path,
    # e.g. for debugging). pin_memory is CUDA-only; on MPS/CPU it is
    # pointless and pinned+non_blocking is the exact combination the
    # dataset.py comment warns is unsafe on MPS, so it stays off there.
    num_workers = _env_int("NUM_WORKERS", 4 if device.type == "cuda" else 0)
    pin_memory = device.type == "cuda"
    loader_kw = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    probe_loader = DataLoader(
        probe_subset, batch_size=cfg.sensitivity_batch_size, shuffle=False, drop_last=False, **loader_kw
    )
    return train_loader, probe_loader, len(dictionary)


def train_one_epoch(
    model: AWDLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    loss_sum = 0.0
    n = 0
    # non_blocking is safe only for CUDA with pinned host memory (see
    # build_loaders); on MPS/CPU an async copy of a non-pinned tensor can
    # silently corrupt data (see shakespeare_lstm/pruning_experiment.py).
    non_blocking = device.type == "cuda"
    batches = tqdm(loader, desc=f"epoch {epoch} [train]", leave=False)
    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bsz = inputs.shape[0]
        loss_sum += float(loss.item()) * bsz
        n += bsz
        batches.set_postfix(loss=loss.item())
    return loss_sum / max(1, n)


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = select_device()
    tune_backend(device)
    print(f"Using device: {device}")
    print(f"Experiment: {cfg.experiment_name} -> {cfg.output_dir}")

    train_loader, probe_loader, vocab_size = build_loaders(cfg, device)
    print(
        f"Vocab size: {vocab_size}, train chunks: {len(train_loader.dataset)}, "
        f"probe chunks: {len(probe_loader.dataset)}, {len(train_loader)} train batches/epoch"
    )

    model = build_model(cfg, vocab_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []
    heatmap_epochs: List[int] = []
    unsigned_heatmap_cols: List[torch.Tensor] = []
    signed_heatmap_cols: List[torch.Tensor] = []
    pooled_boundaries: List = []

    epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="training")
    for epoch in epoch_bar:
        start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip, epoch)
        elapsed = time.time() - start

        row: Dict[str, float] = {"epoch": epoch, "loss": loss, "elapsed_sec": elapsed}

        should_track = (
            (epoch == 1) or (epoch % cfg.sensitivity_interval == 0) or (epoch == cfg.epochs)
        )
        if should_track and len(probe_loader.dataset) > 0:
            unsigned, signed = compute_sensitivity(
                model, probe_loader, device, n_probes=cfg.sensitivity_probes, show_progress=True
            )
            unsigned_summary = summarize_sensitivity(unsigned)
            signed_summary = summarize_sensitivity(signed)
            row.update({f"unsigned_{k}": v for k, v in unsigned_summary.items()})
            row.update({f"signed_{k}": v for k, v in signed_summary.items()})

            unsigned_flat, boundaries = flatten_scores(model, unsigned)
            signed_flat, _ = flatten_scores(model, signed)
            if not pooled_boundaries:
                pooled_boundaries = pooled_group_boundaries(boundaries, unsigned_flat.numel(), cfg.heatmap_rows)
            unsigned_heatmap_cols.append(pool_rows(unsigned_flat, cfg.heatmap_rows))
            signed_heatmap_cols.append(pool_rows(signed_flat, cfg.heatmap_rows))
            heatmap_epochs.append(epoch)

        history.append(row)
        epoch_bar.set_postfix(loss=loss, **({"S": row.get("unsigned_total")} if "unsigned_total" in row else {}))
        msg = f"epoch={epoch:3d} | loss={loss:.4f} | {elapsed:.1f}s"
        if "unsigned_total" in row:
            msg += f" | unsigned_sensitivity_total={row['unsigned_total']:.4e} | signed_sensitivity_total={row['signed_total']:.4e}"
        tqdm.write(msg)

        if epoch % cfg.checkpoint_interval == 0 or epoch == cfg.epochs:
            ckpt_path = out_dir / f"ckpt_epoch{epoch}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": vocab_size,
                    "config": {**asdict(cfg), "output_dir": str(out_dir)},
                    "epoch": epoch,
                },
                ckpt_path,
            )
            tqdm.write(f"Saved checkpoint to {ckpt_path}")

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump({"config": {**asdict(cfg), "output_dir": str(out_dir)}, "history": history}, f, indent=2)
    print(f"Saved training history to {history_path}")

    plot_path = out_dir / "loss_and_sensitivity.png"
    plot_training_history(history, plot_path)
    if plot_path.exists():
        print(f"Saved loss/sensitivity plot to {plot_path}")

    if unsigned_heatmap_cols:
        unsigned_matrix = torch.stack(unsigned_heatmap_cols, dim=1).numpy()
        signed_matrix = torch.stack(signed_heatmap_cols, dim=1).numpy()

        heatmap_data_path = out_dir / "parameter_sensitivity_heatmap_data.npz"
        np.savez(
            heatmap_data_path,
            unsigned=unsigned_matrix,
            signed=signed_matrix,
            epochs=np.array(heatmap_epochs),
            boundary_groups=np.array([g for g, _ in pooled_boundaries]),
            boundary_rows=np.array([r for _, r in pooled_boundaries]),
            n_rows=cfg.heatmap_rows,
        )
        print(f"Saved raw parameter-wise sensitivity matrices to {heatmap_data_path}")

        heatmap_path = out_dir / "parameter_sensitivity_heatmap.png"
        plot_sensitivity_heatmap(
            history, unsigned_matrix, signed_matrix, heatmap_epochs, pooled_boundaries, cfg.heatmap_rows, heatmap_path
        )
        if heatmap_path.exists():
            print(f"Saved parameter-wise sensitivity heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
