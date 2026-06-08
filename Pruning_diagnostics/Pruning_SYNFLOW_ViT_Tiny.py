"""
ViT structured sensitivity-pruning experiment.

This script performs zero-shot structured sensitivity pruning at initialization,
then trains the remaining ViT parameters from scratch. It uses unit-level
score aggregation, layerwise normalization, budgeted selection, and a short
iterative refinement pass before training.

Default target: CIFAR-10 with a compact ViT. Use environment variables to change
model/data/training settings without editing the file.

Example:
    EPOCHS=100 PRUNE_FRACTION=0.25 BATCH_SIZE=128 python ViT_Sensitivity_Pruning_Experiment.py
"""

import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import *

from ViT_Model import *

from sensitivity_metrics import *

from sensitivity_pruning import *

from build_sparse_model import *

from training_tools import *

from pruning_baselines import *

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None

try:
    from torchvision import datasets, transforms
    _TORCHVISION_AVAILABLE = True
except Exception:
    datasets = None
    transforms = None
    _TORCHVISION_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PRUNING_METHOD = 'synflow'

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    seed: int = _env_int("SEED", 0)
    dataset: str = _env_str("DATASET", "CIFAR10")
    data_dir: str = _env_str("DATA_DIR", "./data")
    output_dir: str = _env_str("OUTPUT_DIR", "Plots/vit_synflow_pruning")
    download: bool = _env_bool("DOWNLOAD", True)

    image_size: int = _env_int("IMAGE_SIZE", 32)
    num_classes: int = _env_int("NUM_CLASSES", 10)
    train_subset: int = _env_int("TRAIN_SUBSET", 0)       # 0 means full train set
    test_subset: int = _env_int("TEST_SUBSET", 0)         # 0 means full test set
    sensitivity_samples: int = _env_int("SENSITIVITY_SAMPLES", 16384)
    sensitivity_batch_size: int = _env_int("SENSITIVITY_BATCH_SIZE", 512)
    sensitivity_probes: int = _env_int("SENSITIVITY_PROBES", 16)

    patch_size: int = _env_int("PATCH_SIZE", 4)
    embed_dim: int = _env_int("EMBED_DIM", 192)
    depth: int = _env_int("DEPTH", 12)
    num_heads: int = _env_int("NUM_HEADS", 3)
    mlp_ratio: float = _env_float("MLP_RATIO", 4.0)
    dropout: float = _env_float("DROPOUT", 0.08)

    prune_threshold: float = _env_float("PRUNE_THRESHOLD", 1e-4)
    prune_bias: bool = _env_bool("PRUNE_BIAS", True)
    prune_norm: bool = _env_bool("PRUNE_NORM", True)
    prune_embeddings: bool = _env_bool("PRUNE_EMBEDDINGS", True)
    prune_head: bool = _env_bool("PRUNE_HEAD", True)
    pruning_strategy: str = _env_str("PRUNING_STRATEGY", "structured")  # structured or threshold
    prune_fraction: float = _env_float("PRUNE_FRACTION", 0.99)
    iterative_pruning_rounds: int = _env_int("ITERATIVE_PRUNING_ROUNDS", 10)
    gradual_sparsification: bool = _env_bool("GRADUAL_SPARSIFICATION", True)
    layerwise_normalize_scores: bool = _env_bool("LAYERWISE_NORMALIZE_SCORES", True)
    sensitivity_normalization: str = _env_str("SENSITIVITY_NORMALIZATION", "rank")  # mad, zscore, rank, none
    sensitivity_clip_quantile: float = _env_float("SENSITIVITY_CLIP_QUANTILE", 0.05)
    min_embed_keep_fraction: float = _env_float("MIN_EMBED_KEEP_FRACTION", 0.10)
    min_hidden_keep_fraction: float = _env_float("MIN_HIDDEN_KEEP_FRACTION", 0.05)
    preserve_attention_heads: bool = _env_bool("PRESERVE_ATTENTION_HEADS", False)

    batch_size: int = _env_int("BATCH_SIZE", 256)
    epochs: int = _env_int("EPOCHS", 400)
    lr: float = _env_float("LR", 1e-2)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.08)
    warmup_epochs: int = _env_int("WARMUP_EPOCHS", 20)
    min_lr: float = _env_float("MIN_LR", 1e-8)
    num_workers: int = _env_int("NUM_WORKERS", 4)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    label_smoothing: float = _env_float("LABEL_SMOOTHING", 0.1)
    amp: bool = _env_bool("AMP", True)
    checkpoint_interval: int = _env_int("CHECKPOINT_INTERVAL", 10)

    # Sweep-style sensitivity diagnostics. These are bounded by default so the
    # script remains viable on ViT-scale parameter counts.
    topk_frac: float = _env_float("TOPK_FRAC", 0.10)
    reference_epoch: int = _env_int("REFERENCE_EPOCH", warmup_epochs)  # 0 means first checkpoint after epoch 1
    analysis_probes: int = _env_int("ANALYSIS_PROBES", 4)
    analysis_probe_matrix_rows: int = _env_int("ANALYSIS_PROBE_MATRIX_ROWS", 128)
    max_probe_matrix_elements: int = _env_int("MAX_PROBE_MATRIX_ELEMENTS", 50_000_000)
    save_sensitivity_heatmap: bool = _env_bool("SAVE_SENSITIVITY_HEATMAP", True)
    run_manifold: bool = _env_bool("RUN_MANIFOLD", False)
    full_jacobian_analysis: bool = _env_bool("FULL_JACOBIAN_ANALYSIS", False)
    max_full_jacobian_elements: int = _env_int("MAX_FULL_JACOBIAN_ELEMENTS", 20_000_000)

    # Connectivity-preserving threshold pruning. For dense weights this restores
    # at least this many incident coordinates for empty input/output units.
    connectivity_closure: bool = _env_bool("CONNECTIVITY_CLOSURE", True)
    min_connections_per_unit: int = _env_int("MIN_CONNECTIONS_PER_UNIT", 2)

    def __post_init__(self) -> None:
        if self.dataset.upper() not in {"CIFAR10", "CIFAR100"}:
            raise ValueError("DATASET must be CIFAR10 or CIFAR100.")
        if self.dataset.upper() == "CIFAR100" and self.num_classes == 10:
            self.num_classes = 100
        if self.prune_threshold < 0.0:
            raise ValueError("PRUNE_THRESHOLD must be non-negative.")
        if not (0.0 <= self.prune_fraction < 1.0):
            raise ValueError("PRUNE_FRACTION must be in [0, 1).")
        if self.iterative_pruning_rounds < 1:
            raise ValueError("ITERATIVE_PRUNING_ROUNDS must be at least 1.")
        if self.pruning_strategy.lower() not in {"structured", "threshold"}:
            raise ValueError("PRUNING_STRATEGY must be 'structured' or 'threshold'.")
        if not isinstance(self.gradual_sparsification, bool):
            raise ValueError("GRADUAL_SPARSIFICATION must be a boolean flag.")
        if self.sensitivity_normalization.lower() not in {"mad", "zscore", "rank", "none"}:
            raise ValueError("SENSITIVITY_NORMALIZATION must be one of: mad, zscore, rank, none.")
        if not (0.0 <= self.sensitivity_clip_quantile < 0.5):
            raise ValueError("SENSITIVITY_CLIP_QUANTILE must be in [0, 0.5).")
        if not (0.0 <= self.min_embed_keep_fraction <= 1.0):
            raise ValueError("MIN_EMBED_KEEP_FRACTION must be in [0, 1].")
        if not (0.0 <= self.min_hidden_keep_fraction <= 1.0):
            raise ValueError("MIN_HIDDEN_KEEP_FRACTION must be in [0, 1].")
        if self.image_size % self.patch_size != 0:
            raise ValueError("IMAGE_SIZE must be divisible by PATCH_SIZE.")
        self.checkpoint_interval = max(1, self.checkpoint_interval)


cfg = Config()


# -----------------------------------------------------------------------------
# Reproducibility and device
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(cfg.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

out_dir = Path(cfg.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

train_set, sens_set, test_set = build_datasets(cfg)
pin_memory = device.type == "cuda"
train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=pin_memory,
    persistent_workers=cfg.num_workers > 0,
)
sens_loader = DataLoader(
    sens_set,
    batch_size=cfg.sensitivity_batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=pin_memory,
    persistent_workers=cfg.num_workers > 0,
)
test_loader = DataLoader(
    test_set,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=pin_memory,
    persistent_workers=cfg.num_workers > 0,
)


# -----------------------------------------------------------------------------
# ViT model
# -----------------------------------------------------------------------------

model = VisionTransformer(
    image_size=cfg.image_size,
    patch_size=cfg.patch_size,
    num_classes=cfg.num_classes,
    embed_dim=cfg.embed_dim,
    depth=cfg.depth,
    num_heads=cfg.num_heads,
    mlp_ratio=cfg.mlp_ratio,
    dropout=cfg.dropout,
).to(device)


# -----------------------------------------------------------------------------
# Initial pruning
# -----------------------------------------------------------------------------

print(f"Using device: {device}")
print(f"Dataset: {cfg.dataset.upper()} | train={len(train_set)} | sensitivity={len(sens_set)} | test={len(test_set)}")
print(f"Model trainable parameters before pruning: {trainable_parameter_count(model):,}")
print("Computing zero-shot sensitivity scores at initialization")

_ref_model = copy.deepcopy(model)

if cfg.pruning_strategy.lower() == "structured":
    masks, pruning_stats, embed_sel, hidden_sel, sensitivity_scores = build_structured_masks_iterative(
        _ref_model, sens_loader, cfg, device
    )
else:
    sensitivity_scores, _ = compute_sensitivity_scores(_ref_model, sens_loader, cfg, device, probes=cfg.sensitivity_probes)
    masks, pruning_stats = make_threshold_connectivity_masks(_ref_model, sensitivity_scores, cfg)

# Apply baseline pruning method

prune_images, prune_targets = next(iter(train_loader))
prune_images = prune_images.to(device)
prune_targets = prune_targets.to(device)

# PRUNING SEARCH

target_actual_prune_fraction = float(pruning_stats['actual_prune_fraction_eligible'])

masks, baseline_pruning_stats, baseline_effective_fraction = build_calibrated_baseline_masks(
    model,
    prune_images,
    prune_targets,
    target_actual_prune_fraction,
    cfg,
    device,
    PRUNING_METHOD,
    _ref_model,
)

pruning_stats.update(baseline_pruning_stats)
pruning_stats["baseline_effective_prune_fraction"] = baseline_effective_fraction

# Apply pruning mask

S_init_flat_all = flatten_like_model(model, sensitivity_scores)
apply_masks_(model, masks)
S_init_active = flatten_like_model(model, sensitivity_scores, only_active=masks)
init_topk_idx = topk_indices(S_init_active, cfg.topk_frac)
initial_param_mag_active = flatten_param_magnitudes(model)

print(f"Structured pruning complete via strategy={cfg.pruning_strategy.lower()}")

# -----------------------------------------------------------------------------
# Training, evaluation, and checkpointed sensitivity analysis
# -----------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

# Rebuild a compact model and train that model directly.
model = build_sparse_model_from_masks(model, masks, cfg, device)

pruning_stats.update(compute_eligible_pruning_stats(_ref_model, model, masks, cfg))

print(json.dumps(pruning_stats, indent=2))
print("Training the pruned ViT from scratch")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

history: List[Dict[str, float]] = []
S_ref_active: Optional[torch.Tensor] = None
eig_init: Optional[torch.Tensor] = None
eig_ref: Optional[torch.Tensor] = None
eig_final: Optional[torch.Tensor] = None
ref_topk_idx: Optional[torch.Tensor] = None
ref_epoch: Optional[int] = None
mean_abs_sensitivity_history: List[np.ndarray] = []

# Recompute the baseline on the rebuilt sparse model so tensor shapes match.
S_init_dict, init_probe_matrix = compute_sensitivity_scores(
    model,
    sens_loader,
    cfg,
    device,
    probes=cfg.analysis_probes,
    collect_probe_matrix=True,
)
S_init_active = flatten_like_model(model, S_init_dict)
init_topk_idx = topk_indices(S_init_active, cfg.topk_frac)
initial_param_mag_active = flatten_param_magnitudes(model)
eig_init = probe_covariance_eigvals(init_probe_matrix)

print("\nTraining")
print("========")

for epoch in range(1, cfg.epochs + 1):
    epoch_lr = compute_epoch_lr(epoch, cfg)
    for group in optimizer.param_groups:
        group["lr"] = epoch_lr

    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bsz = images.shape[0]
        loss_sum += float(loss.item()) * bsz
        acc_sum += accuracy(logits.detach(), targets) * bsz
        n += bsz

    train_metrics = {"loss": loss_sum / max(1, n), "accuracy": acc_sum / max(1, n)}
    should_eval = (epoch == 1) or (epoch % cfg.checkpoint_interval == 0) or (epoch == cfg.epochs)
    if should_eval:
        test_metrics = evaluate(model, test_loader, criterion, device)
        S_curr, probe_matrix = compute_sensitivity_scores(
            model, sens_loader, cfg, device, probes=cfg.analysis_probes,
            collect_probe_matrix=(epoch == cfg.epochs),
        )
        S_curr_active = flatten_like_model(model, S_curr)
        rho_init_curr = spearman_corr(S_init_active, S_curr_active)
        init_topk_mass = mass_on_indices(S_curr_active, init_topk_idx)

        if S_ref_active is None and (cfg.reference_epoch <= 0 or epoch >= cfg.reference_epoch):
            S_ref_active = S_curr_active.detach().clone()
            ref_topk_idx = topk_indices(S_ref_active, cfg.topk_frac)
            ref_epoch = epoch
            eig_ref = probe_covariance_eigvals(probe_matrix)
            print(f"Captured sensitivity reference snapshot at epoch={ref_epoch}")

        rho_ref_curr = spearman_corr(S_ref_active, S_curr_active) if S_ref_active is not None else float("nan")
        ref_topk_mass = mass_on_indices(S_curr_active, ref_topk_idx) if ref_topk_idx is not None else float("nan")

        mean_abs_sensitivity_history.append(S_curr_active.detach().cpu().numpy()[None, :])

        row = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "spearman_init_current": float(rho_init_curr),
            "init_topk_mass": float(init_topk_mass),
            "spearman_ref_current": float(rho_ref_curr),
            "ref_topk_mass": float(ref_topk_mass),
        }
        history.append(row)
        print(
            f"epoch={epoch:4d} | lr={row['lr']:.3e} | "
            f"train_loss={row['train_loss']:.4f} | train_acc={100.0 * row['train_accuracy']:.2f}% | "
            f"test_loss={row['test_loss']:.4f} | test_acc={100.0 * row['test_accuracy']:.2f}% | "
            f"rho(init,current)={row['spearman_init_current']:.3f} | "
            f"init-topk-mass={row['init_topk_mass']:.3f} | "
            f"rho(ref,current)={row['spearman_ref_current']:.3f} | "
            f"ref-topk-mass={row['ref_topk_mass']:.3f}"
        )

# Final sensitivity and approximate covariance spectrum.
S_final_dict, final_probe_matrix = compute_sensitivity_scores(
    model, sens_loader, cfg, device, probes=cfg.analysis_probes,
    collect_probe_matrix=True,
)
S_final_active = flatten_like_model(model, S_final_dict)
eig_final = probe_covariance_eigvals(final_probe_matrix)
final_param_mag_active = flatten_param_magnitudes(model)
final_metrics = evaluate(model, test_loader, criterion, device)
module_density = {}
for name, p in model.named_parameters():
    group = name.split(".")[0]
    stats = module_density.setdefault(group, {"retained": 0, "total": 0})
    stats["retained"] += p.numel()
    stats["total"] += p.numel()
module_density = {
    group: {"retained": v["retained"], "total": v["total"], "density": v["retained"] / max(1, v["total"])}
    for group, v in module_density.items()
}


# -----------------------------------------------------------------------------
# Save outputs and sweep-style plots
# -----------------------------------------------------------------------------

run_tag = f"{cfg.dataset.lower()}_vit_tiny_structured_{parameter_count(model)}_params"
summary = {
    "config": asdict(cfg),
    "device": str(device),
    "parameter_count": parameter_count(model),
    "trainable_parameter_count": trainable_parameter_count(model),
    "active_parameter_count": trainable_parameter_count(model),
    "pruning": pruning_stats,
    "module_density": module_density,
    "final_test_loss": final_metrics["loss"],
    "final_test_accuracy": final_metrics["accuracy"],
    "final_spearman_init_final": spearman_corr(S_init_active, S_final_active),
    "final_mass_in_init_topk": mass_on_indices(S_final_active, init_topk_idx),
    "reference_epoch": ref_epoch,
    "final_spearman_ref_final": spearman_corr(S_ref_active, S_final_active) if S_ref_active is not None else None,
    "final_mass_in_ref_topk": mass_on_indices(S_final_active, ref_topk_idx) if ref_topk_idx is not None else None,
    "largest_initial_probe_cov_eigenvalue": float(eig_init[0].item()) if eig_init is not None and eig_init.numel() else None,
    "largest_ref_probe_cov_eigenvalue": float(eig_ref[0].item()) if eig_ref is not None and eig_ref.numel() else None,
    "largest_final_probe_cov_eigenvalue": float(eig_final[0].item()) if eig_final is not None and eig_final.numel() else None,
    "probe_cov_effective_rank_final": effective_rank(eig_final) if eig_final is not None else None,
    "history": history,
    "analysis_note": "Improved structured sensitivity pruning uses robust within-tensor normalization, per-group minimum retention, connectivity closure, and iterative refinement; set PRUNING_STRATEGY=threshold to use a thresholded sensitivity mask.",
}

summary_path = out_dir / "vit_structured_sensitivity_pruning_summary.json"
save_json(summary_path, summary)

if plt is not None and history:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    epochs = [int(h["epoch"]) for h in history]

    def _savefig(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["test_loss"] for h in history], label="test")
    ax.set_yscale("log")
    ax.set_title("Loss evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross entropy")
    ax.legend()
    _savefig(fig, f"Loss_Evolution_{run_tag}.pdf")

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot(epochs, [100.0 * h["train_accuracy"] for h in history], label="train")
    ax.plot(epochs, [100.0 * h["test_accuracy"] for h in history], label="test")
    ax.set_title("Accuracy evolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    _savefig(fig, f"Accuracy_Evolution_{run_tag}.pdf")

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot(epochs, [h["spearman_init_current"] for h in history], label="init/current")
    ax.plot(epochs, [h["spearman_ref_current"] for h in history], label="ref/current")
    ax.set_title("Sensitivity-rank stability")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman correlation")
    ax.legend()
    _savefig(fig, f"Sensitivity_Spearman_Stability_{run_tag}.pdf")

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot(epochs, [h["init_topk_mass"] for h in history], label="initial top-k")
    ax.plot(epochs, [h["ref_topk_mass"] for h in history], label="reference top-k")
    ax.set_title("Sensitivity mass retained in top-k sets")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mass fraction")
    ax.legend()
    _savefig(fig, f"Sensitivity_TopK_Mass_{run_tag}.pdf")

    if eig_init is not None and eig_final is not None:
        fig, ax = plt.subplots(figsize=(7.4, 6.0))
        ax.plot(eig_init.numpy(), label="initial")
        if eig_ref is not None:
            ax.plot(eig_ref.numpy(), label=f"ref @ epoch {ref_epoch}", linestyle="--")
        ax.plot(eig_final.numpy(), label="final")
        ax.set_yscale("log")
        ax.set_title(f"Probe-gradient covariance eigenspectrum (effective rank = {effective_rank(eig_final):.2f})")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.legend()
        _savefig(fig, f"Eigenspectrum_{run_tag}.pdf")

    S_init_np = S_init_active.detach().cpu().numpy()
    S_final_np = S_final_active.detach().cpu().numpy()
    positive_vals = np.concatenate([S_init_np[S_init_np > 0], S_final_np[S_final_np > 0]])
    positive_vals = positive_vals[np.isfinite(positive_vals)]
    bins = np.logspace(np.log10(positive_vals.min()), np.log10(positive_vals.max()), 50) if positive_vals.size else 50
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.hist(S_init_np, bins=bins, density=True, alpha=0.42, label="initial", histtype="stepfilled", linewidth=0.9)
    ax.hist(S_final_np, bins=bins, density=True, alpha=0.42, label="final", histtype="stepfilled", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Active-subnet sensitivity distribution")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Density")
    ax.legend()
    _savefig(fig, f"Sensitivity_Distribution_Initial_vs_Final_{run_tag}.pdf")

    eps = 1e-30
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)
    axes[0].scatter(initial_param_mag_active.clamp_min(eps).numpy(), S_init_active.clamp_min(eps).numpy(), s=8, alpha=0.45, rasterized=True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Before training (active subnet)")
    axes[0].set_xlabel(r"$|\theta_i|$")
    axes[0].set_ylabel(r"$S(\theta_i)$")
    axes[1].scatter(final_param_mag_active.clamp_min(eps).numpy(), S_final_active.clamp_min(eps).numpy(), s=8, alpha=0.45, rasterized=True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("After training (active subnet)")
    axes[1].set_xlabel(r"$|\theta_i|$")
    axes[1].set_ylabel(r"$S(\theta_i)$")
    fig.suptitle("Unnormalised sensitivity vs parameter magnitude", fontsize=13)
    fig.savefig(out_dir / f"Sensitivity_vs_Parameter_Magnitude_{run_tag}.pdf")
    plt.close(fig)

    if cfg.save_sensitivity_heatmap and mean_abs_sensitivity_history:
        Q = np.concatenate(mean_abs_sensitivity_history, axis=0)  # [checkpoints, active_params]
        max_heatmap_params = min(Q.shape[1], 5000)
        if Q.shape[1] > max_heatmap_params:
            # Pick highest final-sensitivity coordinates so the heatmap stays legible and bounded.
            order = np.argsort(S_final_np)[::-1][:max_heatmap_params]
            Q_plot = Q[:, order]
        else:
            Q_plot = Q
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        im = ax.imshow(Q_plot.T, aspect="auto", origin="lower", interpolation="nearest", extent=[epochs[0], epochs[-1], -0.5, Q_plot.shape[1] - 0.5])
        ax.set_title("Absolute sensitivity over training (active subnet; top coordinates if truncated)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Active parameter index")
        fig.colorbar(im, ax=ax, pad=0.04, fraction=0.046).set_label(r"estimated $S(\theta_i)$")
        _savefig(fig, f"Absolute_Sensitivity_Over_Training_{run_tag}.pdf")

    sort_idx = np.argsort(S_final_np)[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(np.arange(S_final_np.size), S_final_np[sort_idx], s=8, alpha=0.65, linewidths=0.0, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Active-subnet sorted sensitivity spectrum")
    ax.set_xlabel("Parameter rank")
    ax.set_ylabel("Sensitivity")
    _savefig(fig, f"Sensitivity_Rank_Spectrum_{run_tag}.pdf")

print("\nFinal summary")
print("=============")
print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
print(f"Wrote summary to {summary_path}")
