"""
Sensitivity-only pruning-at-initialization sweep for a small transformer.

Target architecture
-------------------
ViT-Tiny on CIFAR-10. This is the smallest option in the proposed benchmark
set that is still representative of the parameter-structure patterns that make
larger transformer pruning different from fully connected MLP pruning:

  * patch embedding convolution
  * class token and positional embedding
  * multi-head self-attention Q/K/V/projection matrices
  * transformer MLP blocks
  * LayerNorm and classifier head

This script is intentionally separate from Pruning_Experimental_Sweep.py because
that script is coupled to synthetic regression/vector-field datasets, MSE loss,
MLP-specific masks, and full parameter-Jacobian diagnostics. This version keeps
the same experimental spirit: compute initialization-time parameter sensitivity,
prune once before training, keep masks fixed, and track diagnostic plots during
training.

Only sensitivity pruning is included. SNIP, GraSP, and SynFlow are deliberately
not implemented here.

Typical runs
------------
CPU smoke test:
    EPOCHS=1 N_TRAIN_SUBSET=512 N_TEST_SUBSET=512 N_SENS_BATCHES=2 \
    python Pruning_Sensitivity_ViT_Tiny.py

Main sparse runs:
    SPARSITY=0.95 EPOCHS=200 python Pruning_Sensitivity_ViT_Tiny.py
    SPARSITY=0.99 EPOCHS=200 python Pruning_Sensitivity_ViT_Tiny.py

Optional knobs:
    DEVICE=cuda
    OUTPUT_DIR=Plots/vit_tiny_cifar10_sensitivity
    BATCH_SIZE=128
    LR=3e-4
    WEIGHT_DECAY=0.05
    N_SENS_BATCHES=8
    SENS_PROBES=1
    PRUNE_SCOPE=weights_only          # weights_only or all_prunable
    MASK_CLASSIFIER=0                 # default: keep classifier dense
    MASK_PATCH_EMBED=1                # default: prune patch embedding conv
    RUN_TRAINING=1

Notes on the sensitivity criterion
----------------------------------
For each sensitivity minibatch, the script estimates parameter-level output
Jacobian energy at initialization using a Hutchinson-style probe:

    S_i ~= E_x,v [(d <f_theta(x), v> / d theta_i)^2]

where f_theta(x) are logits and v is a random Rademacher vector with the same
shape as the logits. This is a transformer-compatible analogue of the output
Jacobian sensitivity diagnostics used in the original MLP sweep, without
materialising a full [batch, classes, parameters] Jacobian.

Pruning convention
------------------
At sparsity s, the script keeps the top (1 - s) fraction of prunable parameters
by initialization sensitivity and hard-zeros the rest. The binary masks are
reapplied after every optimiser step, and optimiser state is zeroed on pruned
coordinates.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    import torchvision
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires torchvision for CIFAR-10 loading. Install torchvision "
        "or adapt make_dataloaders() to your local dataset pipeline."
    ) from exc


# ============================================================
# Environment config
# ============================================================


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    seed: int = _env_int("SEED", 0)
    device: str = _env_str("DEVICE", "auto")

    dataset: str = _env_str("DATASET", "cifar10")
    data_dir: str = _env_str("DATA_DIR", "./data")
    output_dir: str = _env_str("OUTPUT_DIR", "Plots/vit_tiny_cifar10_sensitivity")
    num_workers: int = _env_int("NUM_WORKERS", 2)

    # Optional subsets are useful for smoke tests and laptop diagnostics.
    n_train_subset: int = _env_int("N_TRAIN_SUBSET", 0)
    n_test_subset: int = _env_int("N_TEST_SUBSET", 0)

    # ViT-Tiny/CIFAR parameters. The default is intentionally small but still
    # structurally transformer-like.
    image_size: int = _env_int("IMAGE_SIZE", 32)
    patch_size: int = _env_int("PATCH_SIZE", 4)
    embed_dim: int = _env_int("EMBED_DIM", 192)
    depth: int = _env_int("DEPTH", 12)
    num_heads: int = _env_int("NUM_HEADS", 3)
    mlp_ratio: float = _env_float("MLP_RATIO", 4.0)
    dropout: float = _env_float("DROPOUT", 0.0)

    batch_size: int = _env_int("BATCH_SIZE", 128)
    epochs: int = _env_int("EPOCHS", 100)
    lr: float = _env_float("LR", 3e-4)
    weight_decay: float = _env_float("WEIGHT_DECAY", 5e-2)
    warmup_epochs: int = _env_int("WARMUP_EPOCHS", 5)
    lr_eta_min: float = _env_float("LR_ETA_MIN", 1e-6)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)

    # Pruning controls.
    sparsity: float = _env_float("SPARSITY", 0.95)
    prune_scope: str = _env_str("PRUNE_SCOPE", "weights_only")  # weights_only or all_prunable
    mask_classifier: bool = _env_int("MASK_CLASSIFIER", 0) != 0
    mask_patch_embed: bool = _env_int("MASK_PATCH_EMBED", 1) != 0

    # Sensitivity estimation controls.
    n_sens_batches: int = _env_int("N_SENS_BATCHES", 8)
    sens_probes: int = _env_int("SENS_PROBES", 1)
    sensitivity_batch_size: int = _env_int("SENS_BATCH_SIZE", 0)  # 0 -> batch_size

    # Diagnostics.
    checkpoint_interval: int = _env_int("CHECKPOINT_INTERVAL", 5)
    topk_frac: float = _env_float("TOPK_FRAC", 0.01)
    run_training: bool = _env_int("RUN_TRAINING", 1) != 0
    compute_final_curvature_proxy: bool = _env_int("COMPUTE_FINAL_CURVATURE_PROXY", 1) != 0

    def __post_init__(self) -> None:
        if self.dataset.lower() != "cifar10":
            raise ValueError("Only DATASET=cifar10 is implemented in this minimal ViT-Tiny script.")
        if not (0.0 <= self.sparsity < 1.0):
            raise ValueError("SPARSITY must be in [0, 1).")
        if self.prune_scope not in {"weights_only", "all_prunable"}:
            raise ValueError("PRUNE_SCOPE must be 'weights_only' or 'all_prunable'.")
        if self.sensitivity_batch_size <= 0:
            self.sensitivity_batch_size = self.batch_size
        self.checkpoint_interval = max(1, self.checkpoint_interval)


# ============================================================
# Model: compact ViT for CIFAR-10
# ============================================================


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTinyCIFAR(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ============================================================
# Data
# ============================================================


def make_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(cfg.data_dir, train=True, transform=train_tf, download=True)
    test_ds = torchvision.datasets.CIFAR10(cfg.data_dir, train=False, transform=eval_tf, download=True)
    sens_ds = torchvision.datasets.CIFAR10(cfg.data_dir, train=True, transform=eval_tf, download=True)

    rng = np.random.default_rng(cfg.seed)
    if cfg.n_train_subset > 0:
        idx = rng.permutation(len(train_ds))[: cfg.n_train_subset]
        train_ds = Subset(train_ds, idx.tolist())
    if cfg.n_test_subset > 0:
        idx = rng.permutation(len(test_ds))[: cfg.n_test_subset]
        test_ds = Subset(test_ds, idx.tolist())

    # Sensitivity estimation uses deterministic transforms and a fixed subset.
    n_sens = min(len(sens_ds), cfg.n_sens_batches * cfg.sensitivity_batch_size)
    sens_idx = rng.permutation(len(sens_ds))[:n_sens]
    sens_ds = Subset(sens_ds, sens_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    sens_loader = DataLoader(
        sens_ds, batch_size=cfg.sensitivity_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, test_loader, sens_loader


# ============================================================
# Pruning utilities
# ============================================================


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def is_prunable_parameter(name: str, p: nn.Parameter, cfg: Config) -> bool:
    if not p.requires_grad:
        return False
    if not cfg.mask_classifier and name.startswith("head."):
        return False
    if not cfg.mask_patch_embed and name.startswith("patch_embed."):
        return False
    if cfg.prune_scope == "weights_only":
        return p.ndim >= 2
    # all_prunable still keeps position/class tokens and norm/bias dense by default.
    if name in {"cls_token", "pos_embed"}:
        return False
    if ".norm" in name or name.startswith("norm."):
        return False
    if name.endswith("bias"):
        return False
    return True


def make_dense_masks(model: nn.Module, cfg: Config) -> Dict[str, torch.Tensor]:
    masks = {}
    for name, p in model.named_parameters():
        if is_prunable_parameter(name, p, cfg):
            masks[name] = torch.ones_like(p, dtype=torch.bool)
    return masks


def flatten_named_tensors(model: nn.Module, tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for name, _p in model.named_parameters():
        if name in tensors:
            parts.append(tensors[name].reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def unflatten_to_named_masks(
    model: nn.Module,
    cfg: Config,
    flat_keep: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, p in model.named_parameters():
        if not is_prunable_parameter(name, p, cfg):
            continue
        n = p.numel()
        masks[name] = flat_keep[offset : offset + n].view_as(p).to(device=p.device, dtype=torch.bool)
        offset += n
    if offset != flat_keep.numel():
        raise RuntimeError("Mask unflattening consumed an unexpected number of parameters.")
    return masks


def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name].to(device=p.device, dtype=p.dtype))


def zero_optimizer_state_(optimizer: torch.optim.Optimizer, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if name not in masks:
            continue
        state = optimizer.state.get(p)
        if not state:
            continue
        mask = masks[name].to(device=p.device, dtype=p.dtype)
        for value in state.values():
            if torch.is_tensor(value) and value.shape == p.shape:
                value.mul_(mask)


def compute_output_jacobian_sensitivity(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Estimate per-parameter output sensitivity at current weights.

    Uses Hutchinson probes over logits to avoid materialising the full output
    Jacobian. Returned tensors match prunable parameter shapes.
    """
    model.eval()
    scores = {
        name: torch.zeros_like(p, dtype=torch.float32, device=device)
        for name, p in model.named_parameters()
        if is_prunable_parameter(name, p, cfg)
    }
    if not scores:
        raise RuntimeError("No prunable parameters selected. Check PRUNE_SCOPE/MASK_* settings.")

    n_terms = 0
    was_training = model.training
    for batch_idx, (x, _y) in enumerate(loader):
        if batch_idx >= cfg.n_sens_batches:
            break
        x = x.to(device, non_blocking=True)
        for _ in range(cfg.sens_probes):
            model.zero_grad(set_to_none=True)
            logits = model(x)
            # Rademacher probe. Scaling by batch keeps score magnitudes comparable
            # across sensitivity batch sizes.
            probe = torch.empty_like(logits).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            scalar = (logits * probe).sum() / x.shape[0]
            scalar.backward()
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in scores and p.grad is not None:
                        scores[name].add_(p.grad.detach().float().pow(2))
            n_terms += 1
    if n_terms == 0:
        raise RuntimeError("Sensitivity loader produced no batches.")
    for name in scores:
        scores[name].div_(float(n_terms))
    model.train(was_training)
    return scores


def build_sensitivity_prune_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg: Config,
) -> Dict[str, torch.Tensor]:
    flat_scores = flatten_named_tensors(model, scores)
    n_total = flat_scores.numel()
    n_keep = max(1, int(round((1.0 - cfg.sparsity) * n_total)))

    # Keep top sensitivity. Ties are handled by torch.topk deterministically for
    # a fixed seed/device where supported.
    keep_idx = torch.topk(flat_scores, k=n_keep, largest=True, sorted=False).indices
    flat_keep = torch.zeros(n_total, dtype=torch.bool, device=flat_scores.device)
    flat_keep[keep_idx] = True
    return unflatten_to_named_masks(model, cfg, flat_keep)


def layer_mask_statistics(model: nn.Module, masks: Dict[str, torch.Tensor]) -> List[dict]:
    rows = []
    for name, p in model.named_parameters():
        if name not in masks:
            continue
        m = masks[name].detach().bool().cpu()
        retained = int(m.sum().item())
        total = int(m.numel())
        rows.append({
            "name": name,
            "total": total,
            "retained": retained,
            "pruned": total - retained,
            "sparsity": 1.0 - retained / max(1, total),
        })
    return rows


def grouped_mask_statistics(rows: List[dict]) -> List[dict]:
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        name = row["name"]
        if name.startswith("patch_embed"):
            group = "patch_embed"
        elif ".attn." in name:
            group = "attention"
        elif ".mlp." in name:
            group = "transformer_mlp"
        elif name.startswith("head"):
            group = "classifier"
        else:
            group = "other"
        groups.setdefault(group, []).append(row)
    out = []
    for group, rs in groups.items():
        total = sum(r["total"] for r in rs)
        retained = sum(r["retained"] for r in rs)
        out.append({
            "group": group,
            "total": total,
            "retained": retained,
            "pruned": total - retained,
            "sparsity": 1.0 - retained / max(1, total),
        })
    return sorted(out, key=lambda r: r["group"])


# ============================================================
# Training / evaluation
# ============================================================


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return loss_sum / max(1, total), correct / max(1, total)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    if cfg.epochs <= 0:
        return None
    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    decay_epochs = max(1, cfg.epochs - warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=decay_epochs,
        eta_min=cfg.lr_eta_min,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return cosine


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    masks: Dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        apply_masks_(model, masks)
        zero_optimizer_state_(optimizer, model, masks)

        loss_sum += float(loss.item()) * y.numel()
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    if scheduler is not None:
        scheduler.step()
    return loss_sum / max(1, total), correct / max(1, total)


# ============================================================
# Diagnostics / plotting
# ============================================================


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().flatten().float().cpu()
    b = b.detach().flatten().float().cpu()
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = torch.linalg.norm(ar) * torch.linalg.norm(br)
    if float(denom) == 0.0:
        return float("nan")
    return float((ar @ br / denom).item())


def mass_on_topk(current: torch.Tensor, ref_top_idx: torch.Tensor) -> float:
    current = current.detach().flatten().float().clamp_min(0).cpu()
    if current.numel() == 0 or float(current.sum()) == 0.0:
        return float("nan")
    idx = ref_top_idx.detach().cpu()
    return float(current[idx].sum().item() / current.sum().item())


def effective_rank_from_values(values: torch.Tensor) -> float:
    values = values.detach().float().clamp_min(0).cpu()
    total = values.sum()
    if float(total) <= 0:
        return 0.0
    p = values / total
    entropy = -(p[p > 0] * torch.log(p[p > 0])).sum()
    return float(torch.exp(entropy).item())


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, output_dir: Path) -> None:
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["test_loss"], label="test")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("ViT-Tiny loss after initialization-time sensitivity pruning")
    ax.legend()
    savefig(fig, output_dir / "loss_evolution.pdf")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(epochs, history["train_acc"], label="train")
    ax.plot(epochs, history["test_acc"], label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("ViT-Tiny accuracy after initialization-time sensitivity pruning")
    ax.legend()
    savefig(fig, output_dir / "accuracy_evolution.pdf")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(epochs, history["spearman_init_current"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Rank stability: initial vs current sensitivity")
    savefig(fig, output_dir / "sensitivity_rank_stability.pdf")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(epochs, history["init_topk_mass"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("mass on initial top-k")
    ax.set_title("Persistence of initially most sensitive parameters")
    savefig(fig, output_dir / "initial_topk_mass_over_training.pdf")


def plot_sensitivity_distribution(init_scores: torch.Tensor, final_scores: torch.Tensor, output_dir: Path) -> None:
    init_np = init_scores.detach().float().cpu().numpy()
    final_np = final_scores.detach().float().cpu().numpy()
    pos = np.concatenate([init_np[init_np > 0], final_np[final_np > 0]])
    if pos.size > 0:
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 60)
    else:
        bins = 60
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.hist(init_np, bins=bins, density=True, alpha=0.45, label="initial", histtype="stepfilled")
    ax.hist(final_np, bins=bins, density=True, alpha=0.45, label="final", histtype="stepfilled")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sensitivity")
    ax.set_ylabel("density")
    ax.set_title("Active-parameter sensitivity distribution")
    ax.legend()
    savefig(fig, output_dir / "sensitivity_distribution_initial_vs_final.pdf")


def plot_layer_sparsity(rows: List[dict], output_dir: Path) -> None:
    if not rows:
        return
    labels = [r["name"] for r in rows]
    sparsity = [r["sparsity"] for r in rows]
    height = max(5.0, 0.22 * len(labels))
    fig, ax = plt.subplots(figsize=(9.0, height))
    y = np.arange(len(labels))
    ax.barh(y, sparsity)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("sparsity")
    ax.set_title("Layer-wise sparsity induced by initialization sensitivity")
    savefig(fig, output_dir / "layerwise_sparsity.pdf")


def plot_group_sparsity(rows: List[dict], output_dir: Path) -> None:
    if not rows:
        return
    labels = [r["group"] for r in rows]
    sparsity = [r["sparsity"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(labels, sparsity)
    ax.set_ylim(0, 1)
    ax.set_ylabel("sparsity")
    ax.set_title("Component sparsity: patch / attention / MLP / classifier")
    ax.tick_params(axis="x", rotation=20)
    savefig(fig, output_dir / "component_sparsity.pdf")


def plot_sensitivity_heatmap(snapshot_matrix: np.ndarray, epochs: List[int], output_dir: Path) -> None:
    if snapshot_matrix.size == 0:
        return
    # Rows are checkpoints, columns are active parameter sensitivities. Sort by
    # initial sensitivity for readable diagnostics.
    sort_idx = np.argsort(snapshot_matrix[0])[::-1]
    mat = snapshot_matrix[:, sort_idx]
    vmax = np.nanpercentile(mat, 99.5)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = None
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    im = ax.imshow(
        mat.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=vmax,
        extent=[epochs[0], epochs[-1], 0, mat.shape[1]],
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("active parameter rank by initial sensitivity")
    ax.set_title("Sensitivity over training, sorted by initial sensitivity")
    fig.colorbar(im, ax=ax, pad=0.02).set_label("sensitivity")
    savefig(fig, output_dir / "active_sensitivity_over_training_heatmap.pdf")


def plot_curvature_proxy(flat_scores: torch.Tensor, output_dir: Path) -> float:
    # A cheap analogue to the eigenspectrum diagnostic from the MLP script:
    # the diagonal output-Jacobian energy spectrum over active parameters. This
    # does not claim to be the full covariance eigenspectrum; it is a stable
    # transformer-scale proxy.
    vals = torch.sort(flat_scores.detach().float().clamp_min(0).cpu(), descending=True).values
    erank = effective_rank_from_values(vals)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(vals.numpy())
    ax.set_yscale("log")
    ax.set_xlabel("active parameter rank")
    ax.set_ylabel("sensitivity")
    ax.set_title(f"Diagonal sensitivity spectrum, effective rank={erank:.2f}")
    savefig(fig, output_dir / "diagonal_sensitivity_spectrum.pdf")
    return erank


# ============================================================
# Main
# ============================================================


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device(cfg: Config) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    cfg = Config()
    set_reproducibility(cfg.seed)
    device = choose_device(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    print(json.dumps(asdict(cfg), indent=2))

    train_loader, test_loader, sens_loader = make_dataloaders(cfg)

    model = ViTTinyCIFAR(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)
    initial_state = copy.deepcopy(model.state_dict())

    n_params = parameter_count(model)
    n_prunable = sum(
        p.numel() for name, p in model.named_parameters()
        if is_prunable_parameter(name, p, cfg)
    )
    print(f"Model parameters: {n_params:,}")
    print(f"Prunable parameters: {n_prunable:,}")

    print("\nComputing initialization sensitivity")
    print("====================================")
    init_scores = compute_output_jacobian_sensitivity(model, sens_loader, cfg, device)
    masks = build_sensitivity_prune_masks(model, init_scores, cfg)
    apply_masks_(model, masks)

    layer_rows = layer_mask_statistics(model, masks)
    group_rows = grouped_mask_statistics(layer_rows)
    retained = sum(r["retained"] for r in layer_rows)
    pruned = sum(r["pruned"] for r in layer_rows)
    realised_sparsity = pruned / max(1, pruned + retained)

    print(f"Requested sparsity: {cfg.sparsity:.4f}")
    print(f"Realised prunable sparsity: {realised_sparsity:.4f}")
    print(f"Retained prunable parameters: {retained:,} / {retained + pruned:,}")
    print("Component sparsity:")
    for row in group_rows:
        print(f"  {row['group']:16s} sparsity={row['sparsity']:.4f} retained={row['retained']:,}/{row['total']:,}")

    with open(output_dir / "layerwise_sparsity.json", "w") as f:
        json.dump(layer_rows, f, indent=2)
    with open(output_dir / "component_sparsity.json", "w") as f:
        json.dump(group_rows, f, indent=2)

    flat_init_prunable = flatten_named_tensors(model, init_scores).detach().cpu()
    flat_mask = flatten_named_tensors(model, masks).detach().bool().cpu()
    flat_init_active = flat_init_prunable[flat_mask]
    k = max(1, int(round(cfg.topk_frac * flat_init_active.numel())))
    init_top_idx = torch.topk(flat_init_active, k=k, largest=True).indices

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "spearman_init_current": [],
        "init_topk_mass": [],
    }
    sensitivity_snapshots: List[np.ndarray] = []

    def checkpoint(epoch: int, optimizer=None) -> None:
        train_loss, train_acc = evaluate(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        curr_scores = compute_output_jacobian_sensitivity(model, sens_loader, cfg, device)
        flat_curr_prunable = flatten_named_tensors(model, curr_scores).detach().cpu()
        flat_curr_active = flat_curr_prunable[flat_mask]
        rho = spearman_corr(flat_init_active, flat_curr_active)
        top_mass = mass_on_topk(flat_curr_active, init_top_idx)
        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else cfg.lr

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)
        history["spearman_init_current"].append(rho)
        history["init_topk_mass"].append(top_mass)
        sensitivity_snapshots.append(flat_curr_active.numpy())

        print(
            f"epoch={epoch:4d} | lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"rho_init={rho:.3f} init_topk_mass={top_mass:.3f}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)

    print("\nTraining")
    print("========")
    checkpoint(0, optimizer)
    if cfg.run_training:
        for epoch in range(1, cfg.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, scheduler, masks, cfg, device)
            if epoch % cfg.checkpoint_interval == 0 or epoch == cfg.epochs:
                checkpoint(epoch, optimizer)

    print("\nFinal sensitivity diagnostics")
    print("=============================")
    final_scores = compute_output_jacobian_sensitivity(model, sens_loader, cfg, device)
    flat_final_prunable = flatten_named_tensors(model, final_scores).detach().cpu()
    flat_final_active = flat_final_prunable[flat_mask]

    final_erank = plot_curvature_proxy(flat_final_active, output_dir) if cfg.compute_final_curvature_proxy else None

    plot_history(history, output_dir)
    plot_sensitivity_distribution(flat_init_active, flat_final_active, output_dir)
    plot_layer_sparsity(layer_rows, output_dir)
    plot_group_sparsity(group_rows, output_dir)
    if sensitivity_snapshots:
        plot_sensitivity_heatmap(np.stack(sensitivity_snapshots, axis=0), history["epoch"], output_dir)

    final_train_loss, final_train_acc = evaluate(model, train_loader, device)
    final_test_loss, final_test_acc = evaluate(model, test_loader, device)

    summary = {
        "architecture": "ViT-Tiny-CIFAR10",
        "purpose": "minimal transformer sensitivity-pruning diagnostic",
        "config": asdict(cfg),
        "parameter_count": n_params,
        "prunable_parameter_count": n_prunable,
        "retained_prunable_parameter_count": retained,
        "pruned_prunable_parameter_count": pruned,
        "requested_sparsity": cfg.sparsity,
        "realised_prunable_sparsity": realised_sparsity,
        "final_train_loss": final_train_loss,
        "final_train_accuracy": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_acc,
        "final_spearman_init_final": spearman_corr(flat_init_active, flat_final_active),
        "final_mass_in_initial_topk": mass_on_topk(flat_final_active, init_top_idx),
        "final_diagonal_sensitivity_effective_rank": final_erank,
        "history": history,
    }

    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({
        "initial_state": initial_state,
        "final_state": model.state_dict(),
        "masks": {k: v.detach().cpu() for k, v in masks.items()},
        "config": asdict(cfg),
        "summary": summary,
    }, output_dir / "checkpoint_with_masks.pt")

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
