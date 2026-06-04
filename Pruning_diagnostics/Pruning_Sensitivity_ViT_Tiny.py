"""
ViT sensitivity-pruning experiment.

This script performs zero-shot sensitivity pruning at initialization, then trains the
remaining ViT parameters from scratch. It intentionally implements only sensitivity
pruning: no alternative pruning criteria or comparison baselines are included.

Default target: CIFAR-10 with a compact ViT. Use environment variables to change
model/data/training settings without editing the file.

Example:
    EPOCHS=100 PRUNE_FRACTION=0.50 BATCH_SIZE=128 python ViT_Sensitivity_Pruning_Experiment.py
"""

import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None

try:
    from torchvision import datasets, transforms
except Exception as exc:
    raise RuntimeError(
        "This script requires torchvision for CIFAR loading. Install torchvision or "
        "replace build_datasets() with your own dataset loader."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


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
    output_dir: str = _env_str("OUTPUT_DIR", "Plots/vit_sensitivity_pruning")
    download: bool = _env_bool("DOWNLOAD", True)

    image_size: int = _env_int("IMAGE_SIZE", 32)
    num_classes: int = _env_int("NUM_CLASSES", 10)
    train_subset: int = _env_int("TRAIN_SUBSET", 0)       # 0 means full train set
    test_subset: int = _env_int("TEST_SUBSET", 0)         # 0 means full test set
    sensitivity_samples: int = _env_int("SENSITIVITY_SAMPLES", 1024)
    sensitivity_batch_size: int = _env_int("SENSITIVITY_BATCH_SIZE", 64)
    sensitivity_probes: int = _env_int("SENSITIVITY_PROBES", 1)

    patch_size: int = _env_int("PATCH_SIZE", 4)
    embed_dim: int = _env_int("EMBED_DIM", 192)
    depth: int = _env_int("DEPTH", 12)
    num_heads: int = _env_int("NUM_HEADS", 3)
    mlp_ratio: float = _env_float("MLP_RATIO", 4.0)
    dropout: float = _env_float("DROPOUT", 0.0)

    prune_fraction: float = _env_float("PRUNE_FRACTION", 0.50)
    prune_bias: bool = _env_bool("PRUNE_BIAS", False)
    prune_norm: bool = _env_bool("PRUNE_NORM", False)
    prune_embeddings: bool = _env_bool("PRUNE_EMBEDDINGS", True)
    prune_head: bool = _env_bool("PRUNE_HEAD", True)

    batch_size: int = _env_int("BATCH_SIZE", 128)
    epochs: int = _env_int("EPOCHS", 100)
    lr: float = _env_float("LR", 3e-4)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.05)
    warmup_epochs: int = _env_int("WARMUP_EPOCHS", 5)
    min_lr: float = _env_float("MIN_LR", 1e-6)
    num_workers: int = _env_int("NUM_WORKERS", 2)
    grad_clip: float = _env_float("GRAD_CLIP", 1.0)
    label_smoothing: float = _env_float("LABEL_SMOOTHING", 0.1)
    amp: bool = _env_bool("AMP", True)
    checkpoint_interval: int = _env_int("CHECKPOINT_INTERVAL", 1)

    def __post_init__(self) -> None:
        if self.dataset.upper() not in {"CIFAR10", "CIFAR100"}:
            raise ValueError("DATASET must be CIFAR10 or CIFAR100.")
        if self.dataset.upper() == "CIFAR100" and self.num_classes == 10:
            self.num_classes = 100
        if not (0.0 <= self.prune_fraction < 1.0):
            raise ValueError("PRUNE_FRACTION must be in [0, 1).")
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


def _subset(dataset, n: int, seed: int):
    if n <= 0 or n >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, idx)


def build_datasets(cfg: Config):
    mean_std = {
        "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "CIFAR100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    }
    mean, std = mean_std[cfg.dataset.upper()]

    train_tfms = transforms.Compose([
        transforms.RandomCrop(cfg.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_cls = datasets.CIFAR10 if cfg.dataset.upper() == "CIFAR10" else datasets.CIFAR100
    train_set = ds_cls(cfg.data_dir, train=True, transform=train_tfms, download=cfg.download)
    sens_set = ds_cls(cfg.data_dir, train=True, transform=eval_tfms, download=cfg.download)
    test_set = ds_cls(cfg.data_dir, train=False, transform=eval_tfms, download=cfg.download)

    train_set = _subset(train_set, cfg.train_subset, cfg.seed)
    sens_set = _subset(sens_set, cfg.sensitivity_samples, cfg.seed + 1)
    test_set = _subset(test_set, cfg.test_subset, cfg.seed + 2)
    return train_set, sens_set, test_set


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


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


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
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
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
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


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
initial_state = copy.deepcopy(model.state_dict())


# -----------------------------------------------------------------------------
# Sensitivity pruning
# -----------------------------------------------------------------------------


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_prunable_parameter(name: str, p: torch.nn.Parameter, cfg: Config) -> bool:
    if not p.requires_grad:
        return False
    if p.ndim == 1 and not cfg.prune_bias:
        return False
    if ("norm" in name.lower()) and not cfg.prune_norm:
        return False
    if (("patch_embed" in name) or ("pos_embed" in name) or ("cls_token" in name)) and not cfg.prune_embeddings:
        return False
    if name.startswith("head") and not cfg.prune_head:
        return False
    return True


def init_score_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p, device="cpu") for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def _make_probe(logits: torch.Tensor) -> torch.Tensor:
    # Rademacher probe. For logits f(x), E_v[(v^T f)'_theta^2] estimates
    # the sum of squared logit-Jacobian sensitivities without materialising the
    # full parameter Jacobian.
    probe = torch.empty_like(logits).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    probe = probe / math.sqrt(logits.shape[-1])
    return probe


def compute_sensitivity_scores(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model.eval()
    scores = init_score_buffers(model)
    n_accum = 0

    for images, _targets in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]

        for _ in range(cfg.sensitivity_probes):
            model.zero_grad(set_to_none=True)
            logits = model(images)
            probe = _make_probe(logits)
            scalar = (logits * probe).sum() / batch_size
            scalar.backward()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    scores[name].add_(p.grad.detach().float().cpu().pow(2), alpha=batch_size)
            n_accum += batch_size

    if n_accum == 0:
        raise RuntimeError("No samples were available for sensitivity scoring.")
    for name in scores:
        scores[name].div_(n_accum)
    return scores


def make_sensitivity_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg: Config,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    eligible = []
    all_scores = []
    for name, p in model.named_parameters():
        if is_prunable_parameter(name, p, cfg):
            flat = scores[name].reshape(-1).float()
            eligible.append((name, p.shape, flat.numel()))
            all_scores.append(flat)

    if not all_scores:
        raise RuntimeError("No parameters are eligible for pruning under the current config.")

    flat_scores = torch.cat(all_scores)
    n_prunable = flat_scores.numel()
    n_prune = int(math.floor(cfg.prune_fraction * n_prunable))
    n_keep = n_prunable - n_prune

    flat_keep = torch.zeros(n_prunable, dtype=torch.bool)
    if n_keep > 0:
        # Exact global keep set: retain the highest-sensitivity coordinates.
        keep_idx = torch.topk(flat_scores, k=n_keep, largest=True, sorted=False).indices
        flat_keep[keep_idx] = True
    threshold = float(flat_scores[flat_keep].min().item()) if n_keep > 0 else float("inf")

    masks: Dict[str, torch.Tensor] = {}
    cursor = 0
    eligible_names = {name for name, _shape, _numel in eligible}
    for name, p in model.named_parameters():
        if name in eligible_names:
            numel = p.numel()
            mask = flat_keep[cursor:cursor + numel].view_as(p).to(device=p.device)
            cursor += numel
        else:
            mask = torch.ones_like(p, dtype=torch.bool, device=p.device)
        masks[name] = mask

    pruned_eligible = n_prune
    retained_eligible = n_keep
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats = {
        "eligible_parameter_count": float(n_prunable),
        "target_prune_fraction": float(cfg.prune_fraction),
        "actual_pruned_eligible_parameter_count": float(pruned_eligible),
        "actual_retained_eligible_parameter_count": float(retained_eligible),
        "actual_prune_fraction_eligible": float(pruned_eligible / max(1, n_prunable)),
        "actual_prune_fraction_all_trainable": float(pruned_eligible / max(1, total_trainable)),
        "sensitivity_keep_threshold": threshold,
    }
    return masks, stats

def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.mul_(masks[name].to(device=p.device, dtype=p.dtype))


def zero_masked_optimizer_state_(optimizer: torch.optim.Optimizer, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        state = optimizer.state.get(p)
        if not state:
            continue
        mask = masks[name].to(device=p.device, dtype=p.dtype)
        for value in state.values():
            if torch.is_tensor(value) and value.shape == p.shape:
                value.mul_(mask)


def masked_density_by_module(model: nn.Module, masks: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, int | float]]:
    groups: Dict[str, Dict[str, int]] = {}
    for name, mask in masks.items():
        group = name.split(".")[0]
        groups.setdefault(group, {"retained": 0, "total": 0})
        groups[group]["retained"] += int(mask.sum().item())
        groups[group]["total"] += int(mask.numel())
    return {
        k: {"retained": v["retained"], "total": v["total"], "density": v["retained"] / max(1, v["total"])}
        for k, v in groups.items()
    }


print(f"Using device: {device}")
print(f"Dataset: {cfg.dataset.upper()} | train={len(train_set)} | sensitivity={len(sens_set)} | test={len(test_set)}")
print(f"Model trainable parameters before pruning: {trainable_parameter_count(model):,}")
print("Computing zero-shot sensitivity scores at initialization")

sensitivity_scores = compute_sensitivity_scores(model, sens_loader, cfg, device)
masks, pruning_stats = make_sensitivity_masks(model, sensitivity_scores, cfg)
apply_masks_(model, masks)

print("Sensitivity pruning complete")
print(json.dumps(pruning_stats, indent=2))
print("Training the pruned ViT from scratch")


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bsz = images.shape[0]
        loss_sum += float(loss.item()) * bsz
        acc_sum += accuracy(logits, targets) * bsz
        n += bsz
    return {"loss": loss_sum / max(1, n), "accuracy": acc_sum / max(1, n)}


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    if cfg.epochs <= 0:
        return None
    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    decay_epochs = max(1, cfg.epochs - warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=cfg.min_lr)
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


criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = build_scheduler(optimizer, cfg)
scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

history = []

for epoch in range(1, cfg.epochs + 1):
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

        apply_masks_(model, masks)
        zero_masked_optimizer_state_(optimizer, model, masks)

        bsz = images.shape[0]
        loss_sum += float(loss.item()) * bsz
        acc_sum += accuracy(logits.detach(), targets) * bsz
        n += bsz

    if scheduler is not None:
        scheduler.step()

    train_metrics = {"loss": loss_sum / max(1, n), "accuracy": acc_sum / max(1, n)}
    should_eval = (epoch == 1) or (epoch % cfg.checkpoint_interval == 0) or (epoch == cfg.epochs)
    if should_eval:
        test_metrics = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch:4d} | lr={row['lr']:.3e} | "
            f"train_loss={row['train_loss']:.4f} | train_acc={100.0 * row['train_accuracy']:.2f}% | "
            f"test_loss={row['test_loss']:.4f} | test_acc={100.0 * row['test_accuracy']:.2f}%"
        )


# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------

final_metrics = evaluate(model, test_loader, criterion, device)
module_density = masked_density_by_module(model, masks)
summary = {
    "config": asdict(cfg),
    "device": str(device),
    "parameter_count": parameter_count(model),
    "trainable_parameter_count": trainable_parameter_count(model),
    "pruning": pruning_stats,
    "module_density": module_density,
    "final_test_loss": final_metrics["loss"],
    "final_test_accuracy": final_metrics["accuracy"],
    "history": history,
}

summary_path = out_dir / "vit_sensitivity_pruning_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

checkpoint_path = out_dir / "vit_sensitivity_pruned_final.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "initial_state_dict": initial_state,
        "masks": {k: v.detach().cpu() for k, v in masks.items()},
        "sensitivity_scores": sensitivity_scores,
        "config": asdict(cfg),
        "summary": summary,
    },
    checkpoint_path,
)

if plt is not None and history:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["test_loss"] for h in history], label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross entropy")
    ax.set_title("ViT sensitivity-pruned training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(epochs, [100.0 * h["train_accuracy"] for h in history], label="train")
    ax.plot(epochs, [100.0 * h["test_accuracy"] for h in history], label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("ViT sensitivity-pruned accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.pdf")
    plt.close(fig)

print("\nFinal summary")
print("=============")
print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
print(f"Wrote summary to {summary_path}")
print(f"Wrote checkpoint to {checkpoint_path}")
