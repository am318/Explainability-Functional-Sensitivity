"""
ViT tiny SYNFLOW pruning experiment.

This script performs zero-shot SYNFLOW pruning at initialization, then trains the remaining ViT parameters from scratch. It uses a sensitivity-derived pruning budget, global SYNFLOW mask selection, and the same checkpointed sensitivity analysis as the baseline ViT tiny script.

Default target: CIFAR-10 with a compact ViT. Use environment variables to change
model/data/training settings without editing the file.

Example:
    EPOCHS=100 BATCH_SIZE=128 python ViT_Tiny_SYNFLOW_Pruning.py
"""

import copy
import re
import json
import math
import os
import random
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    _TORCHVISION_AVAILABLE = True
except Exception:
    datasets = None
    transforms = None
    _TORCHVISION_AVAILABLE = False

from pathlib import Path
import sys

sys.path.append(str(Path("/Users/alan/Documents/Github/Explainability-Functional-Sensitivity/pruning")))

try:
    from pruning.pruning_baselines import build_snip_masks, build_synflow_masks
except Exception:
    from pruning_baselines import build_snip_masks, build_synflow_masks

PRUNING_METHOD = "synflow"


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
    output_dir: str = _env_str("OUTPUT_DIR", "Plots/vit_tiny_SYNFLOW")
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
    preserve_attention_heads: bool = _env_bool("PRESERVE_ATTENTION_HEADS", False)
    synflow_iterations: int = _env_int("SYNFLOW_ITERATIONS", 5)

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


class _TensorTransformPipeline:
    def __init__(self, image_size: int, mean, std, train: bool):
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.train = train

    @staticmethod
    def _pad(image: torch.Tensor, padding: int = 4) -> torch.Tensor:
        return F.pad(image, (padding, padding, padding, padding), mode="reflect")

    @staticmethod
    def _random_crop(image: torch.Tensor, size: int) -> torch.Tensor:
        _, h, w = image.shape
        if h == size and w == size:
            return image
        if h < size or w < size:
            raise ValueError(f"Cannot crop {size}x{size} from image of shape {(h, w)}")
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return image[:, top:top + size, left:left + size]

    @staticmethod
    def _horizontal_flip(image: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if random.random() < p:
            return torch.flip(image, dims=(2,))
        return image

    @staticmethod
    def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
        if image.shape[-1] == size and image.shape[-2] == size:
            return image
        return F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)

    def __call__(self, image) -> torch.Tensor:
        # torchvision CIFAR datasets yield PIL Images; the custom loaders yield tensors.
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        image = image.float() / 255.0
        if self.train:
            if self.image_size == 32:
                image = self._pad(image, padding=4)
                image = self._random_crop(image, 32)
                image = self._horizontal_flip(image)
            else:
                image = self._resize(image, self.image_size)
        else:
            image = self._resize(image, self.image_size)
        image = (image - self.mean) / self.std
        return image


class CIFAR10Dataset(torch.utils.data.Dataset):
    """Minimal CIFAR-10 loader that does not depend on torchvision."""

    base_folder = "cifar-10-batches-py"
    train_files = [f"data_batch_{i}" for i in range(1, 6)]
    test_files = ["test_batch"]
    label_key = "labels"

    def __init__(self, root: str, train: bool, transform=None, download: bool = False):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.data, self.targets = self._load(download=download)

    def _load(self, download: bool = False):
        folder = self.root / self.base_folder
        files = self.train_files if self.train else self.test_files
        if not folder.exists():
            raise FileNotFoundError(
                f"Could not find {folder}. Set DOWNLOAD=1 or place the extracted CIFAR-10 files there."
            )
        data = []
        targets = []
        for filename in files:
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 file: {path}")
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            batch = entry.get("data")
            if batch is None:
                raise RuntimeError(f"Invalid CIFAR-10 batch format in {path}")
            data.append(batch.reshape(-1, 3, 32, 32))
            targets.extend(entry[self.label_key])
        data = np.concatenate(data, axis=0)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class CIFAR100Dataset(torch.utils.data.Dataset):
    """Minimal CIFAR-100 loader that does not depend on torchvision."""

    base_folder = "cifar-100-python"
    train_files = ["train"]
    test_files = ["test"]
    label_key = "fine_labels"

    def __init__(self, root: str, train: bool, transform=None, download: bool = False):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.data, self.targets = self._load(download=download)

    def _load(self, download: bool = False):
        folder = self.root / self.base_folder
        files = self.train_files if self.train else self.test_files
        if not folder.exists():
            raise FileNotFoundError(
                f"Could not find {folder}. Set DOWNLOAD=1 or place the extracted CIFAR-100 files there."
            )
        data = []
        targets = []
        for filename in files:
            path = folder / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing CIFAR-100 file: {path}")
            with open(path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            batch = entry.get("data")
            if batch is None:
                raise RuntimeError(f"Invalid CIFAR-100 batch format in {path}")
            data.append(batch.reshape(-1, 3, 32, 32))
            targets.extend(entry[self.label_key])
        data = np.concatenate(data, axis=0)
        return data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index])
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


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

    train_tfms = _TensorTransformPipeline(cfg.image_size, mean, std, train=True)
    eval_tfms = _TensorTransformPipeline(cfg.image_size, mean, std, train=False)

    if _TORCHVISION_AVAILABLE:
        ds_cls = datasets.CIFAR10 if cfg.dataset.upper() == "CIFAR10" else datasets.CIFAR100
        train_set = ds_cls(cfg.data_dir, train=True, transform=train_tfms, download=cfg.download)
        sens_set = ds_cls(cfg.data_dir, train=True, transform=eval_tfms, download=cfg.download)
        test_set = ds_cls(cfg.data_dir, train=False, transform=eval_tfms, download=cfg.download)
    else:
        ds_cls = CIFAR10Dataset if cfg.dataset.upper() == "CIFAR10" else CIFAR100Dataset
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
# Sensitivity pruning and sweep-style analysis utilities
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
    return {
        name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
        for name, p in model.named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def _make_probe(logits: torch.Tensor) -> torch.Tensor:
    probe = torch.empty_like(logits).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    probe = probe / math.sqrt(max(1, logits.shape[-1]))
    return probe


def _flatten_like_model(model: nn.Module, tensors: Dict[str, torch.Tensor], only_active: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    parts = []
    for name, p in model.named_parameters():
        if name not in tensors:
            continue
        t = torch.as_tensor(tensors[name]).detach().cpu().reshape(-1)
        if only_active is not None:
            m = torch.as_tensor(only_active[name], dtype=torch.bool).detach().cpu().reshape(-1)
            t = t[m]
        parts.append(t.float())
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts)


def _flatten_param_magnitudes(model: nn.Module, masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    parts = []
    for name, p in model.named_parameters():
        t = p.detach().abs().cpu().reshape(-1)
        if masks is not None:
            m = masks[name].detach().cpu().bool().reshape(-1)
            t = t[m]
        parts.append(t.float())
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().reshape(-1)
    b = b.detach().float().cpu().reshape(-1)
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.numel() < 2:
        return float("nan")
    ar = torch.argsort(torch.argsort(a)).float()
    br = torch.argsort(torch.argsort(b)).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = torch.linalg.norm(ar) * torch.linalg.norm(br)
    if float(denom) == 0.0:
        return float("nan")
    return float((ar @ br / denom).item())


def topk_indices(values: torch.Tensor, frac: float) -> torch.Tensor:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    k = max(1, int(math.ceil(float(frac) * values.numel())))
    k = min(k, values.numel())
    return torch.topk(values, k=k, largest=True, sorted=False).indices.cpu()


def mass_on_indices(values: torch.Tensor, indices: torch.Tensor) -> float:
    values = values.detach().float().cpu().reshape(-1).clamp_min(0)
    if values.numel() == 0 or indices.numel() == 0:
        return float("nan")
    denom = float(values.sum().item())
    if denom <= 0.0:
        return float("nan")
    idx = indices.to(dtype=torch.long).clamp(0, values.numel() - 1)
    return float(values[idx].sum().item() / denom)


def effective_rank(eigvals: torch.Tensor) -> float:
    vals = eigvals.detach().float().cpu().clamp_min(0)
    total = vals.sum()
    if float(total) <= 0.0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()).item())


def compute_sensitivity_scores(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    probes: Optional[int] = None,
    collect_probe_matrix: bool = False,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Hutchinson/Rademacher estimator for parameter sensitivity.

    If collect_probe_matrix is true, this also stores a bounded matrix of active
    probe gradients. Its Gram spectrum is used as a scalable analogue of the
    sweep script's Jacobian covariance eigenspectrum.
    """
    model.eval()
    n_probes = int(probes if probes is not None else cfg.sensitivity_probes)
    scores = init_score_buffers(model)
    n_accum = 0
    probe_rows: List[torch.Tensor] = []
    if masks is not None:
        active_count = int(sum(int(m.detach().cpu().bool().sum().item()) for m in masks.values()))
    else:
        active_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    can_collect = collect_probe_matrix
    if can_collect and active_count * cfg.analysis_probe_matrix_rows > cfg.max_probe_matrix_elements:
        can_collect = False

    for images, _targets in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]
        for _ in range(n_probes):
            model.zero_grad(set_to_none=True)
            logits = model(images)
            probe = _make_probe(logits)
            scalar = (logits * probe).sum() / batch_size
            scalar.backward()
            with torch.no_grad():
                grad_parts = []
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    scores[name].add_(g.cpu().pow(2), alpha=batch_size)
                    if can_collect and len(probe_rows) < cfg.analysis_probe_matrix_rows:
                        if masks is None:
                            grad_parts.append(g.detach().cpu().reshape(-1))
                        else:
                            m = masks[name].to(device=p.device, dtype=torch.bool)
                            grad_parts.append(g[m].detach().cpu().reshape(-1))
                if can_collect and grad_parts and len(probe_rows) < cfg.analysis_probe_matrix_rows:
                    probe_rows.append(torch.cat(grad_parts))
            n_accum += batch_size

    if n_accum == 0:
        raise RuntimeError("No samples were available for sensitivity scoring.")
    for name in scores:
        scores[name].div_(n_accum)
    probe_matrix = torch.stack(probe_rows, dim=0) if probe_rows else None
    return scores, probe_matrix


def _restore_topk_in_vector(mask_vec: torch.Tensor, score_vec: torch.Tensor, k: int) -> int:
    if mask_vec.numel() == 0 or int(mask_vec.sum().item()) >= k:
        return 0
    k = min(k, mask_vec.numel())
    restore_idx = torch.topk(score_vec.float(), k=k, largest=True, sorted=False).indices
    before = int(mask_vec.sum().item())
    mask_vec[restore_idx] = True
    return int(mask_vec.sum().item()) - before


def _connectivity_close_dense_weight(mask: torch.Tensor, scores: torch.Tensor, min_conn: int) -> int:
    """Restore incident edges so dense/conv weights have no empty row/column groups."""
    restored = 0
    if mask.ndim == 2:
        for row in range(mask.shape[0]):
            restored += _restore_topk_in_vector(mask[row, :], scores[row, :], min_conn)
        for col in range(mask.shape[1]):
            restored += _restore_topk_in_vector(mask[:, col], scores[:, col], min_conn)
    elif mask.ndim == 4:
        # Conv2d: flatten spatial kernels when checking output/input channels.
        out_ch, in_ch = mask.shape[:2]
        flat_mask_out = mask.reshape(out_ch, -1)
        flat_scores_out = scores.reshape(out_ch, -1)
        for row in range(out_ch):
            restored += _restore_topk_in_vector(flat_mask_out[row], flat_scores_out[row], min_conn)
        flat_mask_in = mask.permute(1, 0, 2, 3).reshape(in_ch, -1)
        flat_scores_in = scores.permute(1, 0, 2, 3).reshape(in_ch, -1)
        for col in range(in_ch):
            restored += _restore_topk_in_vector(flat_mask_in[col], flat_scores_in[col], min_conn)
    return restored


def make_threshold_connectivity_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg: Config,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    masks: Dict[str, torch.Tensor] = {}
    eligible_total = 0
    raw_retained = 0
    restored_total = 0

    for name, p in model.named_parameters():
        score = scores[name].to(device=p.device, dtype=torch.float32)
        if is_prunable_parameter(name, p, cfg):
            eligible_total += p.numel()
            mask = score >= float(cfg.prune_threshold)
            if cfg.connectivity_closure and p.ndim in (2, 4):
                restored_total += _connectivity_close_dense_weight(mask, score, max(1, cfg.min_connections_per_unit))
            # Never allow a prunable tensor to become completely empty.
            if int(mask.sum().item()) == 0:
                idx = torch.argmax(score.reshape(-1))
                mask.reshape(-1)[idx] = True
                restored_total += 1
            raw_retained += int(mask.sum().item())
        else:
            mask = torch.ones_like(p, dtype=torch.bool, device=p.device)
        masks[name] = mask

    total_trainable = trainable_parameter_count(model)
    retained_eligible = sum(
        int(masks[name].sum().item())
        for name, p in model.named_parameters()
        if is_prunable_parameter(name, p, cfg)
    )
    pruned_eligible = max(0, eligible_total - retained_eligible)
    stats = {
        "eligible_parameter_count": float(eligible_total),
        "threshold": float(cfg.prune_threshold),
        "retained_eligible_parameter_count": float(retained_eligible),
        "pruned_eligible_parameter_count": float(pruned_eligible),
        "actual_prune_fraction_eligible": float(pruned_eligible / max(1, eligible_total)),
        "actual_prune_fraction_all_trainable": float(pruned_eligible / max(1, total_trainable)),
        "connectivity_restored_coordinate_count": float(restored_total),
        "connectivity_closure": bool(cfg.connectivity_closure),
        "min_connections_per_unit": int(cfg.min_connections_per_unit),
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


# --- Structured pruning utilities -------------------------------------------


def _keep_count(total: int, prune_fraction: float) -> int:
    return max(1, min(total, int(math.ceil((1.0 - prune_fraction) * total))))


def _iterative_prune_fraction(cfg: Config, round_idx: int) -> float:
    """Return the prune fraction to use on a given iterative round."""
    target = float(cfg.prune_fraction)
    if not cfg.gradual_sparsification:
        return target
    total_rounds = max(1, int(cfg.iterative_pruning_rounds))
    # Linearly ramp from a mild first-round prune to the requested final sparsity.
    return target * float(round_idx + 1) / float(total_rounds)


def _normalize_unit_scores(unit_scores: torch.Tensor, cfg: Config) -> torch.Tensor:
    unit_scores = unit_scores.detach().float().cpu().reshape(-1)
    if unit_scores.numel() == 0:
        return unit_scores
    if not cfg.layerwise_normalize_scores:
        return unit_scores
    mean = unit_scores.mean()
    scale = unit_scores.std(unbiased=False).clamp_min(1e-12)
    return (unit_scores - mean) / scale


def _tensor_group_scores(name: str, score: torch.Tensor, p: torch.Tensor, cfg: Config) -> Optional[torch.Tensor]:
    score = score.detach().float().cpu()
    if score.numel() == 0:
        return None

    # Embedding/channel-level tensors.
    if name in {"cls_token", "pos_embed"}:
        return score.abs().mean(dim=tuple(range(score.ndim - 1)))
    if "patch_embed.proj.weight" in name and p.ndim == 4:
        return score.abs().mean(dim=(1, 2, 3))
    if "patch_embed.proj.bias" in name and p.ndim == 1:
        return score.abs()
    if "norm" in name.lower() and p.ndim == 1:
        return score.abs()
    if "attn.in_proj_weight" in name and p.ndim == 2:
        return score.abs().mean(dim=1)
    if "attn.in_proj_bias" in name and p.ndim == 1:
        return score.abs()
    if "attn.out_proj.weight" in name and p.ndim == 2:
        return score.abs().mean(dim=0 if cfg.preserve_attention_heads else 1)
    if "attn.out_proj.bias" in name and p.ndim == 1:
        return score.abs()
    if "mlp.fc2.weight" in name and p.ndim == 2:
        return score.abs().mean(dim=0)
    if "mlp.fc2.bias" in name and p.ndim == 1:
        return score.abs()
    if "head.weight" in name and p.ndim == 2:
        # Select classifier input channels, not class rows.
        return score.abs().mean(dim=0)
    if "head.bias" in name and p.ndim == 1:
        return score.abs()
    return None


def _hidden_unit_scores(name: str, score: torch.Tensor, p: torch.Tensor) -> Optional[torch.Tensor]:
    score = score.detach().float().cpu()
    if "mlp.fc1.weight" in name and p.ndim == 2:
        return score.abs().mean(dim=1)
    if "mlp.fc1.bias" in name and p.ndim == 1:
        return score.abs()
    return None


def _build_structured_selection_masks(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    cfg: Config,
    prune_fraction: Optional[float] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor, Dict[int, torch.Tensor]]:
    """Budgeted structured selection over embedding channels and MLP hidden units."""
    effective_prune_fraction = float(cfg.prune_fraction if prune_fraction is None else prune_fraction)
    embed_dim = cfg.embed_dim
    head_dim = max(1, embed_dim // max(1, cfg.num_heads))
    if cfg.preserve_attention_heads and embed_dim % max(1, cfg.num_heads) == 0:
        group_size = head_dim
        n_embed_groups = cfg.num_heads
    else:
        group_size = 1
        n_embed_groups = embed_dim

    embed_group_scores = torch.zeros(n_embed_groups, dtype=torch.float32)
    hidden_scores: Dict[int, torch.Tensor] = {}
    raw_unit_counts = {"embed": 0, "hidden": 0}

    for name, p in model.named_parameters():
        if name not in scores:
            continue
        s = scores[name]
        emb = _tensor_group_scores(name, s, p, cfg)
        if emb is not None:
            emb = _normalize_unit_scores(emb, cfg)
            if "patch_embed.proj.weight" in name and p.ndim == 4:
                if emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif ("cls_token" in name or "pos_embed" in name) and emb.numel() == embed_dim:
                emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif "attn.in_proj_weight" in name and emb.numel() == 3 * embed_dim:
                emb = emb[: 3 * n_embed_groups * group_size].reshape(3, n_embed_groups, group_size).mean(dim=(0, 2))
            elif "attn.in_proj_bias" in name and emb.numel() == 3 * embed_dim:
                emb = emb[: 3 * n_embed_groups * group_size].reshape(3, n_embed_groups, group_size).mean(dim=(0, 2))
            elif "attn.out_proj.weight" in name and p.ndim == 2:
                if emb.numel() == embed_dim:
                    emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif "attn.out_proj.bias" in name and emb.numel() == embed_dim:
                emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif "mlp.fc2.weight" in name and p.ndim == 2 and emb.numel() == embed_dim:
                emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif "mlp.fc2.bias" in name and emb.numel() == embed_dim:
                emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif "head.weight" in name and emb.numel() == embed_dim:
                emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
            elif "head.bias" in name and emb.numel() == cfg.num_classes:
                # Class logits are not pruned; ignore.
                emb = None
            elif "norm" in name.lower() and emb.numel() == embed_dim:
                emb = emb[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)

            if emb is not None and emb.numel() == n_embed_groups:
                embed_group_scores.add_(emb.float())
                raw_unit_counts["embed"] += 1

        # Extra embed-channel contributions from MLP projections.
        if "mlp.fc1.weight" in name and p.ndim == 2:
            embed_contrib = _normalize_unit_scores(s.abs().mean(dim=0), cfg)
            if embed_contrib.numel() == embed_dim:
                if embed_contrib.numel() == n_embed_groups * group_size:
                    embed_contrib = embed_contrib.reshape(n_embed_groups, group_size).mean(dim=1)
                elif embed_contrib.numel() != n_embed_groups:
                    embed_contrib = embed_contrib[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                embed_group_scores.add_(embed_contrib.float())
                raw_unit_counts["embed"] += 1
        if "mlp.fc2.weight" in name and p.ndim == 2:
            embed_contrib = _normalize_unit_scores(s.abs().mean(dim=1), cfg)
            if embed_contrib.numel() == embed_dim:
                if embed_contrib.numel() == n_embed_groups * group_size:
                    embed_contrib = embed_contrib.reshape(n_embed_groups, group_size).mean(dim=1)
                elif embed_contrib.numel() != n_embed_groups:
                    embed_contrib = embed_contrib[:n_embed_groups * group_size].reshape(n_embed_groups, group_size).mean(dim=1)
                embed_group_scores.add_(embed_contrib.float())
                raw_unit_counts["embed"] += 1

        hid = _hidden_unit_scores(name, s, p)
        if hid is not None:
            hid = _normalize_unit_scores(hid, cfg)
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            if block_match:
                idx = int(block_match.group(1))
                hidden_scores.setdefault(idx, torch.zeros_like(hid))
                hidden_scores[idx].add_(hid.float())
                raw_unit_counts["hidden"] += 1

    embed_keep = _keep_count(n_embed_groups, effective_prune_fraction)
    embed_sel = torch.zeros(n_embed_groups, dtype=torch.bool)
    embed_sel[torch.topk(embed_group_scores, k=embed_keep, largest=True, sorted=False).indices] = True
    embed_channel_sel = embed_sel.repeat_interleave(group_size)
    if embed_channel_sel.numel() < embed_dim:
        pad = torch.zeros(embed_dim - embed_channel_sel.numel(), dtype=torch.bool)
        embed_channel_sel = torch.cat([embed_channel_sel, pad], dim=0)
    elif embed_channel_sel.numel() > embed_dim:
        embed_channel_sel = embed_channel_sel[:embed_dim]

    hidden_sel: Dict[int, torch.Tensor] = {}
    for idx, hs in hidden_scores.items():
        keep = _keep_count(hs.numel(), effective_prune_fraction)
        sel = torch.zeros_like(hs, dtype=torch.bool)
        sel[torch.topk(hs, k=keep, largest=True, sorted=False).indices] = True
        hidden_sel[idx] = sel

    masks: Dict[str, torch.Tensor] = {}
    restored_total = 0
    eligible_total = 0
    retained_total = 0

    for name, p in model.named_parameters():
        if not is_prunable_parameter(name, p, cfg):
            masks[name] = torch.ones_like(p, dtype=torch.bool)
            continue
        eligible_total += p.numel()
        if name in {"cls_token", "pos_embed"}:
            mask = embed_channel_sel.view(1, 1, -1).expand_as(p).clone()
        elif "patch_embed.proj.weight" in name and p.ndim == 4:
            mask = embed_channel_sel.view(-1, 1, 1, 1).expand_as(p).clone()
        elif "patch_embed.proj.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "attn.in_proj_weight" in name and p.ndim == 2:
            row_sel = embed_channel_sel.repeat(3)
            if row_sel.numel() != p.shape[0]:
                row_sel = row_sel[:p.shape[0]] if row_sel.numel() > p.shape[0] else torch.cat([row_sel, torch.zeros(p.shape[0] - row_sel.numel(), dtype=torch.bool)], dim=0)
            col_sel = embed_channel_sel
            mask = row_sel.view(-1, 1).expand_as(p).clone() & col_sel.view(1, -1).expand_as(p)
        elif "attn.in_proj_bias" in name and p.ndim == 1:
            mask = embed_channel_sel.repeat(3).clone()
        elif "attn.out_proj.weight" in name and p.ndim == 2:
            mask = embed_channel_sel.view(-1, 1).expand_as(p).clone() & embed_channel_sel.view(1, -1).expand_as(p)
        elif "attn.out_proj.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "mlp.fc2.weight" in name and p.ndim == 2:
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc2", name)
            block_idx = int(block_match.group(1)) if block_match else -1
            hmask = hidden_sel.get(block_idx, torch.ones(p.shape[1], dtype=torch.bool))
            mask = embed_channel_sel.view(-1, 1).expand(p.shape[0], p.shape[1]) & hmask.view(1, -1)
        elif "mlp.fc2.bias" in name and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "mlp.fc1.weight" in name and p.ndim == 2:
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            block_idx = int(block_match.group(1)) if block_match else -1
            hmask = hidden_sel.get(block_idx, torch.ones(p.shape[0], dtype=torch.bool))
            mask = hmask.view(-1, 1).expand_as(p).clone() & embed_channel_sel.view(1, -1).expand_as(p)
        elif "mlp.fc1.bias" in name and p.ndim == 1:
            block_match = re.search(r"blocks\.(\d+)\.mlp\.fc1", name)
            block_idx = int(block_match.group(1)) if block_match else -1
            mask = hidden_sel.get(block_idx, torch.ones_like(p, dtype=torch.bool)).clone()
        elif "norm" in name.lower() and p.ndim == 1:
            mask = embed_channel_sel.clone()
        elif "head.weight" in name and p.ndim == 2:
            mask = embed_channel_sel.view(1, -1).expand_as(p).clone()
        elif "head.bias" in name and p.ndim == 1:
            mask = torch.ones_like(p, dtype=torch.bool)
        else:
            mask = torch.ones_like(p, dtype=torch.bool)

        if cfg.connectivity_closure and p.ndim in (2, 4):
            # Keep at least one coordinate per row/column group after structured masking.
            restored_total += _connectivity_close_dense_weight(mask, scores[name].to(mask.device), max(1, cfg.min_connections_per_unit))
        if int(mask.sum().item()) == 0:
            idx = torch.argmax(scores[name].reshape(-1))
            mask.reshape(-1)[idx] = True
            restored_total += 1
        retained_total += int(mask.sum().item())
        masks[name] = mask

    stats = {
        "eligible_parameter_count": float(eligible_total),
        "retained_eligible_parameter_count": float(retained_total),
        "pruned_eligible_parameter_count": float(max(0, eligible_total - retained_total)),
        "actual_prune_fraction_eligible": float(max(0, eligible_total - retained_total) / max(1, eligible_total)),
        "pruning_strategy": cfg.pruning_strategy.lower(),
        "prune_fraction": float(effective_prune_fraction),
        "iterative_pruning_rounds": int(cfg.iterative_pruning_rounds),
        "layerwise_normalize_scores": bool(cfg.layerwise_normalize_scores),
        "preserve_attention_heads": bool(cfg.preserve_attention_heads),
        "connectivity_restored_coordinate_count": float(restored_total),
        "embed_groups": float(n_embed_groups),
        "embed_groups_retained": float(int(embed_sel.sum().item())),
        "embed_group_keep_fraction": float(int(embed_sel.sum().item()) / max(1, n_embed_groups)),
        "hidden_block_count": float(len(hidden_scores)),
    }
    return masks, stats, embed_sel, hidden_sel


def _intersect_and_keep_one(
    candidate: torch.Tensor,
    previous: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Cumulative pruning step:
    - keep only coordinates that survived previous rounds
    - if that would empty the tensor, keep the best surviving coordinate
    """
    candidate = candidate & previous
    if int(candidate.sum().item()) > 0:
        return candidate

    prev_flat = previous.reshape(-1).bool()
    out = torch.zeros_like(candidate, dtype=torch.bool)

    if int(prev_flat.sum().item()) > 0:
        score_flat = score.reshape(-1).detach().float().clone()
        score_flat = score_flat.masked_fill(~prev_flat, float("-inf"))
        idx = torch.argmax(score_flat)
        out.reshape(-1)[idx] = True
    else:
        # Fallback safety: should not normally happen, but keeps the tensor non-empty.
        idx = torch.argmax(score.reshape(-1))
        out.reshape(-1)[idx] = True

    return out


def build_structured_masks_iterative(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], torch.Tensor, Dict[int, torch.Tensor], Dict[str, torch.Tensor]]:
    """Iteratively recompute sensitivities and refine a structured mask."""
    params = dict(model.named_parameters())
    active_masks: Dict[str, torch.Tensor] = {
        name: torch.ones_like(p, dtype=torch.bool, device=p.device)
        for name, p in params.items()
    }

    final_masks: Dict[str, torch.Tensor] = active_masks
    final_stats: Dict[str, float] = {}
    final_embed_sel = torch.empty(0, dtype=torch.bool)
    final_hidden_sel: Dict[int, torch.Tensor] = {}
    final_scores: Dict[str, torch.Tensor] = {}

    round_fractions: List[float] = []
    for round_idx in range(cfg.iterative_pruning_rounds):
        scores, _ = compute_sensitivity_scores(
            model,
            loader,
            cfg,
            device,
            probes=cfg.sensitivity_probes,
        )
        final_scores = scores

        round_prune_fraction = _iterative_prune_fraction(cfg, round_idx)
        round_fractions.append(round_prune_fraction)

        candidate_masks, stats, embed_sel, hidden_sel = _build_structured_selection_masks(
            model,
            scores,
            cfg,
            prune_fraction=round_prune_fraction,
        )

        # Make pruning cumulative without changing the round schedule.
        for name, cand in candidate_masks.items():
            if is_prunable_parameter(name, params[name], cfg):
                candidate_masks[name] = _intersect_and_keep_one(
                    cand.to(device=params[name].device, dtype=torch.bool),
                    active_masks[name],
                    scores[name].to(device=params[name].device),
                )
            else:
                candidate_masks[name] = cand.to(device=params[name].device, dtype=torch.bool)

        final_masks = candidate_masks
        final_stats = stats
        final_embed_sel = embed_sel
        final_hidden_sel = hidden_sel
        active_masks = candidate_masks
        apply_masks_(model, candidate_masks)

    final_stats["iterative_rounds_completed"] = float(cfg.iterative_pruning_rounds)
    final_stats["gradual_sparsification"] = bool(cfg.gradual_sparsification)
    final_stats["round_prune_fraction_schedule"] = [float(x) for x in round_fractions]
    final_stats["final_round_prune_fraction"] = float(round_fractions[-1]) if round_fractions else float(cfg.prune_fraction)
    final_stats["cumulative_iterative_pruning"] = True
    return final_masks, final_stats, final_embed_sel, final_hidden_sel, final_scores


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


def probe_covariance_eigvals(probe_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if probe_matrix is None or probe_matrix.numel() == 0:
        return None
    X = probe_matrix.float()
    X = X - X.mean(dim=0, keepdim=True)
    gram = (X @ X.T) / max(1, X.shape[0])
    vals = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0).detach().cpu()
    return vals


def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# -----------------------------------------------------------------------------
# Initial pruning
# -----------------------------------------------------------------------------

print(f"Using device: {device}")
print(f"Dataset: {cfg.dataset.upper()} | train={len(train_set)} | sensitivity={len(sens_set)} | test={len(test_set)}")
print(f"Model trainable parameters before pruning: {trainable_parameter_count(model):,}")
print("Computing zero-shot sensitivity scores at initialization")

prune_criterion = nn.CrossEntropyLoss()

sensitivity_scores, _ = compute_sensitivity_scores(model, sens_loader, cfg, device, probes=cfg.sensitivity_probes)
sensitivity_masks, pruning_stats = make_threshold_connectivity_masks(model, sensitivity_scores, cfg)

prune_stats = dict(pruning_stats)
target_sparsity = float(prune_stats["actual_prune_fraction_eligible"])

allowed_masks = {
    name: torch.ones_like(p, dtype=torch.bool) if is_prunable_parameter(name, p, cfg) else torch.zeros_like(p, dtype=torch.bool)
    for name, p in model.named_parameters()
}

prune_images, prune_targets = next(iter(train_loader))
prune_images = prune_images.to(device, non_blocking=True)
prune_targets = prune_targets.to(device, non_blocking=True)

if PRUNING_METHOD == "snip":
    masks = build_snip_masks(
        model,
        prune_images,
        prune_targets,
        prune_criterion,
        target_sparsity,
        allowed_masks=allowed_masks,
    )
elif PRUNING_METHOD == "synflow":
    masks = build_synflow_masks(
        model,
        prune_images,
        prune_targets,
        prune_criterion,
        target_sparsity,
        allowed_masks=allowed_masks,
        iterations=cfg.synflow_iterations,
    )
else:
    raise ValueError(f"Unsupported PRUNING_METHOD={PRUNING_METHOD!r}")

pruning_stats = {
    **pruning_stats,
    "pruning_method": PRUNING_METHOD,
    "target_sparsity": target_sparsity,
    "matched_sensitivity_masks": True,
}

S_init_flat_all = _flatten_like_model(model, sensitivity_scores)
apply_masks_(model, masks)
S_init_active = _flatten_like_model(model, sensitivity_scores, only_active=masks)
init_topk_idx = topk_indices(S_init_active, cfg.topk_frac)
initial_param_mag_active = _flatten_param_magnitudes(model)

print(f"{PRUNING_METHOD.upper()} pruning complete | target_sparsity={100.0 * target_sparsity:.2f}%")
print(json.dumps(pruning_stats, indent=2))
print("Training the pruned ViT from scratch")


# -----------------------------------------------------------------------------
# Training, evaluation, and checkpointed sensitivity analysis
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


def compute_epoch_lr(epoch: int, cfg: Config) -> float:
    if cfg.epochs <= 0:
        return cfg.min_lr
    warmup_epochs = min(cfg.warmup_epochs, cfg.epochs)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        start = cfg.lr * 1e-3
        progress = (epoch - 1) / max(1, warmup_epochs - 1)
        return start + (cfg.lr - start) * progress
    decay_epochs = max(1, cfg.epochs - warmup_epochs)
    decay_progress = (epoch - warmup_epochs) / decay_epochs
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay_progress)))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine_factor


criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


class SparseBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SparseVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SparseBlock(embed_dim, num_heads, hidden_dims[i], dropout) for i in range(depth)]
        )
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


def _build_sparse_model_from_masks(
    src_model: nn.Module,
    masks: Dict[str, torch.Tensor],
    cfg: Config,
    device: torch.device,
) -> SparseVisionTransformer:
    # Active embedding channels are shared across the whole ViT.
    active_embed_idx = masks["cls_token"][0, 0].detach().bool().nonzero(as_tuple=False).flatten().cpu()
    if active_embed_idx.numel() == 0:
        active_embed_idx = torch.tensor([0], dtype=torch.long)

    old_embed_dim = int(src_model.cls_token.shape[-1])
    new_embed_dim = int(active_embed_idx.numel())

    # Keep attention valid: embed_dim must be divisible by num_heads.
    if new_embed_dim % max(1, cfg.num_heads) == 0:
        new_num_heads = cfg.num_heads
    else:
        new_num_heads = max(1, math.gcd(new_embed_dim, cfg.num_heads))

    hidden_dims: List[int] = []
    for b in range(cfg.depth):
        key = f"blocks.{b}.mlp.fc1.bias"
        if key in masks:
            hidden_dims.append(int(masks[key].detach().bool().sum().item()))
        else:
            hidden_dims.append(int(src_model.blocks[b].mlp.fc1.weight.shape[0]))

    dst_model = SparseVisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        num_classes=cfg.num_classes,
        embed_dim=new_embed_dim,
        depth=cfg.depth,
        num_heads=new_num_heads,
        hidden_dims=hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    ae = active_embed_idx.to(device)
    qkv_idx = torch.cat([ae, ae + old_embed_dim, ae + 2 * old_embed_dim], dim=0)

    with torch.no_grad():
        # Patch embedding and tokens.
        dst_model.patch_embed.proj.weight.copy_(src_model.patch_embed.proj.weight[ae, :, :, :])
        if src_model.patch_embed.proj.bias is not None:
            dst_model.patch_embed.proj.bias.copy_(src_model.patch_embed.proj.bias[ae])
        dst_model.cls_token.copy_(src_model.cls_token[:, :, ae])
        dst_model.pos_embed.copy_(src_model.pos_embed[:, :, ae])

        # Per-block transfer.
        for bi, (dst_block, src_block) in enumerate(zip(dst_model.blocks, src_model.blocks)):
            hid_key = f"blocks.{bi}.mlp.fc1.bias"
            hid_mask = masks.get(hid_key, None)
            if hid_mask is None:
                hid_idx = torch.arange(src_block.mlp.fc1.weight.shape[0], device=device)
            else:
                hid_idx = hid_mask.detach().bool().nonzero(as_tuple=False).flatten().to(device)
                if hid_idx.numel() == 0:
                    hid_idx = torch.tensor([0], dtype=torch.long, device=device)

            # LayerNorms
            dst_block.norm1.weight.copy_(src_block.norm1.weight[ae])
            dst_block.norm1.bias.copy_(src_block.norm1.bias[ae])
            dst_block.norm2.weight.copy_(src_block.norm2.weight[ae])
            dst_block.norm2.bias.copy_(src_block.norm2.bias[ae])

            # Attention
            dst_block.attn.in_proj_weight.copy_(src_block.attn.in_proj_weight[qkv_idx][:, ae])
            dst_block.attn.in_proj_bias.copy_(src_block.attn.in_proj_bias[qkv_idx])
            dst_block.attn.out_proj.weight.copy_(src_block.attn.out_proj.weight[ae][:, ae])
            dst_block.attn.out_proj.bias.copy_(src_block.attn.out_proj.bias[ae])

            # MLP
            dst_block.mlp.fc1.weight.copy_(src_block.mlp.fc1.weight[hid_idx][:, ae])
            dst_block.mlp.fc1.bias.copy_(src_block.mlp.fc1.bias[hid_idx])
            dst_block.mlp.fc2.weight.copy_(src_block.mlp.fc2.weight[ae][:, hid_idx])
            dst_block.mlp.fc2.bias.copy_(src_block.mlp.fc2.bias[ae])

        # Final norm and head.
        dst_model.norm.weight.copy_(src_model.norm.weight[ae])
        dst_model.norm.bias.copy_(src_model.norm.bias[ae])
        dst_model.head.weight.copy_(src_model.head.weight[:, ae])
        dst_model.head.bias.copy_(src_model.head.bias)

    return dst_model


# Keep the original ViT parameterization and train with a fixed mask.
# This keeps SNIP, SynFlow, and sensitivity pruning comparable in the same
# model family instead of shrinking the architecture differently by method.
apply_masks_(model, masks)
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

# Recompute the baseline on the masked original model so tensor shapes match.
S_init_dict, init_probe_matrix = compute_sensitivity_scores(
    model,
    sens_loader,
    cfg,
    device,
    probes=cfg.analysis_probes,
    collect_probe_matrix=True,
)
S_init_active = _flatten_like_model(model, S_init_dict)
init_topk_idx = topk_indices(S_init_active, cfg.topk_frac)
initial_param_mag_active = _flatten_param_magnitudes(model)
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
        apply_masks_(model, masks)
        zero_masked_optimizer_state_(optimizer, model, masks)

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
        S_curr_active = _flatten_like_model(model, S_curr)
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
S_final_active = _flatten_like_model(model, S_final_dict)
eig_final = probe_covariance_eigvals(final_probe_matrix)
final_param_mag_active = _flatten_param_magnitudes(model)
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

run_tag = f"{cfg.dataset.lower()}_vit_tiny_SYNFLOW_{parameter_count(model)}_params"
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
    "analysis_note": "SYNFLOW pruning uses a sensitivity-derived sparsity target and the same checkpointed analysis pipeline as the baseline ViT tiny script.",
}

summary_path = out_dir / "vit_tiny_SYNFLOW_pruning_summary.json"
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
