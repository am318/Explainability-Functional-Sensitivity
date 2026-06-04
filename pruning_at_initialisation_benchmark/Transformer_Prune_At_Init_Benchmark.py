"""
Pruning-at-initialization benchmark for ResNet-20, ViT-Tiny, and NanoGPT-style
character language models.

This is a self-contained replacement for the uploaded MLP/synthetic-data sweeps.
It keeps the important experimental interface:

  PRUNING_METHOD=sensitivity | snip | grasp | synflow | random | dense
  SPARSITY=0.95,0.98,0.99 or SPARSITY=0.95
  ARCH=resnet20 | vit_tiny | nanogpt
  DATASET=cifar10 | text

Default benchmark mapping:
  resnet20  -> CIFAR-10
  vit_tiny  -> CIFAR-10
  nanogpt   -> Tiny Shakespeare char-level LM, or TEXT_PATH if supplied

Examples:
  ARCH=vit_tiny PRUNING_METHOD=sensitivity SPARSITY=0.95 EPOCHS=50 python Transformer_Prune_At_Init_Benchmark.py
  ARCH=nanogpt PRUNING_METHOD=snip SPARSITY=0.99 MAX_ITERS=5000 TEXT_PATH=data/input.txt python Transformer_Prune_At_Init_Benchmark.py
  ARCH=vit_tiny PRUNING_METHOD=all SPARSITY=0.95,0.98,0.99 python Transformer_Prune_At_Init_Benchmark.py

Notes:
  - The custom method named "sensitivity" is an initialization-time Jacobian-norm
    score estimated with Hutchinson probes: E_v[(d <f(x), v> / d theta)^2].
    This scales to large output spaces and transformers better than explicit
    full Jacobian materialization.
  - All pruning methods share the same prunable parameter set by default:
    tensors with ndim >= 2, excluding biases and normalization parameters.
  - Masks are enforced after every optimizer update, including optimizer-state
    zeroing for Adam/SGD momentum buffers.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None
    T = None


# -----------------------------------------------------------------------------
# Environment parsing
# -----------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y", "on"}


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


@dataclass
class Config:
    arch: str = env_str("ARCH", "vit_tiny")
    dataset: str = env_str("DATASET", "auto")
    pruning_method: str = env_str("PRUNING_METHOD", "sensitivity")
    sparsity: str = env_str("SPARSITY", "0.95")
    seed: int = env_int("SEED", 0)
    output_dir: str = env_str("OUTPUT_DIR", "PruningBenchResults")
    data_dir: str = env_str("DATA_DIR", "data")

    # Image training
    epochs: int = env_int("EPOCHS", 100)
    batch_size: int = env_int("BATCH_SIZE", 128)
    eval_batch_size: int = env_int("EVAL_BATCH_SIZE", 256)
    lr: float = env_float("LR", 3e-4)
    weight_decay: float = env_float("WEIGHT_DECAY", 5e-2)
    warmup_steps: int = env_int("WARMUP_STEPS", 500)
    train_subset: int = env_int("TRAIN_SUBSET", 0)
    eval_subset: int = env_int("EVAL_SUBSET", 0)
    num_workers: int = env_int("NUM_WORKERS", 2)

    # NanoGPT training
    text_path: str = env_str("TEXT_PATH", "")
    block_size: int = env_int("BLOCK_SIZE", 128)
    max_iters: int = env_int("MAX_ITERS", 5000)
    eval_interval: int = env_int("EVAL_INTERVAL", 500)
    eval_iters: int = env_int("EVAL_ITERS", 100)
    gpt_batch_size: int = env_int("GPT_BATCH_SIZE", 64)
    n_layer: int = env_int("N_LAYER", 6)
    n_head: int = env_int("N_HEAD", 6)
    n_embd: int = env_int("N_EMBD", 384)
    dropout: float = env_float("DROPOUT", 0.0)

    # Pruning score computation
    score_batches: int = env_int("SCORE_BATCHES", 1)
    hutchinson_probes: int = env_int("HUTCHINSON_PROBES", 4)
    synflow_input_batches: int = env_int("SYNFLOW_INPUT_BATCHES", 1)
    prune_biases: bool = env_bool("PRUNE_BIASES", False)
    prune_norms: bool = env_bool("PRUNE_NORMS", False)
    prune_embeddings: bool = env_bool("PRUNE_EMBEDDINGS", False)
    device: str = env_str("DEVICE", "auto")
    compile_model: bool = env_bool("COMPILE", False)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg: Config) -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parameter_count(model: nn.Module, only_trainable: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_trainable))


def infer_norm_parameter_names(model: nn.Module) -> set[str]:
    norm_names: set[str] = set()
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)
    for module_name, module in model.named_modules():
        if isinstance(module, norm_types):
            for child_name, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{child_name}" if module_name else child_name
                norm_names.add(full)
    return norm_names


def infer_embedding_parameter_names(model: nn.Module) -> set[str]:
    emb_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            for child_name, _ in module.named_parameters(recurse=False):
                full = f"{module_name}.{child_name}" if module_name else child_name
                emb_names.add(full)
    return emb_names


def prunable_named_parameters(model: nn.Module, cfg: Config) -> List[Tuple[str, nn.Parameter]]:
    norm_names = infer_norm_parameter_names(model)
    emb_names = infer_embedding_parameter_names(model)
    out: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (not cfg.prune_biases) and name.endswith("bias"):
            continue
        if (not cfg.prune_norms) and name in norm_names:
            continue
        if (not cfg.prune_embeddings) and name in emb_names:
            continue
        if p.ndim < 2 and not cfg.prune_biases:
            continue
        out.append((name, p))
    if not out:
        raise RuntimeError("No prunable parameters selected. Relax PRUNE_* exclusions.")
    return out


def zero_like_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p, dtype=torch.bool, device=p.device) for name, p in model.named_parameters()}


def dense_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: torch.ones_like(p, dtype=torch.bool, device=p.device) for name, p in model.named_parameters()}


def masks_from_scores(model: nn.Module, scores: Dict[str, torch.Tensor], sparsity: float, cfg: Config) -> Dict[str, torch.Tensor]:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    masks = dense_masks(model)
    items = prunable_named_parameters(model, cfg)
    flat_scores = torch.cat([scores[name].detach().abs().flatten().float().cpu() for name, _ in items])
    total = flat_scores.numel()
    n_keep = max(1, total - int(round(sparsity * total)))
    if n_keep >= total:
        return masks

    # Exact global unstructured mask: keep the n_keep largest scores.
    keep_idx = torch.topk(flat_scores, k=n_keep, largest=True, sorted=False).indices
    flat_keep = torch.zeros(total, dtype=torch.bool)
    flat_keep[keep_idx] = True

    offset = 0
    for name, p in items:
        n = p.numel()
        masks[name] = flat_keep[offset:offset + n].view_as(p).to(p.device)
        offset += n
    return masks


def random_masks(model: nn.Module, sparsity: float, cfg: Config, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    masks = dense_masks(model)
    items = prunable_named_parameters(model, cfg)
    total = sum(p.numel() for _, p in items)
    n_prune = int(round(sparsity * total))
    n_prune = max(0, min(total - 1, n_prune))
    flat_mask = torch.ones(total, dtype=torch.bool)
    if n_prune > 0:
        perm = torch.randperm(total, generator=generator)
        flat_mask[perm[:n_prune]] = False
    off = 0
    for name, p in items:
        n = p.numel()
        masks[name] = flat_mask[off:off + n].view_as(p).to(p.device)
        off += n
    return masks


def apply_masks_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in masks:
                p.mul_(masks[name].to(device=p.device, dtype=p.dtype))


def zero_optimizer_state_(optimizer: torch.optim.Optimizer, model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        state = optimizer.state.get(p)
        if not state or name not in masks:
            continue
        m = masks[name].to(device=p.device, dtype=p.dtype)
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == p.shape:
                value.mul_(m)


def mask_stats(model: nn.Module, masks: Dict[str, torch.Tensor], cfg: Config) -> Dict[str, float]:
    total = 0
    retained = 0
    prunable_names = {name for name, _ in prunable_named_parameters(model, cfg)}
    for name, p in model.named_parameters():
        if name not in prunable_names:
            continue
        m = masks[name].detach().bool()
        total += m.numel()
        retained += int(m.sum().item())
    pruned = total - retained
    return {
        "prunable_parameter_count": int(total),
        "retained_parameter_count": int(retained),
        "pruned_parameter_count": int(pruned),
        "actual_sparsity": float(pruned / max(1, total)),
    }


# -----------------------------------------------------------------------------
# ResNet-20 for CIFAR-10
# -----------------------------------------------------------------------------

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks: List[int], num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet20(num_classes: int = 10) -> nn.Module:
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


# -----------------------------------------------------------------------------
# ViT-Tiny for CIFAR-10
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViTTiny(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, num_classes: int = 10,
                 embed_dim: int = 192, depth: int = 12, num_heads: int = 3, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# -----------------------------------------------------------------------------
# NanoGPT-style character model
# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data.long()
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class GPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([GPTBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        if t > self.block_size:
            raise ValueError("Cannot forward sequence longer than block_size")
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def maybe_subset(ds, n: int, seed: int):
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def build_cifar10_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    if torchvision is None:
        raise RuntimeError("torchvision is required for CIFAR-10. Install torchvision or use ARCH=nanogpt.")
    root = Path(cfg.data_dir)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root=str(root), train=False, download=True, transform=test_tf)
    train_ds = maybe_subset(train_ds, cfg.train_subset, cfg.seed)
    test_ds = maybe_subset(test_ds, cfg.eval_subset, cfg.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def load_text_data(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, int], Dict[int, str]]:
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = Path(cfg.text_path) if cfg.text_path else data_dir / "tinyshakespeare.txt"
    if not path.exists():
        if cfg.text_path:
            raise FileNotFoundError(f"TEXT_PATH does not exist: {path}")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Tiny Shakespeare to {path}")
        urllib.request.urlretrieve(url, path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars), stoi, itos


def get_batch_from_tensor(data: torch.Tensor, cfg: Config, device: torch.device, split_batch_size: Optional[int] = None):
    bsz = split_batch_size or cfg.gpt_batch_size
    ix = torch.randint(len(data) - cfg.block_size - 1, (bsz,))
    x = torch.stack([data[i:i + cfg.block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix]).to(device)
    return x, y


# -----------------------------------------------------------------------------
# Model factory and losses
# -----------------------------------------------------------------------------

def build_model_and_data(cfg: Config, device: torch.device):
    arch = cfg.arch.lower()
    if arch in {"resnet20", "vit_tiny"}:
        dataset = "cifar10" if cfg.dataset == "auto" else cfg.dataset.lower()
        if dataset != "cifar10":
            raise ValueError(f"{arch} currently supports DATASET=cifar10, got {dataset}")
        train_loader, test_loader = build_cifar10_loaders(cfg)
        model = resnet20() if arch == "resnet20" else ViTTiny(dropout=cfg.dropout)
        model.to(device)
        return model, {"type": "image", "train_loader": train_loader, "test_loader": test_loader, "dataset": dataset}
    if arch == "nanogpt":
        train_data, val_data, vocab_size, stoi, itos = load_text_data(cfg)
        model = NanoGPT(vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.dropout).to(device)
        return model, {"type": "text", "train_data": train_data, "val_data": val_data, "vocab_size": vocab_size, "dataset": "text"}
    raise ValueError(f"Unknown ARCH={cfg.arch!r}. Use resnet20, vit_tiny, or nanogpt.")


def forward_loss(model: nn.Module, batch, data_info: Dict, device: torch.device):
    if data_info["type"] == "image":
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        return F.cross_entropy(logits, y), logits, y
    x, y = batch
    logits, loss = model(x, y)
    return loss, logits, y


def first_batches(data_info: Dict, cfg: Config, device: torch.device, n: int):
    if data_info["type"] == "image":
        loader = data_info["train_loader"]
        out = []
        it = iter(loader)
        for _ in range(n):
            try:
                out.append(next(it))
            except StopIteration:
                break
        return out
    return [get_batch_from_tensor(data_info["train_data"], cfg, device) for _ in range(n)]


# -----------------------------------------------------------------------------
# Pruning scores
# -----------------------------------------------------------------------------

def init_scores_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p, dtype=torch.float32, device=p.device) for name, p in model.named_parameters()}


def accumulate_grad_scores(model: nn.Module, scores: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    for name, p in model.named_parameters():
        if p.grad is not None:
            scores[name].add_(p.grad.detach().float().pow(2), alpha=scale)


def snip_scores(model: nn.Module, batches, data_info: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss_total = 0.0
    for batch in batches:
        loss, _, _ = forward_loss(model, batch, data_info, device)
        loss_total = loss_total + loss / max(1, len(batches))
    loss_total.backward()
    scores = init_scores_dict(model)
    for name, p in model.named_parameters():
        if p.grad is not None:
            scores[name] = (p.grad.detach() * p.detach()).abs().float()
    model.zero_grad(set_to_none=True)
    return scores


def grasp_scores(model: nn.Module, batches, data_info: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    # GraSP saliency approximation: -theta * H g, using the same mini-batches twice.
    model.zero_grad(set_to_none=True)
    loss = 0.0
    for batch in batches:
        l, _, _ = forward_loss(model, batch, data_info, device)
        loss = loss + l / max(1, len(batches))
    params = [p for p in model.parameters() if p.requires_grad]
    grad = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grad_vec = [g for g in grad if g is not None]
    z = sum((g.detach() * g).sum() for g in grad_vec)
    hv = torch.autograd.grad(z, params, allow_unused=True)
    scores = init_scores_dict(model)
    for (name, p), h in zip(model.named_parameters(), hv):
        if h is not None:
            scores[name] = (-p.detach() * h.detach()).float()
    model.zero_grad(set_to_none=True)
    return scores


def synflow_scores(model: nn.Module, data_info: Dict, cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    signs = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            signs[name] = torch.sign(p)
            p.abs_()
    model.zero_grad(set_to_none=True)
    if data_info["type"] == "image":
        x = torch.ones(1, 3, 32, 32, device=device)
        out = model(x).sum()
    else:
        x = torch.zeros(1, cfg.block_size, dtype=torch.long, device=device)
        out = model(x)[0].sum()
    out.backward()
    scores = init_scores_dict(model)
    for name, p in model.named_parameters():
        if p.grad is not None:
            scores[name] = (p.grad.detach() * p.detach()).abs().float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.mul_(signs[name])
    model.zero_grad(set_to_none=True)
    return scores


def sensitivity_scores(model: nn.Module, batches, data_info: Dict, cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    """Hutchinson estimate of diagonal output-Jacobian covariance at initialization."""
    scores = init_scores_dict(model)
    model.zero_grad(set_to_none=True)
    probes = max(1, cfg.hutchinson_probes)
    for batch in batches:
        _, logits, _ = forward_loss(model, batch, data_info, device)
        # Use all image logits; for LM, use a subsampled time axis to control cost.
        if data_info["type"] == "text" and logits.ndim == 3 and logits.shape[1] > 32:
            step = max(1, logits.shape[1] // 32)
            logits_for_score = logits[:, ::step, :]
        else:
            logits_for_score = logits
        for _ in range(probes):
            v = torch.empty_like(logits_for_score).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            scalar = (logits_for_score * v).sum() / math.sqrt(float(v.numel()))
            model.zero_grad(set_to_none=True)
            scalar.backward(retain_graph=True)
            accumulate_grad_scores(model, scores, scale=1.0 / (len(batches) * probes))
    model.zero_grad(set_to_none=True)
    return scores


def build_masks(model: nn.Module, method: str, sparsity: float, data_info: Dict, cfg: Config, device: torch.device) -> Dict[str, torch.Tensor]:
    method = method.lower()
    if method == "dense" or sparsity <= 0.0:
        return dense_masks(model)
    if method == "random":
        g = torch.Generator().manual_seed(cfg.seed)
        return random_masks(model, sparsity, cfg, g)
    batches = first_batches(data_info, cfg, device, max(1, cfg.score_batches))
    if method == "sensitivity":
        scores = sensitivity_scores(model, batches, data_info, cfg, device)
    elif method == "snip":
        scores = snip_scores(model, batches, data_info, device)
    elif method == "grasp":
        scores = grasp_scores(model, batches, data_info, device)
    elif method == "synflow":
        scores = synflow_scores(model, data_info, cfg, device)
    else:
        raise ValueError(f"Unknown PRUNING_METHOD={method!r}")
    return masks_from_scores(model, scores, sparsity, cfg)


# -----------------------------------------------------------------------------
# Training/eval
# -----------------------------------------------------------------------------

def make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not name.endswith("bias"):
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate_image(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += y.numel()
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def estimate_text_loss(model: nn.Module, data: torch.Tensor, cfg: Config, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch_from_tensor(data, cfg, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    return sum(losses) / max(1, len(losses))


def train_image(model: nn.Module, data_info: Dict, masks: Dict[str, torch.Tensor], cfg: Config, device: torch.device) -> Dict:
    optimizer = make_optimizer(model, cfg)
    train_loader, test_loader = data_info["train_loader"], data_info["test_loader"]
    total_steps = cfg.epochs * len(train_loader)
    step = 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            lr = cosine_lr(step, total_steps, cfg.lr, cfg.warmup_steps)
            set_optimizer_lr(optimizer, lr)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            apply_masks_(model, masks)
            zero_optimizer_state_(optimizer, model, masks)
            step += 1
        train_loss, train_acc = evaluate_image(model, train_loader, device) if cfg.train_subset else (float("nan"), float("nan"))
        test_loss, test_acc = evaluate_image(model, test_loader, device)
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc}
        history.append(row)
        print(json.dumps(row))
    return {"history": history, "final_test_loss": history[-1]["test_loss"], "final_test_acc": history[-1]["test_acc"]}


def train_text(model: nn.Module, data_info: Dict, masks: Dict[str, torch.Tensor], cfg: Config, device: torch.device) -> Dict:
    optimizer = make_optimizer(model, cfg)
    train_data, val_data = data_info["train_data"], data_info["val_data"]
    history = []
    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            train_loss = estimate_text_loss(model, train_data, cfg, device)
            val_loss = estimate_text_loss(model, val_data, cfg, device)
            row = {"iter": step, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": math.exp(min(20.0, val_loss))}
            history.append(row)
            print(json.dumps(row))
        if step == cfg.max_iters:
            break
        lr = cosine_lr(step, cfg.max_iters, cfg.lr, cfg.warmup_steps)
        set_optimizer_lr(optimizer, lr)
        model.train()
        x, y = get_batch_from_tensor(train_data, cfg, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        apply_masks_(model, masks)
        zero_optimizer_state_(optimizer, model, masks)
    return {"history": history, "final_val_loss": history[-1]["val_loss"], "final_val_ppl": history[-1]["val_ppl"]}


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_one(cfg: Config, method: str, sparsity: float, device: torch.device) -> Dict:
    set_seed(cfg.seed)
    model, data_info = build_model_and_data(cfg, device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    n_params = parameter_count(model)
    print(f"arch={cfg.arch} dataset={data_info['dataset']} params={n_params} method={method} sparsity={sparsity}")
    t0 = time.time()
    masks = build_masks(model, method, sparsity, data_info, cfg, device)
    apply_masks_(model, masks)
    stats = mask_stats(model, masks, cfg)
    prune_seconds = time.time() - t0
    print("mask_stats", json.dumps(stats))
    if data_info["type"] == "image":
        train_result = train_image(model, data_info, masks, cfg, device)
    else:
        train_result = train_text(model, data_info, masks, cfg, device)
    result = {
        "config": asdict(cfg),
        "arch": cfg.arch,
        "dataset": data_info["dataset"],
        "method": method,
        "requested_sparsity": sparsity,
        "parameter_count": n_params,
        "mask_stats": stats,
        "prune_seconds": prune_seconds,
        **train_result,
    }
    out_dir = Path(cfg.output_dir) / cfg.arch / data_info["dataset"] / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{cfg.seed}_sparsity{sparsity:.4f}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return result


def main() -> None:
    cfg = Config()
    device = get_device(cfg)
    print(f"Using device: {device}")
    methods = [cfg.pruning_method.lower()]
    if cfg.pruning_method.lower() == "all":
        methods = ["dense", "random", "sensitivity", "snip", "grasp", "synflow"]
    sparsities = parse_float_list(cfg.sparsity)
    all_results = []
    for method in methods:
        for sp in sparsities:
            sp_eff = 0.0 if method == "dense" else sp
            all_results.append(run_one(cfg, method, sp_eff, device))
    out_dir = Path(cfg.output_dir) / cfg.arch
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / f"aggregate_seed{cfg.seed}_{int(time.time())}.json"
    aggregate_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote aggregate {aggregate_path}")


if __name__ == "__main__":
    main()
