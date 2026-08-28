"""Run configuration.

One `RunCfg` == one job. The CLI takes a single config (from flags or a JSON file) and
writes everything it produces into `results/<run_id>/`, so a cluster job array is just
N independent invocations with different flags and no coordination.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelCfg:
    arch: str = "vit"                 # vit | resnet20 | mlp | gpt
    width: int = 192                  # embed dim (vit/gpt), base channels (resnet), hidden (mlp)
    depth: int = 6
    heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0              # kept at 0: sensitivity is an eval-mode quantity
    patch_size: int = 4               # vit
    lazy_alpha: float = 1.0           # Chizat et al. laziness knob (1.0 == ordinary training)
    block_size: int = 128             # gpt context length
    vocab_size: int = 0               # gpt, filled in from the dataset


@dataclass
class DataCfg:
    dataset: str = "cifar10"          # cifar10 | cifar100 | text
    data_dir: str = "./data"
    image_size: int = 32
    download: bool = False
    train_subset: int = 0             # 0 == full
    test_subset: int = 2000
    augment: bool = True
    workers: int = 2            # DataLoader workers; 0 keeps everything in-process
    text_file: str = ""               # character-level corpus for dataset == "text"


@dataclass
class TrainCfg:
    steps: int = 4000
    batch_size: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-5
    warmup_steps: int = 200
    weight_decay: float = 0.05
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9             # sgd only
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    lr_schedule: str = "cosine"       # cosine | constant


@dataclass
class SensCfg:
    """How functional sensitivity S(theta) = E_x ||d f(x)/d theta||^2 is estimated.

    `folds` is the backbone of the C2a noise-floor control: the sensitivity set is split
    into `folds` disjoint subsets, S is accumulated separately on each, and the agreement
    *between folds at the same checkpoint* upper-bounds any agreement we can claim
    *across* checkpoints.
    """
    estimator: str = "auto"           # auto | exact | hutchinson
    exact_max_outputs: int = 32       # "auto" picks exact when output dim <= this
    n_samples: int = 512              # examples used to estimate S
    n_probes: int = 8                 # hutchinson probes per example
    batch_size: int = 32
    folds: int = 2
    include: str = "prunable"         # prunable | all
    prune_bias: bool = False
    prune_norm: bool = False
    prune_embeddings: bool = False
    prune_head: bool = False
    ntk_examples: int = 48            # empirical NTK Gram size for kernel velocity (C4)
    seed: int = 1234
    impl: str = "auto"                # auto | vmap | loop


@dataclass
class RunCfg:
    tag: str = "dev"
    seed: int = 0                     # model initialisation
    data_seed: int = -1               # batch order; -1 == follow `seed`.
                                      # Splitting these is what makes the same-init /
                                      # different-data-order comparison well posed:
                                      # parameterwise rankings from *different* inits are
                                      # not comparable at all under permutation symmetry.
    device: str = "auto"              # auto | cpu | mps | cuda
    out_dir: str = "results"

    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    sens: SensCfg = field(default_factory=SensCfg)

    # checkpoint schedule (log-spaced in optimiser steps, always includes 0 and `steps`)
    n_ckpts: int = 22
    ckpt_first: int = 1

    # sparsity grid at which top-k overlap is evaluated (fraction of weights REMOVED)
    sparsities: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 0.95, 0.99])

    # keep full float32 score vectors: all | none. Masks + subsamples are always kept.
    keep_scores: str = "all"
    track_criteria: bool = True       # also measure fisher/snip/synflow/magnitude
    track_structured: bool = True     # also measure the per-output-unit (structured) ranking
    # save model weights at these steps so C6 can prune-and-retrain from them; -1 == all ckpts
    save_state_at: List[int] = field(default_factory=list)

    def run_id(self) -> str:
        payload = json.dumps(to_dict(self), sort_keys=True).encode()
        return f"{self.tag}-{hashlib.sha1(payload).hexdigest()[:10]}"


def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj) if is_dataclass(obj) else dict(obj)


def _build(cls, payload: Dict[str, Any]):
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def from_dict(payload: Dict[str, Any]) -> RunCfg:
    payload = dict(payload)
    sub = {
        "model": ModelCfg,
        "data": DataCfg,
        "train": TrainCfg,
        "sens": SensCfg,
    }
    kwargs: Dict[str, Any] = {}
    for key, cls in sub.items():
        kwargs[key] = _build(cls, payload.pop(key, {}) or {})
    return _build(RunCfg, {**payload, **kwargs})


def load(path: str) -> RunCfg:
    with open(path) as fh:
        return from_dict(json.load(fh))


def dump(cfg: RunCfg, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(to_dict(cfg), fh, indent=2, sort_keys=True)


def override(cfg: RunCfg, dotted: str, value: str) -> RunCfg:
    """Apply a `train.lr=3e-4` style override, casting to the field's declared type."""
    payload = to_dict(cfg)
    parts = dotted.split(".")
    node: Any = payload
    for part in parts[:-1]:
        if part not in node:
            raise KeyError(f"no config section '{part}' in '{dotted}'")
        node = node[part]
    leaf = parts[-1]
    if leaf not in node:
        raise KeyError(f"no config field '{dotted}'")
    current = node[leaf]
    if isinstance(current, bool):
        node[leaf] = value.lower() in {"1", "true", "yes"}
    elif isinstance(current, int) and not isinstance(current, bool):
        node[leaf] = int(float(value))
    elif isinstance(current, float):
        node[leaf] = float(value)
    elif isinstance(current, list):
        node[leaf] = [float(v) if "." in v or "e" in v.lower() else int(v)
                      for v in value.split(",") if v != ""]
    else:
        node[leaf] = value
    return from_dict(payload)
