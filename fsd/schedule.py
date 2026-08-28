"""Checkpoint and learning-rate schedules.

Checkpoints are log-spaced in optimiser steps because the whole claim of the paper is that
the interesting dynamics happen in the first few hundred steps. A linear grid over a
4000-step run would put ~1 sample inside the phase we care about.
"""
from __future__ import annotations

import math
from typing import List


def log_checkpoints(total_steps: int, n_ckpts: int, first: int = 1) -> List[int]:
    """Log-spaced steps in [0, total_steps], always including 0 and total_steps."""
    if total_steps <= 0:
        return [0]
    n_ckpts = max(2, n_ckpts)
    pts = {0, int(total_steps)}
    first = max(1, min(first, total_steps))
    lo, hi = math.log(first), math.log(total_steps)
    for i in range(n_ckpts - 1):
        frac = i / max(1, n_ckpts - 2)
        pts.add(int(round(math.exp(lo + (hi - lo) * frac))))
    return sorted(p for p in pts if 0 <= p <= total_steps)


def lr_at(step: int, cfg) -> float:
    """Warmup + cosine (or constant) on a per-step basis."""
    warmup = max(0, int(cfg.warmup_steps))
    if warmup > 0 and step < warmup:
        return cfg.lr * (step + 1) / warmup
    if cfg.lr_schedule == "constant":
        return cfg.lr
    decay_steps = max(1, cfg.steps - warmup)
    progress = min(1.0, (step - warmup) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine
