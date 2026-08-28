"""Task abstraction: everything the runner needs that differs between vision and text.

Keeping this behind one interface is what lets the *identical* measurement pipeline run on
a CIFAR classifier and a character-level GPT. If the sensitivity code had to branch on the
task, the claim "this is a property of training, not of image classifiers" would be much
weaker than it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Task:
    is_text = False

    def train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device) -> Dict[str, float]:
        raise NotImplementedError

    def sensitivity_batches(self) -> Tuple[List[torch.Tensor], List[int]]:
        raise NotImplementedError

    def ntk_batch(self) -> Optional[torch.Tensor]:
        raise NotImplementedError


class VisionTask(Task):
    def __init__(self, cfg, seed: int, data_seed: int = None):
        from .data import vision
        self.cfg = cfg
        data_seed = seed if data_seed is None else data_seed
        train_set, clean_set, test_set = vision.build(cfg.data, seed)
        g = torch.Generator().manual_seed(data_seed)
        workers = int(getattr(cfg.data, "workers", 2))
        self.loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                                 num_workers=workers, drop_last=True, generator=g,
                                 persistent_workers=workers > 0)
        self.test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                                      num_workers=workers)
        self._it = iter(self.loader)
        self._sens = vision.sensitivity_batches(
            clean_set, cfg.sens.n_samples, cfg.sens.batch_size, cfg.sens.seed, cfg.sens.folds)
        n_ntk = min(cfg.sens.ntk_examples, self._sens[0][0].shape[0] * len(self._sens[0]))
        self._ntk = torch.cat(self._sens[0])[:n_ntk]

    def train_batch(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)

    def loss(self, logits, y, label_smoothing: float = 0.0):
        return F.cross_entropy(logits, y, label_smoothing=label_smoothing)

    @torch.no_grad()
    def evaluate(self, model, device):
        model.eval()
        tot = correct = 0
        loss_sum = 0.0
        for x, y in self.test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(1) == y).sum())
            tot += y.numel()
        return {"test_loss": loss_sum / max(1, tot), "test_acc": correct / max(1, tot)}

    def sensitivity_batches(self):
        return self._sens

    def ntk_batch(self):
        return self._ntk


class TextTask(Task):
    is_text = True

    def __init__(self, cfg, seed: int, data_seed: int = None):
        from .data.text import CharCorpus
        self.cfg = cfg
        data_seed = seed if data_seed is None else data_seed
        self.corpus = CharCorpus(cfg.data.text_file, cfg.model.block_size)
        self.g = torch.Generator().manual_seed(data_seed)
        self.eval_g = torch.Generator().manual_seed(seed + 1)
        self._sens = self.corpus.sensitivity_batches(
            cfg.sens.n_samples, cfg.sens.batch_size, cfg.sens.seed, cfg.sens.folds)
        self._ntk = torch.cat(self._sens[0])[:cfg.sens.ntk_examples]

    @property
    def vocab_size(self) -> int:
        return self.corpus.vocab_size

    def train_batch(self):
        return self.corpus.batch("train", self.cfg.train.batch_size, self.g)

    def loss(self, logits, y, label_smoothing: float = 0.0):
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                               label_smoothing=label_smoothing)

    @torch.no_grad()
    def evaluate(self, model, device, n_batches: int = 20):
        model.eval()
        g = torch.Generator().manual_seed(1234)
        loss_sum = 0.0
        correct = tot = 0
        for _ in range(n_batches):
            x, y = self.corpus.batch("val", 32, g)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                              y.reshape(-1)))
            correct += int((logits.argmax(-1) == y).sum())
            tot += y.numel()
        loss = loss_sum / n_batches
        return {"test_loss": loss, "test_acc": correct / max(1, tot),
                "test_ppl": float(torch.exp(torch.tensor(loss)))}

    def sensitivity_batches(self):
        return self._sens

    def ntk_batch(self):
        return self._ntk


def build_task(cfg, seed: int, data_seed: int = None) -> Task:
    if cfg.data.dataset.lower() == "text":
        return TextTask(cfg, seed, data_seed)
    return VisionTask(cfg, seed, data_seed)
