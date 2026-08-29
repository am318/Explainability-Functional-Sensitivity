"""Character-level text for the tiny GPT.

Byte-level vocabulary: no tokenizer dependency, no download beyond a plain .txt corpus.
Functional sensitivity is defined identically here -- f(x) is the logit tensor -- which is
the point: nothing about the definition is specific to classification.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch


class CharCorpus:
    def __init__(self, path: str, block_size: int, split: float = 0.9):
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        if not text:
            raise ValueError(f"empty corpus: {path}")
        vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = vocab
        self.vocab_size = len(vocab)
        self.block_size = block_size
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        cut = int(len(data) * split)
        self.train, self.val = data[:cut], data[cut:]

    def batch(self, split: str, batch_size: int, generator) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        hi = len(data) - self.block_size - 1
        ix = torch.randint(0, hi, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y

    def sensitivity_batches(self, n_samples: int, batch_size: int, seed: int,
                            folds: int) -> Tuple[List[torch.Tensor], List[int]]:
        g = torch.Generator().manual_seed(seed)
        batches, fold_ids = [], []
        per_fold = max(1, n_samples // max(1, folds))
        for f in range(folds):
            got = 0
            while got < per_fold:
                b = min(batch_size, per_fold - got)
                x, _ = self.batch("val", b, g)
                batches.append(x)
                fold_ids.append(f)
                got += b
        return batches, fold_ids
