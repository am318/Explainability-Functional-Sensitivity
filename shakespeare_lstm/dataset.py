"""
Char-level dataset utilities for the Shakespeare LSTM language model.

Mirrors the preprocessing used in the original Keras notebook: the corpus is
split into non-overlapping chunks of length `seq_length + 1`, and each chunk
is turned into an (input, target) pair by shifting by one character.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int]]:
    vocab = sorted(set(text))
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    return vocab, char2idx


def encode(text: str, char2idx: Dict[str, int]) -> np.ndarray:
    return np.array([char2idx[c] for c in text], dtype=np.int64)


class CharSequenceDataset(Dataset):
    """Non-overlapping (input, target) chunks of length `seq_length`."""

    def __init__(self, text_as_int: np.ndarray, seq_length: int):
        self.seq_length = seq_length
        num_chunks = len(text_as_int) // (seq_length + 1)
        if num_chunks == 0:
            raise ValueError(
                f"Text of length {len(text_as_int)} is too short for seq_length={seq_length}."
            )
        usable = num_chunks * (seq_length + 1)
        chunks = text_as_int[:usable].reshape(num_chunks, seq_length + 1)
        self.inputs = torch.from_numpy(chunks[:, :-1].copy())
        self.targets = torch.from_numpy(chunks[:, 1:].copy())

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]
