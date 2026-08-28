"""
Word-level dataset utilities for the WikiText-2 AWD-LSTM language model.

Unlike shakespeare_lstm's char dataset, WikiText-2 ships pre-split into
train/valid/test files, so we use the real validation split as the
sensitivity/eval probe set (rather than carving a random slice out of
train) and reserve test for a final, never-touched-during-development
number. The vocabulary is built once, over all three splits pooled
(matching the original awd-lstm-lm/data.py Corpus class), so a word that
only appears in valid/test still gets a stable id.

Chunking follows shakespeare_lstm/dataset.py: each split's token stream is
cut into non-overlapping (input, target) chunks of length `seq_length`,
target = input shifted by one token. This is simpler than the original
repo's single continuous stream with truncated-BPTT hidden carry-over, and
-- as in shakespeare_lstm/model.py -- is the deliberate choice here too:
carrying hidden state only makes sense for temporally contiguous, unshuffled
batches, and we shuffle.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Dictionary:
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)


def _read_words(path: Path) -> List[List[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split() + ["<eos>"] for line in lines]


def build_corpus(data_dir: str) -> Tuple[Dictionary, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dictionary, train_ids, valid_ids, test_ids), each ids array
    a 1D int64 numpy array of token indices for that split."""
    data_dir = Path(data_dir)
    dictionary = Dictionary()
    split_lines = {}
    for split in ("train", "valid", "test"):
        lines = _read_words(data_dir / f"{split}.txt")
        split_lines[split] = lines
        for words in lines:
            for w in words:
                dictionary.add_word(w)

    def encode(lines: List[List[str]]) -> np.ndarray:
        return np.array([dictionary.word2idx[w] for words in lines for w in words], dtype=np.int64)

    return dictionary, encode(split_lines["train"]), encode(split_lines["valid"]), encode(split_lines["test"])


class WordSequenceDataset(Dataset):
    """Non-overlapping (input, target) chunks of length `seq_length`, same
    convention as shakespeare_lstm's CharSequenceDataset."""

    def __init__(self, token_ids: np.ndarray, seq_length: int):
        self.seq_length = seq_length
        num_chunks = len(token_ids) // (seq_length + 1)
        if num_chunks == 0:
            raise ValueError(
                f"Token stream of length {len(token_ids)} is too short for seq_length={seq_length}."
            )
        usable = num_chunks * (seq_length + 1)
        chunks = token_ids[:usable].reshape(num_chunks, seq_length + 1)
        self.inputs = torch.from_numpy(chunks[:, :-1].copy())
        self.targets = torch.from_numpy(chunks[:, 1:].copy())

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]
