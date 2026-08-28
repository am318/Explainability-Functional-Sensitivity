"""Character-level LSTM, adapted from the ShakespeareLSTM branch.

A fourth architecture family that is neither convolutional nor attentional. If the
granularity ladder behaves the same way in a recurrent model, the result is about training
dynamics rather than about a particular inductive bias.

Hidden state starts at zero for every batch: chunks are shuffled, so a carried state would
not correspond to a real continuation of the text. State is only carried during
autoregressive generation, which we do not do here.

The functional sensitivity definition applies unchanged --- f(x) is the logit tensor, and
no label enters --- which is the point of using a label-free score.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

Hidden = Tuple[torch.Tensor, torch.Tensor]


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 64, rnn_units: int = 256,
                 num_layers: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, num_layers=num_layers,
                            batch_first=True)
        self.head = nn.Linear(rnn_units, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.xavier_normal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits only, so the module is a plain x -> f(x) map and the same
        # sensitivity/estimator code applies with no special-casing.
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.head(output)
