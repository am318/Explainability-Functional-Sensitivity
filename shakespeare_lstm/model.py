"""
Char-level LSTM language model, translated from the original Keras
architecture: Embedding -> LSTM -> Linear (logits over the vocabulary).

The Keras model used a `stateful=True` LSTM, carrying hidden state between
consecutive batches within an epoch. That only makes sense when batches are
fed in a fixed temporal order; the original notebook shuffles its chunked
dataset, so the carried state does not correspond to a real continuation of
text. This implementation instead starts every training batch from a
zero hidden state (the standard approach for shuffled sequence chunks) and
only carries hidden state across steps during autoregressive generation,
where a single sequence really is being extended one character at a time.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

Hidden = Tuple[torch.Tensor, torch.Tensor]


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_units: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.head = nn.Linear(rnn_units, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.xavier_normal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(self, batch_size: int, device: torch.device) -> Hidden:
        h0 = torch.zeros(1, batch_size, self.rnn_units, device=device)
        c0 = torch.zeros(1, batch_size, self.rnn_units, device=device)
        return h0, c0

    def forward(
        self, x: torch.Tensor, hidden: Optional[Hidden] = None
    ) -> Tuple[torch.Tensor, Hidden]:
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.head(output)
        return logits, hidden
