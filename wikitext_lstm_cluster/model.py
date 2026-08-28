"""
Word-level AWD-LSTM language model (Merity et al., "Regularizing and
Optimizing LSTM Language Models", https://arxiv.org/abs/1708.02182),
adapted from awd-lstm-lm/model.py for this project's sensitivity/pruning
infrastructure.

The whole point of trying this architecture is that the char-LSTM in
shakespeare_lstm/ turned out to badly overfit (train loss << held-out loss).
AWD-LSTM's regularization -- DropConnect on the hidden-to-hidden weight
matrix (WeightDrop), locked/variational dropout on the embedding and
between-layer activations (LockedDropout), and embedding dropout
(embedded_dropout), plus tied input/output embeddings -- is specifically
designed to close that gap. All four are kept here; the only things
deliberately dropped relative to the original repo are NT-ASGD (we use plain
Adam, consistent with the rest of this project) and the AR/TAR activation
regularization terms and adaptive-softmax (SplitCrossEntropyLoss) -- both
optional extras, not part of the core architecture, and dropping them keeps
`model(x) -> logits` a plain map compatible with common/sensitivity.py.

As in shakespeare_lstm/model.py, every training batch starts from a zero
hidden state rather than carrying state across shuffled chunks (hidden
carry-over is only meaningful for batches that are a real, temporally
ordered continuation of each other).

Internally the recurrent stack is sequence-first (seq, batch, feat), matching
the original AWD-LSTM code and LockedDropout's masking convention (a mask of
shape (1, batch, feat) is broadcast over the time dimension, i.e. shared
across time steps but not across batch elements or features); the public
forward() accepts and returns batch-first tensors, transposing at the
boundary, to match this project's dataset/loss convention (see dataset.py,
train.py).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Hidden = List[Tuple[torch.Tensor, torch.Tensor]]


class LockedDropout(nn.Module):
    """Same dropout mask reused across all time steps of a sequence-first
    tensor (x: (seq, batch, feat)) -- "variational" dropout for RNNs."""

    def forward(self, x: torch.Tensor, dropout: float = 0.5) -> torch.Tensor:
        if not self.training or not dropout:
            return x
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - dropout) / (1 - dropout)
        return mask.expand_as(x) * x


def embedded_dropout(embed: nn.Embedding, words: torch.Tensor, dropout: float = 0.1) -> torch.Tensor:
    """Drops entire rows (whole word types) of the embedding table, same mask
    for every occurrence of a given word in the batch."""
    if dropout:
        mask = embed.weight.new_empty(embed.weight.size(0), 1).bernoulli_(1 - dropout) / (1 - dropout)
        masked_weight = mask.expand_as(embed.weight) * embed.weight
    else:
        masked_weight = embed.weight
    padding_idx = embed.padding_idx if embed.padding_idx is not None else -1
    return F.embedding(
        words, masked_weight, padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse,
    )


class WeightDrop(nn.Module):
    """DropConnect on a named weight of a wrapped module (here, the
    hidden-to-hidden weight of a single-layer LSTM): the raw weight is kept
    as a learnable parameter (`<name>_raw`) and a freshly-dropped-out copy is
    installed as a plain (non-parameter) attribute before every forward
    call, so gradients flow back through the dropout mask to the raw weight.

    PyTorch's RNNBase caches references to its weight tensors in
    `_flat_weights` (rebuilt only at construction / `flatten_parameters()`,
    not looked up dynamically each call) for the fused cuDNN/backend kernel.
    Since we swap in a new tensor by attribute name every forward, that cache
    must be refreshed each time too, or the backend would keep using a stale
    (or in the CUDA case, wrong-memory) weight -- the original 2017-era
    implementation predates this caching and omits the refresh.
    """

    def __init__(self, module: nn.Module, weights: List[str], dropout: float = 0.0):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters = lambda *a, **kw: None
        for name in weights:
            w = getattr(module, name)
            del module._parameters[name]
            module.register_parameter(name + "_raw", nn.Parameter(w.data))

    def _set_weights(self) -> None:
        for name in self.weights:
            raw = getattr(self.module, name + "_raw")
            w = F.dropout(raw, p=self.dropout, training=self.training)
            # Plain `setattr` goes through nn.Module.__setattr__, which (a)
            # auto-registers a Parameter-valued assignment into
            # self.module._parameters -- and in eval mode F.dropout(...,
            # training=False) is a no-op that returns `raw` itself (still a
            # Parameter), so this silently re-registers `name` as a
            # Parameter; and (b) once that registration exists, RNNBase's
            # own __setattr__ then *rejects* the next train-mode call's
            # plain (non-Parameter) dropout output with a TypeError. Neither
            # behaviour is wanted here -- this is meant to be a transient
            # derived tensor, never a persistent registered Parameter -- so
            # bypass nn.Module.__setattr__ entirely via the instance dict,
            # after clearing any such accidental registration.
            self.module._parameters.pop(name, None)
            self.module.__dict__[name] = w
        if isinstance(self.module, nn.RNNBase):
            self.module._flat_weights = [
                getattr(self.module, wn, None) for wn in self.module._flat_weights_names
            ]

    def forward(self, *args):
        self._set_weights()
        return self.module.forward(*args)


class AWDLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        rnn_units: int = 256,
        nlayers: int = 2,
        dropout: float = 0.4,
        dropouth: float = 0.25,
        dropouti: float = 0.4,
        dropoute: float = 0.1,
        wdrop: float = 0.5,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouth = dropouth
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.lockdrop = LockedDropout()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        rnns = []
        for layer in range(nlayers):
            in_size = embedding_dim if layer == 0 else rnn_units
            out_size = rnn_units if layer != nlayers - 1 else (embedding_dim if tie_weights else rnn_units)
            rnn = nn.LSTM(in_size, out_size, num_layers=1)
            if wdrop:
                rnn = WeightDrop(rnn, ["weight_hh_l0"], dropout=wdrop)
            rnns.append(rnn)
        self.rnns = nn.ModuleList(rnns)

        self.head = nn.Linear(embedding_dim if tie_weights else rnn_units, vocab_size)
        if tie_weights:
            self.head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.head.bias)
        if not self.tie_weights:
            nn.init.uniform_(self.head.weight, -initrange, initrange)

    def _layer_hidden_size(self, layer: int) -> int:
        return self.rnn_units if layer != self.nlayers - 1 else (self.embedding_dim if self.tie_weights else self.rnn_units)

    def init_hidden(self, batch_size: int, device: torch.device) -> Hidden:
        return [
            (
                torch.zeros(1, batch_size, self._layer_hidden_size(l), device=device),
                torch.zeros(1, batch_size, self._layer_hidden_size(l), device=device),
            )
            for l in range(self.nlayers)
        ]

    def forward(self, x: torch.Tensor, hidden: Optional[Hidden] = None) -> Tuple[torch.Tensor, Hidden]:
        # x: (batch, seq) -> internally (seq, batch) to match LockedDropout's
        # masking convention (mask shared across dim 0 = time).
        x = x.transpose(0, 1)
        seq_len, batch_size = x.shape
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        emb = embedded_dropout(self.embedding, x, dropout=self.dropoute if self.training else 0.0)
        emb = self.lockdrop(emb, self.dropouti)

        new_hidden: Hidden = []
        raw_output = emb
        for l, rnn in enumerate(self.rnns):
            raw_output, h = rnn(raw_output, hidden[l])
            new_hidden.append(h)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
        output = self.lockdrop(raw_output, self.dropout)

        logits = self.head(output)  # (seq, batch, vocab)
        logits = logits.transpose(0, 1)  # (batch, seq, vocab)
        return logits, new_hidden
