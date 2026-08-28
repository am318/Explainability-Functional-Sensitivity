"""A tiny decoder-only transformer for character-level language modelling.

This is the load-bearing generality check. Functional sensitivity is defined on f(x), the
network's outputs, with no reference to a loss or a label — so it carries over to
next-token prediction unchanged. Any result that only holds for image classifiers is a
result about image classifiers.

The output dimension here is (block_size x vocab_size), far too large for the exact
estimator, so GPT runs always use the Hutchinson path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, block_size: int):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf")).softmax(dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, block_size: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, block_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 128, width: int = 192,
                 depth: int = 6, heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, width))
        self.blocks = nn.ModuleList(
            [Block(width, heads, block_size, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._init)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
