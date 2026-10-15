"""Transformer encoder for next-day prediction from a window of daily features.

Why attention: an LSTM squeezes the whole history through one hidden state, so
information from many days ago must survive every step. Self-attention lets the
last day look DIRECTLY at any earlier day in the window with learned weights,
so distant days are one hop away. Whether that helps depends on the data: if
distant days carry no signal (as in our results), attention has nothing to find.

All tokens in a window are days up to and including the prediction day, so
bidirectional attention inside the window never sees the future.
"""
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding so the model knows the ORDER of days (attention alone is order-blind)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)]


class TransformerNet(nn.Module):
    """Linear embedding -> positional encoding -> N encoder blocks -> last day -> linear head."""

    def __init__(self, input_size: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 64, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pos(self.input_proj(x)))     # (batch, seq_len, d_model)
        return self.head(self.drop(self.norm(h[:, -1, :]))).squeeze(-1)
