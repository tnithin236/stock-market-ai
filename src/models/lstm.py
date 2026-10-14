"""LSTM network for next-day prediction from a window of daily features."""
import torch
from torch import nn


class LSTMNet(nn.Module):
    """LSTM -> last time step -> dropout -> linear.

    One output per sample: a scaled next-day return (regression) or a logit (classification).
    """

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                     # (batch, seq_len, hidden)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)
