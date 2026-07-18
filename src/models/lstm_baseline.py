"""Per-sensor GRU. No graph. THE CONTROL CONDITION."""
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """In:  (B, 12, 207, 2) normalized speed + time-of-day
    Out: (B, 12, 207) normalized speed."""

    def __init__(self, in_dim=2, hidden=64, layers=2, horizon=12, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f'expected (B,T,N,F), got {tuple(x.shape)}'
        b, t, n, f = x.shape
        # Fold nodes into batch -- this is what makes the model graph-blind.
        x = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.gru(x)
        y = self.head(out[:, -1, :])                        # (B*N, horizon)
        y = y.reshape(b, n, self.horizon).permute(0, 2, 1)  # (B, horizon, N)
        assert y.shape == (b, self.horizon, n)
        return y
