"""STGNN = LSTMBaseline + ONE diffusion-convolution step. Nothing else changes.

The point of this project is to isolate the effect of the graph. So the GRU
encoder, hidden size, head, and horizon are IDENTICAL to LSTMBaseline. The only
difference in the forward pass is a single spatial-mixing step over the node
dimension. If anything else differed, the LSTM-vs-STGNN gap would measure
"architecture", not "graph", and the study would be worthless.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_supports(adj: np.ndarray, mode: str = 'road') -> list[torch.Tensor]:
    """Transition matrices for diffusion convolution.

    mode='road'     -> [P_forward, P_backward] from the real road adjacency
    mode='identity' -> [I, I], the PARAMETER-MATCHED control: same architecture,
                       same param count, zero spatial mixing.

    Both modes return 2 supports so param counts match exactly.
    """
    A = torch.from_numpy(np.asarray(adj)).float()
    n = A.shape[0]
    if mode == 'identity':
        I = torch.eye(n)
        return [I, I]
    if mode != 'road':
        raise ValueError(f'unknown mode {mode}')
    p_fwd = A / A.sum(1, keepdim=True).clamp(min=1e-8)
    p_bwd = A.T / A.T.sum(1, keepdim=True).clamp(min=1e-8)
    return [p_fwd, p_bwd]


class STGNN(nn.Module):
    """In:  (B, 12, 207, 2)   Out: (B, 12, 207)"""

    def __init__(self, supports: list[torch.Tensor], in_dim=2, hidden=64,
                 layers=2, horizon=12, dropout=0.1, k_hops=2):
        super().__init__()
        self.horizon = horizon
        self.k_hops = k_hops
        self.register_buffer('supports', torch.stack(supports))  # (S, N, N)

        # --- IDENTICAL to LSTMBaseline ---
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        # --- the only addition ---
        n_blocks = 1 + len(supports) * k_hops
        self.gconv = nn.Linear(hidden * n_blocks, hidden)
        self.drop = nn.Dropout(dropout)
        # --- IDENTICAL to LSTMBaseline ---
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f'expected (B,T,N,F), got {tuple(x.shape)}'
        b, t, n, f = x.shape

        # ---- identical to LSTMBaseline up to here ----
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        out, _ = self.gru(z)
        h = out[:, -1, :].reshape(b, n, -1)              # (B, N, H)

        # ---- THE ONLY DIFFERENCE: spatial mixing over the graph ----
        blocks = [h]
        for p in self.supports:
            xk = h
            for _ in range(self.k_hops):
                xk = torch.einsum('nm,bmd->bnd', p, xk)
                blocks.append(xk)
        g = self.gconv(torch.cat(blocks, dim=-1))        # (B, N, H)
        h = h + self.drop(F.relu(g))   # residual: the model CAN ignore the graph

        # ---- identical to LSTMBaseline from here ----
        y = self.head(h)                                 # (B, N, horizon)
        y = y.permute(0, 2, 1)                           # (B, horizon, N)
        assert y.shape == (b, self.horizon, n)
        return y


class AdaptiveSTGNN(STGNN):
    """Optional experiment: learn the adjacency instead of using road distances.

    If this beats STGNN-road, the hand-built road graph is not the best graph --
    an interesting result about the premise of traffic GNNs.
    """

    def __init__(self, n_nodes: int, emb_dim: int = 10, **kw):
        supports = [torch.eye(n_nodes), torch.eye(n_nodes)]  # placeholders
        super().__init__(supports, **kw)
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def forward(self, x):
        a = F.softmax(F.relu(self.e1 @ self.e2.T), dim=1)
        self.supports = torch.stack([a, a.T])
        return super().forward(x)
