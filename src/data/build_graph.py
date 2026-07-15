"""Build the road-network adjacency for METR-LA (CLAUDE.md 5.1).

DCRNN's thresholded Gaussian kernel over road-network distances:

    A[i, j] = exp( -(dist(i, j) / sigma)^2 )   if that value >= k
    A[i, j] = 0                                 otherwise

sigma is the standard deviation of the observed distances (computed, not chosen).
k = 0.1 sparsifies. The result is directed and asymmetric because road distance
A->B differs from B->A; it is NOT symmetrized here. Row/column order follows
graph_sensor_ids.txt so the matrix aligns with the speed columns.

Distances are in metres; the adjacency is a unitless (207, 207) weight matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_adjacency(
    distances_csv: str, sensor_ids_txt: str, k: float = 0.1
) -> np.ndarray:
    """Build the (N, N) thresholded-Gaussian adjacency from road distances.

    Args:
        distances_csv: from,to,cost pairs; cost is road distance in metres.
        sensor_ids_txt: comma-separated canonical sensor order.
        k: sparsification threshold applied after the kernel.

    Returns:
        (N, N) float32 adjacency, directed/asymmetric, row order == sensor_ids.
    """
    with open(sensor_ids_txt, "r") as f:
        sensor_ids = [s.strip() for s in f.read().strip().split(",") if s.strip()]

    df = pd.read_csv(distances_csv, dtype={"from": str, "to": str})

    n = len(sensor_ids)
    idx = {sid: i for i, sid in enumerate(sensor_ids)}
    dist = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    for f_id, t_id, cost in df.values:
        if f_id in idx and t_id in idx:
            dist[idx[f_id], idx[t_id]] = cost

    finite = dist[~np.isinf(dist)]
    sigma = finite.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < k] = 0.0
    return adj


def symmetrize(adj: np.ndarray) -> np.ndarray:
    """Return (A + A.T) / 2. Use only when a model requires a symmetric graph.

    This is a modelling choice that must be logged in the report (CLAUDE.md 5.1).
    """
    return (adj + adj.T) / 2.0


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    adj = build_adjacency(
        cfg["paths"]["distances_csv"],
        cfg["paths"]["sensor_ids_txt"],
        k=cfg["graph"]["k"],
    )
    is_symmetric = np.allclose(adj, adj.T)
    print("=== adjacency built ===")
    print(f"shape       : {adj.shape} {adj.dtype}")
    print(f"symmetric?  : {is_symmetric}  (expect False)")
    print(f"diagonal[:3]: {adj.diagonal()[:3]}  (expect ~1.0)")
    print(f"value range : [{adj.min():.4f}, {adj.max():.4f}]")
    nnz = (adj > 0).mean()
    print(f"density     : {nnz:.2%} nonzero  (sparse after k={cfg['graph']['k']})")
