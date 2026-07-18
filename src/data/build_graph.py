"""Build the METR-LA adjacency from road-network distances."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_adjacency(distances_csv: str, sensor_ids_txt: str,
                    k: float = 0.1) -> np.ndarray:
    """Thresholded Gaussian kernel on road distances.

    Returns (207, 207) float32. Directed/asymmetric. Row order matches
    sensor_ids_txt -- which MUST match the speed-column order.
    """
    sensor_ids = open(sensor_ids_txt).read().strip().split(',')
    df = pd.read_csv(distances_csv, dtype={'from': str, 'to': str})

    n = len(sensor_ids)
    idx = {sid: i for i, sid in enumerate(sensor_ids)}

    dist = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    for f, t, c in df.values:
        if f in idx and t in idx:
            dist[idx[f], idx[t]] = c

    sigma = dist[~np.isinf(dist)].std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < k] = 0.0
    return adj.astype(np.float32)
