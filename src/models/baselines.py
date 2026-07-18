"""Non-learning baselines. The floor every model must clear."""
from __future__ import annotations
import numpy as np
import pandas as pd

SLOTS_PER_DAY = 288  # 5-min resolution


def copy_last_value(x_norm: np.ndarray, scaler, horizon: int = 12) -> np.ndarray:
    """Repeat the last observed speed for all future steps.

    x_norm: (S, 12, N, 2) normalized -> returns (S, 12, N) in mph.
    """
    last = scaler.inverse_transform(x_norm[:, -1, :, 0])   # (S, N) mph
    return np.repeat(last[:, None, :], horizon, axis=1)


def _key(times_ns: np.ndarray) -> np.ndarray:
    """(dayofweek, 5-min slot) -> flat key in [0, 2016)."""
    idx = pd.DatetimeIndex(times_ns.ravel().astype('datetime64[ns]'))
    k = idx.dayofweek.values * SLOTS_PER_DAY + idx.hour.values * 12 \
        + idx.minute.values // 5
    return k.reshape(times_ns.shape)


def fit_historical_average(df: pd.DataFrame, train_end: pd.Timestamp) -> np.ndarray:
    """Mean speed per (dayofweek, timeslot, sensor), TRAIN PERIOD ONLY.

    Missing readings (0.0) are excluded from the mean -- not averaged in.
    Returns (2016, 207).
    """
    hist = df[df.index <= train_end]
    vals = hist.values
    mask = (vals != 0).astype(np.float32)

    k = (hist.index.dayofweek.values * SLOTS_PER_DAY
         + hist.index.hour.values * 12 + hist.index.minute.values // 5)

    n_nodes = vals.shape[1]
    sums = np.zeros((SLOTS_PER_DAY * 7, n_nodes), np.float64)
    cnts = np.zeros((SLOTS_PER_DAY * 7, n_nodes), np.float64)
    np.add.at(sums, k, vals * mask)
    np.add.at(cnts, k, mask)

    table = sums / np.maximum(cnts, 1)
    # fallback for empty cells: that sensor's overall train mean
    per_sensor = (vals * mask).sum(0) / np.maximum(mask.sum(0), 1)
    empty = cnts == 0
    table[empty] = np.broadcast_to(per_sensor, table.shape)[empty]
    return table.astype(np.float32)


def historical_average(table: np.ndarray, y_time_ns: np.ndarray) -> np.ndarray:
    """table: (2016, N); y_time_ns: (S, 12) -> (S, 12, N) mph."""
    return table[_key(y_time_ns)]
