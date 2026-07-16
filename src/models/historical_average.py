from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalAverage:
    """Per-(day-of-week, time-of-day slot, sensor) mean speed in mph."""

    name = "HistoricalAverage"

    def __init__(self, freq_minutes: int = 5):
        self.freq_minutes = freq_minutes
        self.slots_per_day = 24 * 60 // freq_minutes
        self.table: np.ndarray | None = None

    def _bucket(self, times_ns: np.ndarray) -> np.ndarray:
        """Map ns timestamps to a flat (day-of-week * slots + slot) index."""
        idx = pd.to_datetime(times_ns.ravel())
        slot = (idx.hour * 60 + idx.minute) // self.freq_minutes
        flat = idx.dayofweek.to_numpy() * self.slots_per_day + slot.to_numpy()
        return flat.reshape(times_ns.shape)

    def fit(
        self, speed_mph: np.ndarray, mask: np.ndarray, times_ns: np.ndarray
    ) -> "HistoricalAverage":
        """Build the lookup table from the training series.

        Args:
            speed_mph: (T, N) raw training speeds in mph.
            mask: (T, N) bool, True == genuine reading (zeros excluded).
            times_ns: (T,) int64 ns timestamps of the training steps.
        """
        T, N = speed_mph.shape
        n_buckets = 7 * self.slots_per_day
        flat = self._bucket(times_ns)

        sums = np.zeros((n_buckets, N), dtype=np.float64)
        counts = np.zeros((n_buckets, N), dtype=np.float64)
        valid = mask.astype(np.float64)
        np.add.at(sums, flat, np.where(mask, speed_mph, 0.0))
        np.add.at(counts, flat, valid)

        with np.errstate(invalid="ignore", divide="ignore"):
            table = sums / counts

        empty = counts == 0
        n_empty = int(empty.sum())
        if n_empty:
            sensor_mean = np.nansum(sums, axis=0) / np.maximum(np.nansum(counts, axis=0), 1)
            table[empty] = np.broadcast_to(sensor_mean, table.shape)[empty]
            logger.info(
                "HistoricalAverage: %d/%d (dow,slot,sensor) buckets empty; "
                "filled with per-sensor mean.",
                n_empty, table.size,
            )

        self.table = table.astype(np.float32)
        return self

    def predict(self, y_time_ns: np.ndarray) -> np.ndarray:
        """Predict speeds in mph for target timestamps.

        Args:
            y_time_ns: (num, out_steps) int64 ns timestamps of target steps.

        Returns:
            (num, out_steps, N) predicted speeds in mph.
        """
        assert self.table is not None, "call fit() before predict()"
        flat = self._bucket(y_time_ns)
        return self.table[flat]
