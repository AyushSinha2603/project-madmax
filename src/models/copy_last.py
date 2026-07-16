from __future__ import annotations

import numpy as np

from src.data.load import ZScoreScaler

_MISSING_MPH = 0.1


class CopyLastValue:
    """Repeat the last valid input speed for every horizon."""

    name = "CopyLast"

    def predict(
        self, x: np.ndarray, out_steps: int, scaler: ZScoreScaler
    ) -> np.ndarray:
        """Predict speeds in mph.

        Args:
            x: (num, in_steps, N, F) input features; feature 0 is normalized speed.
            out_steps: number of future steps to emit.
            scaler: fitted train scaler, to map normalized speed back to mph.

        Returns:
            (num, out_steps, N) predicted speeds in mph. Sensors whose entire
            input window is missing fall back to the train mean speed.
        """
        raw_in = scaler.inverse_transform(x[:, :, :, 0])
        valid = raw_in > _MISSING_MPH

        num, in_steps, n = raw_in.shape
        last = np.full((num, n), np.nan, dtype=np.float32)
        for t in range(in_steps):
            last = np.where(valid[:, t, :], raw_in[:, t, :], last)
        last = np.where(np.isnan(last), np.float32(scaler.mean), last)

        return np.repeat(last[:, None, :], out_steps, axis=1)
