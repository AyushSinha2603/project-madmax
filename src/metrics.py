from __future__ import annotations

import numpy as np


def _valid_mask(
    target: np.ndarray, null_val: float, mask: np.ndarray | None
) -> np.ndarray:
    """Boolean mask of genuine readings; from target != null_val unless given."""
    if mask is None:
        return target != null_val
    return mask.astype(bool)


def masked_mae(
    preds: np.ndarray,
    target: np.ndarray,
    null_val: float = 0.0,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute error over valid entries, in mph. NaN if none valid."""
    m = _valid_mask(target, null_val, mask)
    if not m.any():
        return float("nan")
    return float(np.abs(preds[m] - target[m]).mean())


def masked_rmse(
    preds: np.ndarray,
    target: np.ndarray,
    null_val: float = 0.0,
    mask: np.ndarray | None = None,
) -> float:
    """Root mean squared error over valid entries, in mph. NaN if none valid."""
    m = _valid_mask(target, null_val, mask)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.square(preds[m] - target[m]).mean()))


def masked_mape(
    preds: np.ndarray,
    target: np.ndarray,
    null_val: float = 0.0,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute percentage error over valid entries, in percent.

    Denominator is |target|; valid entries have target != null_val so no
    division by zero. NaN if none valid.
    """
    m = _valid_mask(target, null_val, mask)
    if not m.any():
        return float("nan")
    return float((np.abs(preds[m] - target[m]) / np.abs(target[m])).mean() * 100.0)


def all_metrics(
    preds: np.ndarray,
    target: np.ndarray,
    null_val: float = 0.0,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return {'mae', 'rmse', 'mape'} for one prediction/target pair."""
    return {
        "mae": masked_mae(preds, target, null_val, mask),
        "rmse": masked_rmse(preds, target, null_val, mask),
        "mape": masked_mape(preds, target, null_val, mask),
    }
