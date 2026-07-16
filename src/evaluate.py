from __future__ import annotations

import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from src.data.load import ZScoreScaler
from src.metrics import masked_mae, masked_mape, masked_rmse

METRICS_COLUMNS = ["model", "split", "horizon_min", "mae", "rmse", "mape"]


def per_horizon_metrics(
    preds_mph: np.ndarray,
    y_norm: np.ndarray,
    y_mask: np.ndarray,
    scaler: ZScoreScaler,
    horizons_steps: list[int],
    freq_minutes: int,
) -> list[dict]:
    """Compute masked MAE/RMSE/MAPE at each reported horizon.

    Args:
        preds_mph: (num, out_steps, N) predictions in mph.
        y_norm: (num, out_steps, N) normalized targets; de-normalized here.
        y_mask: (num, out_steps, N) bool validity mask.
        scaler: train scaler to map targets back to mph.
        horizons_steps: 1-indexed future steps to report (e.g. [3, 6, 12]).
        freq_minutes: minutes per step (5), for the horizon->minute label.

    Returns:
        One dict per horizon with horizon_min, mae, rmse, mape.
    """
    target_mph = scaler.inverse_transform(y_norm)
    rows = []
    for h in horizons_steps:
        i = h - 1
        p, t, m = preds_mph[:, i, :], target_mph[:, i, :], y_mask[:, i, :]
        rows.append(
            {
                "horizon_min": h * freq_minutes,
                "mae": masked_mae(p, t, mask=m),
                "rmse": masked_rmse(p, t, mask=m),
                "mape": masked_mape(p, t, mask=m),
            }
        )
    return rows


def upsert_metrics(path: str, model: str, split: str, rows: list[dict]) -> pd.DataFrame:
    """Insert/replace this (model, split)'s rows in results/metrics.csv.

    Reads any existing table, drops matching (model, split) rows, appends the new
    ones, and rewrites sorted. Returns the full table.
    """
    new = pd.DataFrame(rows)
    new.insert(0, "split", split)
    new.insert(0, "model", model)
    new = new[METRICS_COLUMNS]

    if os.path.exists(path):
        old = pd.read_csv(path)
        keep = ~((old["model"] == model) & (old["split"] == split))
        table = pd.concat([old[keep], new], ignore_index=True)
    else:
        table = new

    table = table.sort_values(["split", "model", "horizon_min"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_csv(path, index=False, float_format="%.4f")
    return table


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def write_run_log(
    results_dir: str, config: dict, seed: int, metrics: dict, extra: dict | None = None
) -> str:
    """Write results/run_<timestamp>.json with config, git commit, seed, metrics."""
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{stamp}.json")
    payload = {
        "timestamp": stamp,
        "git_commit": _git_commit(),
        "seed": seed,
        "config": config,
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
