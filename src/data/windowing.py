from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from src.data.load import MetrLADataset, load_config, load_metrla

logger = logging.getLogger(__name__)


def time_features(
    timestamps: pd.DatetimeIndex, add_day_of_week: bool
) -> np.ndarray:
    """Return (T, F_time) time features: time-of-day, optionally day-of-week.

    time-of-day = seconds since midnight / 86400, in [0, 1).
    day-of-week = weekday index / 7, in [0, 1).
    """
    seconds = (
        timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second
    ).to_numpy(dtype=np.float32)
    tod = seconds / 86400.0
    feats = [tod]
    if add_day_of_week:
        feats.append(timestamps.dayofweek.to_numpy(dtype=np.float32) / 7.0)
    return np.stack(feats, axis=-1)


def _make_split_windows(
    speed_norm: np.ndarray,
    time_feat: np.ndarray,
    mask: np.ndarray,
    times_ns: np.ndarray,
    in_steps: int,
    out_steps: int,
) -> dict[str, np.ndarray]:
    """Build sliding windows over one split. All inputs are that split's slice."""
    L, N = speed_norm.shape
    num = L - (in_steps + out_steps) + 1
    if num <= 0:
        raise ValueError(f"split too short ({L}) for {in_steps}+{out_steps} windows")

    feat = np.concatenate(
        [speed_norm[:, :, None], np.broadcast_to(time_feat[:, None, :], (L, N, time_feat.shape[-1]))],
        axis=-1,
    ).astype(np.float32)

    base = np.arange(num)[:, None]
    idx_in = base + np.arange(in_steps)[None, :]
    idx_out = base + (in_steps + np.arange(out_steps)[None, :])

    x = feat[idx_in]
    y = speed_norm[idx_out]
    y_mask = mask[idx_out]
    y_time = times_ns[idx_out]

    assert x.shape == (num, in_steps, N, feat.shape[-1]), x.shape
    assert y.shape == (num, out_steps, N), y.shape
    assert y_mask.shape == y.shape and y_time.shape == (num, out_steps)
    return {"x": x, "y": y.astype(np.float32), "y_mask": y_mask, "y_time": y_time}


def build_windows(config: dict, ds: MetrLADataset | None = None) -> None:
    """Build per-split windows from METR-LA and save them to data/processed/.

    Writes {split}.npz for each split plus metadata.json (scaler, sensor order,
    feature names, horizons). Asserts the shape convention and horizon mapping.
    """
    if ds is None:
        ds = load_metrla(config)

    wcfg = config["windowing"]
    in_steps = wcfg["input_steps"]
    out_steps = wcfg["output_steps"]
    freq = config["data"]["freq_minutes"]

    for h in wcfg["report_horizons"]:
        assert 1 <= h <= out_steps, f"horizon step {h} outside 1..{out_steps}"
    logger.info(
        "Horizons: %s",
        {h: f"{h * freq} min" for h in wcfg["report_horizons"]},
    )

    tfeat = time_features(ds.timestamps, wcfg["add_day_of_week"])
    times_ns = ds.timestamps.to_numpy(dtype="datetime64[ns]").view("int64")

    out_dir = config["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    feature_names = ["speed_norm", "time_of_day"] + (
        ["day_of_week"] if wcfg["add_day_of_week"] else []
    )
    meta = {
        "mean": ds.scaler.mean,
        "std": ds.scaler.std,
        "sensor_ids": ds.sensor_ids,
        "feature_names": feature_names,
        "in_steps": in_steps,
        "out_steps": out_steps,
        "freq_minutes": freq,
        "report_horizons": {str(h): h * freq for h in wcfg["report_horizons"]},
        "splits": {},
    }

    for name, sl in ds.splits.items():
        w = _make_split_windows(
            ds.speed_norm[sl], tfeat[sl], ds.mask[sl], times_ns[sl], in_steps, out_steps
        )
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(path, **w)
        ts = pd.to_datetime(w["y_time"][[0, -1], 0])
        meta["splits"][name] = {
            "num_samples": int(w["x"].shape[0]),
            "first_target": str(ts[0]),
            "last_target": str(ts[1]),
        }
        logger.info(
            "%-5s x=%s y=%s  valid targets=%.2f%%  -> %s",
            name, w["x"].shape, w["y"].shape, 100.0 * w["y_mask"].mean(), path,
        )

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata.json (F=%d features: %s)", len(feature_names), feature_names)


def load_windows(processed_dir: str, split: str) -> dict[str, np.ndarray]:
    """Load one split's windowed arrays: x, y, y_mask, y_time."""
    data = np.load(os.path.join(processed_dir, f"{split}.npz"))
    return {k: data[k] for k in data.files}


def load_metadata(processed_dir: str) -> dict:
    """Load metadata.json written by build_windows."""
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    build_windows(cfg)
