"""Load METR-LA speeds, mask missing readings, split chronologically, z-score.

Enforces four correctness rules (CLAUDE.md 2.1-2.3, 6.1): mask zeros as missing,
split in time order, z-score from train stats only, align columns to the canonical
sensor order. See PROGRESS.md for the reasoning behind each.

Units: speeds in mph. Shapes annotated as (T, N): T timesteps, N sensors.
Reads the .h5 directly via PyTables (the local pandas 3.0 read_hdf regression).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tables
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Read config.yaml into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_speed_dataframe(h5_path: str) -> pd.DataFrame:
    """Read metr-la.h5 into a (T, N) DataFrame of speeds in mph.

    Index is a 5-minute DatetimeIndex; columns are sensor ids as strings.
    Reads the pandas "fixed" layout directly via PyTables.
    """
    with tables.open_file(h5_path, "r") as h5:
        columns = h5.root.df.axis0.read()
        index = h5.root.df.axis1.read()
        values = h5.root.df.block0_values.read()

    columns = [c.decode() if isinstance(c, bytes) else str(c) for c in columns]
    return pd.DataFrame(
        values.astype(np.float32),
        index=pd.to_datetime(index),
        columns=columns,
    )


def load_sensor_ids(txt_path: str) -> list[str]:
    """Read the canonical 207 sensor ids, in order, from graph_sensor_ids.txt.

    This ordering is the single source of truth aligning adjacency rows and
    speed columns.
    """
    with open(txt_path, "r") as f:
        ids = f.read().strip().split(",")
    return [s.strip() for s in ids if s.strip()]


@dataclass
class ZScoreScaler:
    """Z-score scaler fit on the train split only.

    mean/std are over all training values (zeros included), matching DCRNN so
    results stay comparable to the reference table. Metrics are reported in mph
    after inverse_transform; the zero-mask governs loss/metrics separately.
    """

    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        """mph -> standardized."""
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """standardized -> mph."""
        return x * self.std + self.mean


def chronological_split(
    n: int, train_frac: float, val_frac: float
) -> dict[str, slice]:
    """Return contiguous, in-order train/val/test slices tiling [0, n)."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }


@dataclass
class MetrLADataset:
    """Bundle consumed by windowing, baselines, and models.

    sensor_ids : list[str]                canonical order (graph_sensor_ids.txt)
    timestamps : pd.DatetimeIndex  (T,)   5-minute resolution
    speed      : np.ndarray (T, N) f32    raw speeds in mph (0.0 == missing)
    mask       : np.ndarray (T, N) bool   True == genuine reading
    speed_norm : np.ndarray (T, N) f32    z-scored with train stats
    splits     : dict[str, slice]         train/val/test time ranges
    scaler     : ZScoreScaler             fit on the train split
    """

    sensor_ids: list[str]
    timestamps: pd.DatetimeIndex
    speed: np.ndarray
    mask: np.ndarray
    speed_norm: np.ndarray
    splits: dict[str, slice]
    scaler: ZScoreScaler


def load_metrla(config: dict) -> MetrLADataset:
    """Load, align, mask, split, and normalize METR-LA end to end.

    Emits INFO logs for the missing-data count and train/val/test time ranges.
    """
    paths = config["paths"]
    dcfg = config["data"]

    df = load_speed_dataframe(paths["speeds_h5"])
    sensor_ids = load_sensor_ids(paths["sensor_ids_txt"])

    if list(df.columns) != sensor_ids:
        missing = set(sensor_ids) - set(df.columns)
        assert not missing, f"{len(missing)} sensor ids missing from h5: {list(missing)[:5]}"
        logger.warning("Speed columns not in canonical order; reindexing to align.")
        df = df[sensor_ids]
    else:
        logger.info("Speed columns already match graph_sensor_ids.txt order.")

    speed = df.values.astype(np.float32)
    timestamps = df.index
    n_t, n_s = speed.shape

    assert n_s == dcfg["n_sensors"], f"expected {dcfg['n_sensors']} sensors, got {n_s}"
    assert n_t == dcfg["n_timesteps"], f"expected {dcfg['n_timesteps']} steps, got {n_t}"
    assert list(df.columns) == sensor_ids, "column alignment failed after reindex"

    missing_value = dcfg["missing_value"]
    mask = speed != missing_value
    n_missing = int((~mask).sum())
    logger.info(
        "Missing readings: %d of %d (%.2f%%) stored as %.1f",
        n_missing, speed.size, 100.0 * n_missing / speed.size, missing_value,
    )

    split_cfg = dcfg["split"]
    splits = chronological_split(n_t, split_cfg["train"], split_cfg["val"])
    for name, sl in splits.items():
        ts = timestamps[sl]
        logger.info("Split %-5s: %5d steps  %s -> %s",
                    name, sl.stop - sl.start, ts[0], ts[-1])

    train_speed = speed[splits["train"]]
    scaler = ZScoreScaler(mean=float(train_speed.mean()), std=float(train_speed.std()))
    logger.info("Train z-score stats (all values, DCRNN-style): mean=%.4f std=%.4f",
                scaler.mean, scaler.std)
    speed_norm = scaler.transform(speed).astype(np.float32)

    return MetrLADataset(
        sensor_ids=sensor_ids,
        timestamps=timestamps,
        speed=speed,
        mask=mask,
        speed_norm=speed_norm,
        splits=splits,
        scaler=scaler,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    ds = load_metrla(cfg)
    print("\n=== METR-LA loaded ===")
    print(f"speed      : {ds.speed.shape} {ds.speed.dtype}  (mph)")
    print(f"mask       : {ds.mask.shape} {ds.mask.dtype}  ({ds.mask.mean():.2%} valid)")
    print(f"speed_norm : {ds.speed_norm.shape} {ds.speed_norm.dtype}")
    print(f"scaler     : mean={ds.scaler.mean:.4f} std={ds.scaler.std:.4f}")
    tr = ds.speed_norm[ds.splits["train"]]
    print(f"norm train : mean={tr.mean():.4f} std={tr.std():.4f}  (expect ~0, ~1)")
