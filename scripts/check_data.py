"""Fail-loud sanity gate for the raw METR-LA data (CLAUDE.md 6.1).

Run before writing any model. Asserts the speed matrix shape, 5-minute
resolution, date range, missing rate, non-zero speed range, and the exact
column-order match against graph_sensor_ids.txt (the silent killer). Also
sanity-checks the road-distance CSV that becomes the graph.

Adjacency shape/asymmetry are verified where the graph is built
(build_graph.py / test_graph.py, Step 5), against our own builder rather than
the oracle, so they are intentionally not re-checked here.

Exits non-zero if any check fails. Read-only: writes nothing.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.load import load_config, load_sensor_ids, load_speed_dataframe

_results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    """Record a check and print a PASS/FAIL line."""
    _results.append((name, passed, detail))
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name:<28} {detail}")


def main() -> int:
    cfg = load_config()
    dcfg = cfg["data"]

    df = load_speed_dataframe(cfg["paths"]["speeds_h5"])
    sensor_ids = load_sensor_ids(cfg["paths"]["sensor_ids_txt"])
    speed = df.values
    idx = df.index

    exp_shape = (dcfg["n_timesteps"], dcfg["n_sensors"])
    check("speed shape", speed.shape == exp_shape,
          f"expected {exp_shape}, got {speed.shape}")

    diffs = idx.to_series().diff().dropna().unique()
    only_5min = len(diffs) == 1 and diffs[0] == pd.Timedelta(minutes=dcfg["freq_minutes"])
    check("time resolution", only_5min,
          f"expected uniform {dcfg['freq_minutes']}min, got {[str(d) for d in diffs]}")

    start, end = idx[0], idx[-1]
    date_ok = (start == pd.Timestamp("2012-03-01 00:00:00")
               and pd.Timestamp("2012-06-01") <= end <= pd.Timestamp("2012-07-01"))
    check("date range", date_ok, f"{start} -> {end}")

    missing_rate = float((speed == dcfg["missing_value"]).mean())
    check("missing rate ~8%", 0.05 <= missing_rate <= 0.12,
          f"expected 5-12%, got {missing_rate:.2%}")

    nz = speed[speed != dcfg["missing_value"]]
    speed_ok = nz.min() > 0.0 and nz.max() <= 75.0
    check("non-zero speed range", speed_ok,
          f"expected (0, 75] mph, got [{nz.min():.1f}, {nz.max():.1f}]")

    order_ok = list(df.columns) == sensor_ids
    n_shared = len(set(df.columns) & set(sensor_ids))
    check("column order == sensor_ids", order_ok,
          f"exact-order match; {n_shared}/{len(sensor_ids)} ids shared")

    dist = pd.read_csv(cfg["paths"]["distances_csv"], dtype={"from": str, "to": str})
    cols_ok = list(dist.columns[:3]) == ["from", "to", "cost"]
    costs = dist["cost"].to_numpy(dtype=float) if cols_ok else np.array([np.nan])
    cost_ok = cols_ok and np.isfinite(costs).all() and (costs >= 0).all()
    check("distances csv", cost_ok,
          f"cols={list(dist.columns[:3])}, rows={len(dist)}, "
          f"cost=[{np.nanmin(costs):.1f}, {np.nanmax(costs):.1f}]")

    ids_set = set(sensor_ids)
    referenced = set(dist["from"]) | set(dist["to"])
    coverage = len(ids_set & referenced)
    check("sensor ids in distances", coverage == len(ids_set),
          f"{coverage}/{len(ids_set)} sensor ids referenced in distances csv")

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print("-" * 60)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
