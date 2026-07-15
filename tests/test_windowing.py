"""Tests for the 12-in/12-out windowing (CLAUDE.md 6.2).

Pins the off-by-one alignment (target = the steps immediately after the input),
the shape convention, and the time-of-day feature, on tiny synthetic arrays so
the check is fast and file-independent.
"""

import numpy as np
import pandas as pd

from src.data.windowing import _make_split_windows, time_features


def test_window_alignment_and_shapes():
    L, N = 5, 1
    speed = np.arange(L, dtype=np.float32)[:, None]
    tfeat = (np.arange(L, dtype=np.float32) / 10.0)[:, None]
    mask = np.ones((L, N), dtype=bool)
    times = np.arange(L, dtype=np.int64)

    w = _make_split_windows(speed, tfeat, mask, times, in_steps=2, out_steps=2)

    assert w["x"].shape == (2, 2, 1, 2)
    assert w["y"].shape == (2, 2, 1)

    np.testing.assert_array_equal(w["x"][0, :, 0, 0], [0.0, 1.0])
    np.testing.assert_allclose(w["x"][0, :, 0, 1], [0.0, 0.1], atol=1e-6)
    np.testing.assert_array_equal(w["y"][0, :, 0], [2.0, 3.0])
    np.testing.assert_array_equal(w["y_time"][0], [2, 3])

    np.testing.assert_array_equal(w["y"][1, :, 0], [3.0, 4.0])
    np.testing.assert_array_equal(w["y_time"][1], [3, 4])


def test_target_immediately_follows_input():
    L, N = 30, 3
    speed = np.random.RandomState(0).randn(L, N).astype(np.float32)
    tfeat = np.zeros((L, 1), dtype=np.float32)
    mask = np.ones((L, N), dtype=bool)
    times = np.arange(L, dtype=np.int64)

    w = _make_split_windows(speed, tfeat, mask, times, in_steps=12, out_steps=12)
    for i in range(w["x"].shape[0]):
        np.testing.assert_array_equal(w["x"][i, :, :, 0], speed[i : i + 12])
        np.testing.assert_array_equal(w["y"][i], speed[i + 12 : i + 24])


def test_time_of_day_feature():
    idx = pd.to_datetime(
        ["2012-03-01 00:00", "2012-03-01 06:00", "2012-03-01 12:00", "2012-03-01 18:00"]
    )
    tf = time_features(idx, add_day_of_week=False)
    assert tf.shape == (4, 1)
    np.testing.assert_allclose(tf[:, 0], [0.0, 0.25, 0.5, 0.75], atol=1e-6)


def test_day_of_week_optional():
    idx = pd.to_datetime(["2012-03-01 00:00"])
    assert time_features(idx, add_day_of_week=False).shape == (1, 1)
    assert time_features(idx, add_day_of_week=True).shape == (1, 2)
