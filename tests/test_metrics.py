"""Hand-computed tests for the masked metrics (CLAUDE.md 10).

These functions are the ruler every model is judged by, so correctness is proven
by arithmetic on a tiny example that includes missing (0.0) entries.

Fixture (targets in mph; 0.0 == missing at (0,1) and (1,2)):

    target = [[10,  0, 20],
              [30, 40,  0]]
    preds  = [[12,  5, 18],
              [33, 36, 100]]

Valid cells and abs errors:
    (0,0) 10 -> 12 : 2
    (0,2) 20 -> 18 : 2
    (1,0) 30 -> 33 : 3
    (1,1) 40 -> 36 : 4

    MAE  = (2+2+3+4)/4                 = 2.75
    RMSE = sqrt((4+4+9+16)/4)          = sqrt(8.25) = 2.8722813...
    MAPE = mean(2/10,2/20,3/30,4/40)*100 = 12.5

The masked-out cells carry a 5 mph and a 100 mph error; if masking is broken
these values wreck the results, so the test fails loudly.
"""

import math

import numpy as np
import pytest

from src.metrics import masked_mae, masked_mape, masked_rmse

TARGET = np.array([[10.0, 0.0, 20.0], [30.0, 40.0, 0.0]])
PREDS = np.array([[12.0, 5.0, 18.0], [33.0, 36.0, 100.0]])


def test_masked_mae_hand_computed():
    assert masked_mae(PREDS, TARGET) == pytest.approx(2.75)


def test_masked_rmse_hand_computed():
    assert masked_rmse(PREDS, TARGET) == pytest.approx(math.sqrt(8.25))


def test_masked_mape_hand_computed():
    assert masked_mape(PREDS, TARGET) == pytest.approx(12.5)


def test_masking_actually_excludes_zeros():
    naive_mae = np.abs(PREDS - TARGET).mean()
    assert naive_mae == pytest.approx((2 + 5 + 2 + 3 + 4 + 100) / 6)
    assert masked_mae(PREDS, TARGET) != pytest.approx(naive_mae)


def test_explicit_mask_overrides_null_val():
    mask = TARGET != 0.0
    assert masked_mae(PREDS, TARGET, mask=mask) == pytest.approx(2.75)


def test_all_valid_matches_plain_mean():
    t = np.array([[10.0, 20.0], [30.0, 40.0]])
    p = np.array([[11.0, 22.0], [27.0, 44.0]])
    assert masked_mae(p, t) == pytest.approx(np.abs(p - t).mean())


def test_no_valid_entries_returns_nan():
    t = np.zeros((2, 2))
    p = np.ones((2, 2))
    assert math.isnan(masked_mae(p, t))
    assert math.isnan(masked_rmse(p, t))
    assert math.isnan(masked_mape(p, t))
