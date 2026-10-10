"""
Tests for the scores.
"""

import numpy as np
import pytest
from upordowntrader.backtest import metrics


def test_a_coin_flip_scores_log_2_and_a_quarter():
    up = np.array([True, False, True, False])
    assert metrics.log_loss(np.full(4, 0.5), up) == pytest.approx(np.log(2))
    assert metrics.brier(np.full(4, 0.5), up) == pytest.approx(0.25)


def test_a_confident_miss_costs_far_more_than_a_confident_hit_earns():
    assert metrics.log_loss(np.array([0.99]), np.array([False])) > 100 * metrics.log_loss(np.array([0.99]), np.array([True]))


def test_favorites_fold_up_and_down_together():
    p = np.array([0.92, 0.08, 0.93, 0.07, 0.6])
    up = np.array([True, False, False, False, True])
    rows = metrics.favorites(p, up, windows=np.array([0, 1, 2, 2, 3]))
    nineties = next(row for row in rows if row["low"] == 0.9)
    assert (nineties["evaluations"], nineties["windows"]) == (4, 3)
    assert nineties["won"] == pytest.approx(3 / 4) and nineties["said"] == pytest.approx(0.925)
