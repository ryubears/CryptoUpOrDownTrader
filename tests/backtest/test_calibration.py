"""
Tests for the calibration backtest, on a made up random walk whose variance is known, so the right model is too.
"""

import numpy as np
import pytest
import synthetic
from dataclasses import replace
from upordowntrader.backtest import calibration, metrics

SIGMA = 1e-4
DAYS, WARMUP, TEST_FROM = 8, 2, 6


@pytest.fixture(scope="module")
def history():
    return synthetic.random_walk_history(DAYS, SIGMA, seed=7)


@pytest.fixture(scope="module")
def points(history):
    return calibration.evaluation_points(history, 900, 60, WARMUP, TEST_FROM * calibration.DAY)


def test_windows_are_15_minutes_on_the_quarter_hour_after_the_warmup(history, points):
    starts = np.unique(points.window_start)
    assert len(starts) == (DAYS - WARMUP) * 96 and starts[0] == WARMUP * calibration.DAY and np.all(starts % 900 == 0)
    assert points.offsets.tolist() == [*range(60, 841, 60), 870, 885, 895]
    assert points.test.mean() == pytest.approx((DAYS - TEST_FROM) / (DAYS - WARMUP))


def test_each_evaluation_knows_only_the_strike_the_price_and_the_readings_taken(history, points):
    i = np.flatnonzero((points.window == 5) & (points.seconds_left == 15))[0]
    start = int(points.window_start[i])
    seconds = history.second_close
    assert points.strike[i] == pytest.approx(seconds[start - 60:start].mean())
    assert points.settlement[i] == pytest.approx(seconds[start + 840:start + 900].mean())
    assert points.price[i] == seconds[start + 884]                          # The last whole second's close.
    assert points.observed_sum[i] == pytest.approx(seconds[start + 840:start + 885].sum())
    assert points.minute[i] == (start + 885) // 60 - 1


def test_a_window_missing_an_averaging_second_or_more_than_a_minute_is_skipped(history):
    gap = history.minute_missing.copy()
    gap[WARMUP * 1440 + 14] = 1         # The first window's settlement minute, and so the second's strike minute.
    gap[WARMUP * 1440 + 37] = 61        # More than a minute missing in the middle of the third window.
    points = calibration.evaluation_points(replace(history, minute_missing=gap), 900, 60, WARMUP, TEST_FROM * calibration.DAY)
    assert points.skipped == 3 and points.window_start.min() == WARMUP * calibration.DAY + 3 * 900


def test_the_true_model_is_calibrated_on_a_random_walk(history, points):
    results = calibration.calibrate(history, points, ["realized", "close"], half_lives=(1440,), weightings=("time",), seasons=(False,),
                                    distributions=("normal", "empirical"), log=lambda message: None)
    best = results[("realized", 1440, "time", False, "normal")]
    assert best["test_spread"] == pytest.approx(1, abs=0.06)               # The forecast variance is right.
    assert best["test_log_loss"] < 0.55                                    # Far better than a coin flip's 0.693.
    p, _ = calibration.evaluate(history, points, "realized", 1440, "time", False, "normal")
    for row in metrics.favorites(p[points.test], points.up[points.test], points.window[points.test]):
        assert abs(row["won"] - row["said"]) < 4 * row["error"] + 0.01


def test_the_empirical_distribution_takes_its_residuals_from_other_windows(points):
    train = ~points.test
    first = train & (points.window_start < np.median(points.window_start[train]))
    residuals = np.select([first, train], [5.0, -5.0], 0.0)    # Each part of the history marked by its residuals.
    p = calibration.chances("empirical", np.zeros(len(points.up)), residuals, points)
    assert np.all(p[first] < 0.01)                      # From the second half of the train period, all below 0.
    assert np.all(p[train & ~first] > 0.99)             # From the first half, all above.
    assert np.allclose(p[points.test], 0.5, atol=0.01)  # From the whole train period, half and half.


def test_the_rolling_distribution_takes_its_residuals_only_from_the_days_before(points):
    day = points.window_start // calibration.DAY
    residuals = np.where(day <= 4, -5.0, 5.0)                   # The history turns on day 5.
    p = calibration.rolling_chances(np.zeros(len(points.up)), residuals, points, days=60, step=1)
    assert np.all(p[day == WARMUP] == 0.5)                      # No past yet, so the normal distribution's.
    assert np.all(p[day == 5] < 0.01)                           # Only days 2 to 4 before it, all below 0, though its own are above.
    assert np.allclose(p[day == 7], (2 * 96 + 0.5) / (5 * 96 + 1))  # Days 2 to 6 before it, two of the five above.
