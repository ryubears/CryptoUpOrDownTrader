"""
Tests for the variance measures and forecasts, on made up histories.
"""

import numpy as np
import pytest
import synthetic
from dataclasses import replace
from upordowntrader.model import volatility

SIGMA = 1e-4


@pytest.fixture(scope="module")
def history():
    return synthetic.random_walk_history(3, SIGMA)


def test_ewma_of_a_constant_is_the_constant_and_a_jump_halves_each_half_life():
    assert np.allclose(volatility.ewma(np.full(2000, 3.0), 30), 3.0)
    values = np.zeros(3000)
    values[2000] = 1.0
    averaged = volatility.ewma(values, 30)
    assert averaged[2030] == pytest.approx(averaged[2000] / 2)


def test_every_input_measures_about_the_true_variance_per_minute(history):
    true = SIGMA ** 2 * 60
    assert np.mean(volatility.measures(history, "realized")) == pytest.approx(true, rel=0.01)
    assert np.mean(volatility.measures(history, "close")) == pytest.approx(true, rel=0.08)      # One noisy return a minute.
    assert np.mean(volatility.measures(history, "close_15m")) == pytest.approx(true, rel=0.15)  # Overlapping, so noisier still.
    # The range of 60 one second prices misses the moves between them, so the range measures come out lower.
    for input in ("parkinson", "garman_klass"):
        assert 0.5 * true < np.mean(volatility.measures(history, input)) < true


def test_a_forecast_uses_nothing_after_its_minute(history):
    later = replace(history, minute_realized=history.minute_realized.copy())
    later.minute_realized[3000:] *= 10
    for weighting in volatility.WEIGHTINGS:
        before, _ = volatility.forecasts(history, "realized", 60, weighting, season=True)
        after, _ = volatility.forecasts(later, "realized", 60, weighting, season=True)
        assert np.array_equal(before[:3000], after[:3000]) and not np.allclose(before[3000:], after[3000:])


def test_the_season_finds_the_half_hour_that_moves_twice_as_much():
    days = 10
    values = np.ones(days * volatility.MINUTES)
    for day in range(days):
        values[day * volatility.MINUTES + 600:day * volatility.MINUTES + 630] = 2.0      # 10:00 to 10:30 every day.
    shares = volatility.season_shares(values, np.zeros(days, dtype=bool)).reshape(days, -1)
    assert np.all(shares[:volatility.SEASON_MIN_DAYS] == 1)                               # Not enough history yet.
    later = shares[volatility.SEASON_MIN_DAYS:]
    assert np.allclose(later[:, 615], 2 * 48 / 49) and np.allclose(later[:, 0], 48 / 49)


def test_volume_weighting_leans_toward_the_busy_minutes(history):
    busy = replace(history, minute_volume=np.where(np.arange(len(history.minute_volume)) % 2 == 0, 9.0, 1.0),
                   minute_realized=np.where(np.arange(len(history.minute_volume)) % 2 == 0, 2.0, 1.0) * SIGMA ** 2 * 60)
    time_weighted, _ = volatility.forecasts(busy, "realized", 600, "time", season=False)
    volume_weighted, _ = volatility.forecasts(busy, "realized", 600, "volume", season=False)
    assert time_weighted[-1] * 60 == pytest.approx(1.5 * SIGMA ** 2 * 60, rel=0.01)
    assert volume_weighted[-1] * 60 == pytest.approx(1.9 * SIGMA ** 2 * 60, rel=0.01)    # (9 * 2 + 1 * 1) / 10.
