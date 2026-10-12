"""
Tests for the fair price as traded: its chances, the fee and edge, and the recalibration and blend fitted on settled marks.
"""

import numpy as np
import pytest
from upordowntrader.model import fair, pricing


def settled_marks(scale, per_minute=3000, seed=0):
    """
    Made up settled marks at every whole minute left whose outcomes come true as a Student t of z / scale says, and a
    market that prices them exactly so.
    """
    rng = np.random.default_rng(seed)
    seconds_left = np.repeat(60 * np.arange(1, fair.MINUTES + 1), per_minute)
    z = rng.normal(0, 1.5, len(seconds_left))
    p = pricing.up_student_t(z / scale, 8)
    return z, p, rng.random(len(z)) < p, seconds_left


def test_a_moment_reads_the_next_marks_values_and_the_averaging_minute_the_last_ones():
    assert fair.minutes_left([0.5, 30, 60, 61, 840, 899]).tolist() == [1, 1, 1, 2, 14, 14]


def test_without_a_fit_the_chance_is_the_models_own_student_t():
    z = np.array([-1.0, 0.0, 2.0])
    assert fair.chance(z, [100, 400, 800]).tolist() == pricing.up_student_t(z, fair.DOF).tolist()


def test_the_fee_and_edge_are_kalshis():
    assert fair.fee(0.5) == pytest.approx(0.0175)
    assert fair.edge(0.70, 0.60) == pytest.approx(0.70 - 0.60 - 0.07 * 0.6 * 0.4)


def test_a_fit_finds_the_spread_the_outcomes_were_drawn_with():
    z, p, up, seconds_left = settled_marks(scale=1.3)
    fit = fair.fit(z, p, up, seconds_left)
    assert abs(np.median(fit.scale) - 1.3) <= 0.1                   # Each minute's, from its 3,000 marks, within a few steps.
    recalibrated = fair.chance(z, seconds_left, fit)
    assert np.abs(recalibrated - p).mean() < np.abs(fair.chance(z, seconds_left) - p).mean()


def test_the_blend_trusts_a_market_that_knows_and_not_a_model_that_does_not():
    z, p, up, seconds_left = settled_marks(scale=1.3)
    unknowing = np.random.default_rng(1).normal(0, 1.5, len(z))
    fit = fair.fit(unknowing, p, up, seconds_left)
    assert np.all(np.abs(fit.blend[:, 1] - 1) < 0.25) and np.all(np.abs(fit.blend[:, 2]) < 0.25)


def test_a_minute_with_too_few_marks_keeps_the_model_and_the_markets_price():
    z, p, up, seconds_left = settled_marks(scale=1.3, per_minute=100)
    fit = fair.fit(z, p, up, seconds_left)
    assert fit.scale.tolist() == [1.0] * fair.MINUTES and fit.dof.tolist() == [fair.DOF] * fair.MINUTES
    market = np.array([0.2, 0.5, 0.9])
    assert fair.blended(market, np.array([0.9, 0.1, 0.5]), [100, 400, 800], fit) == pytest.approx(market)


def test_the_tracking_variance_is_the_mean_square_of_the_errors_of_the_windows_settled_before():
    opens = 900.0 * np.arange(7)
    errors = 1e-5 * np.array([1, -1, 2, np.nan, -2, 1, 1])
    variances = fair.tracking_variances(errors, opens, opens + 900, windows=4, least=2)
    # Each window opens as the one before closes, so that one is left out: window 3 has 0 and 1, window 6 the last 4, 1 to 4.
    assert variances == pytest.approx(1e-10 * np.array([0, 0, 0, 1, 2, 2, 3]))
