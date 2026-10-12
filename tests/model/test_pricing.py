"""
Tests for the fair price: the settlement average's variance, checked against simulation, and the chances of Up.
"""

import numpy as np
import pytest
from upordowntrader.model import pricing

SIGMA = 1e-4    # Log return standard deviation per second.


def simulated_settlements(seconds_left, taken, paths=20_000, seed=1):
    """
    Settlement averages of a random walk starting at price 1, seconds_left before the close, with the readings already
    taken all at 1. Returns them with their expected value and the observed sum.
    """
    rng = np.random.default_rng(seed)
    steps = int(seconds_left)
    prices = np.exp(np.cumsum(rng.normal(0, SIGMA, (paths, steps)), axis=1))
    to_come = pricing.READINGS - taken
    observed_sum = float(taken)
    settlements = (observed_sum + prices[:, steps - to_come:].sum(axis=1)) / pricing.READINGS
    return settlements, pricing.expected_settlement(1.0, observed_sum, seconds_left), observed_sum


@pytest.mark.parametrize(("seconds_left", "taken"), [(300, 0), (90, 0), (60, 0), (20, 40), (5, 55)])
def test_the_averaging_variance_matches_a_simulated_random_walk(seconds_left, taken):
    settlements, expected, _ = simulated_settlements(seconds_left, taken)
    variance = np.var(np.log(settlements / expected))
    assert variance / (SIGMA ** 2 * pricing.averaging_seconds(seconds_left)) == pytest.approx(1, abs=0.04)


def test_the_averaging_counts_about_a_third_of_its_minute():
    assert pricing.averaging_seconds(60) == pytest.approx(61 * 121 / 360)
    assert pricing.averaging_seconds(900) == pytest.approx(840 + 61 * 121 / 360)
    assert pricing.averaging_seconds(1) == pytest.approx(1 / 3600)       # One reading left, a sixtieth of the average.


def test_expected_settlement_mixes_the_readings_taken_with_the_price_now():
    assert pricing.expected_settlement(100.0, 0.0, 600) == 100.0
    assert pricing.expected_settlement(100.0, 40 * 90.0, 20) == pytest.approx((40 * 90 + 20 * 100) / 60)


def test_up_wins_exactly_when_the_residual_is_at_least_minus_z():
    rng = np.random.default_rng(2)
    price, strike, settlement = 100 * np.exp(rng.normal(0, 0.01, (3, 5000)))
    seconds_left = rng.integers(1, 900, 5000)
    observed_sum = np.where(seconds_left < 60, (60 - seconds_left) * price, 0.0)
    z = pricing.z_scores(price, observed_sum, strike, seconds_left, SIGMA ** 2)
    residuals = pricing.residuals(settlement, price, observed_sum, seconds_left, SIGMA ** 2)
    assert np.array_equal(residuals >= -z, settlement >= strike)


def test_a_simulated_window_is_priced_at_the_rate_up_wins():
    settlements, _, observed_sum = simulated_settlements(300, 0, seed=3)
    strike = 1.0005                                                     # Half a spread or so above the price now.
    p = pricing.up_normal(pricing.z_scores(1.0, observed_sum, strike, 300, SIGMA ** 2))
    assert p == pytest.approx(np.mean(settlements >= strike), abs=0.01)


def test_the_distributions_agree_at_the_middle_and_the_t_has_fatter_tails():
    for up in (pricing.up_normal, lambda z: pricing.up_student_t(z, 4), lambda z: pricing.up_empirical(z, np.linspace(-3, 3, 10001))):
        assert up(0.0) == pytest.approx(0.5, abs=1e-3)
    assert 1 - pricing.up_student_t(3.0, 4) > 1 - pricing.up_normal(3.0)       # A 3 spread miss is likelier under a t.
    assert pricing.up_student_t(0.5, 4) > pricing.up_normal(0.5)              # And a half spread lead safer.


def test_the_empirical_chance_is_the_share_of_past_residuals_above_minus_z():
    past = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert pricing.up_empirical(0.5, past) == pytest.approx((3 + 0.5) / 6)  # -0.5 leaves 0, 1, and 2 above.
    assert 0 < pricing.up_empirical(-10.0, past) < pricing.up_empirical(10.0, past) < 1


def test_the_tracking_variance_keeps_a_window_decided_by_less_than_it_uncertain():
    price, tracking = 1.00002, 4e-5 ** 2                    # Every reading but the last taken, half a tracking error up.
    exact = pricing.z_scores(price, 59 * price, 1.0, 1, SIGMA ** 2)
    tracked = pricing.z_scores(price, 59 * price, 1.0, 1, SIGMA ** 2, tracking=tracking)
    assert exact > 10 and tracked == pytest.approx(np.log(price) / 4e-5, rel=1e-3)
    early = pricing.z_scores(1.001, 0.0, 1.0, 600, SIGMA ** 2, tracking=tracking)    # Ten minutes out it hardly matters.
    assert early == pytest.approx(pricing.z_scores(1.001, 0.0, 1.0, 600, SIGMA ** 2), rel=1e-3)
