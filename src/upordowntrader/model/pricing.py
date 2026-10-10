"""
The fair price of an up or down contract: the chance its settlement average ends at or above its strike.

Kalshi and Polymarket US settle a window on the average of 60 one second readings of CF Benchmarks' BRTI in the minute
before it closes, against the same average in the minute before it opens, the strike, Up winning a tie. Inside a window
the strike is known, and so are the price now and any readings already taken. The model treats the log price as a
random walk without drift, so the settlement average lands around its expected value with a spread set by the variance
per second and the seconds of variance the averaging leaves, see averaging_seconds(). The chance of Up is the share of
that spread above the strike, found from the z score, how far the expected settlement is above the strike in spreads:

    z = log(expected settlement / strike) / sqrt(variance per second * averaging seconds)

A normal distribution is the textbook choice. Bitcoin jumps more often than a normal allows, so a Student t, with fatter
tails, or the residuals of past windows themselves, may fit better. Every distribution here has a variance of 1, so a z
score means the same spread under each.
"""

import numpy as np
from scipy import special, stats

READINGS = 60   # One second readings averaged at each end of a window.


def averaging_seconds(seconds_left, readings=READINGS):
    """
    The seconds of random walk variance left in the settlement average, seconds_left before the window closes. Before the
    averaging minute, the time until it begins counts in full, and the average of its n readings adds (n+1)(2n+1)/6n,
    about a third of the minute, since the readings share the moves before them. Inside the minute, only the r readings
    still to come are uncertain, each a 1/n share of the average, which leaves r(r+1)(2r+1)/6n².
    """
    seconds_left = np.asarray(seconds_left, dtype=float)
    r = np.minimum(seconds_left, readings)
    return np.maximum(seconds_left - readings, 0) + r * (r + 1) * (2 * r + 1) / (6 * readings ** 2)


def expected_settlement(price, observed_sum, seconds_left, readings=READINGS):
    """
    The expected settlement average: the readings already taken, whose sum is observed_sum, with the price now standing in
    for each one still to come. Before the averaging minute, no readings are taken and it is the price now.
    """
    to_come = np.minimum(np.asarray(seconds_left, dtype=float), readings)
    return (observed_sum + to_come * price) / readings


def spreads(variance_per_second, seconds_left, readings=READINGS):
    """
    The standard deviation of the log settlement average around its expected value.
    """
    return np.sqrt(variance_per_second * averaging_seconds(seconds_left, readings))


def z_scores(price, observed_sum, strike, seconds_left, variance_per_second, readings=READINGS):
    """
    How far the expected settlement is above the strike, in spreads. The chance of Up rises with it.
    """
    expected = expected_settlement(price, observed_sum, seconds_left, readings)
    return np.log(expected / strike) / spreads(variance_per_second, seconds_left, readings)


def residuals(settlement, price, observed_sum, seconds_left, variance_per_second, readings=READINGS):
    """
    How far each settlement ended from the expected one, in spreads, the z score's counterpart after the fact: Up won when
    the residual is at least -z. Residuals have a spread of 1 when the variance forecast is right on average.
    """
    expected = expected_settlement(price, observed_sum, seconds_left, readings)
    return np.log(settlement / expected) / spreads(variance_per_second, seconds_left, readings)


def up_normal(z):
    """
    The chance of Up under a normal distribution.
    """
    return special.ndtr(z)


def up_student_t(z, dof):
    """
    The chance of Up under a Student t with dof degrees of freedom, scaled to a variance of 1: fatter tails than a normal,
    so more weight on big jumps and less on middling moves.
    """
    return stats.t.cdf(np.asarray(z) * np.sqrt(dof / (dof - 2)), dof)


def up_empirical(z, past):
    """
    The chance of Up from past residuals: the share of them at least -z. Half a residual is added above and below, since
    a finite past cannot rule an outcome out.
    """
    past = np.sort(past)
    above = len(past) - np.searchsorted(past, -np.asarray(z), side="left")
    return (above + 0.5) / (len(past) + 1)
