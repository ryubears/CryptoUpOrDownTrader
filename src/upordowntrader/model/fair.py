"""
The fair price of a window's Up as the replay and the benchmark compute it, and its corrections learned from Kalshi.

The variance forecast is the calibration's best, see backtest/calibration.py, pulled DAY_SHARE of the way toward the
day's, in log variance: on Kalshi's windows of 2026 the market priced more variance than the 30 minute forecast, most
when the day had been busier than the last half hour, and the outcomes came out between the two. The chance of Up is
then a Student t of the z score, see model/pricing.py, whose spread includes how far the price's stand-in for BRTI missed
the settlements of the windows before, see tracking_variances().

Two corrections are fitted on Kalshi's settled windows, at each whole minute left, see backtest/benchmark.py:

    recalibration  a scale on the z score's spread and the Student t's degrees of freedom, those that gave the lowest
                   log loss, so the model's chances come true as often as they say on the windows it will trade
    blend          a logistic of the market's mid price and the model's chance, both in log odds, fitted the same way:
                   how far the model's chance should move the market's toward it. Where the model knows nothing the
                   market does not, the model's weight comes out near 0.

A fit holds one value for each whole minute left, 1 to MINUTES. A moment between marks takes the next mark's, and a
moment inside the averaging minute the last one's, 1 minute left.

A fair price's edge on a contract is what it is worth by that price less what it costs, with Kalshi's taker fee.
"""

import numpy as np
from dataclasses import dataclass
from upordowntrader.backtest import metrics
from upordowntrader.model import pricing, volatility

FORECAST = ("close", 30, "scaled", False)   # The calibration's best variance forecast: input, half life, weighting, season.
DAY_HALF_LIFE = 1440        # Minutes. The half life of the day's forecast the variance is pulled toward.
DAY_SHARE = 0.15            # How far the forecast is pulled toward the day's, in log variance. 0.15 and 0.25 tied best.
DOF = 8                     # The Student t's degrees of freedom without a recalibration, the calibration's best.
MINUTES = 14                # Whole minutes left a fit has values for: every minute mark after the first.
SCALES = np.round(np.arange(0.8, 1.61, 0.05), 2)    # Spread scales a recalibration chooses from.
DOFS = (4, 6, 8, 12, 20, 50)                        # Degrees of freedom it chooses from.
MIN_MARKS = 500             # Settled marks a minute left needs before it is fitted; with fewer it keeps the model as is.
CLIP = 1e-4                 # Chances are kept this far from 0 and 1 in log odds.
TINY = 1e-14                # Stands in for a variance of zero in a logarithm.
FEE_RATE = 0.07             # Kalshi's taker fee is FEE_RATE * price * (1 - price) a contract, for KXBTC15M's multiplier of 1.
TRACKING_WINDOWS = 96       # Windows settled before one whose tracking errors set its tracking variance: a day of 15 minutes.
TRACKING_MIN = 8            # Of them a window needs before it has a tracking variance; with fewer the price is taken as exact.


@dataclass
class Fit:
    """
    The corrections at each whole minute left, minutes left - 1 indexing each array.
    """
    scale: np.ndarray           # The spread's scale.
    dof: np.ndarray             # The Student t's degrees of freedom.
    blend: np.ndarray           # (intercept, market weight, model weight) of the blend's logistic, one row each.


# VARIANCE AND CHANCE

def variance(history):
    """
    The variance per second forecast at the end of each minute of the history, see the module's docstring.
    """
    short, _ = volatility.forecasts(history, *FORECAST)
    day, _ = volatility.forecasts(history, "close", DAY_HALF_LIFE, "time", False)
    return np.exp((1 - DAY_SHARE) * np.log(np.maximum(short, TINY)) + DAY_SHARE * np.log(np.maximum(day, TINY)))


def minutes_left(seconds_left):
    """
    The whole minutes left a fit's values are read at for each moment: the next mark's, at least 1, at most MINUTES.
    """
    return np.clip(np.ceil(np.asarray(seconds_left, dtype=float) / 60).astype(int), 1, MINUTES)


def chance(z, seconds_left, fit=None):
    """
    The chance of Up for z scores, see pricing.z_scores(), recalibrated by a Fit at each moment's minutes left, or with
    the model's own Student t without one.
    """
    if fit is None:
        return pricing.up_student_t(z, DOF)
    i = minutes_left(seconds_left) - 1
    return pricing.up_student_t(np.asarray(z) / fit.scale[i], fit.dof[i])


def tracking_variances(errors, open_times, close_times, windows=TRACKING_WINDOWS, least=TRACKING_MIN):
    """
    The tracking variance of each window, see model/pricing.py: the mean square of the tracking errors of the last
    windows settled before it opened, at least least of them, else 0, so the price is taken as exact. errors are each
    window's log of the stand-in's settlement over the window's own, NaN where unknown, and the windows are oldest first.
    A window closing as another opens is left out of that one's, as its settlement comes seconds later. On Kalshi's
    windows to 2026-10-12, the six exchanges' average missed by $3.26 in root mean square, and Coinbase's minutes over
    2026 by $9.69: little beside 15 minutes of bitcoin's moves, but most of the spread in the averaging minute.
    """
    errors = np.asarray(errors, dtype=float)
    known = np.isfinite(errors)
    squares = np.concatenate([[0.0], np.cumsum(np.where(known, errors, 0.0) ** 2)])
    counts = np.concatenate([[0], np.cumsum(known)])
    settled = np.searchsorted(np.asarray(close_times), np.asarray(open_times), side="left")   # Windows closed before each opened.
    first = np.maximum(settled - windows, 0)
    count = counts[settled] - counts[first]
    return np.where(count >= least, (squares[settled] - squares[first]) / np.maximum(count, 1), 0.0)


def log_odds(p):
    """
    Chances in log odds, kept CLIP from certainty.
    """
    p = np.clip(p, CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def blended(market, p, seconds_left, fit):
    """
    The blend of the market's mid prices and the model's chances by a Fit, see the module's docstring.
    """
    b = fit.blend[minutes_left(seconds_left) - 1]
    return 1 / (1 + np.exp(-np.clip(b[..., 0] + b[..., 1] * log_odds(market) + b[..., 2] * log_odds(p), -30, 30)))


def fee(price):
    """
    Kalshi's taker fee a contract at a price.
    """
    return FEE_RATE * price * (1 - price)


def edge(chance, price):
    """
    The edge after the fee on a contract bought at price, given the chance it pays.
    """
    return chance - price - fee(price)


# FITTING

def logistic(x, up, iterations=30):
    """
    The coefficients, intercept first, of a logistic regression of whether Up won on the columns of x, by Newton's method.
    """
    x = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ b, -30, 30)))
        b += np.linalg.solve(x.T @ (x * (p * (1 - p) + 1e-9)[:, None]) + 1e-6 * np.eye(len(b)), x.T @ (up - p))
    return b


def fit(z, market, up, seconds_left):
    """
    A Fit from settled marks: their z scores, the market's mid at each, NaN where it had none, whether Up won, and the
    seconds left. A minute left with fewer than MIN_MARKS keeps the model as is, and the market's price alone.
    """
    left = minutes_left(seconds_left)
    scale, dof = np.ones(MINUTES), np.full(MINUTES, DOF)
    blend = np.tile([0.0, 1.0, 0.0], (MINUTES, 1))
    for m in range(1, MINUTES + 1):
        at = left == m
        if at.sum() < MIN_MARKS:
            continue
        _, scale[m - 1], dof[m - 1] = min((metrics.log_loss(pricing.up_student_t(z[at] / s, d), up[at]), s, d) for s in SCALES for d in DOFS)
        quoted = at & ~np.isnan(market)
        if quoted.sum() >= MIN_MARKS:
            p = pricing.up_student_t(z[quoted] / scale[m - 1], dof[m - 1])
            blend[m - 1] = logistic(np.column_stack([log_odds(market[quoted]), log_odds(p)]), up[quoted].astype(float))
    return Fit(scale, dof, blend)
