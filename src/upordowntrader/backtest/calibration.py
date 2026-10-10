"""
Offline calibration backtest: do the fair price model's chances come true as often as they say?

Every window of the chosen length in the price history is priced at a grid of times inside it, from what was known then:
the strike, the price, the settlement readings already taken, and a variance forecast from the minutes before. Each
chance of Up is scored against how the window settled. A calibrated model's 80% comes true 80% of the time. No market
prices are needed, only bitcoin's.

Every combination of variance input, half life, weighting, season, and distribution is scored, see model/volatility.py
and model/pricing.py. Two distributions are learned from the residuals of past windows: empirical from the whole train
period, and rolling from the ROLLING_DAYS days before each week, so it follows bitcoin as it changes. The best is chosen
on the earlier days, the train period, and checked on the later ones, the test period, so a choice that fit the past by
luck shows up worse on the test. Scores, see backtest/metrics.py:

    log loss   the main score, lower is better, a coin flip scores 0.693
    brier      lower is better, a coin flip scores 0.25
    spread     the standard deviation of the residuals, see pricing.residuals(): 1 when the variance forecast is right on
               average, above 1 when bitcoin moved more than forecast

The settlement averages come from the source's own prices, not BRTI, which is published only under license. Binance's
are the average of the minute's 60 one second closes, as BRTI's settlement averages 60 one second readings. Coinbase's
1 minute candles have no seconds, so its averages are (open + high + low + close) / 4, and the late evaluations inside
the averaging minute are left out.

Run from src/ with:
    python3 -m upordowntrader.backtest.calibration --source binance --minutes 15

The scores of every combination are saved as a CSV under data/calibration/.
"""

import argparse
import csv
import itertools
import numpy as np
import time
from dataclasses import dataclass
from datetime import date, timedelta
from upordowntrader.backtest import metrics
from upordowntrader.common.log import log
from upordowntrader.common.paths import DATA_DIR
from upordowntrader.common.sources import SOURCES
from upordowntrader.db import database
from upordowntrader.history import load
from upordowntrader.model import pricing, volatility

HALF_LIVES = (10, 30, 120, 480, 1440, 10080)    # Minutes: 10 and 30 minutes, 2 and 8 hours, a day, and a week.
DISTRIBUTIONS = ("normal", "t4", "t8", "empirical", "rolling")
ROLLING_DAYS = 60           # Days of windows before an evaluation the rolling distribution takes its residuals from.
ROLLING_STEP = 7            # Days between refreshes of the rolling distribution's residuals.
LATE = (30, 15, 5)          # Seconds before the close also scored, inside the averaging minute, for a source with seconds.
MAX_MISSING = 60            # Seconds a window may lack and still be scored, so long as its two averaging minutes lack none.
TEST_SHARE = 0.25           # The share of the scored days, the latest, held out as the test period by default.
MIN_VARIANCE = 1e-14        # Variance per second a forecast is kept above, so a dead quiet stretch cannot divide by zero.
TOP = 10                    # Combinations listed in the report.
RESULTS_DIR = DATA_DIR / "calibration"
DAY = 86400


@dataclass
class Points:
    """
    The evaluations a calibration scores, one per window and offset, as flat arrays: what the model knew at each, and how
    its window settled. Times are in seconds since the history's start.
    """
    window_seconds: int
    offsets: np.ndarray         # The seconds after a window opens that are scored.
    skipped: int                # Windows left out for missing data.
    window_start: np.ndarray    # When each evaluation's window opened.
    window: np.ndarray          # Which window, counting from 0.
    offset: np.ndarray          # Which offset, an index into offsets.
    seconds_left: np.ndarray
    price: np.ndarray           # The price at the evaluation.
    observed_sum: np.ndarray    # The sum of the settlement readings already taken, 0 before the averaging minute.
    strike: np.ndarray
    settlement: np.ndarray
    up: np.ndarray              # Whether Up won.
    minute: np.ndarray          # The last whole minute before the evaluation, whose forecast the model uses.
    horizon_start: np.ndarray   # The first and last minutes left in the window, whose season shares the forecast takes.
    horizon_end: np.ndarray
    test: np.ndarray            # Whether the evaluation's window is in the test period.


# EVALUATIONS

def scored_windows(history, window_seconds, warmup_days):
    """
    When each scored window opens, in seconds since the history's start, with its strike and settlement, and how many
    windows were skipped. Windows open on multiples of their length, as Kalshi's and Polymarket US's do, from warmup_days
    in. A window is skipped when either averaging minute lacks a second, or the window lacks more than MAX_MISSING.
    """
    total = len(history.minute_close) * 60
    starts = np.arange(max(warmup_days * DAY, window_seconds), total - window_seconds + 1, window_seconds)
    strike_minute = starts // 60 - 1
    settle_minute = (starts + window_seconds) // 60 - 1
    missing = np.concatenate([[0], np.cumsum(history.minute_missing)])
    whole = ((history.minute_missing[strike_minute] == 0) & (history.minute_missing[settle_minute] == 0)
             & (missing[settle_minute + 1] - missing[strike_minute] <= MAX_MISSING))
    return starts[whole], history.minute_average[strike_minute[whole]], history.minute_average[settle_minute[whole]], int((~whole).sum())


def window_offsets(window_seconds, every, seconds):
    """
    The seconds after a window opens that are scored: every `every` seconds until a minute before the close, and LATE
    seconds before the close too when the source has seconds.
    """
    grid = list(range(every, window_seconds - pricing.READINGS + 1, every))
    return np.array(grid + ([window_seconds - s for s in LATE] if seconds else []))


def readings_taken(history, close, seconds_left):
    """
    The sum of the settlement readings already taken, seconds_left before each close: the 1 second closes from the start
    of the averaging minute up to the evaluation.
    """
    taken = pricing.READINGS - seconds_left
    index = (close - pricing.READINGS)[:, None] + np.arange(pricing.READINGS)[None, :]
    return np.where(np.arange(pricing.READINGS)[None, :] < taken[:, None], history.second_close[index], 0.0).sum(axis=1)


def evaluation_points(history, window_seconds, every, warmup_days, test_from):
    """
    The Points to score: every scored window, see scored_windows(), at every offset, see window_offsets(). test_from is in
    seconds since the history's start.
    """
    if DAY % window_seconds or window_seconds % 60 or every % 60:
        raise ValueError("the window must divide a day into whole minutes, and the evaluations fall on whole minutes")
    starts, strike, settlement, skipped = scored_windows(history, window_seconds, warmup_days)
    offsets = window_offsets(window_seconds, every, history.second_close is not None)
    window = np.repeat(np.arange(len(starts)), len(offsets))
    offset = np.tile(np.arange(len(offsets)), len(starts))
    at = starts[window] + offsets[offset]
    close = starts[window] + window_seconds
    seconds_left = close - at
    price = history.second_close[at - 1] if history.second_close is not None else history.minute_close[at // 60 - 1]
    observed_sum = np.zeros(len(at))
    late = seconds_left < pricing.READINGS
    if late.any():
        observed_sum[late] = readings_taken(history, close[late], seconds_left[late])
    return Points(window_seconds=window_seconds, offsets=offsets, skipped=skipped, window_start=starts[window], window=window,
                  offset=offset, seconds_left=seconds_left, price=price, observed_sum=observed_sum, strike=strike[window],
                  settlement=settlement[window], up=settlement[window] >= strike[window], minute=at // 60 - 1,
                  horizon_start=at // 60, horizon_end=close // 60 - 1, test=starts[window] >= test_from)


# SCORING

def variances(history, points, input, half_life, weighting, season):
    """
    The variance per second each evaluation's model uses: the forecast at its last whole minute, times the average season
    share of the minutes left in its window.
    """
    per_second, shares = volatility.forecasts(history, input, half_life, weighting, season)
    total = np.concatenate([[0], np.cumsum(shares)])
    horizon = (total[points.horizon_end + 1] - total[points.horizon_start]) / (points.horizon_end - points.horizon_start + 1)
    return np.maximum(per_second[points.minute] * horizon, MIN_VARIANCE)


def rolling_chances(z, residuals, points, days=ROLLING_DAYS, step=ROLLING_STEP):
    """
    The chance of Up from recent residuals: each step days' evaluations take the residuals at their offset from the days
    of windows before them, so the distribution follows bitcoin as it changes, and uses only the past, which makes its
    train scores as honest as its test ones. The first step days, with no past yet, take the normal distribution's.
    """
    p = pricing.up_normal(z)
    day = points.window_start // DAY
    for k in range(len(points.offsets)):
        at = np.flatnonzero(points.offset == k)         # In window order, so in order of day.
        for start in range(int(day.min()) + step, int(day.max()) + 1, step):
            past, first, end = np.searchsorted(day[at], [start - days, start, start + step])
            if end > first:
                p[at[first:end]] = pricing.up_empirical(z[at[first:end]], residuals[at[past:first]])
    return p


def chances(distribution, z, residuals, points):
    """
    The chance of Up at each evaluation under one of DISTRIBUTIONS. The empirical one takes each offset's residuals from
    windows other than the evaluation's: a test evaluation's from the whole train period, and a train evaluation's from
    the other half of it, so that no score sees the outcome it is scoring. The rolling one takes them from the recent
    past, see rolling_chances().
    """
    if distribution == "normal":
        return pricing.up_normal(z)
    if distribution.startswith("t"):
        return pricing.up_student_t(z, int(distribution[1:]))
    if distribution == "rolling":
        return rolling_chances(z, residuals, points)
    train = ~points.test
    first = train & (points.window_start < np.median(points.window_start[train]))
    second = train & ~first
    p = np.empty(len(z))
    for k in range(len(points.offsets)):
        at = points.offset == k
        for scored, past in ((at & first, at & second), (at & second, at & first), (at & points.test, at & train)):
            if scored.any():
                p[scored] = pricing.up_empirical(z[scored], residuals[past])
    return p


def scores(p, residuals, points):
    """
    The log loss, Brier score, and residuals' spread of chances of Up, on the train and test periods apart.
    """
    out = {}
    for period, inside in (("train", ~points.test), ("test", points.test)):
        out[f"{period}_log_loss"] = metrics.log_loss(p[inside], points.up[inside]) if inside.any() else np.nan
        out[f"{period}_brier"] = metrics.brier(p[inside], points.up[inside]) if inside.any() else np.nan
        out[f"{period}_spread"] = float(np.std(residuals[inside])) if inside.any() else np.nan
    return out


def z_and_residuals(history, points, input, half_life, weighting, season):
    """
    Every evaluation's z score and residual under one variance forecast, see pricing.z_scores() and pricing.residuals().
    """
    variance = variances(history, points, input, half_life, weighting, season)
    z = pricing.z_scores(points.price, points.observed_sum, points.strike, points.seconds_left, variance)
    return z, pricing.residuals(points.settlement, points.price, points.observed_sum, points.seconds_left, variance)


def evaluate(history, points, input, half_life, weighting, season, distribution):
    """
    One combination's chances of Up and residuals at every evaluation.
    """
    z, residuals = z_and_residuals(history, points, input, half_life, weighting, season)
    return chances(distribution, z, residuals, points), residuals


def calibrate(history, points, inputs, half_lives=HALF_LIVES, weightings=volatility.WEIGHTINGS, seasons=(False, True),
              distributions=DISTRIBUTIONS, log=print):
    """
    Score every combination, as {(input, half_life, weighting, season, distribution): scores()}.
    """
    results, t0 = {}, time.time()
    # The scaled weighting is the time one when its half life is no longer than the look at how busy the market is now.
    forecasts = [choice for choice in itertools.product(inputs, half_lives, weightings, seasons)
                 if choice[2] != "scaled" or choice[1] > volatility.FAST_HALF_LIFE]
    for i, (input, half_life, weighting, season) in enumerate(forecasts, 1):
        z, residuals = z_and_residuals(history, points, input, half_life, weighting, season)
        for distribution in distributions:
            results[(input, half_life, weighting, season, distribution)] = scores(chances(distribution, z, residuals, points), residuals, points)
        if i % 12 == 0 or i == len(forecasts):
            log(f"scored {i} of {len(forecasts)} forecasts, {len(results)} combinations, in {time.time() - t0:.0f}s")
    return results


# REPORT

def half_life_label(minutes):
    """
    A half life in minutes as 30m, 8h, or 7d.
    """
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}m"


def combination_label(key):
    """
    A combination's choices as one line of the report's columns.
    """
    input, half_life, weighting, season, distribution = key
    return f"{input:<13}{half_life_label(half_life):>5}  {weighting:<8}{'season' if season else 'flat':<8}{distribution:<10}"


def option_label(position, option):
    """
    One choice of a combination, by its position in the key, as the report shows it.
    """
    if position == 1:
        return half_life_label(option)
    if position == 3:
        return "on" if option else "off"
    return option


def seconds_label(seconds):
    """
    Seconds as minutes and seconds, 14:00 or 0:05.
    """
    return f"{seconds // 60}:{seconds % 60:02d}"


def report(history, points, results, log=print):
    """
    Print the scores: the best combinations, each choice's effect with the others held at the best's, and the best's
    calibration on the test period by the favorite's chance and by the time left.
    """
    windows = len(np.unique(points.window))
    test_windows = len(np.unique(points.window[points.test]))
    first_test = history.first_day + timedelta(days=int(points.window_start[points.test].min() // DAY)) if points.test.any() else None
    log(f"\n{points.window_seconds // 60} minute windows of {history.source}, from {history.first_day} for {history.days()} days")
    log(f"{windows:,} windows scored, {windows - test_windows:,} train and {test_windows:,} test from {first_test}, "
        f"{points.skipped:,} skipped for missing data. Up won {points.up[points.offset == 0].mean():.1%} of them.")
    log(f"Scored at {', '.join(seconds_label(int(s)) for s in points.window_seconds - points.offsets)} before the close.")

    scored = slice(int(points.window_start.min()) // 60, None)
    scales = [(input, np.mean(volatility.measures(history, input)[scored])) for input in ("realized", *volatility.RETURN_MINUTES)
              if input != "realized" or history.minute_realized is not None]
    log("Variance per minute measured over different returns, against the first: "
        + ", ".join(f"{input} {variance / scales[0][1]:.2f}" for input, variance in scales)
        + ". Above 1, moves carry on in the same direction more than a random walk's do.")

    ranked = sorted(results, key=lambda key: results[key]["train_log_loss"])
    header = f"{'input':<13}{'half':>5}  {'weight':<8}{'season':<8}{'dist':<10}{'train LL':>9}{'test LL':>9}{'brier':>8}{'spread':>8}"
    log(f"\nBest {TOP} of {len(results)} combinations by train log loss, with their test scores:")
    log(header)
    for key in ranked[:TOP]:
        r = results[key]
        log(f"{combination_label(key)}{r['train_log_loss']:>9.4f}{r['test_log_loss']:>9.4f}{r['test_brier']:>8.4f}{r['test_spread']:>8.3f}")
    log(f"{'coin flip':<44}{np.log(2):>9.4f}{np.log(2):>9.4f}{0.25:>8.4f}")

    best = ranked[0]
    log("\nEach choice on its own, the others held at the best's, test log loss:")
    for position, name in enumerate(("input", "half life", "weighting", "season", "distribution")):
        cells = []
        for option in sorted({key[position] for key in results}):
            key = (*best[:position], option, *best[position + 1:])
            if key in results:
                cells.append(f"{option_label(position, option)} {results[key]['test_log_loss']:.4f}")
        log(f"  {name:<13}" + "   ".join(cells))

    p, residuals = evaluate(history, points, *best)
    test = points.test
    log(f"\nThe best, {combination_label(best).strip()}, on the test period by the favorite's chance:")
    log(f"{'favorite':<12}{'evaluations':>12}{'windows':>9}{'said':>8}{'won':>8}{'gap':>8}{'error':>8}")
    for row in metrics.favorites(p[test], points.up[test], points.window[test]):
        log(f"{row['low']:>4.0%} to {row['high']:<4.0%}{row['evaluations']:>12,}{row['windows']:>9,}{row['said']:>8.1%}{row['won']:>8.1%}"
            f"{100 * (row['won'] - row['said']):>+7.1f}c{100 * row['error']:>7.1f}c")
    log("A positive gap means the favorite won more often than the model said, so it priced favorites too cheap.")

    log("\nThe best on the test period by time left:")
    log(f"{'left':>6}{'evaluations':>12}{'log loss':>10}{'brier':>8}{'spread':>8}")
    for k, offset in enumerate(points.offsets):
        inside = test & (points.offset == k)
        if inside.any():
            log_loss, brier = metrics.log_loss(p[inside], points.up[inside]), metrics.brier(p[inside], points.up[inside])
            log(f"{seconds_label(int(points.window_seconds - offset)):>6}{int(inside.sum()):>12,}{log_loss:>10.4f}{brier:>8.4f}"
                f"{np.std(residuals[inside]):>8.3f}")


def save(results, path):
    """
    Write every combination's scores to a CSV, best train log loss first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        names = list(next(iter(results.values())))
        writer.writerow(["input", "half_life_minutes", "weighting", "season", "distribution", *names])
        for key in sorted(results, key=lambda key: results[key]["train_log_loss"]):
            writer.writerow([*key, *(f"{results[key][name]:.6f}" for name in names)])


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score the fair price model's chances against how bitcoin's windows settled.")
    ap.add_argument("--source", default="binance", choices=sorted(SOURCES))
    ap.add_argument("--minutes", type=int, default=15, help="window length, 15 for Kalshi's and Polymarket US's 15 minute markets")
    ap.add_argument("--start", type=date.fromisoformat, help="first UTC day of history to read, YYYY-MM-DD, by default the first stored")
    ap.add_argument("--end", type=date.fromisoformat, help="UTC day to stop before, by default the day after the last stored")
    ap.add_argument("--warmup-days", type=int, default=volatility.SEASON_DAYS, help="days read before the first window scored")
    ap.add_argument("--test-from", type=date.fromisoformat, help=f"first UTC day of the test period, by default the latest {TEST_SHARE:.0%}% of days")
    ap.add_argument("--every", type=int, help="seconds between evaluations in a window, a multiple of 60, by default a fifteenth of it")
    args = ap.parse_args()
    t0 = time.time()
    history = load.load(database.connect(), args.source, args.start, args.end, log=log)
    log(f"loaded {history.days()} days in {time.time() - t0:.0f}s")
    window_seconds = args.minutes * 60
    scored_days = history.days() - args.warmup_days
    test_day = args.test_from or history.first_day + timedelta(days=args.warmup_days + round(scored_days * (1 - TEST_SHARE)))
    points = evaluation_points(history, window_seconds, args.every or max(60, window_seconds // 15 // 60 * 60), args.warmup_days,
                               (test_day - history.first_day).days * DAY)
    log(f"{len(points.up):,} evaluations")
    inputs = [i for i in volatility.INPUTS if i != "realized" or history.minute_realized is not None]
    results = calibrate(history, points, inputs, log=log)
    report(history, points, results)
    path = RESULTS_DIR / f"{args.source}-{args.minutes}m-{history.first_day}-{history.days()}d.csv"
    save(results, path)
    print(f"\nEvery combination's scores are in {path}")
