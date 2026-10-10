"""
How much bitcoin's price is moving: each minute's measure of variance, and forecasts of it from the minutes before.

Each minute gives a measure of its variance, in squared log returns, from one of these inputs:

    close         the squared return from one minute's close to the next, the textbook measure and the noisiest
    close_5m      the squared return over the 5 minutes to the minute's close, over 5, so still a variance per minute
    close_15m     the same over 15 minutes, a window's length
    realized      the sum of the minute's 60 squared 1 second returns, which sees every second's move
    parkinson     the squared log range from the minute's high to its low, scaled to a variance
    garman_klass  the range and the open to close move together

For a random walk every input measures the same variance per minute. Bitcoin's moves are not quite a random walk: when
seconds' moves tend to carry on in the same direction, they add up to more over a minute than their squares say, and
minutes' moves likewise over 15 minutes. Then the inputs measured over longer returns come out larger, and the one
measured over the window's own length is the one to trust.

A forecast is an exponentially weighted average of the measures, recent minutes counting more, each minute's weight
halving every half life. Three weightings:

    time     by age alone
    volume   by age times the minute's volume, so busy minutes count more
    scaled   by age alone, then scaled by how busy the last few minutes were against the half life's average, on the idea
             that variance comes with volume

Bitcoin moves more at some hours than others, around the US open and on the hour, and less at weekends. With season on,
each measure is divided by its half hour's usual share of the day's variance before the averaging, and the forecast for
a later minute is multiplied by that minute's share. The shares come from the SEASON_DAYS days before, weekdays and
weekends apart, so no forecast uses anything from after its minute.
"""

import numpy as np
from scipy.signal import lfilter

INPUTS = ("close", "close_5m", "close_15m", "realized", "parkinson", "garman_klass")
RETURN_MINUTES = {"close": 1, "close_5m": 5, "close_15m": 15}   # The minutes each close input's returns span.
WEIGHTINGS = ("time", "volume", "scaled")
FAST_HALF_LIFE = 10         # Minutes. How far back the scaled weighting looks to say how busy the market is now.
SEASON_DAYS = 28            # Days of history the season's shares come from.
SEASON_MINUTES = 30         # Minutes in each part of the day the season tells apart.
SEASON_MIN_DAYS = 4         # Days of the same kind, weekday or weekend, needed before a day gets shares; until then all are 1.
SHARE_LIMITS = (0.2, 5.0)   # Shares are kept inside these, so a half hour with almost no history cannot blow a forecast up.
MINUTES = 1440              # Minutes in a day.
TINY = 1e-300               # Stands in for zero volume in a division.


def measures(history, input):
    """
    Each minute's measure of variance, in squared log returns, from one of INPUTS.
    """
    if input in RETURN_MINUTES:
        minutes = RETURN_MINUTES[input]
        log_close = np.log(history.minute_close)
        earlier = np.concatenate([np.full(minutes, log_close[0]), log_close[:-minutes]])
        return (log_close - earlier) ** 2 / minutes
    if input == "realized":
        if history.minute_realized is None:
            raise ValueError(f"{history.source} has no 1 second bars, so no realized variance")
        return history.minute_realized
    log_range = np.log(history.minute_high / history.minute_low)
    if input == "parkinson":
        return log_range ** 2 / (4 * np.log(2))
    if input == "garman_klass":
        return 0.5 * log_range ** 2 - (2 * np.log(2) - 1) * np.log(history.minute_close / history.minute_open) ** 2
    raise ValueError(f"unknown input {input!r}, expected one of {INPUTS}")


def ewma(values, half_life):
    """
    The exponentially weighted average of the values up to each point, each one's weight halving every half_life points.
    It starts from the average of the first day, so the first few points are not ruled by the first value.
    """
    decay = 0.5 ** (1 / half_life)
    averaged, _ = lfilter([1 - decay], [1, -decay], values, zi=[decay * np.mean(values[:MINUTES])])
    return averaged


def season_shares(values, weekend):
    """
    Each minute's half hour's share of the day's variance, as a multiple of an average half hour's: the half hour's total
    over the SEASON_DAYS days before the minute's day that are of its kind, weekday or weekend, against an average half
    hour's total over the same days. values are per minute from the start of a UTC day, weekend per day.
    """
    days = len(values) // MINUTES
    per_part = values[:days * MINUTES].reshape(days, -1, SEASON_MINUTES).sum(axis=2)
    shares = np.ones_like(per_part)
    for day in range(days):
        before = [d for d in range(max(0, day - SEASON_DAYS), day) if weekend[d] == weekend[day]]
        if len(before) >= SEASON_MIN_DAYS:
            totals = per_part[before].sum(axis=0)
            shares[day] = np.clip(totals / max(totals.mean(), TINY), *SHARE_LIMITS)
    return np.repeat(shares, SEASON_MINUTES, axis=1).reshape(-1)


def forecasts(history, input, half_life, weighting, season):
    """
    The variance per second forecast at the end of each minute, from it and the minutes before, and each minute's season
    share, all 1 without season. The forecast for a later minute is the forecast times that minute's share. half_life is
    in minutes.
    """
    values = measures(history, input)
    shares = season_shares(values, history.weekend()) if season else np.ones_like(values)
    values = values / shares
    volume = history.minute_volume
    if weighting == "time":
        level = ewma(values, half_life)
    elif weighting == "volume":
        level = ewma(values * volume, half_life) / np.maximum(ewma(volume, half_life), TINY)
    elif weighting == "scaled":
        level = ewma(values, half_life) * ewma(volume, FAST_HALF_LIFE) / np.maximum(ewma(volume, half_life), TINY)
    else:
        raise ValueError(f"unknown weighting {weighting!r}, expected one of {WEIGHTINGS}")
    return level / 60, shares
