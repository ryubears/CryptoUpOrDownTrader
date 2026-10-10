"""
Made up price histories with a known variance, for the tests that need one without the network or the database.
"""

import numpy as np
from datetime import UTC, date, datetime
from upordowntrader.history import load

FIRST_DAY = date(2026, 1, 5)    # A Monday.


def day_start(first_day, day):
    """
    Seconds since 1970 at the start of the day-th day after first_day.
    """
    return int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp()) + day * load.DAY


def random_walk_rows(days, sigma, seed=0, price=80_000.0, first_day=FIRST_DAY):
    """
    1 second bars of a random walk in log price without drift, sigma per second, as one array of read_bars() rows per day.
    """
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, sigma, days * load.DAY)))
    open_ = np.concatenate([[price], close[:-1]])
    times = day_start(first_day, 0) + np.arange(days * load.DAY)
    rows = np.column_stack([times, open_, np.maximum(open_, close), np.minimum(open_, close), close, rng.exponential(1.0, days * load.DAY)])
    return [rows[d * load.DAY:(d + 1) * load.DAY] for d in range(days)]


def random_walk_history(days, sigma, seed=0, first_day=FIRST_DAY):
    """
    A History of 1 second bars of a random walk, see random_walk_rows().
    """
    parts, before = [], np.nan
    for d, rows in enumerate(random_walk_rows(days, sigma, seed, first_day=first_day)):
        parts.append(load.from_seconds(rows, day_start(first_day, d), before))
        before = parts[-1][1]["close"][-1]
    return load.assemble("binance", first_day, parts)
