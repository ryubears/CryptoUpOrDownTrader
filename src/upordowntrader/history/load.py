"""
Bitcoin's price history as numpy arrays, read from the database for the calibration.

A History covers whole UTC days. It has 1 minute bars for every minute, and for a source with 1 second bars, the price
every second too. A second or minute without a bar, from an exchange outage or a day not downloaded, carries the last
price forward with no volume, and is counted as missing, so the calibration can skip the windows it falls in.
"""

import numpy as np
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from upordowntrader.common.sources import SOURCES
from upordowntrader.db import database

DAY = 86400                 # Seconds in a day.
MINUTES = 1440              # Minutes in a day.


@dataclass
class History:
    """
    A source's prices from the start of first_day, every array indexed by seconds or minutes since then.
    """
    source: str
    first_day: date
    second_close: np.ndarray | None     # The last price in each second, None for a source with only minute bars.
    minute_open: np.ndarray
    minute_high: np.ndarray
    minute_low: np.ndarray
    minute_close: np.ndarray
    minute_volume: np.ndarray           # Bitcoin traded in each minute.
    minute_average: np.ndarray          # Each minute's average price as a settlement takes it, see from_seconds() and from_minutes().
    minute_realized: np.ndarray | None  # The sum of each minute's squared 1 second log returns, None without seconds.
    minute_missing: np.ndarray          # Seconds of each minute without a bar, 0 to 60.

    def days(self):
        return len(self.minute_close) // MINUTES

    def weekend(self):
        """
        Whether each day is a Saturday or Sunday, in UTC.
        """
        return np.array([(self.first_day + timedelta(days=d)).weekday() >= 5 for d in range(self.days())])


def forward_fill(values, before):
    """
    The values with each NaN replaced by the last number before it, or by before where none comes earlier.
    """
    last = np.where(np.isnan(values), -1, np.arange(len(values)))
    np.maximum.accumulate(last, out=last)
    return np.where(last >= 0, values[np.maximum(last, 0)], before)


def by_minute(values):
    """
    A day of values per second, as a row of 60 per minute.
    """
    return values.reshape(MINUTES, 60)


def placed(rows, column, index, size, default):
    """
    One column of the rows spread over size slots by index, default where no row falls.
    """
    out = np.full(size, default, dtype=float) if np.isscalar(default) else default.copy()
    out[index] = rows[:, column]
    return out


def from_seconds(rows, day_start, before):
    """
    One day of 1 second bars, as read_bars() gives them, turned into the day's price every second and its minute bars.
    before is the last price of the day before, NaN at the start of the history, when the day's first price fills in.
    A minute's average is the average of its 60 closes, as a settlement averages 60 one second readings.
    """
    index = (rows[:, 0] - day_start).astype(int)
    if np.isnan(before) and len(rows):
        before = rows[0, 4]
    close = forward_fill(placed(rows, 4, index, DAY, np.nan), before)
    present = np.zeros(DAY, dtype=bool)
    present[index] = True
    returns = np.diff(np.log(close), prepend=np.log(before))
    return close, {
        "open": by_minute(placed(rows, 1, index, DAY, close))[:, 0],
        "high": by_minute(placed(rows, 2, index, DAY, close)).max(axis=1),
        "low": by_minute(placed(rows, 3, index, DAY, close)).min(axis=1),
        "close": by_minute(close)[:, -1],
        "volume": by_minute(placed(rows, 5, index, DAY, 0.0)).sum(axis=1),
        "average": by_minute(close).mean(axis=1),
        "realized": by_minute(returns ** 2).sum(axis=1),
        "missing": 60 - by_minute(present).sum(axis=1),
    }


def from_minutes(rows, day_start, before):
    """
    One day of 1 minute bars, as read_bars() gives them, turned into the day's minute bars, see from_seconds(). Without the
    seconds, a minute's average is taken as (open + high + low + close) / 4.
    """
    index = ((rows[:, 0] - day_start) // 60).astype(int)
    if np.isnan(before) and len(rows):
        before = rows[0, 4]
    close = forward_fill(placed(rows, 4, index, MINUTES, np.nan), before)
    present = np.zeros(MINUTES, dtype=bool)
    present[index] = True
    minute = {"open": placed(rows, 1, index, MINUTES, close), "high": placed(rows, 2, index, MINUTES, close),
              "low": placed(rows, 3, index, MINUTES, close), "close": close, "volume": placed(rows, 5, index, MINUTES, 0.0)}
    minute["average"] = (minute["open"] + minute["high"] + minute["low"] + minute["close"]) / 4
    minute["missing"] = np.where(present, 0, 60)
    return None, minute


def assemble(source, first_day, days):
    """
    A History from each day's (seconds, minutes) pair, as from_seconds() or from_minutes() give them, in order.
    """
    minutes = {key: np.concatenate([m[key] for _, m in days]) for key in days[0][1]}
    seconds = np.concatenate([s for s, _ in days]) if days[0][0] is not None else None
    return History(source=source, first_day=first_day, second_close=seconds, minute_open=minutes["open"], minute_high=minutes["high"],
                   minute_low=minutes["low"], minute_close=minutes["close"], minute_volume=minutes["volume"],
                   minute_average=minutes["average"], minute_realized=minutes.get("realized"), minute_missing=minutes["missing"])


def load(conn, source, first_day=None, end_day=None, log=print):
    """
    A source's History from first_day up to but not including end_day, by default from the first day stored to the
    last. Days not stored inside the range count as missing.
    """
    symbol, seconds = SOURCES[source]
    series = database.series_id(conn, source, symbol, seconds)
    stored = database.stored_days(conn, series)
    if not stored:
        raise ValueError(f"no {source} history stored, download it first with upordowntrader.history.download")
    first_day = first_day or min(stored)
    end_day = end_day or max(stored) + timedelta(days=1)
    convert = from_seconds if seconds == 1 else from_minutes
    days, before = [], np.nan
    for i in range((end_day - first_day).days):
        day = first_day + timedelta(days=i)
        day_start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
        days.append(convert(database.read_bars(conn, series, day_start, day_start + DAY), day_start, before))
        before = days[-1][1]["close"][-1]
        if (i + 1) % 30 == 0:
            log(f"loaded {i + 1} of {(end_day - first_day).days} days of {source}")
    return assemble(source, first_day, days)
