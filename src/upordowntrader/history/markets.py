"""
Download Kalshi's settled Bitcoin 15 minute windows: each one's exact BRTI strike and settlement, and the market's best
Yes bid and ask minute by minute, for the benchmark, see backtest/benchmark.py.

Kalshi keeps the markets settled before a cutoff, a few weeks back, in a historical store, read whole, and the later ones
in its live store, read by close time. Both go into the markets table, beside the windows the recorder stores. Each
settled window's minutes come from the candlesticks call, a run of windows a call, see runs(), or, for a window settled
long ago, which that call returns nothing for, from the historical store's call, a window a call. Kalshi holds that one
to about 4 calls a second, so the 20,000 windows before 2026-08-04 take an hour and a half. Windows with minutes stored
already are skipped, so an interrupted download picks up where it stopped, and a rerun adds those settled since.

Run from src/ with:
    python3 -m upordowntrader.history.markets
"""

import time
from upordowntrader.api import kalshi
from upordowntrader.common.log import log
from upordowntrader.db import database

SERIES = "KXBTC15M"         # Kalshi's Bitcoin 15 minute up or down windows.
LOG_EVERY = 100             # Calls between progress lines.


def runs(markets, limit=kalshi.CANDLE_LIMIT):
    """
    The markets, oldest first, in runs one candlesticks call can return: Kalshi counts every minute from a run's first open
    to its last close once for each of its markets, which must come within limit.
    """
    found, run = [], []
    for m in markets:
        if run and (len(run) + 1) * ((m.close_time - run[0].open_time) // 60 + 1) > limit:
            found.append(run)
            run = []
        run.append(m)
    return found + ([run] if run else [])


def download(conn, series=SERIES, log=print):
    """
    Store every market of the series in either store, and the minutes of each settled one without them. Returns how many
    markets' minutes were stored.
    """
    t0 = time.time()
    found = kalshi.historical_windows(series)
    found += kalshi.windows(series, max((m.close_time for m in found), default=0), time.time())
    database.upsert_markets(conn, found)
    have = database.tickers_with_minutes(conn, "kalshi")
    wanted = sorted((m for m in database.load_markets(conn, "kalshi") if m.series == series and m.result and m.ticker not in have),
                    key=lambda m: m.open_time)
    calls = runs(wanted)
    log(f"{series}: {len(found):,} markets read in {time.time() - t0:.0f}s, {len(wanted):,} settled without minutes, in {len(calls):,} calls")
    stored = 0
    for i, run in enumerate(calls, 1):
        minutes = kalshi.minute_quotes([m.ticker for m in run], run[0].open_time, run[-1].close_time)
        for m in run:
            if not minutes.get(m.ticker):
                minutes[m.ticker] = kalshi.historical_minute_quotes(m.ticker, m.open_time, m.close_time)
        database.store_minutes(conn, "kalshi", minutes)
        stored += sum(1 for rows in minutes.values() if rows)
        if i % LOG_EVERY == 0 or i == len(calls):
            log(f"{i:,} of {len(calls):,} calls, minutes of {stored:,} markets stored, in {time.time() - t0:.0f}s")
    return stored


# MAIN

if __name__ == "__main__":
    download(database.connect(), log=log)
