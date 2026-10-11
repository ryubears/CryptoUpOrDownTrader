"""
Download bitcoin's price history into the database, one UTC day at a time, skipping the days already stored.

Coinbase's 1 minute candles come from its REST API, 300 to a call. Each day is stored in one transaction, so an
interrupted download leaves only whole days, and running it again picks up where it stopped. A few days are fetched at
once, since most of the time goes to waiting on the network.

Run from src/ with:
    python3 -m upordowntrader.history.download --start 2025-09-01
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from upordowntrader.api import coinbase
from upordowntrader.api.http import RequestFailed
from upordowntrader.common.log import log
from upordowntrader.common.sources import SOURCES
from upordowntrader.db import database

WORKERS = 4             # Days fetched at once.
DEFAULT_DAYS = 400      # Days back from today the download starts by default: a year to score, and four weeks of warmup.


def coinbase_day(day):
    """
    One day of Coinbase's 1 minute BTC-USD candles, as bar tuples for database.store_day(). Coinbase does not count trades.
    """
    symbol, seconds = SOURCES["coinbase"]
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return [(int(c.open_time.timestamp()), c.open, c.high, c.low, c.close, c.volume, None)
            for c in coinbase.candles(symbol, seconds, start, start + timedelta(days=1))]


FETCHERS = {"coinbase": coinbase_day}


def fetch_day(source, day):
    """
    One day of a source's bars, or None when the source has not published the day yet.
    """
    try:
        return FETCHERS[source](day)
    except RequestFailed as e:
        if e.status == 404:
            return None
        raise


def days_between(first, end):
    """
    Every date from first up to but not including end.
    """
    return [first + timedelta(days=i) for i in range((end - first).days)]


def download(conn, source, first, end, log=print, workers=WORKERS):
    """
    Download and store every day from first up to but not including end that the source's series lacks. Today, still
    under way, is never downloaded. Returns how many days were stored.
    """
    symbol, seconds = SOURCES[source]
    series = database.series_id(conn, source, symbol, seconds)
    stored = database.stored_days(conn, series)
    wanted = [day for day in days_between(first, min(end, datetime.now(UTC).date())) if day not in stored]
    log(f"{source}: {len(wanted)} days to download from {first} to {end}, {len(stored)} stored already")
    done, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(wanted), workers * 2):        # A few days ahead at a time, so fetched days never pile up.
            chunk = wanted[i:i + workers * 2]
            for day, bars in zip(chunk, pool.map(lambda d: fetch_day(source, d), chunk), strict=True):
                if bars is None:
                    log(f"{source} {day}: not published yet")
                    continue
                database.store_day(conn, series, day, bars)
                done += 1
                log(f"{source} {day}: {len(bars):,} bars, {done}/{len(wanted)} days in {time.time() - t0:.0f}s")
    return done


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download bitcoin's price history into SQLite.")
    ap.add_argument("--source", default="coinbase", choices=["all", *SOURCES])
    ap.add_argument("--start", type=date.fromisoformat, default=datetime.now(UTC).date() - timedelta(days=DEFAULT_DAYS),
                    help="first UTC day, YYYY-MM-DD")
    ap.add_argument("--end", type=date.fromisoformat, default=datetime.now(UTC).date(), help="UTC day to stop before, YYYY-MM-DD")
    args = ap.parse_args()
    conn = database.connect()
    for source in SOURCES if args.source == "all" else [args.source]:
        download(conn, source, args.start, args.end, log=log)
