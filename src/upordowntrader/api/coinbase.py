"""
Coinbase Exchange public market data: candles, read without an API key.

Coinbase is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on, so its
BTC-USD price is a close stand-in for the settlement price. Its shortest candle is 1 minute. Binance has 1 second klines.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from upordowntrader.api.http import RateLimiter, get_json, utc

API = "https://api.exchange.coinbase.com"
PAGE = 300              # The most candle lengths one call may span.
GRANULARITIES = (60, 300, 900, 3600, 21600, 86400)  # The candle lengths Coinbase offers, in seconds.
LIMITER = RateLimiter(5)    # Coinbase allows 10 public calls a second per IP, and this keeps to half.
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass
class Candle:
    """
    One candle. Volume is in the base currency, BTC for BTC-USD.
    """
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_candle(row):
    """
    A Candle from Coinbase's row: time, low, high, open, close, volume.
    """
    time, low, high, open_, close, volume = row
    return Candle(datetime.fromtimestamp(time, tz=UTC), float(open_), float(high), float(low), float(close), float(volume))


def check_granularity(granularity):
    """
    Raise ValueError for a candle length Coinbase does not offer.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity}, expected one of {GRANULARITIES}")


def ceil_to(dt, step):
    """
    The first multiple of step since 1970 at or after dt.
    """
    return EPOCH - ((EPOCH - dt) // step) * step


def fetch_candles(product_id, granularity, start, end):
    """
    One call: the candles opening from start to end, both included, oldest first. The range may span at most PAGE
    candles, and a period with no trades has no candle.
    """
    check_granularity(granularity)
    start, end = utc(start), utc(end)
    if end < start:
        raise ValueError("end is before start")
    if end - start > timedelta(seconds=granularity * PAGE):
        raise ValueError(f"range spans more than {PAGE} candles")
    params = {"granularity": granularity, "start": int(start.timestamp()), "end": int(end.timestamp())}
    rows = get_json(f"{API}/products/{product_id}/candles", params, LIMITER)
    return sorted((parse_candle(row) for row in rows), key=lambda c: c.open_time)     # Coinbase sends the newest first.


def candles(product_id, granularity, start, end):
    """
    Every candle opening from start up to but not including end, oldest first, PAGE to a call. A period with no trades
    has no candle, so check for gaps rather than assume every period is there.
    """
    check_granularity(granularity)
    step = timedelta(seconds=granularity)
    start, end = ceil_to(utc(start), step), utc(end)    # On a candle boundary, so no candle falls between two calls.
    while start < end:
        window_end = min(start + step * PAGE, end)
        batch = fetch_candles(product_id, granularity, start, max(start, window_end - step))     # Coinbase's end is included.
        yield from (c for c in batch if start <= c.open_time < window_end)
        start = window_end
