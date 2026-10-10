"""
Binance spot market data: klines, Binance's word for candles, read from the REST API or the bulk archive.

api.binance.com refuses US IPs with HTTP 451, so the REST calls go to data-api.binance.vision, Binance's public mirror of
its market data endpoints. The archive at data.binance.vision keeps one zipped CSV per day or per month, and is the fast
way to read years of 1 second klines: a day takes about 3 seconds from it and about 36 from the REST API.

Binance prices BTC in USDT rather than USD, so its prices differ from the USD index Kalshi and Polymarket US settle on by
the USDT/USD rate, typically under 0.1%.
"""

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from upordowntrader.api.http import RateLimiter, get_bytes, get_json, utc

API = "https://data-api.binance.vision/api/v3"
ARCHIVE = "https://data.binance.vision/data/spot"
PAGE = 1000             # The most klines one REST call returns.
# Binance allows 6000 request weight a minute per IP and a klines call weighs 2, so 10 calls a second stays well inside it.
API_LIMITER = RateLimiter(10)
ARCHIVE_LIMITER = RateLimiter(5)
# Seconds in each interval Binance offers, but for "1M", a calendar month, which has no fixed length.
INTERVALS = {"1s": 1, "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
             "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800}
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class ChecksumMismatch(Exception):
    """
    An archive file whose SHA-256 differs from the one Binance publishes beside it.
    """


@dataclass
class Kline:
    """
    One kline. Volumes are in BTC, the base asset, but for the quote ones, in USDT.
    """
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trade_count: int
    taker_buy_volume: float         # The part of volume bought by orders that crossed the spread.
    taker_buy_quote_volume: float


# PARSING

def from_timestamp(value):
    """
    The datetime of a Binance timestamp: epoch milliseconds, or microseconds in the spot archive's files from 2025 on.
    10**14 milliseconds is the year 5138 and 10**14 microseconds is 1973, so the size tells them apart.
    """
    if value >= 10**14:
        return EPOCH + timedelta(microseconds=value)
    return EPOCH + timedelta(milliseconds=value)


def to_millis(dt):
    """
    Epoch milliseconds for a datetime, as the REST API takes times.
    """
    return (utc(dt) - EPOCH) // timedelta(milliseconds=1)


def interval_length(interval):
    """
    The length of an interval such as '1s' or '1m'. Raises ValueError for one not in INTERVALS.
    """
    if interval not in INTERVALS:
        raise ValueError(f"unsupported interval {interval!r}, expected one of {', '.join(INTERVALS)}")
    return timedelta(seconds=INTERVALS[interval])


def parse_kline(row):
    """
    A Kline from a REST or archive row: open time, open, high, low, close, volume, close time, quote volume, trade count,
    taker buy volume, taker buy quote volume, and a column Binance ignores.
    """
    return Kline(
        open_time=from_timestamp(int(row[0])), open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]),
        volume=float(row[5]), quote_volume=float(row[7]), trade_count=int(row[8]), taker_buy_volume=float(row[9]),
        taker_buy_quote_volume=float(row[10]))


# REST

def fetch_klines(symbol, interval, start=None, end=None, limit=PAGE):
    """
    One call: up to limit klines opening from start to end, both included, oldest first. Without start, the latest ones.
    """
    interval_length(interval)
    if not 1 <= limit <= PAGE:
        raise ValueError(f"limit must be from 1 to {PAGE}")
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start is not None:
        params["startTime"] = to_millis(start)
    if end is not None:
        params["endTime"] = to_millis(end)
    return [parse_kline(row) for row in get_json(f"{API}/klines", params, API_LIMITER)]


def klines(symbol, interval, start, end):
    """
    Every kline opening from start up to but not including end, oldest first, PAGE to a call. An outage on Binance's side
    leaves a gap, so consecutive klines can be more than one interval apart.
    """
    step = interval_length(interval)
    start, end = utc(start), utc(end)
    last_open = end - timedelta(milliseconds=1)     # The REST API's end is included.
    while start < end:
        batch = fetch_klines(symbol, interval, start, last_open)
        yield from batch
        if len(batch) < PAGE:
            return
        start = batch[-1].open_time + step


# ARCHIVE

def parse_archive(payload):
    """
    The klines in an archive file, a zip holding one CSV, parsed as they are read, so a month of 1 second klines, 2.6
    million of them, need not be held at once.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"expected one CSV in the archive, found {names}")
        with archive.open(names[0]) as raw:
            for row in csv.reader(io.TextIOWrapper(raw, encoding="ascii")):
                if row and row[0].isdigit():        # Skips blank lines and any header row.
                    yield parse_kline(row)


def fetch_archive(path):
    """
    The klines of the archive file at a path under ARCHIVE. The file is downloaded and checked against its published
    SHA-256 at once, and parsed lazily, see parse_archive(). A file not yet published raises RequestFailed with status 404.
    """
    payload = get_bytes(f"{ARCHIVE}/{path}", limiter=ARCHIVE_LIMITER)
    expected = get_bytes(f"{ARCHIVE}/{path}.CHECKSUM", limiter=ARCHIVE_LIMITER).decode().split()[0].lower()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ChecksumMismatch(f"{path}: expected sha256 {expected}, got {actual}")
    return parse_archive(payload)


def daily_archive(symbol, interval, day):
    """
    The klines of one UTC day, from the archive, see fetch_archive(). A day's file is published the day after.
    """
    interval_length(interval)
    return fetch_archive(f"daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip")


def monthly_archive(symbol, interval, year, month):
    """
    The klines of one calendar month in UTC, from the archive, see fetch_archive(). A month's file is published early in
    the month after.
    """
    interval_length(interval)
    return fetch_archive(f"monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip")
