"""
Smoke tests against the real exchange APIs, skipped unless asked for with `python3 -m pytest -m network`.
"""

import pytest
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from upordowntrader.api import binance, coinbase

pytestmark = pytest.mark.network

DAY = datetime(2026, 10, 1, tzinfo=UTC)


def test_binance_rest_klines_are_whole_and_consistent():
    klines = list(binance.klines("BTCUSDT", "1s", DAY, DAY + timedelta(seconds=1500)))
    assert len(klines) == 1500 and klines[0].open_time == DAY
    assert all(k.low <= min(k.open, k.close) <= max(k.open, k.close) <= k.high for k in klines)


def test_binance_archive_matches_the_rest_api():
    archive = list(binance.daily_archive("BTCUSDT", "1m", date(2026, 10, 1)))
    rest = list(binance.klines("BTCUSDT", "1m", DAY, DAY + timedelta(minutes=5)))
    assert len(archive) == 1440 and archive[:5] == rest      # Microsecond archive timestamps against millisecond REST ones.


def test_coinbase_candles_come_oldest_first():
    candles = list(coinbase.candles("BTC-USD", 60, DAY, DAY + timedelta(hours=6)))
    assert 350 <= len(candles) <= 360 and candles[0].open_time == DAY
    assert all(a.open_time < b.open_time for a, b in pairwise(candles))
