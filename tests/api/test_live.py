"""
Smoke tests against the real exchange APIs, skipped unless asked for with `python3 -m pytest -m network`.
"""

import pytest
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from upordowntrader.api import coinbase, kalshi

pytestmark = pytest.mark.network

DAY = datetime(2026, 10, 1, tzinfo=UTC)


def test_coinbase_candles_come_oldest_first():
    candles = list(coinbase.candles("BTC-USD", 60, DAY, DAY + timedelta(hours=6)))
    assert 350 <= len(candles) <= 360 and candles[0].open_time == DAY
    assert all(a.open_time < b.open_time for a, b in pairwise(candles))


def test_kalshi_lists_the_window_trading_now_and_the_next_and_settles_the_last():
    now = datetime.now(UTC).timestamp()
    found = kalshi.windows("KXBTC15M", now, now + 1800)
    trading = [m for m in found if m.open_time <= now < m.close_time]
    assert len(trading) == 1 and trading[0].strike is not None and trading[0].close_time - trading[0].open_time == 900
    assert any(m.open_time >= trading[0].close_time and m.strike is None for m in found)      # Listed ahead, no strike yet.
    (last,) = kalshi.windows("KXBTC15M", trading[0].open_time - 300, trading[0].open_time)
    if last.result:                                                                         # Settled a few seconds after closing.
        assert last.settlement is not None and last.result == ("yes" if last.settlement >= last.strike else "no")

