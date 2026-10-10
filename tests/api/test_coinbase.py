"""
Tests for the Coinbase client, with a stand in for its candles endpoint.
"""

import httpx
import pytest
from datetime import UTC, datetime, timedelta
from upordowntrader.api import coinbase

T0 = datetime(2026, 10, 1, tzinfo=UTC)


def minute(i):
    """
    Epoch seconds of the minute i minutes after T0.
    """
    return int((T0 + timedelta(minutes=i)).timestamp())


def candles_api(times):
    """
    A stand in for the candles endpoint, holding a candle at each of the times: start and end both included, at most 300
    candle lengths between them, the newest first.
    """
    def answer(request):
        granularity, start, end = (int(request.url.params[k]) for k in ("granularity", "start", "end"))
        if (end - start) / granularity > 300:
            return httpx.Response(400, json={"message": "granularity too small for the requested time range"})
        rows = [[t, 99.5, 101.5, 100.0, 100.5, 1.25] for t in sorted(times, reverse=True) if start <= t <= end]
        return httpx.Response(200, json=rows)

    return answer


def test_a_row_parses_into_a_candle_from_coinbases_column_order():
    candle = coinbase.parse_candle([minute(0), 99.5, 101.5, 100.0, 100.5, 1.25])
    assert candle == coinbase.Candle(T0, open=100.0, high=101.5, low=99.5, close=100.5, volume=1.25)


def test_candles_come_oldest_first_without_overlap_and_keep_the_gaps(serve):
    times = {minute(i) for i in range(700)} - {minute(5), minute(450)}      # Two minutes with no trades.
    sent = serve(candles_api(times))
    candles = list(coinbase.candles("BTC-USD", 60, T0, T0 + timedelta(minutes=650)))
    assert [int(c.open_time.timestamp()) for c in candles] == sorted(t for t in times if t < minute(650))
    assert len(sent) == 3                                   # 300, 300, then 50 minutes.
    assert sent[0].url.path == "/products/BTC-USD/candles"


def test_a_start_between_candles_moves_to_the_next_candle(serve):
    serve(candles_api({minute(i) for i in range(700)}))
    candles = list(coinbase.candles("BTC-USD", 60, T0 + timedelta(seconds=30), T0 + timedelta(minutes=601)))
    assert candles[0].open_time == T0 + timedelta(minutes=1) and len(candles) == 600


def test_one_call_may_not_span_more_than_300_candles():
    with pytest.raises(ValueError, match="more than 300"):
        coinbase.fetch_candles("BTC-USD", 60, T0, T0 + timedelta(minutes=301))


def test_a_candle_length_coinbase_does_not_offer_is_refused():
    with pytest.raises(ValueError, match="unsupported granularity"):
        list(coinbase.candles("BTC-USD", 30, T0, T0 + timedelta(minutes=1)))
