"""
Tests for the Kalshi client: signing, reading the windows' markets, and the book stream, with no network.
"""

import base64
import httpx
import json
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from upordowntrader.api import bookstream, kalshi

# A window's market as the markets call gives it, trading, and the one before it, settled. Cut down from real answers.
TRADING = {"ticker": "KXBTC15M-26OCT101515-15", "event_ticker": "KXBTC15M-26OCT101515", "open_time": "2026-10-10T19:00:00Z",
           "close_time": "2026-10-10T19:15:00Z", "floor_strike": 82961.78, "expiration_value": "", "result": "", "status": "active"}
UPCOMING = {**TRADING, "ticker": "KXBTC15M-26OCT101530-30", "event_ticker": "KXBTC15M-26OCT101530", "open_time": "2026-10-10T19:15:00Z",
            "close_time": "2026-10-10T19:30:00Z", "floor_strike": None, "status": "initialized"}
SETTLED = {"ticker": "KXBTC15M-26OCT101500-00", "event_ticker": "KXBTC15M-26OCT101500", "open_time": "2026-10-10T18:45:00Z",
           "close_time": "2026-10-10T19:00:00Z", "floor_strike": 82958.65, "expiration_value": "82961.78", "result": "yes",
           "settlement_ts": "2026-10-10T19:00:03.686718Z", "status": "finalized"}


def test_headers_are_signed_with_the_accounts_key(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(kalshi, "credentials", lambda: ("key-id", key))
    headers = kalshi.signed_headers("GET", kalshi.WS_PATH)
    assert headers["KALSHI-ACCESS-KEY"] == "key-id"
    message = (headers["KALSHI-ACCESS-TIMESTAMP"] + "GET" + kalshi.WS_PATH).encode()
    key.public_key().verify(base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]), message,      # Raises when it does not match.
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH), hashes.SHA256())


def test_a_market_carries_its_times_strike_and_settlement():
    trading, upcoming, settled = (kalshi.market(m) for m in (TRADING, UPCOMING, SETTLED))
    assert (trading.series, trading.open_time, trading.close_time - trading.open_time) == ("KXBTC15M", 1791658800, 900)
    assert (trading.strike, trading.settlement, trading.result, trading.settled_at) == (82961.78, None, None, None)
    assert upcoming.strike is None                              # Set as it opens.
    assert (settled.strike, settled.settlement, settled.result) == (82958.65, 82961.78, "yes")
    assert settled.settled_at == pytest.approx(1791658803.686718)
    assert kalshi.market({**SETTLED, "expiration_value": "77,362.10"}).settlement == 77362.10      # As a few markets write it.


def test_windows_asks_for_the_series_closing_in_the_range_and_follows_the_cursor(serve):
    pages = iter([{"markets": [TRADING], "cursor": "next"}, {"markets": [UPCOMING], "cursor": ""}])
    sent = serve(lambda request: httpx.Response(200, json=next(pages)))
    found = kalshi.windows("KXBTC15M", 1791140000.5, 1791142000)
    assert [m.ticker for m in found] == [TRADING["ticker"], UPCOMING["ticker"]]
    params = sent[0].url.params
    assert (params["series_ticker"], params["min_close_ts"], params["max_close_ts"]) == ("KXBTC15M", "1791140000", "1791142000")
    assert sent[1].url.params["cursor"] == "next"


def test_markets_reads_tickers_in_batches(serve, monkeypatch):
    monkeypatch.setattr(kalshi, "MARKETS_BATCH", 2)
    sent = serve(lambda request: httpx.Response(200, json={"markets": [SETTLED]}))
    kalshi.markets(["c", "a", "b", "a"])
    assert [request.url.params["tickers"] for request in sent] == ["a,b", "c"]


def test_historical_windows_reads_the_whole_historical_store(serve):
    sent = serve(lambda request: httpx.Response(200, json={"markets": [SETTLED], "cursor": ""}))
    assert [m.ticker for m in kalshi.historical_windows("KXBTC15M")] == [SETTLED["ticker"]]
    assert sent[0].url.path.endswith("/historical/markets") and "min_close_ts" not in sent[0].url.params


def test_minute_quotes_reads_each_minutes_closing_bid_and_ask_with_empty_sides_as_none(serve):
    candle = {"end_period_ts": 1786491060, "volume_fp": "90924.95", "price": {"close_dollars": "0.5000"},
              "yes_bid": {"close_dollars": "0.5000"}, "yes_ask": {"close_dollars": "0.5100"}}
    empty = {"end_period_ts": 1786491120, "volume": "0", "price": {"close": None}, "yes_bid": {"close": "0.0000"}, "yes_ask": {"close": "1.0000"}}
    sent = serve(lambda request: httpx.Response(200, json={"markets": [{"market_ticker": "A", "candlesticks": [candle, empty]}]}))
    found = kalshi.minute_quotes(["A", "B"], 1786491000.0, 1786491900)
    assert found == {"A": [(1786491060, 0.5, 0.51, 0.5, 90924.95), (1786491120, None, None, None, 0.0)]}
    params = sent[0].url.params
    assert (params["market_tickers"], params["start_ts"], params["end_ts"], params["period_interval"]) == ("A,B", "1786491000", "1786491900", "1")


def stream(tickers=("T",), depth=5):
    seen = []
    s = kalshi.KalshiBookStream(list(tickers), lambda ticker, bids, asks, sent: seen.append((ticker, bids, asks, sent)))
    s.depth = depth
    s.reset()
    return s, seen


def test_the_stream_restates_no_side_as_yes_asks_and_tracks_its_subscription():
    s, seen = stream()
    s.handle('{"type": "subscribed", "msg": {"channel": "orderbook_delta", "sid": 4}}')
    assert s.sid == 4 and s.subscribed.is_set()
    # With use_yes_price, a resting No order comes priced as the Yes ask it is.
    s.apply({"type": "orderbook_snapshot", "msg": {"market_ticker": "T", "yes_dollars_fp": [["0.28", "10"], ["0.30", "4"]],
                                                   "no_dollars_fp": [["0.35", "7"], ["0.32", "5"]]}})
    s.apply({"type": "orderbook_delta", "msg": {"market_ticker": "T", "price_dollars": "0.32", "delta_fp": "-5", "side": "no",
                                                "ts_ms": 1791140401234}})
    assert seen[0] == ("T", [[0.30, 4.0], [0.28, 10.0]], [[0.32, 5.0], [0.35, 7.0]], None)    # A snapshot says no time.
    assert seen[1] == ("T", [[0.30, 4.0], [0.28, 10.0]], [[0.35, 7.0]], 1791140401.234)


def test_a_delta_for_a_market_not_wanted_or_without_a_snapshot_is_ignored():
    s, seen = stream()
    s.apply({"type": "orderbook_delta", "msg": {"market_ticker": "T", "price_dollars": "0.3", "delta_fp": "1", "side": "yes"}})
    s.apply({"type": "orderbook_snapshot", "msg": {"market_ticker": "OTHER", "yes_dollars_fp": [["0.3", "1"]]}})
    assert seen == []


def test_a_skipped_sequence_number_forces_a_reconnect():
    s, _ = stream()
    s.handle(json.dumps({"type": "orderbook_delta", "sid": 1, "seq": 7, "msg": {}}))
    with pytest.raises(bookstream.Reconnect):
        s.handle(json.dumps({"type": "orderbook_delta", "sid": 1, "seq": 9, "msg": {}}))


def test_the_lifecycle_says_when_a_market_pauses_trades_again_and_is_decided():
    s, _ = stream()
    states = []
    s.on_state = lambda ticker, why: states.append((ticker, why))
    for body in ({"event_type": "deactivated", "is_deactivated": True}, {"event_type": "activated"}, {"event_type": "determined"},
                 {"event_type": "activated"}):
        s.apply({"type": kalshi.LIFECYCLE, "msg": {"market_ticker": "T", **body}})
    assert states == [("T", "paused"), ("T", None), ("T", "decided")]      # Nothing after it is decided.
