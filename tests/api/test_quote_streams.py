"""
Tests for the exchanges' quote streams, Coinbase's order book and Kraken's ticker, fed messages shaped as each sends them.
"""

import json
import pytest
from upordowntrader.api import coinbase, kraken


def quote_stream(stream_class, products):
    seen = []
    stream = stream_class(products, lambda product, bids, asks, sent: seen.append((product, bids, asks, sent)))
    stream.reset()
    return stream, seen


def test_coinbase_keeps_the_whole_book_and_passes_on_its_best_level_a_side():
    stream, seen = quote_stream(coinbase.CoinbaseQuoteStream, ["BTC-USD"])
    assert not stream.handle(json.dumps({"type": "subscriptions", "channels": [{"name": "level2_50", "product_ids": ["BTC-USD"]}]}))
    stream.handle(json.dumps({"type": "snapshot", "product_id": "BTC-USD", "bids": [["82961.83", "0.5"], ["82961.00", "2.0"]],
                              "asks": [["82961.84", "0.2"], ["82963.00", "1.0"]]}))
    assert seen[-1] == ("BTC-USD", [[82961.83, 0.5]], [[82961.84, 0.2]], None)          # A snapshot says no time.
    stream.handle(json.dumps({"type": "l2update", "product_id": "BTC-USD", "time": "2026-10-11T03:08:26.583783Z",
                              "changes": [["buy", "82961.83", "0.00000000"], ["sell", "82961.50", "0.3"]]}))
    assert seen[-1][:3] == ("BTC-USD", [[82961.00, 2.0]], [[82961.50, 0.3]])          # The best bid emptied, a better ask came.
    assert seen[-1][3] == pytest.approx(1791688106.583783)
    stream.handle(json.dumps({"type": "l2update", "product_id": "BTC-USD", "time": "2026-10-11T03:08:27Z",
                              "changes": [["sell", "82963.00", "0"], ["buy", "82000", "1"]]}))       # Changes away from the top.
    assert seen[-1][:3] == ("BTC-USD", [[82961.00, 2.0]], [[82961.50, 0.3]])


def test_coinbase_ignores_a_product_it_has_no_snapshot_for():
    stream, seen = quote_stream(coinbase.CoinbaseQuoteStream, ["BTC-USD"])
    assert not stream.handle(json.dumps({"type": "l2update", "product_id": "BTC-USD", "changes": [["buy", "1", "1"]]}))
    assert seen == []


def test_kraken_passes_on_each_best_bid_and_offer_and_heartbeats_are_not_data():
    stream, seen = quote_stream(kraken.KrakenQuoteStream, ["BTC/USD"])
    assert not stream.handle(json.dumps({"channel": "heartbeat"}))
    assert not stream.handle(json.dumps({"method": "subscribe", "success": True, "result": {"channel": "ticker"}}))
    assert stream.handle(json.dumps({"channel": "ticker", "type": "update", "data": [
        {"symbol": "BTC/USD", "bid": 82962.2, "bid_qty": 0.35412564, "ask": 82962.3, "ask_qty": 1.16484052,
         "timestamp": "2026-10-11T03:08:08.962607Z"}]}))
    assert seen == [("BTC/USD", [[82962.2, 0.35412564]], [[82962.3, 1.16484052]], pytest.approx(1791688088.962607))]
    assert kraken.ticker_frame("subscribe", ["BTC/USD"])["params"] == {"channel": "ticker", "symbol": ["BTC/USD"], "event_trigger": "bbo"}


def test_kraken_logs_a_refused_subscription():
    logs = []
    stream = kraken.KrakenQuoteStream(["BTC/USD"], lambda *a: None, log=logs.append)
    stream.handle(json.dumps({"method": "subscribe", "success": False, "error": "Currency pair not supported"}))
    assert logs == ["kraken stream refused subscribe: Currency pair not supported"]
