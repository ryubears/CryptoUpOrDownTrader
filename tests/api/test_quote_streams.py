"""
Tests for the exchanges' quote streams, Coinbase's order book, Kraken's ticker, Bitstamp's order book, Gemini's top of
book, Bullish's best level, and Crypto.com's book changes, fed messages shaped as each sends them.
"""

import json
import pytest
from upordowntrader.api import bitstamp, bullish, coinbase, cryptocom, gemini, kraken
from upordowntrader.api.bookstream import Reconnect


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


def test_bitstamp_passes_on_the_best_level_of_each_book_it_sends():
    stream, seen = quote_stream(bitstamp.BitstampQuoteStream, ["btcusd"])
    assert not stream.handle(json.dumps({"event": "bts:subscription_succeeded", "channel": "order_book_btcusd", "data": {}}))
    assert stream.handle(json.dumps({"event": "data", "channel": "order_book_btcusd", "data": {
        "timestamp": "1791734796", "microtimestamp": "1791734796206682",
        "bids": [["83705.24", "0.35410802"], ["83704.85", "0.25000000"]], "asks": [["83705.25", "0.41366012"], ["83708.72", "0.1"]]}}))
    assert seen == [("btcusd", [[83705.24, 0.35410802]], [[83705.25, 0.41366012]], pytest.approx(1791734796.206682))]
    assert not stream.handle(json.dumps({"event": "data", "channel": "order_book_ethusd", "data": {"bids": [], "asks": []}}))


def test_bitstamp_asking_to_reconnect_rebuilds_the_connection():
    stream, _ = quote_stream(bitstamp.BitstampQuoteStream, ["btcusd"])
    with pytest.raises(Reconnect):
        stream.handle(json.dumps({"event": "bts:request_reconnect", "channel": "", "data": ""}))


def test_gemini_keeps_both_sides_and_passes_on_each_change_with_its_time():
    stream, seen = quote_stream(gemini.GeminiQuoteStream, ["BTCUSD"])
    stream.handle(json.dumps({"type": "update", "socket_sequence": 0, "events": [
        {"type": "change", "side": "ask", "price": "83725.59", "remaining": "0.0028", "reason": "initial"},
        {"type": "change", "side": "bid", "price": "83725.58", "remaining": "0.03583296", "reason": "initial"}]}))
    assert seen[-1] == ("BTCUSD", [[83725.58, 0.03583296]], [[83725.59, 0.0028]], None)     # The first says no time.
    stream.handle(json.dumps({"type": "update", "socket_sequence": 1, "timestampms": 1791734795679, "events": [
        {"type": "change", "side": "ask", "price": "83725.60", "remaining": "0.5", "reason": "top-of-book"}]}))
    assert seen[-1] == ("BTCUSD", [[83725.58, 0.03583296]], [[83725.60, 0.5]], pytest.approx(1791734795.679))
    assert not stream.handle(json.dumps({"type": "heartbeat", "socket_sequence": 2}))
    assert len(seen) == 2


def test_gemini_rebuilds_the_connection_when_a_message_was_lost():
    stream, _ = quote_stream(gemini.GeminiQuoteStream, ["BTCUSD"])
    stream.handle(json.dumps({"type": "heartbeat", "socket_sequence": 0}))
    with pytest.raises(Reconnect):
        stream.handle(json.dumps({"type": "update", "socket_sequence": 2, "events": []}))


def test_bullish_passes_on_each_best_level_with_its_time_and_logs_a_refusal():
    logs = []
    seen = []
    stream = bullish.BullishQuoteStream(["BTCUSD"], lambda *a: seen.append(a), log=logs.append)
    assert not stream.handle(json.dumps({"jsonrpc": "2.0", "id": "BTCUSD", "result": {
        "responseCode": "200", "responseCodeName": "OK", "message": "Successfully subscribed ( BTCUSD 1)"}}))
    assert stream.handle(json.dumps({"type": "update", "dataType": "V1TALevel1", "data": {
        "symbol": "BTCUSD", "bid": ["83556.4000", "0.00299199"], "ask": ["83556.5000", "0.00233757"], "sequenceNumber": "2529809794",
        "datetime": "2026-10-11T22:34:38.260Z", "timestamp": "1791758078260", "publishedAtTimestamp": "1791758078263"}}))
    assert seen == [("BTCUSD", [[83556.4, 0.00299199]], [[83556.5, 0.00233757]], pytest.approx(1791758078.260))]
    assert not stream.handle(json.dumps({"jsonrpc": "2.0", "id": "BTCUSD", "result": {"responseCode": "400", "message": "Bad symbol"}}))
    assert logs and logs[0].startswith("bullish stream refused BTCUSD")
    assert bullish.command_frame("unsubscribe", "BTCUSD")["params"] == {"topic": "l1Orderbook", "symbol": "BTCUSD"}


def cryptocom_message(channel, data):
    return json.dumps({"id": -1, "method": "subscribe", "code": 0, "result": {
        "instrument_name": "BTC_USD", "subscription": "book.BTC_USD.10", "channel": channel, "depth": 10, "data": [data]}})


def test_cryptocom_keeps_the_book_from_its_snapshot_and_changes_and_answers_heartbeats():
    stream, seen = quote_stream(cryptocom.CryptoComQuoteStream, ["BTC_USD"])
    assert not stream.handle(json.dumps({"id": 1, "method": "subscribe", "code": 0, "channel": "book.BTC_USD.10"}))
    stream.handle(cryptocom_message("book", {"asks": [["83578.46", "0.12006", "3"], ["83579.30", "0.12", "1"]],
                                             "bids": [["83578.45", "0.83356", "9"], ["83578.35", "0.02991", "1"]], "t": 1791758080078, "u": 10}))
    assert seen[-1] == ("BTC_USD", [[83578.45, 0.83356]], [[83578.46, 0.12006]], pytest.approx(1791758080.078))
    stream.handle(cryptocom_message("book.update", {"update": {"asks": [["83578.46", "0", "0"]], "bids": [["83578.45", "0.9", "10"]]},
                                                    "t": 1791758080519, "u": 12, "pu": 10}))
    assert seen[-1] == ("BTC_USD", [[83578.45, 0.9]], [[83579.30, 0.12]], pytest.approx(1791758080.519))   # The best ask emptied.
    assert not stream.handle(json.dumps({"id": 1791758845879, "method": "public/heartbeat", "code": 0}))
    assert stream.commands.get_nowait() == ("heartbeat", [1791758845879])


def test_cryptocom_rebuilds_the_connection_when_a_change_was_lost():
    stream, seen = quote_stream(cryptocom.CryptoComQuoteStream, ["BTC_USD"])
    assert not stream.handle(cryptocom_message("book.update", {"update": {"asks": [], "bids": []}, "t": 1, "u": 5, "pu": 4}))
    stream.handle(cryptocom_message("book", {"asks": [["2", "1", "1"]], "bids": [["1", "1", "1"]], "t": 1, "u": 10}))
    with pytest.raises(Reconnect):
        stream.handle(cryptocom_message("book.update", {"update": {"asks": [], "bids": []}, "t": 2, "u": 13, "pu": 11}))
    assert len(seen) == 1                                   # A change before the snapshot is not passed on.
