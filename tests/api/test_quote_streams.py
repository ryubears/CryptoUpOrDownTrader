"""
Tests for the exchanges' quote streams, Coinbase's order book, Kraken's ticker, Bitstamp's order book, Gemini's top of
book, Bullish's best level, and Crypto.com's book changes, and the perpetuals', Binance's, Bybit's, OKX's, Deribit's, and
Coinbase Derivatives', fed messages shaped as each sends them.
"""

import json
import pytest
from upordowntrader.api import binance, bitstamp, bullish, bybit, coinbase, cryptocom, deribit, gemini, kraken, okx
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


def logged_stream(stream_class, products):
    """
    A quote stream whose quotes and log lines are kept.
    """
    seen, logs = [], []
    stream = stream_class(products, lambda product, bids, asks, sent: seen.append((product, bids, asks, sent)), log=logs.append)
    stream.reset()
    return stream, seen, logs


def test_binance_passes_on_each_book_ticker_with_its_time_and_logs_a_refusal():
    stream, seen, logs = logged_stream(binance.BinanceQuoteStream, ["BTCUSDT"])
    assert not stream.handle(json.dumps({"result": None, "id": 1}))
    assert stream.handle(json.dumps({"e": "bookTicker", "u": 11798458399813, "s": "BTCUSDT", "ps": "BTCUSDT", "b": "82490.30", "B": "19.934",
                                     "a": "82490.40", "A": "5.465", "T": 1791829284397, "E": 1791829284398, "st": 1}))
    assert seen == [("BTCUSDT", [[82490.30, 19.934]], [[82490.40, 5.465]], pytest.approx(1791829284.397))]   # The change's time.
    stream.handle(json.dumps({"error": {"code": 2, "msg": "Invalid request"}, "id": 1}))
    assert logs == ["binance stream refused: {'code': 2, 'msg': 'Invalid request'}"]
    assert binance.command_frame("SUBSCRIBE", ["BTCUSDT"])["params"] == ["btcusdt@bookTicker"]


def test_bybit_passes_on_each_best_level_with_the_engines_time_and_pings():
    stream, seen, logs = logged_stream(bybit.BybitQuoteStream, ["BTCUSDT"])
    assert not stream.handle(json.dumps({"success": True, "ret_msg": "", "conn_id": "c", "req_id": "", "op": "subscribe"}))
    assert not stream.handle(json.dumps({"success": True, "ret_msg": "pong", "conn_id": "c", "req_id": "", "op": "ping"}))
    assert stream.handle(json.dumps({"topic": "orderbook.1.BTCUSDT", "ts": 1791829284499, "type": "snapshot", "cts": 1791829284498,
                                     "data": {"s": "BTCUSDT", "b": [["82488.1", "3.711"]], "a": [["82488.2", "3.471"]], "u": 1, "seq": 2}}))
    assert seen[-1] == ("BTCUSDT", [[82488.1, 3.711]], [[82488.2, 3.471]], pytest.approx(1791829284.498))
    stream.handle(json.dumps({"topic": "orderbook.1.BTCUSDT", "ts": 1791829284600, "type": "delta", "cts": 1791829284599,
                              "data": {"s": "BTCUSDT", "b": [], "a": [["82488.3", "1.5"]], "u": 2, "seq": 3}}))
    assert seen[-1][:3] == ("BTCUSDT", [[82488.1, 3.711]], [[82488.3, 1.5]])           # A delta changes only the side it names.
    stream.handle(json.dumps({"success": False, "ret_msg": "error:handler not found,topic:orderbook.1.NOSUCH", "op": "subscribe"}))
    assert logs == ["bybit stream refused subscribe: error:handler not found,topic:orderbook.1.NOSUCH"]
    assert json.loads(stream.keepalive()) == {"op": "ping"} and stream.keepalive_seconds == 20


def test_okx_passes_on_each_best_level_in_bitcoin_and_pongs_are_not_data():
    stream, seen, logs = logged_stream(okx.OkxQuoteStream, ["BTC-USDT-SWAP"])
    assert not stream.handle("pong")
    assert not stream.handle(json.dumps({"event": "subscribe", "arg": {"channel": "bbo-tbt", "instId": "BTC-USDT-SWAP"}, "connId": "c"}))
    assert stream.handle(json.dumps({"arg": {"channel": "bbo-tbt", "instId": "BTC-USDT-SWAP"}, "data": [
        {"asks": [["82490.9", "299.66", "0", "51"]], "bids": [["82490.8", "928.02", "0", "58"]], "ts": "1791829284608", "seqId": 1}]}))
    assert seen == [("BTC-USDT-SWAP", [[82490.8, pytest.approx(9.2802)]], [[82490.9, pytest.approx(2.9966)]], pytest.approx(1791829284.608))]
    stream.handle(json.dumps({"event": "error", "msg": "Subscribe failed, instId:NOSUCH doesn't exist.", "code": "60018", "connId": "c"}))
    assert logs == ["okx stream refused: Subscribe failed, instId:NOSUCH doesn't exist."]
    assert stream.keepalive() == "ping"


def test_deribit_passes_on_each_quote_in_bitcoin_and_logs_a_subscription_to_nothing():
    stream, seen, logs = logged_stream(deribit.DeribitQuoteStream, ["BTC-PERPETUAL"])
    assert not stream.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "result": ["quote.BTC-PERPETUAL"], "testnet": False}))
    assert stream.handle(json.dumps({"jsonrpc": "2.0", "method": "subscription", "params": {"channel": "quote.BTC-PERPETUAL", "data": {
        "timestamp": 1791829284096, "instrument_name": "BTC-PERPETUAL", "best_ask_price": 82454.5, "best_bid_price": 82454.0,
        "best_ask_amount": 31480.0, "best_bid_amount": 14270.0}}}))
    assert seen == [("BTC-PERPETUAL", [[82454.0, pytest.approx(14270 / 82454)]], [[82454.5, pytest.approx(31480 / 82454.5)]],
                     pytest.approx(1791829284.096))]                                    # Dollars turned into bitcoin.
    stream.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "result": [], "testnet": False}))
    assert logs == ["deribit stream refused: no channel subscribed"]


def frame(n, events, channel="l2_data"):
    """
    A message of Coinbase's Advanced Trade websocket, the n-th of its connection.
    """
    return json.dumps({"channel": channel, "timestamp": "2026-10-12T18:21:23.9Z", "sequence_num": n, "events": events})


def test_coinbase_perp_keeps_the_book_in_bitcoin_and_passes_on_its_best_level_with_the_changes_time():
    stream, seen, logs = logged_stream(coinbase.CoinbasePerpQuoteStream, ["BIP-20DEC30-CDE"])
    assert not stream.handle(frame(0, [{"type": "update", "product_id": "BIP-20DEC30-CDE", "updates": [
        {"side": "bid", "event_time": "2026-10-12T18:21:23.8Z", "price_level": "1", "new_quantity": "1"}]}]))      # No snapshot yet.
    assert stream.handle(frame(1, [{"type": "snapshot", "product_id": "BIP-20DEC30-CDE", "updates": [
        {"side": "bid", "event_time": "2026-10-12T18:21:23.712102Z", "price_level": "82430", "new_quantity": "11"},
        {"side": "bid", "event_time": "2026-10-12T18:21:23.712102Z", "price_level": "82425", "new_quantity": "298"},
        {"side": "offer", "event_time": "2026-10-12T18:21:23.712102Z", "price_level": "82450", "new_quantity": "608"}]}]))
    assert seen[-1] == ("BIP-20DEC30-CDE", [[82430.0, pytest.approx(0.11)]], [[82450.0, pytest.approx(6.08)]], None)   # A snapshot says no time.
    assert not stream.handle(frame(2, [{"subscriptions": {"level2": ["BIP-20DEC30-CDE"]}}], channel="subscriptions"))
    assert not stream.handle(frame(3, [{"current_time": "2026-10-12 18:21:24", "heartbeat_counter": 1}], channel="heartbeats"))
    stream.handle(frame(4, [{"type": "update", "product_id": "BIP-20DEC30-CDE", "updates": [
        {"side": "bid", "event_time": "2026-10-12T18:21:24.349409Z", "price_level": "82430", "new_quantity": "0"},
        {"side": "offer", "event_time": "2026-10-12T18:21:24.430162Z", "price_level": "82440", "new_quantity": "5"}]}]))
    assert seen[-1][:3] == ("BIP-20DEC30-CDE", [[82425.0, pytest.approx(2.98)]], [[82440.0, pytest.approx(0.05)]])
    assert seen[-1][3] == pytest.approx(1791829284.430162)                              # The latest change's time.
    with pytest.raises(Reconnect):
        stream.handle(frame(6, []))                                                     # Message 5 was lost.
    stream.reset()
    assert not stream.handle(frame(0, [], channel="subscriptions")) and logs == []      # A new connection counts from 0 again.
