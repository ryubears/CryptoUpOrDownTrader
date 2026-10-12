"""
Tests for Hyperliquid's up or down windows read from its market list, their settlements read from its explorer, and its
streams, fed messages shaped as Hyperliquid sends them.
"""

import httpx
import json
import pytest
from datetime import UTC, datetime
from upordowntrader.api import hyperliquid

FIFTEEN = {"outcome": 11376, "name": "template:binaryPrice", "venue": "out", "quoteToken": "USDC",
           "description": "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:60|threshold:83984|time:20261012-0545"}
DAILY = {"outcome": 10997, "name": "Recurring", "quoteToken": "USDC",
         "description": "class:priceBinary|underlying:BTC|expiry:20261012-0600|targetPrice:82994|period:1d"}
OTHERS = [
    {"outcome": 11235, "name": "template:binaryPrice", "venue": "out", "description":
     "perp:xyz:SP500|priceDescription:xyz:SP500-USDC perp mark|seconds:60|threshold:7822.5|time:20261012-2000"},       # Not crypto.
    {"outcome": 11370, "name": "template:binaryPrice", "venue": "skew", "description":
     "perp:BTC|priceDescription:the Hyperliquid BTC perp trade|seconds:90|threshold:84130|time:20261012-0600"},        # Another deployer.
    {"outcome": 7176, "name": "template:priceTouch", "venue": "out", "description":
     "perp:BTC|priceDescription:BTC-USDC perp mark|seconds:3|target:80000|time:20261101-0000"},                         # Not up or down.
    {"outcome": 11002, "name": "Recurring Named Outcome", "description": "index:0"},
]
SETTLE = {"time": 1791783008788, "user": "0xe505", "error": None, "action": {"type": "outcomeDeploy", "venue": "out", "operation": {
    "settleOutcome": {"outcome": 11376, "settleFraction": "1", "details": "", "nameAndDescription": [FIFTEEN["name"], FIFTEEN["description"]]}}}}
REGISTER = {"time": 1791783014442, "user": "0xe505", "error": None, "action": {"type": "outcomeDeploy", "venue": "out", "operation": {
    "registerStandaloneOutcomeFromTemplate": {"id": "binaryPrice", "keywordToValue": [["perp", "BTC"], ["threshold", "83474"]]}}}}


def test_the_15_minute_and_daily_windows_are_read_and_every_other_market_is_not():
    fifteen = hyperliquid.window(FIFTEEN)
    close = int(datetime(2026, 10, 12, 5, 45, tzinfo=UTC).timestamp())
    assert (fifteen.ticker, fifteen.series, fifteen.event) == ("#113760", "out-BTC-15m", "out-BTC-15m-20261012-0545")
    assert (fifteen.open_time, fifteen.close_time, fifteen.strike, fifteen.result) == (close - 900, close, 83984.0, None)
    daily = hyperliquid.window(DAILY)
    assert (daily.series, daily.close_time - daily.open_time, daily.strike) == ("recurring-BTC-1d", 86400, 82994.0)
    assert all(hyperliquid.window(o) is None for o in OTHERS)
    assert hyperliquid.window({**FIFTEEN, "description": FIFTEEN["description"].replace("0545", "0550")}) is None   # Off the quarter hour.
    assert hyperliquid.window({**FIFTEEN, "description": FIFTEEN["description"].replace("BTC", "ETH")}) is None     # BTC only, for now.
    assert hyperliquid.outcome_of("#113760") == 11376 and hyperliquid.coin(11376, 1) == "#113761"


def test_windows_asks_for_the_market_list(serve):
    sent = serve(lambda request: httpx.Response(200, json={"outcomes": [FIFTEEN, DAILY, *OTHERS], "questions": [], "deployers": []}))
    assert [m.ticker for m in hyperliquid.windows()] == ["#113760", "#109970"]
    assert json.loads(sent[0].content) == {"type": "outcomeMeta"}


def test_settlements_reads_each_settling_account_of_the_deployer_from_the_explorer(serve):
    deployers = [{"venue": "out", "subDeployers": [["registerStandaloneOutcomeFromTemplate", ["0xa"]], ["settleOutcome", ["0xe505", "0xf192"]]]},
                 {"venue": "skew", "subDeployers": [["settleOutcome", ["0x1c86"]]]}]
    failed = {**SETTLE, "error": "not allowed"}
    settle = {**SETTLE["action"]["operation"]["settleOutcome"], "nameAndDescription": [OTHERS[0]["name"], OTHERS[0]["description"]]}
    other = {**SETTLE, "action": {**SETTLE["action"], "operation": {"settleOutcome": settle}}}           # Not one of the series.

    def answer(request):
        body = json.loads(request.content)
        if body["type"] == "outcomeMeta":
            return httpx.Response(200, json={"outcomes": [], "deployers": deployers})
        return httpx.Response(200, json={"type": "userDetails", "txs": [SETTLE, REGISTER, failed, other] if body["user"] == "0xe505" else []})

    sent = serve(answer)
    found = hyperliquid.settlements()
    assert [(m.ticker, m.series, m.strike, m.result, m.settled_at) for m in found] == [("#113760", "out-BTC-15m", 83984.0, "yes", 1791783008.788)]
    assert [json.loads(r.content).get("user") for r in sent] == [None, "0xe505", "0xf192"]
    assert sent[1].url.host == "rpc.hyperliquid.xyz"


def stream(contract_ids):
    books, marks = [], []
    s = hyperliquid.HyperliquidBookStream(contract_ids, lambda *a: books.append(a))
    s.on_mark = lambda *a: marks.append(a)
    return s, books, marks


def test_the_book_stream_passes_on_outcome_books_and_perp_marks():
    s, books, marks = stream(["#113760", "BTC"])
    assert not s.handle(json.dumps({"channel": "subscriptionResponse", "data": {"method": "subscribe", "subscription": {"type": "l2Book"}}}))
    assert s.handle(json.dumps({"channel": "l2Book", "data": {"coin": "#113760", "time": 1791783143961, "levels": [
        [{"px": "0.1345", "sz": "20.0", "n": 1}, {"px": "0.115", "sz": "52.0", "n": 1}], [{"px": "0.19", "sz": "7.0", "n": 1}]]}}))
    assert books == [("#113760", [[0.1345, 20.0], [0.115, 52.0]], [[0.19, 7.0]], pytest.approx(1791783143.961))]
    assert s.handle(json.dumps({"channel": "activeAssetCtx", "data": {"coin": "BTC", "ctx": {"markPx": "83802.3", "oraclePx": "83840.4",
                                                                                              "midPx": "83788.5", "funding": "0.0000068"}}}))
    assert marks == [("BTC", 83802.3, 83840.4)]
    assert not s.handle(json.dumps({"channel": "l2Book", "data": {"coin": "#999990", "time": 1, "levels": [[], []]}}))   # Let go.
    assert hyperliquid.subscription_frame("unsubscribe", "#113760") == {"method": "unsubscribe",
                                                                       "subscription": {"type": "l2Book", "coin": "#113760"}}
    assert hyperliquid.subscription_frame("subscribe", "ETH")["subscription"] == {"type": "activeAssetCtx", "coin": "ETH"}


def test_the_quote_stream_passes_on_the_perps_best_bid_and_ask_with_its_time():
    seen = []
    s = hyperliquid.HyperliquidQuoteStream(["BTC"], lambda *a: seen.append(a))
    assert s.handle(json.dumps({"channel": "bbo", "data": {"coin": "BTC", "time": 1791783144081, "bbo": [
        {"px": "83788.0", "sz": "0.20106", "n": 2}, {"px": "83789.0", "sz": "8.76776", "n": 24}]}}))
    assert seen == [("BTC", [[83788.0, 0.20106]], [[83789.0, 8.76776]], pytest.approx(1791783144.081))]
    s.handle(json.dumps({"channel": "bbo", "data": {"coin": "BTC", "time": 1791783145000, "bbo": [None, {"px": "83789.0", "sz": "1", "n": 1}]}}))
    assert seen[-1][1] == []                                    # An empty side.


def test_listings_say_when_each_window_was_listed_and_candles_come_oldest_first(serve):
    deployers = [{"venue": "out", "subDeployers": [["registerStandaloneOutcomeFromTemplate", ["0xa"]]]}]
    keywords = [["perp", "BTC"], ["priceDescription", "BTC-USDC perp mark"], ["seconds", "60"], ["threshold", "83474"], ["time", "20261012-1115"]]
    operation = {"registerStandaloneOutcomeFromTemplate": {"id": "binaryPrice", "keywordToValue": keywords}}
    register = {**REGISTER, "action": {**REGISTER["action"], "operation": operation}}

    def answer(request):
        body = json.loads(request.content)
        if body["type"] == "outcomeMeta":
            return httpx.Response(200, json={"outcomes": [], "deployers": deployers})
        if body["type"] == "userDetails":
            return httpx.Response(200, json={"txs": [register, SETTLE]})
        return httpx.Response(200, json=[{"t": 1791783060000, "o": "2", "h": "3", "l": "1", "c": "2.5", "v": "10", "n": 4},
                                         {"t": 1791783000000, "o": "1", "h": "2", "l": "1", "c": "2", "v": "5", "n": 2}])

    serve(answer)
    close = int(datetime(2026, 10, 12, 11, 15, tzinfo=UTC).timestamp())
    assert hyperliquid.listings() == {("out-BTC-15m", close, 83474.0): pytest.approx(1791783014.442)}
    assert [c[0] for c in hyperliquid.minute_candles("BTC", 1791783000, 1791783120)] == [1791783000, 1791783060]
