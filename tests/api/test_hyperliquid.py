"""
Tests for Hyperliquid's up or down windows read from its market list, their settlements read from its explorer, its
streams, fed messages shaped as Hyperliquid sends them, and its orders, signed as its own SDK signs them.
"""

import eth_account
import httpx
import json
import pytest
from datetime import UTC, datetime
from upordowntrader.api import hyperliquid

TEST_KEY = "0x0123456789012345678901234567890123456789012345678901234567890123"    # A throwaway key, Hyperliquid's SDK's tests'.
CLOID = "0x00000000000000000000000000000001"
NONCE = 1792070000000

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
    assert hyperliquid.subscription_frames("unsubscribe", "#113760") == [
        {"method": "unsubscribe", "subscription": {"type": "l2Book", "coin": "#113760"}},
        {"method": "unsubscribe", "subscription": {"type": "bbo", "coin": "#113760"}}]
    assert [f["subscription"] for f in hyperliquid.subscription_frames("subscribe", "ETH")] == [{"type": "activeAssetCtx", "coin": "ETH"}]


def test_the_best_levels_come_between_snapshots_laid_over_the_newest_one():
    s, books, _ = stream(["#113760"])
    s.handle(json.dumps({"channel": "l2Book", "data": {"coin": "#113760", "time": 1000_000, "levels": [
        [{"px": "0.50", "sz": "10", "n": 1}, {"px": "0.49", "sz": "20", "n": 1}, {"px": "0.47", "sz": "30", "n": 1}],
        [{"px": "0.53", "sz": "5", "n": 1}, {"px": "0.55", "sz": "15", "n": 1}]]}}))
    # The bid rose to 0.51 and the ask's best was taken, both after the snapshot: the snapshot's levels behind them stay.
    assert s.handle(json.dumps({"channel": "bbo", "data": {"coin": "#113760", "time": 1002_000, "bbo": [
        {"px": "0.51", "sz": "4", "n": 1}, {"px": "0.55", "sz": "15", "n": 1}]}}))
    assert books[-1] == ("#113760", [[0.51, 4.0], [0.50, 10.0], [0.49, 20.0], [0.47, 30.0]], [[0.55, 15.0]], 1002.0)
    # The bid fell to 0.48: the snapshot's 0.50 and 0.49, gone by then, are dropped.
    s.handle(json.dumps({"channel": "bbo", "data": {"coin": "#113760", "time": 1003_000, "bbo": [{"px": "0.48", "sz": "2", "n": 1}, None]}}))
    assert books[-1] == ("#113760", [[0.48, 2.0], [0.47, 30.0]], [], 1003.0)                  # No ask left at all.
    # A newer snapshot is taken whole, as the best levels it has are newer than those of the bbo.
    s.handle(json.dumps({"channel": "l2Book", "data": {"coin": "#113760", "time": 1005_000, "levels": [
        [{"px": "0.46", "sz": "1", "n": 1}], [{"px": "0.60", "sz": "2", "n": 1}]]}}))
    assert books[-1] == ("#113760", [[0.46, 1.0]], [[0.60, 2.0]], 1005.0)


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


# ORDERS: the signatures below are what Hyperliquid's own SDK, hyperliquid-python-sdk 0.24.0, makes of the same actions
# with TEST_KEY: sign_l1_action(wallet, action, None, NONCE, expires_after, True).

def test_an_order_on_an_outcome_side_is_written_and_signed_as_hyperliquids_sdk_does():
    account = eth_account.Account.from_key(TEST_KEY)
    action = hyperliquid.order_action("#1230", True, 0.54, 10.0, "Ioc", CLOID)
    assert action == {"type": "order", "orders": [{"a": 100_001_230, "b": True, "p": "0.54", "s": "10", "r": False,
                                                   "t": {"limit": {"tif": "Ioc"}}, "c": CLOID}], "grouping": "na"}
    assert hyperliquid.signature(action, NONCE, account) == {"r": "0x3d7dcd0ba2e6c1a9868e866825883241f91fe1e59a23454e4c22689d4eba8476",
                                                             "s": "0x3c163e50e659ffb9f41e0c2e815305f1848a28d5ad74728a71a9bb08df44e6b0", "v": 27}
    assert hyperliquid.signature(action, NONCE, account, NONCE + 5000) == {
        "r": "0xf0e24feeb5f722899b8537dec483d8ebf18f46cb14d56aebe5493ae03f2323d3",
        "s": "0x36667a01097626d727cb2cb3b9d11eabbe26d8507ae67935c2645853ba407f4e", "v": 27}


def test_a_cancel_by_our_id_on_the_no_side_is_signed_as_hyperliquids_sdk_does():
    account = eth_account.Account.from_key(TEST_KEY)
    action = hyperliquid.cancel_action(hyperliquid.side_coin("#1230", "no"), CLOID)
    assert action == {"type": "cancelByCloid", "cancels": [{"asset": 100_001_231, "cloid": CLOID}]}
    assert hyperliquid.signature(action, NONCE + 1, account) == {"r": "0xffa02e468ba996f0dfda44e00cc77cabe1e1d57a2da86aa09728ebdafea26e82",
                                                                 "s": "0x195b6f187c481e3b262e7dd48a22df4d342a618b05a345fa91650122ed2c9", "v": 27}


def test_prices_and_sizes_are_written_without_trailing_zeros_and_prices_to_five_figures():
    assert [hyperliquid.wire(x) for x in (10.0, 0.5, 0.54073, 3.0)] == ["10", "0.5", "0.54073", "3"]
    assert hyperliquid.price_wire(0.540734) == "0.54073" and hyperliquid.price_wire(0.05) == "0.05"
    with pytest.raises(ValueError):
        hyperliquid.wire(0.123456789)


def test_nonces_never_repeat_within_a_millisecond():
    a, b = hyperliquid.nonce(), hyperliquid.nonce()
    assert b > a


def test_an_order_goes_out_once_signed_and_its_answer_is_read(serve):
    account = eth_account.Account.from_key(TEST_KEY)
    filled = {"filled": {"totalSz": "7", "avgPx": "0.531", "oid": 77}}
    answers = iter([{"status": "ok", "response": {"type": "order", "data": {"statuses": [filled]}}},
                    {"status": "ok", "response": {"type": "order", "data": {"statuses": [
                        {"error": "Order could not immediately match against any resting orders. asset=100001230"}]}}},
                    {"status": "err", "response": "User or API Wallet 0x0 does not exist."}])
    sent = serve(lambda request: httpx.Response(200, json=next(answers)))
    first = hyperliquid.place("#1230", True, 0.54, 10.0, "Ioc", CLOID, account)
    body = json.loads(sent[0].content)
    assert str(sent[0].url) == hyperliquid.EXCHANGE_URL and body["action"]["orders"][0]["c"] == CLOID and body["vaultAddress"] is None
    assert body["signature"] == hyperliquid.signature(body["action"], body["nonce"], account)
    assert hyperliquid.answer(first, 10.0) == ("partial", 7.0, 0.531, 77, None)
    assert hyperliquid.answer(hyperliquid.place("#1230", True, 0.54, 10.0, account=account), 10.0)[0] == "unfilled"
    assert hyperliquid.answer(hyperliquid.place("#1230", True, 0.54, 10.0, account=account), 10.0)[:2] == ("rejected", 0.0)
    assert len(sent) == 3                                       # Each sent once, an error not retried.
    assert hyperliquid.answer("<html>502</html>", 10.0)[0] == "error"                  # An answer that cannot be read.
