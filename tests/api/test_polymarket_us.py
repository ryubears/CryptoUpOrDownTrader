"""
Tests for the Polymarket US client: signing, reading the windows' markets, and the book stream, with no network.
"""

import asyncio
import base64
import httpx
import json
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from upordowntrader.api import bookstream, polymarket_us

OPEN = 1791896400       # 2026-10-13 13:00 UTC.

# A window's market as the gateway gives it, resolved Up, and the next, trading. Cut down from real answers.
RESOLVED = {"slug": "cpc-btc-updown-15m-2026-10-13-1245z", "endDate": "2026-10-13T13:00:10Z", "status": "MARKET_STATUS_RESOLVED",
            "marketSides": [{"description": "Yes", "price": "1", "long": True}, {"description": "No", "price": "0", "long": False}],
            "assetPriceTerms": {"windowStart": "2026-10-13T12:45:00Z", "windowEnd": "2026-10-13T13:00:00Z",
                                "priceToBeat": {"value": "83161.39", "currency": "USD"}, "settlementPrice": {"value": "83249.07", "currency": "USD"}}}
TRADING = {"slug": "cpc-btc-updown-15m-2026-10-13-1300z", "endDate": "2026-10-13T13:45:00Z", "status": "MARKET_STATUS_OPEN",
           "marketSides": [{"description": "Yes", "price": "0.4700", "long": True}, {"description": "No", "price": "0.54", "long": False}],
           "assetPriceTerms": {"windowStart": "2026-10-13T13:00:00Z", "windowEnd": "2026-10-13T13:15:00Z",
                               "priceToBeat": {"value": "83249.07", "currency": "USD"}, "settlementPrice": None}}


def test_headers_are_signed_with_the_accounts_key(monkeypatch):
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(polymarket_us, "credentials", lambda: ("key-id", key))
    headers = polymarket_us.signed_headers("GET", polymarket_us.WS_PATH)
    assert headers["X-PM-Access-Key"] == "key-id"
    message = (headers["X-PM-Timestamp"] + "GET" + polymarket_us.WS_PATH).encode()
    key.public_key().verify(base64.b64decode(headers["X-PM-Signature"]), message)      # Raises when it does not match.


def test_a_window_is_named_for_its_open():
    assert polymarket_us.slug(OPEN) == "cpc-btc-updown-15m-2026-10-13-1300z"


def test_a_market_carries_its_times_strike_and_once_resolved_how_it_settled():
    resolved, trading = polymarket_us.market(RESOLVED), polymarket_us.market(TRADING)
    assert (resolved.venue, resolved.ticker, resolved.open_time, resolved.close_time) == ("polymarket_us", RESOLVED["slug"], OPEN - 900, OPEN)
    assert (resolved.strike, resolved.settlement, resolved.result, resolved.settled_at) == (83161.39, 83249.07, "yes", OPEN + 10)
    assert (trading.strike, trading.settlement, trading.result, trading.settled_at) == (83249.07, None, None, None)
    down = {**RESOLVED, "marketSides": [{"price": "0", "long": True}, {"price": "1", "long": False}]}
    assert polymarket_us.market(down).result == "no"


def test_windows_reads_each_window_closing_in_the_range_and_skips_those_not_listed(serve):
    sent = serve(lambda request: httpx.Response(200, json={"market": TRADING}) if request.url.path.endswith("1300z")
                 else httpx.Response(404, json={}))
    found = polymarket_us.windows(OPEN + 1, OPEN + 1800)       # Closing from 13:00:01 to 13:30: the 13:00 and 13:15 windows.
    assert [request.url.path.rsplit("/", 1)[1] for request in sent] == ["cpc-btc-updown-15m-2026-10-13-1300z", "cpc-btc-updown-15m-2026-10-13-1315z"]
    assert [m.ticker for m in found] == [TRADING["slug"]]


def level(price, contracts):
    """
    A price level as the markets websocket sends it.
    """
    return {"px": {"value": f"{price:.4f}", "currency": "USD"}, "qty": f"{contracts:.4f}"}


def book_message(name, bids, offers, at="2026-10-13T13:00:01.123456789Z", state="MARKET_STATE_OPEN"):
    """
    A book as the markets websocket sends it, the levels as (price, contracts) pairs, in any order.
    """
    return json.dumps({"requestId": name, "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA",
                       "marketData": {"marketSlug": name, "bids": [level(*b) for b in bids], "offers": [level(*o) for o in offers],
                                      "state": state, "transactTime": at}})


def stream(names=("M",), depth=5):
    seen = []
    s = polymarket_us.PolymarketUSBookStream(list(names), lambda name, bids, asks, sent: seen.append((name, bids, asks, sent)))
    s.depth = depth
    s.reset()
    return s, seen


def test_each_book_replaces_the_last_best_first_and_the_first_on_a_connection_says_no_time():
    s, seen = stream(depth=2)
    assert s.handle(book_message("M", [(0.46, 5), (0.47, 3), (0.45, 0)], [(0.50, 2), (0.49, 4), (0.52, 1)]))
    assert s.handle(book_message("M", [(0.48, 1)], [], at="2026-10-13T13:00:02.5Z"))
    assert seen[0] == ("M", [[0.47, 3.0], [0.46, 5.0]], [[0.49, 4.0], [0.50, 2.0]], None)      # A snapshot, its time maybe long ago.
    assert seen[1] == ("M", [[0.48, 1.0]], [], OPEN + 2.5)


def test_a_market_not_trading_is_said_so_and_messages_without_a_book_are_not_data():
    s, seen = stream()
    states = []
    s.on_state = lambda name, why: states.append((name, why))
    s.handle(book_message("M", [], [], state="MARKET_STATE_SUSPENDED"))
    s.handle(book_message("M", [], []))
    assert states == [("M", "suspended"), ("M", None)]
    assert not s.handle(json.dumps({"heartbeat": {}})) and not s.handle(json.dumps({"requestId": "M", "unsubscribed": True}))
    assert s.handle(book_message("OTHER", [(0.5, 1)], [])) and len(seen) == 2       # A market not wanted is dropped.


def test_a_subscription_refused_as_one_too_many_rebuilds_the_connection():
    s, _ = stream()
    with pytest.raises(bookstream.Reconnect):
        s.handle(json.dumps({"requestId": "M", "error": "max subscriptions per connection reached"}))


def test_each_market_is_its_own_subscription_and_a_dropped_one_is_ended():
    s, _ = stream(["B", "A"])
    sent = []

    class Socket:
        async def send(self, frame):
            sent.append(json.loads(frame))

    asyncio.run(s.subscribe(Socket()))
    asyncio.run(s.send_command(Socket(), "add", ["C"]))
    asyncio.run(s.send_command(Socket(), "remove", ["A"]))
    assert [f["subscribe"]["marketSlugs"] for f in sent[:3]] == [["A"], ["B"], ["C"]]
    assert sent[0]["subscribe"]["requestId"] == "A" and sent[0]["subscribe"]["subscriptionType"] == "SUBSCRIPTION_TYPE_MARKET_DATA"
    assert sent[3] == {"unsubscribe": {"requestId": "A"}}
