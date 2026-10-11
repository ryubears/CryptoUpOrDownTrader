"""
Kalshi API client, for the Bitcoin up or down windows.

Two jobs. The query half reads a series' window markets from the public
API: when each opens and closes, its strike, and once it settles, how, and
the market's best Yes bid and ask minute by minute. Kalshi moves markets
settled before a cutoff, a few weeks back, to a historical store with its
own markets call.
The streaming half opens one websocket with a signed API key, subscribes
to order book updates, and keeps a live book for each ticker restated from
the Yes side, and to the markets' lifecycle, to say when one is paused,
closed, or decided. Tickers can be added and removed while the connection
runs. This is the only file that knows Kalshi's field names and message
formats. It follows SportsArb's Kalshi client, less the trading.

A window's market carries the exact CF Benchmarks BRTI averages it settles
on: floor_strike, the average of the 60 seconds before it opened, set as it
opens, and expiration_value, the average of the 60 seconds before it
closed, set as it settles.

The websocket needs an API key, the same files as SportsArb's: the key id
in data/kalshi_key_id.txt and the RSA private key in
data/kalshi_private_key.pem. The market reads need none.
"""

import asyncio
import base64
import functools
import json
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from upordowntrader.api.bookstream import BookStream, Reconnect
from upordowntrader.api.http import RateLimiter, get_json
from upordowntrader.common.paths import DATA_DIR
from upordowntrader.common.timeutil import epoch
from upordowntrader.db.models import Market

# The hosts Kalshi dedicates to API traders, in place of the legacy api.elections.kalshi.com, as SportsArb uses.
BASE = "https://external-api.kalshi.com/trade-api/v2"
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"    # WS_URL's path, which the handshake's signature covers.
KEY_ID_FILE = DATA_DIR / "kalshi_key_id.txt"
PRIVATE_KEY_FILE = DATA_DIR / "kalshi_private_key.pem"
LIMITER = RateLimiter(10)       # Kalshi's basic tier allows 20 reads a second, and this keeps to half.
PAGE = 200                      # Markets per page of the markets call.
HISTORICAL_PAGE = 1000          # Markets per page of the historical store's markets call, its most.
CANDLE_LIMIT = 10_000           # Minutes one candlesticks call may span, every minute of its range counted for each market.
MARKETS_BATCH = 50              # Tickers per markets call, see markets().
LIFECYCLE = "market_lifecycle_v2"   # The channel saying when any market opens, pauses, closes, or is decided. It takes no tickers.


# SIGNING

@functools.cache
def credentials():
    """
    The account's key id and RSA private key, read from the data folder once.
    """
    return KEY_ID_FILE.read_text().strip(), serialization.load_pem_private_key(PRIVATE_KEY_FILE.read_bytes(), password=None)


def signed_headers(method, path):
    """
    The three headers that authenticate a request. Kalshi wants the timestamp,
    the method, and the path signed with the account's RSA key. The websocket
    handshake uses the same scheme.
    """
    key_id, key = credentials()
    ts = str(int(time.time() * 1000))
    message = (ts + method + path).encode()
    signature = key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


# QUERY

def number_or_none(value):
    """
    A number Kalshi sends, as a float, or None for a missing or empty one, as a market's not yet set expiration_value is.
    A few markets write thousands with commas, as 77,362.10.
    """
    return float(str(value).replace(",", "")) if value not in (None, "") else None


def market(m):
    """
    A Market from one market of the markets call.
    """
    return Market(
        venue="kalshi",
        ticker=m["ticker"],
        event=m["event_ticker"],
        series=m["event_ticker"].split("-")[0],
        open_time=int(epoch(m["open_time"])),
        close_time=int(epoch(m["close_time"])),
        strike=number_or_none(m.get("floor_strike")),
        settlement=number_or_none(m.get("expiration_value")),
        result=m.get("result") or None,
        settled_at=epoch(m.get("settlement_ts")),
    )


def dollars(prices, name):
    """
    One of a candlestick's prices, such as its yes_bid's close, in dollars. The live store sends close_dollars, the
    historical store close.
    """
    return number_or_none(prices.get(f"{name}_dollars", prices.get(name)))


def paged(params, path="/markets"):
    """
    Follow a markets call's cursor and return every market.
    """
    found, cursor = [], None
    while True:
        data = get_json(f"{BASE}{path}", {**params, **({"cursor": cursor} if cursor else {})}, LIMITER)
        found.extend(data.get("markets") or [])
        cursor = data.get("cursor")
        if not cursor:
            return found


def windows(series, start, end):
    """
    The series' markets closing from start to end, in seconds since 1970, as Markets: the one trading now and those
    listed ahead of it, which have no strike until they open.
    """
    return [market(m) for m in paged({"series_ticker": series, "min_close_ts": int(start), "max_close_ts": int(end), "limit": PAGE})]


def historical_windows(series):
    """
    The series' markets in Kalshi's historical store, those settled before its cutoff, as Markets. The store's markets
    call takes no times, so this reads them all, newest first.
    """
    return [market(m) for m in paged({"series_ticker": series, "limit": HISTORICAL_PAGE}, "/historical/markets")]


def minute(candle):
    """
    One minute of a candlesticks call as (time, bid, ask, price, volume): time when the minute ended, bid and ask the best
    Yes bid and ask as it ended, None for an empty side, which Kalshi sends as a bid of 0 or an ask of 1, price the last
    trade's, None before any, and volume the contracts traded in the minute.
    """
    bid, ask = dollars(candle["yes_bid"], "close"), dollars(candle["yes_ask"], "close")
    return (int(candle["end_period_ts"]), bid if bid else None, ask if ask is not None and ask < 1 else None, dollars(candle["price"], "close"),
            number_or_none(candle.get("volume_fp", candle.get("volume"))))


def minute_quotes(tickers, start, end):
    """
    Each market's minutes from start to end, in seconds since 1970, from the candlesticks call, as {ticker: [minute()]}.
    Kalshi counts every minute of the range for each ticker against CANDLE_LIMIT, so a long range takes few tickers. The
    call returns nothing for a market settled long ago, on 2026-10-12 for any before 2026-08-04, see
    historical_minute_quotes().
    """
    params = {"market_tickers": ",".join(tickers), "start_ts": int(start), "end_ts": int(end), "period_interval": 1}
    return {m["market_ticker"]: [minute(c) for c in m.get("candlesticks") or []]
            for m in get_json(f"{BASE}/markets/candlesticks", params, LIMITER).get("markets") or []}


def historical_minute_quotes(ticker, start, end):
    """
    The minutes of one market in the historical store from start to end, as [minute()], from the store's own
    candlesticks call, which takes one market a call.
    """
    params = {"start_ts": int(start), "end_ts": int(end), "period_interval": 1}
    return [minute(c) for c in get_json(f"{BASE}/historical/markets/{ticker}/candlesticks", params, LIMITER).get("candlesticks") or []]


def markets(tickers):
    """
    The markets of the tickers, as Markets, each once. The call takes a list of tickers, so this costs one call per
    MARKETS_BATCH of them. A ticker the call does not return is left out.
    """
    tickers = sorted(set(tickers))
    found = []
    for i in range(0, len(tickers), MARKETS_BATCH):
        batch = tickers[i:i + MARKETS_BATCH]
        found.extend(market(m) for m in get_json(f"{BASE}/markets", {"tickers": ",".join(batch), "limit": len(batch)}, LIMITER).get("markets") or [])
    return found


# STREAMING

def update_frame(message_id, sid, tickers, action):
    """
    The command that adds or removes tickers on a live subscription. action is
    'add_markets' or 'delete_markets'.
    """
    return {"id": message_id, "cmd": "update_subscription",
            "params": {"sids": [sid], "market_tickers": tickers, "action": action}}


class KalshiBookStream(BookStream):
    """
    Kalshi's order book channel over a signed connection, as SportsArb
    streams it. Books are given from the Yes side, best first. The
    subscription asks for use_yes_price, so a resting No order arrives
    already priced as the Yes ask it is. Every message counts as data
    because the feed has no keepalive replies. A skipped sequence number,
    each subscription counting its own, forces a reconnect. Each side's best
    levels are kept with its book and sorted again only when a delta reaches
    them, so a delta deeper in the book costs no sort.

    The same connection follows every market's lifecycle, see LIFECYCLE,
    and says when one of the wanted markets is paused, closed, or decided,
    and when a pause ends, see lifecycle().
    """

    name = "kalshi"

    def reset(self):
        self.sid = None                     # The order book subscription's id, needed for update commands.
        self.last_seq = {}                  # Each subscription's last sequence number, by its id.
        self.subscribed = asyncio.Event()   # Set once the order book subscription's acknowledgement arrives.
        self.message_id = 3

    def connect(self):
        return self.open_connection(WS_URL, signed_headers("GET", WS_PATH))

    async def subscribe(self, ws):
        await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                  "params": {"channels": ["orderbook_delta"], "market_tickers": sorted(self.wanted), "use_yes_price": True}}))
        await ws.send(json.dumps({"id": 2, "cmd": "subscribe", "params": {"channels": [LIFECYCLE]}}))

    def lifecycle(self, body):
        """
        Note what one lifecycle message says of a wanted market: paused, or
        trading again, when it is deactivated or activated, closed when its
        close moves to now or earlier, trading again when a close moves
        later, and decided when it is determined or settled, which no later
        message changes.
        """
        ticker, event = body.get("market_ticker"), body.get("event_type")
        if ticker not in self.wanted or self.states.get(ticker) == "decided":
            return
        if event == "deactivated":
            self.set_state(ticker, "paused" if body.get("is_deactivated") else None)
        elif event == "activated":
            self.set_state(ticker, None)
        elif event == "close_date_updated" and body.get("close_ts") is not None:
            if body["close_ts"] <= time.time():
                self.set_state(ticker, "closed")
            elif self.states.get(ticker) == "closed":
                self.set_state(ticker, None)
        elif event in ("determined", "settled"):
            self.set_state(ticker, "decided")

    async def send_command(self, ws, action, tickers):
        await self.subscribed.wait()
        kalshi_action = "add_markets" if action == "add" else "delete_markets"
        await ws.send(json.dumps(update_frame(self.message_id, self.sid, tickers, kalshi_action)))
        self.message_id += 1

    def best(self, book, side):
        """
        The best self.depth levels of one side of a book as [price, size]
        lists: the yes bids highest first, the asks, on the no side, lowest first.
        """
        return [[p, s] for p, s in sorted(book[side].items(), reverse=side == "yes")[:self.depth]]

    def apply(self, m):
        """
        Update the local books from one feed message and pass the ticker's
        best levels on, with the matching engine's time of a delta. A delta
        deeper than the levels shown leaves them as they were.
        """
        kind, body = m.get("type"), m.get("msg") or {}
        ticker = body.get("market_ticker")
        if kind == "subscribed":
            if body.get("channel") != LIFECYCLE:
                self.sid = body.get("sid")
                self.subscribed.set()
            return
        if kind == "error":
            self.log(f"kalshi stream error {body}")
            return
        if kind == LIFECYCLE:
            self.lifecycle(body)
            return
        if kind == "orderbook_snapshot" and ticker in self.wanted:
            book = self.books[ticker] = {
                "yes": {float(p): float(s) for p, s in body.get("yes_dollars_fp") or [] if float(s) > 0},
                "no": {float(p): float(s) for p, s in body.get("no_dollars_fp") or [] if float(s) > 0},
            }
            book["shown"] = {side: self.best(book, side) for side in ("yes", "no")}    # The levels last passed on.
        elif kind == "orderbook_delta" and ticker in self.books:
            book = self.books[ticker]
            side = body["side"]
            levels = book[side]
            price = float(body["price_dollars"])
            # Round to hundredths of a contract so summing many deltas does not leave floating point residue.
            size = round(levels.get(price, 0.0) + float(body["delta_fp"]), 2)
            if size > 0:
                levels[price] = size
            else:
                levels.pop(price, None)
            shown = book["shown"][side]
            if len(shown) < self.depth or (price >= shown[-1][0] if side == "yes" else price <= shown[-1][0]):
                book["shown"][side] = self.best(book, side)
        else:
            return
        sent = body.get("ts_ms") if kind == "orderbook_delta" else None      # A snapshot carries no time.
        self.on_book(ticker, book["shown"]["yes"], book["shown"]["no"], sent / 1000 if sent else None)

    def handle(self, raw):
        m = json.loads(raw)
        seq, sid = m.get("seq"), m.get("sid")
        if seq is not None:
            last = self.last_seq.get(sid)
            if last is not None and seq != last + 1:
                raise Reconnect(f"skipped from seq {last} to {seq}")
            self.last_seq[sid] = seq
        self.apply(m)
        return True
