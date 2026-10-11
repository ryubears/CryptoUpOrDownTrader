"""
Coinbase Exchange public market data, read without an API key: candles from the REST API, and the live order book's
best bid and ask from the websocket.

Coinbase is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on, so its
BTC-USD price is a close stand-in for the settlement price. Its shortest candle is 1 minute; the websocket gives every
change of the book, in batches every 50 ms.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from upordowntrader.api.bookstream import BookStream
from upordowntrader.api.http import RateLimiter, get_json, utc
from upordowntrader.common.timeutil import epoch

API = "https://api.exchange.coinbase.com"
WS_URL = "wss://ws-feed.exchange.coinbase.com"
BOOK_CHANNEL = "level2_batch"   # The whole book, then each change, batched every 50 ms. Public, unlike the unbatched level2.
PAGE = 300              # The most candle lengths one call may span.
GRANULARITIES = (60, 300, 900, 3600, 21600, 86400)  # The candle lengths Coinbase offers, in seconds.
LIMITER = RateLimiter(5)    # Coinbase allows 10 public calls a second per IP, and this keeps to half.
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass
class Candle:
    """
    One candle. Volume is in the base currency, BTC for BTC-USD.
    """
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_candle(row):
    """
    A Candle from Coinbase's row: time, low, high, open, close, volume.
    """
    time, low, high, open_, close, volume = row
    return Candle(datetime.fromtimestamp(time, tz=UTC), float(open_), float(high), float(low), float(close), float(volume))


def check_granularity(granularity):
    """
    Raise ValueError for a candle length Coinbase does not offer.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity}, expected one of {GRANULARITIES}")


def ceil_to(dt, step):
    """
    The first multiple of step since 1970 at or after dt.
    """
    return EPOCH - ((EPOCH - dt) // step) * step


def fetch_candles(product_id, granularity, start, end):
    """
    One call: the candles opening from start to end, both included, oldest first. The range may span at most PAGE
    candles, and a period with no trades has no candle.
    """
    check_granularity(granularity)
    start, end = utc(start), utc(end)
    if end < start:
        raise ValueError("end is before start")
    if end - start > timedelta(seconds=granularity * PAGE):
        raise ValueError(f"range spans more than {PAGE} candles")
    params = {"granularity": granularity, "start": int(start.timestamp()), "end": int(end.timestamp())}
    rows = get_json(f"{API}/products/{product_id}/candles", params, LIMITER)
    return sorted((parse_candle(row) for row in rows), key=lambda c: c.open_time)     # Coinbase sends the newest first.


def candles(product_id, granularity, start, end):
    """
    Every candle opening from start up to but not including end, oldest first, PAGE to a call. A period with no trades
    has no candle, so check for gaps rather than assume every period is there.
    """
    check_granularity(granularity)
    step = timedelta(seconds=granularity)
    start, end = ceil_to(utc(start), step), utc(end)    # On a candle boundary, so no candle falls between two calls.
    while start < end:
        window_end = min(start + step * PAGE, end)
        batch = fetch_candles(product_id, granularity, start, max(start, window_end - step))     # Coinbase's end is included.
        yield from (c for c in batch if start <= c.open_time < window_end)
        start = window_end


# STREAMING

class CoinbaseQuoteStream(BookStream):
    """
    The best bid and ask of Coinbase's order books, from the batched level 2 channel. The first message of each product
    is the whole book, every level, and each one after it the levels that changed, with Coinbase's time. The book is
    kept whole, with its best price a side remembered, so a change away from the top costs nothing and only the best
    level emptying sends for a new best, see apply(). Each change passes the best level a side on, as the book streams
    do, with Coinbase's time for it.
    """

    name = "coinbase"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await ws.send(json.dumps({"type": "subscribe", "product_ids": sorted(self.wanted), "channels": [BOOK_CHANNEL]}))

    async def send_command(self, ws, action, product_ids):
        await ws.send(json.dumps({"type": "subscribe" if action == "add" else "unsubscribe", "product_ids": product_ids,
                                  "channels": [BOOK_CHANNEL]}))

    def apply(self, book, side, price, size):
        """
        Set one level of a side, "bids" or "asks", and keep the side's best price: a new level better than it becomes it,
        and the best level emptying makes the best of the rest it.
        """
        levels, best = book[side], book["best"]
        if size > 0:
            levels[price] = size
            if best[side] is None or (price > best[side] if side == "bids" else price < best[side]):
                best[side] = price
        else:
            levels.pop(price, None)
            if price == best[side]:
                best[side] = (max if side == "bids" else min)(levels, default=None)

    def top(self, book, side):
        """
        The best level of a side as a one level list, empty when the side is.
        """
        price = book["best"][side]
        return [[price, book[side][price]]] if price is not None else []

    def handle(self, raw):
        m = json.loads(raw)
        kind, product = m.get("type"), m.get("product_id")
        if kind == "error":
            self.log(f"coinbase stream error {m.get('message')}: {m.get('reason')}")
            return False
        if kind == "snapshot" and product in self.wanted:
            book = self.books[product] = {"bids": {}, "asks": {}, "best": {"bids": None, "asks": None}}
            for side in ("bids", "asks"):
                for price, size in m.get(side) or []:
                    self.apply(book, side, float(price), float(size))
            sent = None                                         # A snapshot carries no time.
        elif kind == "l2update" and product in self.books:
            book = self.books[product]
            for side, price, size in m.get("changes") or []:
                self.apply(book, "bids" if side == "buy" else "asks", float(price), float(size))
            sent = epoch(m.get("time"))
        else:
            return False                                        # The subscription's acknowledgement.
        self.on_book(product, self.top(book, "bids"), self.top(book, "asks"), sent)
        return True

