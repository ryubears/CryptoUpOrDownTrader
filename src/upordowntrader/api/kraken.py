"""
Kraken public market data: the live best bid and ask from its websocket, read without an API key.

Kraken is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on. Its version
2 websocket's ticker channel, asked to trigger on the best bid and offer, sends the best bid and ask, with their sizes and
a timestamp, every time either changes, and a heartbeat every second besides. The timestamp is not always the quote's
time: on 2026-10-11 a fifth of the quotes carried the stamp of an earlier one, 1 in 20 over a second stale, so it is
passed on as Kraken's time but read with care, see backtest/replay.py. This is the only file that knows Kraken's field
names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream
from upordowntrader.common.timeutil import epoch

WS_URL = "wss://ws.kraken.com/v2"


def ticker_frame(method, symbols):
    """
    The frame that subscribes to, or unsubscribes from, the symbols' best bid and offer. method is 'subscribe' or
    'unsubscribe'.
    """
    return {"method": method, "params": {"channel": "ticker", "symbol": symbols, "event_trigger": "bbo"}}


class KrakenQuoteStream(BookStream):
    """
    Kraken's best bid and ask for each wanted symbol, such as 'BTC/USD', passed on as one level a side, as the book streams
    pass their books, with Kraken's time. Heartbeats and status messages keep the connection open but do not count as
    data, so a feed sending only them goes stale and is rebuilt.
    """

    name = "kraken"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await ws.send(json.dumps(ticker_frame("subscribe", sorted(self.wanted))))

    async def send_command(self, ws, action, symbols):
        await ws.send(json.dumps(ticker_frame("subscribe" if action == "add" else "unsubscribe", symbols)))

    def handle(self, raw):
        m = json.loads(raw)
        if m.get("method") in ("subscribe", "unsubscribe"):
            if not m.get("success"):
                self.log(f"kraken stream refused {m.get('method')}: {m.get('error')}")
            return False
        if m.get("channel") != "ticker":
            return False                                        # A heartbeat or the system's status.
        for quote in m.get("data") or []:
            if quote.get("symbol") in self.wanted:
                bids = [[float(quote["bid"]), float(quote["bid_qty"])]] if quote.get("bid") else []
                asks = [[float(quote["ask"]), float(quote["ask_qty"])]] if quote.get("ask") else []
                self.on_book(quote["symbol"], bids, asks, epoch(quote.get("timestamp")))
        return True
