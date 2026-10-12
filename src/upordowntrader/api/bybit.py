"""
Bybit public market data: the live best bid and ask of a USDT perpetual from its websocket, read without an API key.

Bybit's BTCUSDT perpetual is among the busiest bitcoin markets, so its price may move before those of the exchanges BRTI
is built from, which backtest/lead.py measures. It is priced in tether, so its moves count, not its level. Its level 1
order book topic sends the best level a side whole, each time either changes and every 3 seconds without one, with the
matching engine's time in milliseconds. Bybit asks for a ping every 20 seconds. This is the only file that knows
Bybit's field names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream

WS_URL = "wss://stream.bybit.com/v5/public/linear"


def topic(symbol):
    """
    The level 1 order book topic of a symbol, such as 'BTCUSDT'.
    """
    return f"orderbook.1.{symbol}"


def level(levels):
    """
    A side's best level from Bybit's [[price, size], ...] as strings, as a one level list, empty when the side is.
    """
    return [[float(levels[0][0]), float(levels[0][1])]] if levels and float(levels[0][1]) > 0 else []


class BybitQuoteStream(BookStream):
    """
    Bybit's best bid and ask of each wanted perpetual, such as 'BTCUSDT', passed on as one level a side, as the book
    streams pass their books, with the matching engine's time. Level 1 sent only snapshots on 2026-10-13; a delta, should
    one come, changes only the sides it names.
    """

    name = "bybit perp"
    keepalive_seconds = 20

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await ws.send(json.dumps({"op": "subscribe", "args": [topic(symbol) for symbol in sorted(self.wanted)]}))

    async def send_command(self, ws, action, symbols):
        await ws.send(json.dumps({"op": "subscribe" if action == "add" else "unsubscribe", "args": [topic(symbol) for symbol in symbols]}))

    def keepalive(self):
        return json.dumps({"op": "ping"})

    def handle(self, raw):
        m = json.loads(raw)
        if m.get("success") is False:
            self.log(f"bybit stream refused {m.get('op')}: {m.get('ret_msg')}")
            return False
        data = m.get("data") or {}
        symbol = data.get("s")
        if not str(m.get("topic", "")).startswith("orderbook.1.") or symbol not in self.wanted:
            return False                                        # A subscription's acknowledgement, or a ping's.
        if m.get("type") == "snapshot" or symbol not in self.books:
            self.books[symbol] = {"b": level(data.get("b")), "a": level(data.get("a"))}
        else:
            for side in ("b", "a"):
                if data.get(side):
                    self.books[symbol][side] = level(data[side])
        book = self.books[symbol]
        self.on_book(symbol, book["b"], book["a"], int(m.get("cts") or m["ts"]) / 1000)
        return True
