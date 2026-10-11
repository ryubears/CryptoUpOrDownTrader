"""
Bullish public market data: the live best bid and ask from its websocket, read without an API key.

Bullish is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on, since
2024-12-30. Its multi order book websocket, subscribed to a symbol's level 1 topic, sends the best level a side as a
snapshot, then again each time either changes, with Bullish's time in milliseconds. Its sequence numbers count every
change of the whole book, so those of the best level skip and cannot show a lost message. This is the only file that
knows Bullish's field names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream

WS_URL = "wss://api.exchange.bullish.com/trading-api/v1/market-data/orderbook"
TOPIC = "l1Orderbook"


def command_frame(method, symbol):
    """
    The frame that subscribes to, or unsubscribes from, a symbol's best level. method is 'subscribe' or 'unsubscribe'.
    """
    return {"jsonrpc": "2.0", "type": "command", "method": method, "id": symbol, "params": {"topic": TOPIC, "symbol": symbol}}


def level(pair):
    """
    A side's best level, [price, size] as strings, as a one level list, empty when the side is.
    """
    return [[float(pair[0]), float(pair[1])]] if pair and len(pair) >= 2 else []


class BullishQuoteStream(BookStream):
    """
    Bullish's best bid and ask for each wanted symbol, such as 'BTCUSD', passed on as one level a side, as the book streams
    pass their books, with Bullish's time.
    """

    name = "bullish"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        for symbol in sorted(self.wanted):
            await ws.send(json.dumps(command_frame("subscribe", symbol)))

    async def send_command(self, ws, action, symbols):
        for symbol in symbols:
            await ws.send(json.dumps(command_frame("subscribe" if action == "add" else "unsubscribe", symbol)))

    def handle(self, raw):
        m = json.loads(raw)
        if "error" in m or str((m.get("result") or {}).get("responseCode", 200)) != "200":
            self.log(f"bullish stream refused {m.get('id')}: {m.get('error') or m.get('result')}")
            return False
        data = m.get("data") or {}
        if m.get("dataType") != "V1TALevel1" or data.get("symbol") not in self.wanted:
            return False                                        # A subscription's acknowledgement, or a keepalive's.
        ms = data.get("timestamp")
        self.on_book(data["symbol"], level(data.get("bid")), level(data.get("ask")), int(ms) / 1000 if ms else None)
        return True
