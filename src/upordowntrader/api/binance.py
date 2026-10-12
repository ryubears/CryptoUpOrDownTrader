"""
Binance USD-M futures public market data: the live best bid and ask of a perpetual from its websocket, read without an
API key.

Binance's BTCUSDT perpetual trades more bitcoin than any other market, so its price may move before those of the
exchanges BRTI is built from, which backtest/lead.py measures. It is priced in tether, which trades within a few
hundredths of a percent of the dollar, so its moves count, not its level. Its book ticker stream sends the best level a
side each time either changes, with the time of the change in milliseconds. Binance refuses its spot market's websocket
to US addresses, the AWS instance's among them, but not this one. This is the only file that knows Binance's field names
and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream

WS_URL = "wss://fstream.binance.com/ws"


def command_frame(method, symbols):
    """
    The frame that subscribes to, or unsubscribes from, symbols' book tickers. method is 'SUBSCRIBE' or 'UNSUBSCRIBE'.
    """
    return {"method": method, "params": [f"{symbol.lower()}@bookTicker" for symbol in symbols], "id": 1}


class BinanceQuoteStream(BookStream):
    """
    Binance's best bid and ask of each wanted perpetual, such as 'BTCUSDT', passed on as one level a side, as the book
    streams pass their books, with Binance's time. Binance closes each connection after 24 hours, and the loop connects
    again.
    """

    name = "binance perp"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await ws.send(json.dumps(command_frame("SUBSCRIBE", sorted(self.wanted))))

    async def send_command(self, ws, action, symbols):
        await ws.send(json.dumps(command_frame("SUBSCRIBE" if action == "add" else "UNSUBSCRIBE", symbols)))

    def handle(self, raw):
        m = json.loads(raw)
        if "error" in m:
            self.log(f"binance stream refused: {m['error']}")
            return False
        if m.get("e") != "bookTicker" or m.get("s") not in self.wanted:
            return False                                        # A subscription's acknowledgement.
        self.on_book(m["s"], [[float(m["b"]), float(m["B"])]], [[float(m["a"]), float(m["A"])]], int(m["T"]) / 1000)
        return True
