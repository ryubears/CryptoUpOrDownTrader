"""
Deribit public market data: the live best bid and ask of a perpetual from its websocket, read without an API key.

Deribit's BTC-PERPETUAL is priced in dollars and settled in bitcoin, and its price may move before those of the
exchanges BRTI is built from, which backtest/lead.py measures. Its servers are in London, a few tens of milliseconds
from the AWS instance, nearer than the other perpetuals'. Its quote channel sends the best level a side whole each time
either changes, with Deribit's time in milliseconds and sizes in dollars. This is the only file that knows Deribit's
field names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream

WS_URL = "wss://www.deribit.com/ws/api/v2"


def command_frame(method, instruments):
    """
    The JSON-RPC frame that subscribes to, or unsubscribes from, instruments' quotes. method is 'subscribe' or
    'unsubscribe'.
    """
    return {"jsonrpc": "2.0", "id": 1, "method": f"public/{method}", "params": {"channels": [f"quote.{i}" for i in instruments]}}


def level(price, amount):
    """
    A side's best level, its size in dollars turned into bitcoin, as a one level list, empty when the side is.
    """
    return [[float(price), float(amount) / float(price)]] if price and amount else []


class DeribitQuoteStream(BookStream):
    """
    Deribit's best bid and ask of each wanted instrument, such as 'BTC-PERPETUAL', passed on as one level a side, as the
    book streams pass their books, with Deribit's time. A subscription to an instrument Deribit does not have is answered
    with no channels, and logged.
    """

    name = "deribit perp"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await ws.send(json.dumps(command_frame("subscribe", sorted(self.wanted))))

    async def send_command(self, ws, action, instruments):
        await ws.send(json.dumps(command_frame("subscribe" if action == "add" else "unsubscribe", instruments)))

    def handle(self, raw):
        m = json.loads(raw)
        if "error" in m or m.get("result") == []:
            self.log(f"deribit stream refused: {m.get('error') or 'no channel subscribed'}")
            return False
        quote = (m.get("params") or {}).get("data") or {}
        if m.get("method") != "subscription" or quote.get("instrument_name") not in self.wanted:
            return False                                        # A subscription's reply.
        self.on_book(quote["instrument_name"], level(quote.get("best_bid_price"), quote.get("best_bid_amount")),
                     level(quote.get("best_ask_price"), quote.get("best_ask_amount")), int(quote["timestamp"]) / 1000)
        return True
