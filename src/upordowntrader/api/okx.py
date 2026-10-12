"""
OKX public market data: the live best bid and ask of a perpetual swap from its websocket, read without an API key.

OKX's BTC-USDT-SWAP is among the busiest bitcoin markets, so its price may move before those of the exchanges BRTI is
built from, which backtest/lead.py measures. It is priced in tether, so its moves count, not its level. Its tick by tick
best bid and offer channel sends the best level a side whole each time either changes, with OKX's time in milliseconds
and sizes in contracts, see CONTRACT_BTC. OKX closes a connection it sends nothing on for 30 seconds unless pinged. This
is the only file that knows OKX's field names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream

WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
CHANNEL = "bbo-tbt"
CONTRACT_BTC = {"BTC-USDT-SWAP": 0.01}     # Bitcoin a contract, OKX's ctVal on 2026-10-13. Others are left in contracts.


def command_frame(op, instruments):
    """
    The frame that subscribes to, or unsubscribes from, instruments' best levels. op is 'subscribe' or 'unsubscribe'.
    """
    return {"op": op, "args": [{"channel": CHANNEL, "instId": instrument} for instrument in instruments]}


def level(levels, contract):
    """
    A side's best level from OKX's [[price, contracts, 0, orders], ...] as strings, its size in bitcoin, as a one level
    list, empty when the side is.
    """
    return [[float(levels[0][0]), float(levels[0][1]) * contract]] if levels else []


class OkxQuoteStream(BookStream):
    """
    OKX's best bid and ask of each wanted instrument, such as 'BTC-USDT-SWAP', passed on as one level a side, as the book
    streams pass their books, with OKX's time.
    """

    name = "okx perp"
    keepalive_seconds = 25

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await ws.send(json.dumps(command_frame("subscribe", sorted(self.wanted))))

    async def send_command(self, ws, action, instruments):
        await ws.send(json.dumps(command_frame("subscribe" if action == "add" else "unsubscribe", instruments)))

    def keepalive(self):
        return "ping"

    def handle(self, raw):
        if raw == "pong":
            return False
        m = json.loads(raw)
        if m.get("event") == "error":
            self.log(f"okx stream refused: {m.get('msg')}")
            return False
        instrument = (m.get("arg") or {}).get("instId")
        if (m.get("arg") or {}).get("channel") != CHANNEL or instrument not in self.wanted or not m.get("data"):
            return False                                        # A subscription's acknowledgement.
        quote, contract = m["data"][0], CONTRACT_BTC.get(instrument, 1.0)
        self.on_book(instrument, level(quote.get("bids"), contract), level(quote.get("asks"), contract), int(quote["ts"]) / 1000)
        return True
