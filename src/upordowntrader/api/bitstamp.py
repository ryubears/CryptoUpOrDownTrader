"""
Bitstamp public market data: the live best bid and ask from its websocket, read without an API key.

Bitstamp is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on. Its
websocket's order book channel sends the top 100 levels a side, with Bitstamp's time to the microsecond, every time the
book changes, about ten times a second. Bitstamp asks clients to reconnect now and then, before maintenance, with a
bts:request_reconnect event. This is the only file that knows Bitstamp's field names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream, Reconnect

WS_URL = "wss://ws.bitstamp.net"
CHANNEL_PREFIX = "order_book_"      # A pair's channel is this and the pair, order_book_btcusd for 'btcusd'.


def subscription_frame(event, pair):
    """
    The frame that subscribes to, or unsubscribes from, a pair's order book. event is 'bts:subscribe' or 'bts:unsubscribe'.
    """
    return {"event": event, "data": {"channel": CHANNEL_PREFIX + pair}}


class BitstampQuoteStream(BookStream):
    """
    Bitstamp's best bid and ask for each wanted pair, such as 'btcusd', passed on as one level a side, as the book streams
    pass their books, with Bitstamp's time.
    """

    name = "bitstamp"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        for pair in sorted(self.wanted):
            await ws.send(json.dumps(subscription_frame("bts:subscribe", pair)))

    async def send_command(self, ws, action, pairs):
        for pair in pairs:
            await ws.send(json.dumps(subscription_frame("bts:subscribe" if action == "add" else "bts:unsubscribe", pair)))

    def handle(self, raw):
        m = json.loads(raw)
        event = m.get("event")
        if event == "bts:request_reconnect":
            raise Reconnect("asked to reconnect by Bitstamp")
        if event == "bts:error":
            self.log(f"bitstamp stream error: {m.get('data')}")
            return False
        pair = (m.get("channel") or "").removeprefix(CHANNEL_PREFIX)
        if event != "data" or pair not in self.wanted:
            return False                                        # A subscription's acknowledgement.
        data = m.get("data") or {}
        bids = [[float(price), float(size)] for price, size in (data.get("bids") or [])[:1]]
        asks = [[float(price), float(size)] for price, size in (data.get("asks") or [])[:1]]
        micro = data.get("microtimestamp")
        self.on_book(pair, bids, asks, int(micro) / 1e6 if micro else None)
        return True
