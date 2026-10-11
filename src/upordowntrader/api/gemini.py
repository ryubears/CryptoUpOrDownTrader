"""
Gemini public market data: the live best bid and ask from its websocket, read without an API key.

Gemini is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on. Its version
1 market data websocket, asked for the top of book only, sends both sides' best levels first, then each change of either,
with Gemini's time in milliseconds, and a heartbeat every 5 seconds. Every message carries a sequence number, so a lost
one shows. The symbol is part of the address, so a connection carries one. This is the only file that knows Gemini's
field names and message formats.
"""

import json
from upordowntrader.api.bookstream import BookStream, Reconnect

WS_URL = "wss://api.gemini.com/v1/marketdata/{symbol}?top_of_book=true&trades=false&heartbeat=true"
SIDES = {"bid": "bids", "ask": "asks"}


class GeminiQuoteStream(BookStream):
    """
    Gemini's best bid and ask of one symbol, such as 'BTCUSD', passed on as one level a side, as the book streams pass their
    books, with Gemini's time. A skipped sequence number means a change was lost, so the connection is rebuilt.
    """

    name = "gemini"

    def reset(self):
        self.top = {"bids": [], "asks": []}     # Each side's best level, as a one level list, empty when the side is.
        self.sequence = None                    # The last message's sequence number on this connection.

    def connect(self):
        return self.open_connection(WS_URL.format(symbol=min(self.wanted)))

    async def subscribe(self, ws):
        pass                                    # The address subscribes.

    async def send_command(self, ws, action, symbols):
        """
        The address names the symbol, so another symbol takes another connection: close this one, and the loop connects
        again for what is wanted.
        """
        await ws.close()

    def handle(self, raw):
        m = json.loads(raw)
        sequence = m.get("socket_sequence")
        if sequence is not None:
            if self.sequence is not None and sequence != self.sequence + 1:
                raise Reconnect(f"skipped from sequence {self.sequence} to {sequence}")
            self.sequence = sequence
        if m.get("type") != "update":
            return False                                        # A heartbeat.
        changed = False
        for event in m.get("events") or []:
            if event.get("type") == "change" and event.get("side") in SIDES:
                size = float(event["remaining"])
                self.top[SIDES[event["side"]]] = [[float(event["price"]), size]] if size > 0 else []
                changed = True
        if changed:
            ms = m.get("timestampms")
            self.on_book(min(self.wanted), self.top["bids"], self.top["asks"], ms / 1000 if ms else None)
        return True
