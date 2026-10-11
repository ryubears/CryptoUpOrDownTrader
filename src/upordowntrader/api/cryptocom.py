"""
Crypto.com Exchange public market data: the live best bid and ask from its websocket, read without an API key.

Crypto.com is one of the exchanges CF Benchmarks builds BRTI from, the index Kalshi and Polymarket US settle on, since
2025-03-31. Its market data websocket, subscribed to an instrument's book of DEPTH levels as a snapshot and then its
changes, sends the snapshot, then every 10 ms the levels changed, an empty change when none did, with Crypto.com's time in
milliseconds. Each change names the sequence number of the one before, so a lost one shows. Every 30 seconds it sends a
heartbeat, and closes a connection that does not answer it. It asks clients to wait a second after connecting before
sending, since its rate limits count from the second the connection opened. This is the only file that knows Crypto.com's
field names and message formats.
"""

import asyncio
import json
from upordowntrader.api.bookstream import BookStream, Reconnect

WS_URL = "wss://stream.crypto.com/exchange/v1/market"
DEPTH = 10                  # Levels a side kept. Crypto.com sends 10 or 50.
UPDATE_MS = 10              # How often it sends changes, the fastest it offers.
OPEN_PAUSE = 1              # Seconds to wait after connecting before subscribing.
SIDES = ("bids", "asks")


def channel(instrument):
    """
    The channel of an instrument's book, such as 'book.BTC_USD.10'.
    """
    return f"book.{instrument}.{DEPTH}"


def command_frame(method, instruments):
    """
    The frame that subscribes to, or unsubscribes from, instruments' books. method is 'subscribe' or 'unsubscribe'.
    """
    return {"id": 1, "method": method, "params": {"channels": [channel(i) for i in instruments],
                                                  "book_subscription_type": "SNAPSHOT_AND_UPDATE", "book_update_frequency": UPDATE_MS}}


class CryptoComQuoteStream(BookStream):
    """
    Crypto.com's best bid and ask for each wanted instrument, such as 'BTC_USD', passed on as one level a side, as the book
    streams pass their books, with Crypto.com's time. Each instrument's book is kept from its snapshot and changes, and
    a change that does not follow the last one means one was lost, so the connection is rebuilt.
    """

    name = "cryptocom"

    def reset(self):
        self.levels = {}            # Instrument maps to {"bids": {price: size}, "asks": {...}}, from its snapshot on.
        self.sequence = {}          # Instrument maps to the sequence number of its last snapshot or change.

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        await asyncio.sleep(OPEN_PAUSE)
        await ws.send(json.dumps(command_frame("subscribe", sorted(self.wanted))))

    async def send_command(self, ws, action, instruments):
        """
        Apply an add or remove, or answer a heartbeat, queued by handle() as ('heartbeat', [its id]).
        """
        if action == "heartbeat":
            await ws.send(json.dumps({"id": instruments[0], "method": "public/respond-heartbeat"}))
        else:
            await ws.send(json.dumps(command_frame("subscribe" if action == "add" else "unsubscribe", instruments)))

    def handle(self, raw):
        m = json.loads(raw)
        if m.get("method") == "public/heartbeat":
            self.commands.put_nowait(("heartbeat", [m.get("id")]))
            return False
        if m.get("code"):
            self.log(f"cryptocom stream error {m.get('code')}: {m.get('message')}")
            return False
        result = m.get("result") or {}
        instrument = result.get("instrument_name")
        if instrument not in self.wanted or not result.get("data"):
            return False                                        # A subscription's acknowledgement.
        data = result["data"][0]
        if result.get("channel") == "book":
            self.levels[instrument] = {side: {float(p): float(s) for p, s, *_ in data.get(side) or []} for side in SIDES}
        elif result.get("channel") == "book.update" and instrument in self.levels:
            if data.get("pu") != self.sequence.get(instrument):
                raise Reconnect(f"{instrument} skipped from sequence {self.sequence.get(instrument)} to {data.get('pu')}")
            for side in SIDES:
                book = self.levels[instrument][side]
                for p, s, *_ in (data.get("update") or {}).get(side) or []:
                    if float(s) > 0:
                        book[float(p)] = float(s)
                    else:
                        book.pop(float(p), None)
        else:
            return False                                        # A change before its snapshot.
        self.sequence[instrument] = data.get("u")
        bids, asks = self.levels[instrument]["bids"], self.levels[instrument]["asks"]
        best_bid = [[max(bids), bids[max(bids)]]] if bids else []
        best_ask = [[min(asks), asks[min(asks)]]] if asks else []
        ms = data.get("t")
        self.on_book(instrument, best_bid, best_ask, ms / 1000 if ms else None)
        return True
