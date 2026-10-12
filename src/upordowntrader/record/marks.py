"""
Record a venue's perps' mark prices, which some of its window markets settle on.

Hyperliquid sends each perp's mark and oracle prices about once a second, the same ones again when neither moved. Each
change is stored with the time it reached us, as Hyperliquid's messages carry none, and written with the other rows
once a second.
"""

import time
from collections import Counter
from upordowntrader.db import database


class MarkRecorder:
    """
    Keeps each perp's newest mark and oracle price, and stores each change.
    """

    def __init__(self, conn, venue, log=print):
        self.conn = conn
        self.venue = venue
        self.log = log
        self.last = {}              # Coin maps to the (mark, oracle) last stored.
        self.pending = []           # Rows not written yet, see flush().
        self.updates = Counter()    # Prices each coin sent since the start.
        self.stored = Counter()     # Rows written for each coin since the start.

    def on_mark(self, coin, mark, oracle):
        """
        Note a perp's prices from the stream, stored when either moved.
        """
        self.updates[coin] += 1
        if self.last.get(coin) == (mark, oracle):
            return
        self.last[coin] = (mark, oracle)
        self.pending.append((self.venue, coin, time.time(), mark, oracle))
        self.stored[coin] += 1

    def flush(self):
        """
        Write every row queued, in one transaction. Called once a second.
        """
        if self.pending:
            database.store_marks(self.conn, self.pending)
            self.pending = []

    def status(self):
        """
        One line with each coin's prices and the rows stored, since the start.
        """
        parts = [f"{coin} {self.updates[coin]:,} marks, {self.stored[coin]:,} stored" for coin in sorted(self.updates)]
        return f"{self.venue} " + (", ".join(parts) or "no marks yet")
