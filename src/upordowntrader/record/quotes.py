"""
Record exchanges' best bid and ask, for a stand-in for BRTI.

The windows settle on CF Benchmarks' BRTI, built once a second from the order books of a handful of exchanges, and
published only under license. Coinbase and Kraken are two of them, and their books are public, so their best bid and
ask, recorded beside Kalshi's books, let the windows be priced from much what BRTI sees. The quote streams send every
change of the best bid or ask's price or size, many a second.

As the Kalshi recorder does with books, see record/recorder.py, this stores a row when a source's best bid or ask price
moves, at once, and otherwise its newest quote once a second when only a size changed, see flush(). So every price move
is stored to the moment, which matters: on 2026-10-11 Kalshi's book moved within tens of milliseconds of bitcoin, and a
model reading bitcoin a second late took the market's moves for edges. Each row says when the quote reached us and when
the exchange says it was, the clock to line quotes up by, though Kraken's stamps are at times stale, see api/kraken.py.

When an exchange's connection is lost, the stretch until the next is subscribed is stored as a gap under the exchange's
name, and its quote is dropped, so its next one is stored whatever it is.
"""

import time
from collections import Counter
from upordowntrader.db import database
from upordowntrader.db.models import Gap


class QuoteRecorder:
    """
    Keeps each exchange's newest quote, and stores its price moves at once and the rest once a second.
    """

    def __init__(self, conn, log=print):
        self.conn = conn
        self.log = log
        self.quotes = {}            # Source maps to the (bid, bid_size, ask, ask_size) last stored.
        self.waiting = {}           # Source maps to its newest (received, sent, bid, bid_size, ask, ask_size), when it differs
                                    # from the stored quote only in sizes, stored at the next flush.
        self.pending = []           # Rows not written yet, see flush().
        self.updates = Counter()    # Quotes each source sent since the start.
        self.stored = Counter()     # Rows written for each source since the start.
        self.gaps = Counter()

    def store(self, source, received, sent, bid, bid_size, ask, ask_size):
        """
        Queue a row for a source's quote, which becomes the one stored, see flush().
        """
        self.quotes[source] = (bid, bid_size, ask, ask_size)
        self.waiting.pop(source, None)
        self.pending.append((source, received, sent, bid, bid_size, ask, ask_size))
        self.stored[source] += 1

    def on_quote(self, source, bids, asks, sent=None):
        """
        Note a quote from a source's stream: the best level a side, and the exchange's time for it, in seconds since 1970,
        or None. Stored at once when it is the source's first or moves the best bid or ask price, kept for the next flush
        when only a size changed, and dropped when it is the quote stored already.
        """
        now = time.time()
        self.updates[source] += 1
        bid, bid_size = bids[0] if bids else (None, None)
        ask, ask_size = asks[0] if asks else (None, None)
        stored = self.quotes.get(source)
        if stored == (bid, bid_size, ask, ask_size):
            self.waiting.pop(source, None)          # Back to what is stored, so nothing is left to store.
        elif stored is None or (stored[0], stored[2]) != (bid, ask):
            self.store(source, now, sent, bid, bid_size, ask, ask_size)
        else:
            self.waiting[source] = (now, sent, bid, bid_size, ask, ask_size)

    def flush(self):
        """
        Queue each source's newest quote with a size change waiting, then write every row queued, in one transaction.
        Called once a second, so sizes are stored to the second.
        """
        for source, quote in list(self.waiting.items()):
            self.store(source, *quote)
        if self.pending:
            database.store_quotes(self.conn, self.pending)
            self.pending = []

    def on_gap(self, source, start_time, end_time):
        """
        Store a stretch when a source's stream was down, and drop its quote, so the next is stored whatever it is.
        """
        database.insert_gap(self.conn, Gap(source, start_time, end_time))
        self.gaps[source] += 1
        self.quotes.pop(source, None)
        self.waiting.pop(source, None)

    def status(self):
        """
        One line with each source's quotes, the rows stored, and gaps, since the start.
        """
        sources = sorted(set(self.updates) | set(self.gaps))
        parts = [f"{s} {self.updates[s]:,} quotes, {self.stored[s]:,} stored, {self.gaps[s]} gaps" for s in sources]
        return ", ".join(parts) or "no quotes yet"
