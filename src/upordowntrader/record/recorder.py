"""
Record the live order books of a venue's window markets.

The venue's stream pushes every book change, hundreds a second on a busy
Kalshi window, almost all of them a size changing at a price that stays.
The recorder keeps each market's newest best levels in memory and stores
a row when its best bid or best ask price moves, at once, and otherwise
the newest book once a second when anything else in its kept levels
changed, see flush(). So every move of the best prices is stored to the
moment, and sizes and depth to the second, which is as finely as bitcoin's
price is kept to line them up with. On 2026-10-10 a window's book changed
about 200 times a second, 1 in 200 of them a best price, so this stores a
row or two a second instead of 5.7 GB a day. Each row says when the change
reached us and when the venue's matching engine says it made it, which is
the clock to line books up by: this Mac's ran about 0.25s slow that day.

When the connection is lost, the stretch until the next one is subscribed
is stored as a gap, and the books of its markets are dropped until it
sends them again, so a quiet book can be told from one that went unseen.
The status line says how many changes came and how many were stored, and
how far behind the venue they reached us.
"""

import json
import time
from upordowntrader.common.stats import quantile
from upordowntrader.db import database
from upordowntrader.db.models import Gap

BOOK_LEVELS = 5             # Levels a side the streams pass on, and the recorder keeps and stores.


def book_row(venue, ticker, received, sent, bids, asks):
    """
    A books table row for one change: its times, the best bid and ask with their sizes, and every level kept.
    """
    bid, bid_size = bids[0] if bids else (None, None)
    ask, ask_size = asks[0] if asks else (None, None)
    return (venue, ticker, received, sent, bid, bid_size, ask, ask_size, json.dumps(bids), json.dumps(asks))


def best_prices(book):
    """
    The best bid and best ask prices of a (bids, asks) book, None for an empty side.
    """
    bids, asks = book
    return (bids[0][0] if bids else None, asks[0][0] if asks else None)


class Recorder:
    """
    Keeps the newest book of every market a venue streams, and stores its best price moves at once and the rest once a
    second.
    """

    def __init__(self, conn, venue, log=print):
        self.conn = conn
        self.venue = venue
        self.log = log
        self.books = {}             # Ticker maps to the (bids, asks) last stored.
        self.waiting = {}           # Ticker maps to its newest (received, sent, bids, asks), when it differs from the stored
                                    # book by more than the best prices, stored at the next flush.
        self.pending = []           # Rows not written yet, see flush().
        self.updates = 0            # Changes the stream passed on since the start.
        self.stored = 0             # Rows written since the start.
        self.gaps = 0
        self.last_update = None     # Wall clock seconds of the newest change.
        self.behind = []            # Seconds each change reached us after the venue made it, since the last status.

    def store(self, ticker, received, sent, bids, asks):
        """
        Queue a row for a market's book, which becomes the one stored, see flush().
        """
        self.books[ticker] = (bids, asks)
        self.waiting.pop(ticker, None)
        self.pending.append(book_row(self.venue, ticker, received, sent, bids, asks))

    def on_book(self, ticker, bids, asks, sent=None):
        """
        Note a book from the stream: its best levels a side, and when the venue made the change, in seconds since 1970, or
        None when it did not say. Stored at once when it is a market's first or moves a best price, kept for the next
        flush when it changes anything else, and dropped when it is the book stored already.
        """
        now = self.last_update = time.time()
        self.updates += 1
        if sent is not None:
            self.behind.append(now - sent)
        stored = self.books.get(ticker)
        if stored == (bids, asks):
            self.waiting.pop(ticker, None)          # Back to what is stored, so nothing is left to store.
        elif stored is None or best_prices(stored) != best_prices((bids, asks)):
            self.store(ticker, now, sent, bids, asks)
        else:
            self.waiting[ticker] = (now, sent, bids, asks)

    def flush(self):
        """
        Queue the newest book of each market with a change waiting, then write every row queued, in one transaction. Called
        once a second, so sizes and depth are stored to the second.
        """
        for ticker, (received, sent, bids, asks) in list(self.waiting.items()):
            self.store(ticker, received, sent, bids, asks)
        if self.pending:
            database.store_books(self.conn, self.pending)
            self.stored += len(self.pending)
            self.pending = []

    def forget(self, tickers):
        """
        Drop the books of markets no longer streamed, or not seen for a while, so their next book is stored whatever it is.
        """
        for ticker in tickers:
            self.books.pop(ticker, None)
            self.waiting.pop(ticker, None)

    def on_gap(self, start_time, end_time, tickers):
        """
        Store a stretch when the stream was down, and drop the books of the markets it carries until it sends them again.
        """
        database.insert_gap(self.conn, Gap(self.venue, start_time, end_time))
        self.gaps += 1
        self.forget(tickers)

    def on_state(self, ticker, why):
        """
        Log what the venue says of a market's trading: why it stopped, in a word, or None when it trades again.
        """
        self.log(f"{self.venue} {ticker} {'trading again' if why is None else f'not trading: {why}'}")

    def status(self):
        """
        One line with what has happened so far, and how far behind the venue its books came since the last status line.
        A negative delay means this machine's clock runs behind the venue's.
        """
        last = f"last {time.time() - self.last_update:.0f}s ago" if self.last_update else "none yet"
        line = f"{self.venue} tracking {len(self.books)} books, {self.updates:,} changes ({last}), {self.stored:,} stored, {self.gaps} gaps"
        if self.behind:
            line += f", {1000 * quantile(self.behind, 0.5):.0f} ms behind the venue, {1000 * quantile(self.behind, 0.9):.0f} at 90%"
            self.behind = []
        return line
