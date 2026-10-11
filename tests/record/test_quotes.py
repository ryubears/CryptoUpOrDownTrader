"""
Tests for storing the exchanges' quotes: price moves at once, size changes once a second.
"""

from upordowntrader.db import database
from upordowntrader.record.quotes import QuoteRecorder


def stored(conn):
    return conn.execute("SELECT source, sent, bid, bid_size, ask, ask_size FROM quotes ORDER BY rowid").fetchall()


def recorder(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    return conn, QuoteRecorder(conn)


def test_a_first_quote_and_a_price_move_are_stored_at_once(tmp_path):
    conn, quotes = recorder(tmp_path)
    quotes.on_quote("coinbase", [[82961.83, 0.5]], [[82961.84, 0.2]], 1000.1)
    quotes.on_quote("coinbase", [[82961.80, 0.4]], [[82961.84, 0.2]], 1000.3)      # The bid moves.
    quotes.on_quote("kraken", [[82962.2, 1.0]], [], None)                          # No ask for now.
    assert stored(conn) == []                                                      # Nothing written until the flush.
    quotes.flush()
    assert stored(conn) == [("coinbase", 1000.1, 82961.83, 0.5, 82961.84, 0.2), ("coinbase", 1000.3, 82961.80, 0.4, 82961.84, 0.2),
                            ("kraken", None, 82962.2, 1.0, None, None)]


def test_size_changes_wait_for_the_flush_and_only_the_newest_is_stored(tmp_path):
    conn, quotes = recorder(tmp_path)
    quotes.on_quote("kraken", [[82962.2, 1.0]], [[82962.3, 1.0]], 1000.0)
    quotes.flush()
    for size in (2.0, 3.0, 4.0):
        quotes.on_quote("kraken", [[82962.2, size]], [[82962.3, 1.0]], 1000.5)
    quotes.flush()
    quotes.flush()                                                                 # Nothing new since.
    assert [row[3] for row in stored(conn)] == [1.0, 4.0]
    assert quotes.status() == "kraken 4 quotes, 2 stored, 0 gaps"


def test_a_gap_is_stored_under_the_exchange_and_its_next_quote_is_stored_whatever_it_is(tmp_path):
    conn, quotes = recorder(tmp_path)
    quotes.on_quote("kraken", [[82962.2, 1.0]], [[82962.3, 1.0]])
    quotes.on_gap("kraken", 1000.0, 1004.5)
    quotes.on_quote("kraken", [[82962.2, 1.0]], [[82962.3, 1.0]])                  # The same quote, after the gap.
    quotes.flush()
    assert conn.execute("SELECT venue, start_time, end_time FROM gaps").fetchall() == [("kraken", 1000.0, 1004.5)]
    assert len(stored(conn)) == 2
