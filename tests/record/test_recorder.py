"""
Tests for storing the recorded books: best price moves at once, everything else once a second.
"""

import json
from upordowntrader.db import database
from upordowntrader.record.recorder import Recorder

BIDS, ASKS = [[0.56, 120.0], [0.55, 40.0]], [[0.58, 75.0]]


def stored(conn):
    return conn.execute("SELECT ticker, sent, bid, bid_size, ask, ask_size, bids, asks FROM books ORDER BY rowid").fetchall()


def recorder(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    return conn, Recorder(conn, "kalshi", log=lambda message: None)


def test_a_first_book_and_a_best_price_move_are_stored_at_once(tmp_path):
    conn, r = recorder(tmp_path)
    r.on_book("T", BIDS, ASKS, None)                                     # The snapshot.
    r.on_book("T", BIDS, [[0.57, 10.0]], 1791140401.5)                   # The best ask moves.
    r.on_book("U", [], [], None)                                         # A market with no orders yet.
    assert stored(conn) == []                                            # Nothing written until the flush.
    r.flush()
    assert stored(conn) == [("T", None, 0.56, 120.0, 0.58, 75.0, json.dumps(BIDS), json.dumps(ASKS)),
                            ("T", 1791140401.5, 0.56, 120.0, 0.57, 10.0, json.dumps(BIDS), "[[0.57, 10.0]]"),
                            ("U", None, None, None, None, None, "[]", "[]")]


def test_size_and_depth_changes_wait_for_the_flush_and_only_the_newest_is_stored(tmp_path):
    conn, r = recorder(tmp_path)
    r.on_book("T", BIDS, ASKS)
    r.flush()
    for size in (80.0, 90.0, 100.0):                                     # The best ask's size changes three times in a second.
        r.on_book("T", BIDS, [[0.58, size]], 1791140402.0)
    r.flush()
    r.flush()                                                            # Nothing new since.
    assert [row[5] for row in stored(conn)] == [75.0, 100.0]
    assert (r.updates, r.stored) == (4, 2)


def test_a_book_back_to_the_one_stored_stores_nothing(tmp_path):
    conn, r = recorder(tmp_path)
    r.on_book("T", BIDS, ASKS)
    r.on_book("T", [[0.56, 130.0], [0.55, 40.0]], ASKS)                  # A size change, waiting.
    r.on_book("T", BIDS, ASKS)                                           # Undone before the flush.
    r.flush()
    assert len(stored(conn)) == 1


def test_a_gap_is_stored_and_drops_the_books_it_carried(tmp_path):
    conn, r = recorder(tmp_path)
    r.on_book("T", BIDS, ASKS)
    r.on_gap(1791140400.0, 1791140403.5, ["T"])
    r.on_book("T", BIDS, ASKS)                                           # The new connection's snapshot, the same book.
    r.flush()
    assert conn.execute("SELECT venue, start_time, end_time FROM gaps").fetchall() == [("kalshi", 1791140400.0, 1791140403.5)]
    assert len(stored(conn)) == 2                                        # Stored again, since it went unseen for a while.


def test_the_status_line_says_how_far_behind_the_venue_books_came(tmp_path, monkeypatch):
    _, r = recorder(tmp_path)
    monkeypatch.setattr("upordowntrader.record.recorder.time.time", lambda: 1000.25)
    r.on_book("T", BIDS, ASKS, 1000.0)
    line = r.status()
    assert line.startswith("kalshi tracking 1 books, 1 changes (last 0s ago), 0 stored, 0 gaps") and "250 ms behind the venue" in line
    assert "behind" not in r.status()                                    # Forgotten once said.
