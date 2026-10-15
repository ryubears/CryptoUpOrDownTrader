"""
Tests for storing and reading price bars, and for the columns added to a database made before them.
"""

import sqlite3
from datetime import date
from upordowntrader.db import database


def test_a_series_is_added_once_and_found_again(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    first = database.series_id(conn, "coinbase", "BTC-USD", 60)
    assert database.series_id(conn, "coinbase", "BTC-USD", 60) == first
    assert database.series_id(conn, "coinbase", "BTC-USD", 1) != first          # Another bar length is another series.


def test_a_day_is_stored_whole_and_read_back_in_order(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    series = database.series_id(conn, "coinbase", "BTC-USD", 60)
    database.store_day(conn, series, date(2026, 10, 1), [(t, 1.0, 2.0, 0.5, 1.5, 3.0, 7) for t in (102, 100, 101)])
    database.store_day(conn, series, date(2026, 10, 1), [(100, 9.0, 9.0, 9.0, 9.0, 9.0, 1)])      # A rerun replaces the bar.
    assert database.stored_days(conn, series) == {date(2026, 10, 1): 1}
    bars = database.read_bars(conn, series, 100, 102)
    assert bars.tolist() == [[100, 9.0, 9.0, 9.0, 9.0, 9.0], [101, 1.0, 2.0, 0.5, 1.5, 3.0]]
    assert database.read_bars(conn, series, 200, 300).shape == (0, len(database.BAR_COLUMNS))


def test_a_table_made_before_a_column_was_added_gains_it_with_its_rows_kept(tmp_path):
    path = tmp_path / "history.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE paper_orders (venue TEXT NOT NULL, ticker TEXT NOT NULL, close_time INTEGER NOT NULL, decided REAL NOT NULL, "
                "side TEXT NOT NULL, profit REAL, PRIMARY KEY (venue, ticker, decided, side))")
    old.execute("INSERT INTO paper_orders VALUES ('kalshi', 'K', 1000, 500.0, 'yes', 1.5)")
    old.commit()
    old.close()
    conn = database.connect(path)
    assert conn.execute("SELECT venue, profit, cause FROM paper_orders").fetchall() == [("kalshi", 1.5, None)]
    database.connect(path)                                      # Once added, connecting again leaves it.
