"""
Tests for storing and reading price bars.
"""

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
