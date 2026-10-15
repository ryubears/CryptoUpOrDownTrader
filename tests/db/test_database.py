"""
Tests for storing and reading price bars, and for the columns added to a database made before them.
"""

import pytest
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


def test_opportunities_made_before_their_outcome_columns_gain_them_with_the_takes_return_filled_in(tmp_path):
    path = tmp_path / "history.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE opportunities (venue TEXT NOT NULL, ticker TEXT NOT NULL, close_time INTEGER NOT NULL, side TEXT NOT NULL, "
                "start_time REAL NOT NULL, take_size REAL, take_capital REAL, take_profit REAL, PRIMARY KEY (venue, ticker, side, start_time))")
    old.execute("INSERT INTO opportunities VALUES ('kalshi', 'K', 1000, 'yes', 500.0, 10, 5.0, 0.4)")
    old.commit()
    old.close()
    conn = database.connect(path)
    assert conn.execute("SELECT take_return_pct, min_edge_total, result, actual_profit FROM opportunities").fetchone() == (8.0, None, None, None)
    assert "idx_opportunities_unsettled" in {row[1] for row in conn.execute("PRAGMA index_list(opportunities)")}


def opportunity(side="yes", take=(10.0, 5.0, 0.4), venue="hyperliquid", ticker="#1230", start=500.0):
    size, capital, profit = take or (None, None, None)
    row = {c: None for c in database.OPPORTUNITY_COLUMNS}
    return {**row, "venue": venue, "ticker": ticker, "close_time": 1000, "side": side, "start_time": start, "end_time": start + 1, "seconds": 1.0,
            "peak_time": start, "peak_edge": 0.04, "peak_price": 0.5, "peak_fair": 0.54, "peak_size": 10.0, "peak_capital": 5.0, "peak_profit": 0.4,
            "book_changed": 0, "days_held": 0.006, "return_pct": 8.0, "annual_pct": 480.0, "min_edge": 0.03, "take_size": size,
            "take_capital": capital, "take_profit": profit}


def test_a_markets_opportunities_settle_once_with_what_their_take_made_less_a_venues_fee(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_opportunities(conn, [opportunity(), opportunity("no", start=501.0), opportunity(take=None, start=502.0)])
    assert database.unsettled_markets(conn, 999) == [] and database.unsettled_markets(conn, 1000) == [("hyperliquid", "#1230")]
    assert database.settle_opportunities(conn, "hyperliquid", "#1230", "yes", 0.001) == 3
    assert database.settle_opportunities(conn, "hyperliquid", "#1230", "yes", 0.001) == 0
    rows = database.load_opportunities(conn)
    assert [(o["result"], o["actual_profit"], o["actual_return_pct"]) for o in rows] == [
        ("yes", pytest.approx(10 * 0.999 - 5.0), pytest.approx(100 * (10 * 0.999 - 5.0) / 5.0)), ("yes", -5.0, -100.0), ("yes", None, None)]
    assert database.unsettled_markets(conn, 2000) == []
