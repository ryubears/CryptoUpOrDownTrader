"""
SQLite storage for bitcoin's price history.

One database file holds every table:

    series   each price series, a source's bars of one length for one symbol
    bars     every bar of every series, written by history/download.py
    days     the UTC days each series has wholly stored, so a rerun of the download skips them

The tables themselves are in schema.sql. This file holds the reads and writes.
"""

import numpy as np
import sqlite3
from datetime import date
from pathlib import Path
from upordowntrader.common.paths import DATA_DIR

DB_PATH = DATA_DIR / "history.sqlite"
SCHEMA = Path(__file__).with_name("schema.sql")
BAR_COLUMNS = ("time", "open", "high", "low", "close", "volume")    # The columns read_bars() gives, in order.


# CONNECTION

def connect(db_path=None):
    """
    Open the database, creating the file and tables if needed. Uses DB_PATH unless a path is given.
    """
    db_path = Path(db_path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    # Write ahead logging lets a calibration read while a download writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA.read_text())
    return conn


# SERIES

def series_id(conn, source, symbol, seconds):
    """
    The id of a price series, added on first use.
    """
    with conn:
        conn.execute("INSERT OR IGNORE INTO series (source, symbol, seconds) VALUES (?, ?, ?)", (source, symbol, seconds))
    return conn.execute("SELECT id FROM series WHERE source = ? AND symbol = ? AND seconds = ?", (source, symbol, seconds)).fetchone()[0]


# BARS

def store_day(conn, series, day, bars):
    """
    Store one UTC day's bars of a series, each a (time, open, high, low, close, volume, trades) tuple, and mark the day
    stored, in one transaction, so a day is either wholly stored or not at all. Bars already stored are replaced.
    """
    with conn:
        conn.executemany("INSERT OR REPLACE INTO bars (series_id, time, open, high, low, close, volume, trades) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         [(series, *bar) for bar in bars])
        conn.execute("INSERT OR REPLACE INTO days (series_id, day, bars) VALUES (?, ?, ?)", (series, day.isoformat(), len(bars)))


def stored_days(conn, series):
    """
    The days a series has wholly stored, as {date: bars}.
    """
    return {date.fromisoformat(day): bars for day, bars in conn.execute("SELECT day, bars FROM days WHERE series_id = ?", (series,))}


def read_bars(conn, series, start, end):
    """
    The bars of a series opening from start up to but not including end, both in seconds since 1970, oldest first, as a
    numpy array with a column for each of BAR_COLUMNS.
    """
    rows = conn.execute(f"SELECT {', '.join(BAR_COLUMNS)} FROM bars WHERE series_id = ? AND time >= ? AND time < ? ORDER BY time",
                        (series, start, end)).fetchall()
    return np.array(rows, dtype=float).reshape(-1, len(BAR_COLUMNS))
