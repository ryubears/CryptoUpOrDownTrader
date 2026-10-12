"""
SQLite storage for bitcoin's price history and the venues' recorded books.

One database file holds every table:

    series   each price series, a source's bars of one length for one symbol
    bars     every bar of every series, written by history/download.py
    days     the UTC days each series has wholly stored, so a rerun of the download skips them
    markets  each window market recorded, its strike and how it settled, written by record/run.py and history/markets.py
    minutes  a market's best Yes bid and ask minute by minute, from the venue's candlesticks, by history/markets.py
    books    a recorded market's book at each best price move, and once a second besides, by record/recorder.py
    gaps     stretches when a venue's feed was down, by record/recorder.py and record/quotes.py
    quotes   each exchange's best bid and ask at each price move, and once a second besides, by record/quotes.py

The tables themselves are in schema.sql. This file holds the reads and writes.
"""

import numpy as np
import sqlite3
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from upordowntrader.common.paths import DATA_DIR
from upordowntrader.db.models import Market

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
    # Write ahead logging lets a calibration read while a download or the recorder writes.
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


# MARKETS

def upsert_markets(conn, markets):
    """
    Insert new markets or refresh known ones. A strike or settlement already stored is kept when a read lacks it.
    """
    names = [f.name for f in fields(Market)]
    kept = ("strike", "settlement", "result", "settled_at")
    with conn:
        conn.executemany(f"""
            INSERT INTO markets ({', '.join(names)}) VALUES ({', '.join(':' + n for n in names)})
            ON CONFLICT (venue, ticker) DO UPDATE SET open_time = excluded.open_time, close_time = excluded.close_time,
                {', '.join(f'{n} = COALESCE(excluded.{n}, markets.{n})' for n in kept)}
        """, [asdict(m) for m in markets])


def unsettled(conn, venue, now):
    """
    The tickers of a venue's markets closed by now, in seconds since 1970, that have not settled yet.
    """
    return [row[0] for row in conn.execute("SELECT ticker FROM markets WHERE venue = ? AND close_time <= ? AND result IS NULL", (venue, now))]


def load_markets(conn, venue):
    """
    A venue's markets, oldest first.
    """
    names = [f.name for f in fields(Market)]
    return [Market(*row) for row in conn.execute(f"SELECT {', '.join(names)} FROM markets WHERE venue = ? ORDER BY close_time", (venue,))]


# MINUTES

def store_minutes(conn, venue, minutes):
    """
    Store markets' minutes, as {ticker: [(time, bid, ask, price, volume)]}, in one transaction. Minutes stored already are
    replaced.
    """
    with conn:
        conn.executemany("INSERT OR REPLACE INTO minutes (venue, ticker, time, bid, ask, price, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         [(venue, ticker, *row) for ticker, rows in minutes.items() for row in rows])


def tickers_with_minutes(conn, venue):
    """
    The tickers of a venue's markets with minutes stored.
    """
    return {row[0] for row in conn.execute("SELECT DISTINCT ticker FROM minutes WHERE venue = ?", (venue,))}


def read_minutes(conn, venue):
    """
    Every stored minute of a venue's markets, as {ticker: numpy array of (time, bid, ask) rows, oldest first}, NaN for an
    empty side.
    """
    found = {}
    for ticker, time, bid, ask in conn.execute("SELECT ticker, time, bid, ask FROM minutes WHERE venue = ? ORDER BY ticker, time", (venue,)):
        found.setdefault(ticker, []).append((time, bid, ask))
    return {ticker: np.array(rows, dtype=float) for ticker, rows in found.items()}



# BOOKS

def store_books(conn, rows):
    """
    Store book changes, each a (venue, ticker, received, sent, bid, bid_size, ask, ask_size, bids, asks) tuple, in one
    transaction.
    """
    with conn:
        conn.executemany("INSERT INTO books (venue, ticker, received, sent, bid, bid_size, ask, ask_size, bids, asks) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def store_quotes(conn, rows):
    """
    Store quotes, each a (source, received, sent, bid, bid_size, ask, ask_size) tuple, in one transaction.
    """
    with conn:
        conn.executemany("INSERT INTO quotes (source, received, sent, bid, bid_size, ask, ask_size) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def store_marks(conn, rows):
    """
    Store perps' marks, each a (venue, coin, received, mark, oracle) tuple, in one transaction.
    """
    with conn:
        conn.executemany("INSERT INTO marks (venue, coin, received, mark, oracle) VALUES (?, ?, ?, ?, ?)", rows)


def read_marks(conn, venue, coin, start, end):
    """
    A perp's marks received from start up to end, in seconds since 1970, as a numpy array of (received, mark, oracle) rows,
    oldest first, NaN for an oracle price not sent.
    """
    rows = conn.execute("SELECT received, mark, oracle FROM marks WHERE venue = ? AND coin = ? AND received >= ? AND received < ? ORDER BY received",
                        (venue, coin, float(start), float(end))).fetchall()
    return np.array([(r, m, np.nan if o is None else o) for r, m, o in rows], dtype=float).reshape(-1, 3)


def insert_gap(conn, gap):
    """
    Record a stretch when a venue's feed was down.
    """
    with conn:
        conn.execute("INSERT OR REPLACE INTO gaps (venue, start_time, end_time) VALUES (:venue, :start_time, :end_time)", asdict(gap))
