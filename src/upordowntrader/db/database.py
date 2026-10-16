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
    marks    a venue's perps' marks, which some of its windows settle on, by record/marks.py
    paper_orders  the paper trader's orders, by paper/trader.py, settled by paper/run.py
    opportunities  each stretch of positive edge the paper trader saw, by paper/trader.py, settled by paper/run.py
    trades   each decision to trade with real money, or its paper twin, by live/executor.py
    orders   every real order sent to a venue, by live/executor.py
    settlements  how each trade paid out, by live/executor.py

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
# Columns schema.sql gained after its table had rows, as (table, column, type, an UPDATE that fills the rows before or None).
ADDED_COLUMNS = (("paper_orders", "cause", "TEXT", None),
                 ("opportunities", "take_return_pct", "REAL", "UPDATE opportunities SET take_return_pct = 100 * take_profit / take_capital "
                                                              "WHERE take_capital > 0"),
                 ("opportunities", "min_edge_total", "REAL", None),
                 ("opportunities", "result", "TEXT", None),
                 ("opportunities", "actual_profit", "REAL", None),
                 ("opportunities", "actual_return_pct", "REAL", None))
# Indexes on such columns, made once they are there.
ADDED_INDEXES = ("CREATE INDEX IF NOT EXISTS idx_opportunities_unsettled ON opportunities (result, close_time)",)
DROPPED_COLUMNS = (("opportunities", "annual_pct"),)    # Columns schema.sql lost after its table had rows, as (table, column).


# CONNECTION

def connect(db_path=None):
    """
    Open the database, creating the file and tables if needed, and adding the columns a table made before them lacks. Uses
    DB_PATH unless a path is given.
    """
    db_path = Path(db_path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    # Write ahead logging lets a calibration read while a download or the recorder writes.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA.read_text())
    add_columns(conn)
    return conn


def add_columns(conn):
    """
    Bring a table made before up to schema.sql, which CREATE TABLE IF NOT EXISTS leaves as it was: add each of
    ADDED_COLUMNS it lacks, filling its rows as it says, then ADDED_INDEXES, and drop each of DROPPED_COLUMNS it has.
    """
    for table, column in DROPPED_COLUMNS:
        if column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}:
            with conn:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    for table, column, kind, fill in ADDED_COLUMNS:
        if column not in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}:
            with conn:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                if fill:
                    conn.execute(fill)
    with conn:
        for index in ADDED_INDEXES:
            conn.execute(index)


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


# PAPER ORDERS

PAPER_COLUMNS = ("venue", "ticker", "close_time", "decided", "side", "price", "model", "mid", "fair", "edge", "limit_price", "offered",
                 "arrived", "contracts", "cost", "best", "fees", "expected", "result", "profit", "cause")


def store_paper_order(conn, order):
    """
    Store a paper order, a dict with the paper_orders columns but result and profit, which settle_paper_orders() sets.
    """
    names = [n for n in PAPER_COLUMNS if n not in ("result", "profit")]
    with conn:
        conn.execute(f"INSERT OR REPLACE INTO paper_orders ({', '.join(names)}) VALUES ({', '.join(':' + n for n in names)})", order)


def unsettled_markets(conn, now):
    """
    The (venue, ticker) of each market with paper orders or opportunities not settled yet, closed by now, in seconds since
    1970.
    """
    return conn.execute("SELECT venue, ticker FROM paper_orders WHERE result IS NULL AND close_time <= :now UNION "
                        "SELECT venue, ticker FROM opportunities WHERE result IS NULL AND close_time <= :now ORDER BY venue, ticker",
                        {"now": now}).fetchall()


def settle_paper_orders(conn, venue, ticker, result, settle_fee=0.0):
    """
    Settle a market's paper orders as the window went, 'yes' or 'no': each one's contracts pay a dollar when its side won,
    less settle_fee of it, a venue's fee at settlement, less what they cost and their fees. Returns how many orders were
    settled.
    """
    with conn:
        return conn.execute("UPDATE paper_orders SET result = :result, profit = contracts * (side = :result) * (1 - :fee) "
                            "- contracts * COALESCE(cost, 0) - fees WHERE venue = :venue AND ticker = :ticker AND result IS NULL",
                            {"venue": venue, "ticker": ticker, "result": result, "fee": settle_fee}).rowcount


def load_paper_orders(conn):
    """
    Every paper order as a dict of the paper_orders columns, oldest first.
    """
    rows = conn.execute(f"SELECT {', '.join(PAPER_COLUMNS)} FROM paper_orders ORDER BY decided").fetchall()
    return [dict(zip(PAPER_COLUMNS, row, strict=True)) for row in rows]


# OPPORTUNITIES

OPPORTUNITY_COLUMNS = ("venue", "ticker", "close_time", "side", "start_time", "end_time", "seconds", "peak_time", "peak_edge", "peak_price",
                       "peak_fair", "peak_size", "peak_capital", "peak_profit", "book_changed", "days_held", "return_pct",
                       "min_edge", "min_edge_seconds", "min_edge_size", "min_edge_capital", "min_edge_profit", "take_size", "take_capital",
                       "take_profit", "take_return_pct", "min_edge_total")
OUTCOME_COLUMNS = ("result", "actual_profit", "actual_return_pct")      # Set as the window settles, see settle_opportunities().


def store_opportunities(conn, rows):
    """
    Store opportunities, each a dict with the opportunities columns, in one transaction.
    """
    with conn:
        conn.executemany(f"INSERT OR REPLACE INTO opportunities ({', '.join(OPPORTUNITY_COLUMNS)}) "
                         f"VALUES ({', '.join(':' + n for n in OPPORTUNITY_COLUMNS)})", rows)


def load_opportunities(conn, since=0.0):
    """
    The opportunities starting from since, in seconds since 1970, as dicts of the opportunities columns, oldest first.
    """
    names = (*OPPORTUNITY_COLUMNS, *OUTCOME_COLUMNS)
    rows = conn.execute(f"SELECT {', '.join(names)} FROM opportunities WHERE start_time >= ? ORDER BY start_time", (since,)).fetchall()
    return [dict(zip(names, row, strict=True)) for row in rows]


def settle_opportunities(conn, venue, ticker, result, settle_fee=0.0):
    """
    Set a market's opportunities' outcome as the window went, 'yes' or 'no': what each one's take made, its contracts
    paying a dollar when their side won, less settle_fee of it, a venue's fee at settlement, less what they cost, and that
    over what they cost. Returns how many were settled.
    """
    made = "take_size * (side = :result) * (1 - :fee) - take_capital"
    with conn:
        return conn.execute(f"UPDATE opportunities SET result = :result, actual_profit = CASE WHEN take_size > 0 THEN {made} END, "
                            f"actual_return_pct = CASE WHEN take_capital > 0 THEN 100.0 * ({made}) / take_capital END "
                            "WHERE venue = :venue AND ticker = :ticker AND result IS NULL",
                            {"venue": venue, "ticker": ticker, "result": result, "fee": settle_fee}).rowcount


# TRADES, ORDERS, AND SETTLEMENTS

TRADE_COLUMNS = ("id", "mode", "venue", "ticker", "close_time", "side", "signal_ts", "fair", "edge", "quantity", "limit_price", "filled",
                 "cost", "fees", "status", "pays_at")
ORDER_COLUMNS = ("id", "trade_id", "venue", "ticker", "coin", "action", "outcome", "quantity", "limit_price", "tif", "client_id", "sent_at",
                 "status", "venue_order_id", "answered_at", "latency_ms", "filled", "dollars", "fees", "note", "response", "book_at", "book_ts")
SETTLEMENT_COLUMNS = ("trade_id", "mode", "settled_at", "result", "payout", "fees", "profit")


def insert(conn, table, columns, row):
    """
    Store a row, a dict with the table's columns but its id, and return its id.
    """
    names = [n for n in columns if n != "id"]
    with conn:
        return conn.execute(f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join(':' + n for n in names)})", row).lastrowid


def update(conn, table, row_id, changes):
    """
    Set some columns of a row, by id: changes is a dict of them.
    """
    with conn:
        conn.execute(f"UPDATE {table} SET {', '.join(f'{n} = :{n}' for n in changes)} WHERE id = :id", {**changes, "id": row_id})


def store_trade(conn, trade):
    """
    Store a trade, a dict with the trades columns but its id, and return its id.
    """
    return insert(conn, "trades", TRADE_COLUMNS, trade)


def store_order(conn, order):
    """
    Store an order before it goes out, a dict with the orders columns but its id, and return its id.
    """
    return insert(conn, "orders", ORDER_COLUMNS, order)


def load_trades(conn, where="1", params=()):
    """
    The trades a WHERE clause picks, as dicts of the trades columns, oldest first.
    """
    rows = conn.execute(f"SELECT {', '.join(TRADE_COLUMNS)} FROM trades WHERE {where} ORDER BY signal_ts, id", params).fetchall()
    return [dict(zip(TRADE_COLUMNS, row, strict=True)) for row in rows]


def load_orders(conn, trade_id):
    """
    A trade's orders, as dicts of the orders columns, in the order they were sent.
    """
    rows = conn.execute(f"SELECT {', '.join(ORDER_COLUMNS)} FROM orders WHERE trade_id = ? ORDER BY id", (trade_id,)).fetchall()
    return [dict(zip(ORDER_COLUMNS, row, strict=True)) for row in rows]


def open_trades(conn, venue=None, ticker=None, now=None):
    """
    The trades that hold contracts and have not settled, of one market or of every market closed by now.
    """
    where = "filled > 0 AND id NOT IN (SELECT trade_id FROM settlements)"
    if ticker is not None:
        return load_trades(conn, f"{where} AND venue = ? AND ticker = ?", (venue, ticker))
    return load_trades(conn, f"{where} AND close_time <= ?", (now,))


def store_settlement(conn, row):
    """
    Store how a trade paid out, a dict with the settlements columns.
    """
    names = ", ".join(SETTLEMENT_COLUMNS)
    with conn:
        conn.execute(f"INSERT OR REPLACE INTO settlements ({names}) VALUES ({', '.join(':' + n for n in SETTLEMENT_COLUMNS)})", row)


def load_settlements(conn):
    """
    Every settlement, as dicts of the settlements columns, oldest first.
    """
    rows = conn.execute(f"SELECT {', '.join(SETTLEMENT_COLUMNS)} FROM settlements ORDER BY settled_at, trade_id").fetchall()
    return [dict(zip(SETTLEMENT_COLUMNS, row, strict=True)) for row in rows]
