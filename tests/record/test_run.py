"""
Tests for the recording process, with stand ins for Kalshi's market reads and stream.
"""

import asyncio
import time
from dataclasses import replace
from upordowntrader.api import hyperliquid, kalshi
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.record import run
from upordowntrader.record.quotes import QuoteRecorder
from upordowntrader.record.recorder import Recorder


def window(ticker, close_time, strike=None, settlement=None, result=None):
    """
    A Market for one of the series' windows, closing at close_time.
    """
    return Market("kalshi", ticker, ticker.rsplit("-", 1)[0], run.SERIES, int(close_time) - 900, int(close_time), strike, settlement,
                  result, None)


class FakeStream:
    """
    Stands in for a stream, Kalshi's or an exchange's: keeps the wanted set, and while running passes on a book for each
    five times a second.
    """

    name = "fake"

    def __init__(self, contract_ids, on_book, on_gap=None, log=print):
        self.wanted = set(contract_ids)
        self.on_book = on_book
        self.on_gap = on_gap
        self.reconnects = 0

    def reconnect(self):
        self.reconnects += 1

    def add(self, tickers):
        self.wanted |= set(tickers)

    def remove(self, tickers):
        self.wanted -= set(tickers)

    async def run(self):
        price = 0.5
        while True:
            for ticker in sorted(self.wanted):
                self.on_book(ticker, [[price, 10.0]], [[price + 0.01, 5.0]], time.time())
            price += 0.01
            await asyncio.sleep(0.2)


def test_a_refresh_stores_the_windows_and_streams_only_those_not_closed(tmp_path, monkeypatch):
    now = time.time()
    found = [window("KXBTC15M-A-00", now - 30, 80000.0), window("KXBTC15M-B-15", now + 870, 80010.0), window("KXBTC15M-C-30", now + 1770)]
    monkeypatch.setattr(kalshi, "windows", lambda series, start, end: found)
    conn = database.connect(tmp_path / "history.sqlite")
    stream = FakeStream(["KXBTC15M-A-00"], None)
    recorder = Recorder(conn, "kalshi")
    recorder.on_book("KXBTC15M-A-00", [[0.9, 1.0]], [[0.95, 1.0]])
    asyncio.run(run.refresh(conn, stream, recorder, {}))
    assert stream.wanted == {"KXBTC15M-B-15", "KXBTC15M-C-30"}            # The closed window dropped,
    assert recorder.books == {}                                          # and its book forgotten.
    assert [m.ticker for m in database.load_markets(conn, "kalshi")] == ["KXBTC15M-A-00", "KXBTC15M-B-15", "KXBTC15M-C-30"]


def test_a_read_after_a_sleep_reaches_back_to_before_it(tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(kalshi, "windows", lambda series, start, end: asked.append((start, end)) or [])
    conn = database.connect(tmp_path / "history.sqlite")
    reads = {"at": time.time() - 7200}                                   # Two hours asleep since the last read.
    asyncio.run(run.refresh(conn, FakeStream([], None), Recorder(conn, "kalshi"), reads))
    asyncio.run(run.refresh(conn, FakeStream([], None), Recorder(conn, "kalshi"), reads))
    (first_start, first_end), (second_start, _) = asked
    assert round(first_end - first_start) == 7200 + run.WINDOW_SECONDS + run.LOOKAHEAD_SECONDS
    assert round(second_start - reads["at"]) == -run.WINDOW_SECONDS      # Then a window back from the last read only.


def test_a_late_tick_stores_the_sleep_as_a_gap_for_every_stream_and_reconnects_them(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    recorder, quotes = Recorder(conn, "kalshi"), QuoteRecorder(conn)
    recorder.on_book("KXBTC15M-B-15", [[0.5, 1.0]], [[0.51, 1.0]])
    book_stream = FakeStream(["KXBTC15M-B-15"], None, recorder.on_gap)
    quote_stream = FakeStream(["BTC-USD"], None, lambda start, end, products: quotes.on_gap("coinbase", start, end))
    logs = []
    assert not run.woke([book_stream, quote_stream], 1000.0, 1001.2, logs.append)    # An ordinary tick.
    assert run.woke([book_stream, quote_stream], 1000.0, 4600.0, logs.append)        # An hour asleep.
    gaps = conn.execute("SELECT venue, start_time, end_time FROM gaps ORDER BY venue").fetchall()
    assert gaps == [("coinbase", 1000.0, 4600.0), ("kalshi", 1000.0, 4600.0)]
    assert (book_stream.reconnects, quote_stream.reconnects) == (1, 1) and recorder.books == {}
    assert logs == ["no tick for 3600s, this machine slept or the process stalled, storing a gap and reconnecting"]


def test_settling_reads_only_closed_windows_and_keeps_the_strike(tmp_path, monkeypatch):
    now = time.time()
    conn = database.connect(tmp_path / "history.sqlite")
    database.upsert_markets(conn, [window("KXBTC15M-A-00", now - 30, 80000.0), window("KXBTC15M-B-15", now + 870, 80010.0)])
    asked, logs = [], []
    monkeypatch.setattr(kalshi, "markets", lambda tickers: asked.append(tickers) or [window("KXBTC15M-A-00", now - 30, None, 80012.5, "yes")])
    asyncio.run(run.settle(conn, logs.append))
    assert asked == [["KXBTC15M-A-00"]]
    settled = database.load_markets(conn, "kalshi")[0]
    assert (settled.strike, settled.settlement, settled.result) == (80000.0, 80012.5, "yes")     # The strike stored is kept.
    assert logs == ["kalshi KXBTC15M-A-00 settled yes, 80,012.50 against a strike of 80,000.00"]
    asyncio.run(run.settle(conn, logs.append))
    assert len(asked) == 1                                               # Nothing left to settle.


def hyperliquid_window(ticker, close_time, series="out-BTC-15m", strike=83984.0):
    """
    A Market for one of Hyperliquid's windows, closing at close_time.
    """
    period = 86400 if series.startswith("recurring") else 900
    return Market("hyperliquid", ticker, f"{series}-{ticker}", series, int(close_time) - period, int(close_time), strike, None, None, None)


def test_a_hyperliquid_refresh_streams_the_windows_not_closed_and_keeps_the_perps(tmp_path, monkeypatch):
    now = time.time()
    found = [hyperliquid_window("#113720", now - 30), hyperliquid_window("#113760", now + 870)]
    monkeypatch.setattr(hyperliquid, "windows", lambda: found)
    conn = database.connect(tmp_path / "history.sqlite")
    stream = FakeStream(["BTC", "#113000", "#113720"], None)
    recorder = Recorder(conn, "hyperliquid")
    recorder.books = {"#113000": object(), "#113720": object()}
    asyncio.run(run.refresh_hyperliquid(conn, stream, recorder))
    assert stream.wanted == {"BTC", "#113760"} and recorder.books == {}
    assert [m.ticker for m in database.load_markets(conn, "hyperliquid")] == ["#113720", "#113760"]


def test_the_15_minute_windows_settle_as_the_explorer_says_and_the_daily_by_the_marks(tmp_path, monkeypatch):
    now = time.time()
    close = int(now) - 100
    conn = database.connect(tmp_path / "history.sqlite")
    database.upsert_markets(conn, [hyperliquid_window("#113720", close), hyperliquid_window("#113730", close),
                                   hyperliquid_window("#109970", close, "recurring-BTC-1d", 82994.0),
                                   hyperliquid_window("#100000", now - run.SETTLE_REACH - 60), hyperliquid_window("#113760", now + 800),
                                   hyperliquid_window("#115400", close, "out-ETH-15m", 2545.1)])
    database.store_marks(conn, [("hyperliquid", "BTC", close - 2.0, 82990.0, None), ("hyperliquid", "BTC", close + 2.0, 83000.0, None)])
    backfilled = hyperliquid_window("#112000", close - 7200)
    monkeypatch.setattr(hyperliquid, "settlements", lambda: [replace(hyperliquid_window("#113720", close), result="yes", settled_at=close + 8.0),
                                                             replace(backfilled, result="no", settled_at=close - 7192.0)])
    logs = []
    asyncio.run(run.settle_hyperliquid(conn, logs.append))
    settled = {m.ticker: m for m in database.load_markets(conn, "hyperliquid")}
    assert (settled["#113720"].result, settled["#113720"].settled_at) == ("yes", close + 8.0)
    assert settled["#113730"].result is None                            # Not settled by its deployer yet.
    assert (settled["#109970"].settlement, settled["#109970"].result) == (82995.0, "yes")     # Halfway between the marks around.
    assert settled["#100000"].result is None and settled["#113760"].result is None          # Too long closed, and not closed.
    assert settled["#112000"].result == "no"                            # Never seen listed, stored from the explorer.
    assert settled["#115400"].result is None                            # On a perp no longer tracked.
    assert sorted(logs) == ["hyperliquid out-BTC-15m #113720 settled yes against a strike of 83,984",
                            "hyperliquid recurring-BTC-1d #109970 settled yes against a strike of 82,994, the mark 82,995.00"]


def test_a_daily_window_without_marks_on_both_sides_of_its_close_waits(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_marks(conn, [("hyperliquid", "BTC", 1000.0, 82990.0, None)])
    assert run.interpolated_mark(conn, "BTC", 1010) is None and run.interpolated_mark(conn, "ETH", 1010) is None


def test_a_short_run_records_the_open_windows_books_and_the_exchanges_quotes(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr(kalshi, "KalshiBookStream", FakeStream)
    monkeypatch.setattr(hyperliquid, "HyperliquidBookStream", FakeStream)
    monkeypatch.setattr(run, "QUOTE_STREAMS", {"coinbase": (FakeStream, "BTC-USD"), "kraken": (FakeStream, "BTC/USD")})
    monkeypatch.setattr(kalshi, "windows", lambda series, start, end: [window("KXBTC15M-B-15", now + 870, 80010.0)])
    monkeypatch.setattr(kalshi, "markets", lambda tickers: [])
    monkeypatch.setattr(hyperliquid, "windows", lambda: [hyperliquid_window("#113760", now + 870)])
    monkeypatch.setattr(run, "code_version", lambda: "6132fc7-dirty")
    conn = database.connect(tmp_path / "history.sqlite")
    logs = []
    asyncio.run(run.record(conn, seconds=2.5, log=logs.append))
    tickers = {row[0] for row in conn.execute("SELECT ticker FROM books")} - set(hyperliquid.UNDERLYINGS)    # The fake sends the perps books.
    assert tickers == {"KXBTC15M-B-15", "#113760"} and conn.execute("SELECT COUNT(*) FROM books WHERE venue = 'kalshi'").fetchone()[0] >= 5
    quotes = conn.execute("SELECT source, COUNT(*), MIN(bid), MAX(ask) FROM quotes GROUP BY source").fetchall()
    assert [q[0] for q in quotes] == ["coinbase", "kraken"] and all(q[1] >= 1 and q[2] < q[3] for q in quotes)
    assert logs[0] == ("recording KXBTC15M, 5 levels a side, the quotes of coinbase, kraken, and Hyperliquid's up or down windows on BTC, "
                       "with code 6132fc7-dirty")
    assert [line.split()[0] for line in logs[-4:]] == ["kalshi", "coinbase", "hyperliquid", "hyperliquid"]
