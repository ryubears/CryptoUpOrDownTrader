"""
Tests for the paper trading process's periodic work, the paper orders' storage and settling, and the report, with stand ins
for the venues' reads and streams.
"""

import asyncio
import pytest
import time
from upordowntrader.api import hyperliquid, kalshi, polymarket_us
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.model import fair
from upordowntrader.paper import report, run
from upordowntrader.paper.trader import SETTLE_FEES, PaperTrader


def order(venue="kalshi", ticker="K", close_time=1000, decided=500.0, side="yes", contracts=10.0, cost=0.4, edge=0.03, cause="coinbase",
          **more):
    """
    A paper order as the trader stores it, filled at cost a contract.
    """
    rate = fair.FEE_RATE if venue == "kalshi" else polymarket_us.TAKER_FEE_RATE
    return {"venue": venue, "ticker": ticker, "close_time": close_time, "decided": decided, "side": side, "price": 80_000.0, "model": 0.5,
            "mid": 0.4, "fair": 0.45, "edge": edge, "limit_price": cost, "offered": 50.0, "arrived": decided + 0.02, "contracts": contracts,
            "cost": cost if contracts else None, "best": cost if contracts else None, "fees": contracts * fair.fee(cost, rate),
            "expected": contracts * fair.edge(0.45, cost, rate), "cause": cause, **more}


def test_settling_pays_the_side_that_won_less_cost_and_fees_once(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    stored = (order(), order(decided=501.0, side="no", cost=0.55), order(decided=502.0, contracts=0.0),
              order(ticker="OPEN", close_time=9e9, decided=503.0))
    for o in stored:
        database.store_paper_order(conn, o)
    assert database.unsettled_markets(conn, 1000) == [("kalshi", "K")]
    assert database.settle_paper_orders(conn, "kalshi", "K", "yes") == 3
    assert database.settle_paper_orders(conn, "kalshi", "K", "yes") == 0
    profits = [o["profit"] for o in database.load_paper_orders(conn)[:3]]
    assert profits == pytest.approx([10 * (1 - 0.4) - 10 * fair.fee(0.4), -10 * 0.55 - 10 * fair.fee(0.55), 0.0])


class FakeStream:
    """
    Stands in for a venue's book stream: keeps the wanted set.
    """

    def __init__(self):
        self.wanted = set()

    def add(self, names):
        self.wanted |= set(names)

    def remove(self, names):
        self.wanted -= set(names)


def window(close_time, strike=None):
    return Market("kalshi", f"K-{close_time}", f"K-{close_time}", "KXBTC15M", close_time - 900, close_time, strike, None, None, None)


def test_a_refresh_prices_and_streams_the_windows_not_closed_and_reads_the_tracking_of_the_one_opened(tmp_path, monkeypatch):
    close = (int(time.time()) // 900 + 1) * 900
    monkeypatch.setattr(kalshi, "windows", lambda series, start, end: [window(close - 900, 1.0), window(close, 80_000.0), window(close + 900)])
    monkeypatch.setattr(run, "read_tracking", lambda db_path, market, errors: 1e-9)
    monkeypatch.setattr(run, "read_offset", lambda db_path, market: 12.5)
    logs = []
    trader = PaperTrader(database.connect(tmp_path / "history.sqlite"), logs.append)
    streams = {"kalshi": FakeStream(), "polymarket_us": FakeStream()}
    streams["kalshi"].wanted = {"K-OLD"}
    asyncio.run(run.refresh(trader, streams, tmp_path / "history.sqlite"))
    assert streams["kalshi"].wanted == {f"K-{close}", f"K-{close + 900}"}
    assert streams["polymarket_us"].wanted == {polymarket_us.slug(close - 900), polymarket_us.slug(close)}
    assert trader.windows[close].tracking == 1e-9 and trader.windows[close + 900].tracking is None   # Not opened yet.
    assert trader.windows[close].offset == 12.5         # The trader has no readings from before it opened.
    assert logs == [f"paper: K-{close} opened at a strike of 80,000.00, tracking error $2.53",
                    f"paper: K-{close} opened before the trader started, its price's offset read from the recorder's quotes: +12.50"]


def test_settling_reads_each_venues_markets_and_settles_their_orders(tmp_path, monkeypatch):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_paper_order(conn, order())
    database.store_paper_order(conn, order(venue="polymarket_us", ticker="P"))
    monkeypatch.setattr(kalshi, "markets", lambda tickers: [Market("kalshi", "K", "K", "S", 100, 1000, 1.0, 2.0, "yes", 1001.0)])
    monkeypatch.setattr(polymarket_us, "window", lambda name: Market("polymarket_us", "P", "P", "S", 100, 1000, 1.0, None, None, None))
    logs = []
    asyncio.run(run.settle(conn, logs.append))
    assert [o["result"] for o in database.load_paper_orders(conn)] == ["yes", None]       # Polymarket US has not resolved yet.
    assert logs == [f"paper kalshi K settled yes: 1 orders, 10 contracts, {10 * (0.6 - fair.fee(0.4)):+.2f}$"]


def test_a_refresh_streams_hyperliquids_market_of_a_window_once_listed_beside_its_perp(tmp_path, monkeypatch):
    close = (int(time.time()) // 900 + 1) * 900
    monkeypatch.setattr(kalshi, "windows", lambda series, start, end: [window(close, 80_000.0), window(close + 900)])
    monkeypatch.setattr(run, "read_tracking", lambda db_path, market, errors: 1e-9)
    monkeypatch.setattr(run, "read_offset", lambda db_path, market: 12.5)
    listed = [Market("hyperliquid", "#1230", "x", run.HYPERLIQUID_SERIES, close - 900, close, 80_062.0, None, None, None),
              Market("hyperliquid", "#5550", "y", "recurring-BTC-1d", close - 900, close, 80_100.0, None, None, None)]   # Not a 15 minute one.
    monkeypatch.setattr(hyperliquid, "windows", lambda: listed)
    trader = PaperTrader(database.connect(tmp_path / "history.sqlite"), lambda line: None)
    streams = {"kalshi": FakeStream(), "hyperliquid": FakeStream()}
    streams["hyperliquid"].wanted = {"BTC", "#999"}
    asyncio.run(run.refresh(trader, streams, tmp_path / "history.sqlite"))
    assert streams["hyperliquid"].wanted == {"BTC", "#1230"} and trader.windows[close].names["hyperliquid"] == "#1230"
    assert "hyperliquid" not in trader.windows[close + 900].names                  # Not listed yet.


def test_settling_reads_hyperliquids_from_its_explorer_and_settles_the_opportunities_too_less_its_fee(tmp_path, monkeypatch):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_paper_order(conn, order(venue="hyperliquid", ticker="#1230", side="no", cost=0.45))
    database.store_opportunities(conn, [opportunity(venue="hyperliquid", ticker="#1230", start=500.0, close_time=1000)])
    monkeypatch.setattr(hyperliquid, "settlements", lambda: [Market("hyperliquid", "#1230", "x", "out-BTC-15m", 100, 1000, 1.0, None, "no", 1008.0)])
    logs = []
    asyncio.run(run.settle(conn, logs.append))
    (o,) = database.load_paper_orders(conn)
    fee = SETTLE_FEES["hyperliquid"]
    assert o["profit"] == pytest.approx(10 * (1 - fee) - 10 * 0.45 - o["fees"])
    (opp,) = database.load_opportunities(conn)
    assert (opp["result"], opp["actual_profit"]) == ("no", -50.0)       # Its take bought Yes, 100 contracts for 50$.
    assert logs[0].startswith("paper hyperliquid #1230 settled no: 1 orders, 10 contracts")


def test_the_report_gives_each_venue_by_edge_time_left_what_brought_the_decision_and_day(tmp_path):
    causes = {1: "kalshi", 2: "coinbase", 3: "time", 4: None}
    orders = [{**order(close_time=1000 * k, decided=1000 * k - 400.0, edge=0.02 + 0.01 * k, cause=causes[k]), "result": "yes" if k % 2 else "no"}
              for k in range(1, 5)]
    for o in orders:
        o["profit"] = o["contracts"] * (o["side"] == o["result"]) - o["contracts"] * o["cost"] - o["fees"]
    lines = []
    late = {**order(venue="hyperliquid", ticker="#1230", close_time=4000, decided=3880.0, cause="kalshi"), "result": "yes"}
    late["profit"] = late["contracts"] * (1 - late["cost"]) - late["fees"]
    report.report([*orders, late, {**order(decided=5000.0), "result": None, "profit": None}], lines.append)
    text = "\n".join(lines)
    assert "6 paper orders from 1970-01-01 00:10 to 1970-01-01 01:23 UTC, 5 settled, 6 filled" in text
    assert "kalshi" in text and "polymarket_us" not in text
    # Kalshi's 5 orders of 10 contracts at 40c, one not settled yet, two of them in the first window.
    assert ("  spent a window, fees included: 5.21$ on average over the 4 windows priced, 5.21$ in each of the 4 it bought in, 8.34$ at "
            "most; 1.2 filled orders a window") in lines
    assert "How to read it, in cents a contract, after each venue's fees:" in lines and "top of book" in lines[1]
    start = next(i for i, line in enumerate(lines) if line.startswith("\nhyperliquid\n"))
    assert any(line.strip().startswith("1-5 min left") for line in lines[start:])          # 2 minutes left, on Hyperliquid,
    assert not any(line.strip().startswith("1-5 min left") for line in lines[:start])      # and none on Kalshi.
    for label in ("all", "edge 2-3c", "edge 3-5c", "edge 5c+", "5-10 min left", "on its book", "on an exchange", "on the clock", "not recorded",
                  "1970-01-01"):
        assert any(line.strip().startswith(label) for line in lines), label
    assert not any(line.strip().startswith("on Kalshi's book") for line in lines[:start])     # Kalshi's own orders cannot come on it,
    assert any(line.strip().startswith("on Kalshi's book") for line in lines[start:])          # Hyperliquid's can.


def test_an_orders_cause_is_grouped_from_its_venue():
    assert [report.cause(order(venue=v, cause=c)) for v, c in (("kalshi", "kalshi"), ("polymarket_us", "polymarket_us"),
                                                               ("polymarket_us", "kalshi"), ("kalshi", "polymarket_us"), ("kalshi", "other"),
                                                               ("kalshi", "kraken"), ("kalshi", "time"), ("kalshi", None))] == \
        ["on its book", "on its book", "on Kalshi's book", "on another book", "on another book", "on an exchange", "on the clock", "not recorded"]


def opportunity(venue="polymarket_us", start=1000.0, take=(100.0, 50.0, 2.0), lasted=0.5, book_changed=1, left=500.0, **more):
    """
    An opportunity as the trader stores it, one order at the threshold or more taking take, (contracts, capital, profit),
    peaking left seconds before its window closes, or none with take None.
    """
    size, capital, profit = take or (None, None, None)
    days = (left + 10) / 86400
    return {"venue": venue, "ticker": "P", "close_time": int(start + left), "side": "yes", "start_time": start, "end_time": start + 1,
            "seconds": 1.0, "peak_time": start, "peak_edge": 0.03, "peak_price": 0.5, "peak_fair": 0.55, "peak_size": 120.0, "peak_capital": 60.0,
            "peak_profit": 2.2, "book_changed": book_changed, "days_held": days, "return_pct": 5.6,
            "min_edge": 0.02, "min_edge_seconds": lasted if take else None, "min_edge_size": 40.0 if take else None,
            "min_edge_capital": 20.0 if take else None, "min_edge_profit": 0.8 if take else None, "take_size": size, "take_capital": capital,
            "take_profit": profit, "take_return_pct": 100 * profit / capital if take else None, "min_edge_total": lasted if take else None,
            "result": None, "actual_profit": None, "actual_return_pct": None, **more}


def test_the_opportunities_report_gives_what_one_order_each_could_have_taken_and_how_long_the_edge_lasted(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    rows = [opportunity(start=1000.0), opportunity(start=2000.0, take=(10.0, 5.0, 0.5), lasted=2.0, book_changed=0, left=800.0),
            opportunity(start=3000.0, take=None)]
    database.store_opportunities(conn, rows)
    assert [o["start_time"] for o in database.load_opportunities(conn, 1500.0)] == [2000.0, 3000.0]
    lines = []
    report.report_opportunities(database.load_opportunities(conn), "since the first", lines.append)
    text = "\n".join(lines)
    assert "3 opportunities at a positive edge after the fee, since the first, over 3 windows priced, 2 reaching the threshold" in text
    assert ("  a window: 0.7 edges on average, in 2 of the 3 windows priced, 1 at most, of 1.0 at any positive edge; "
            "buying them would have cost 18$ a window, expecting 0.83$") in lines
    assert "polymarket_us opportunities (edges of 2c or more)" in text and "kalshi opportunities" not in text
    held = (50.0 * rows[0]["days_held"] + 5.0 * rows[1]["days_held"]) / 55 * 1440
    assert (f"  2 edges: buying each in full, in one order at its best moment, would have cost 55$ with the fees, expecting 2.50$, "
            f"{100 * 2.5 / 55:,.1f}% of the cost, held {held:,.1f} minutes on average until the window paid") in lines
    assert "year" not in text and "annual" not in text
    assert "at 2c or more for 2.000s at the median, 2.000s at the 90th percentile, 2.000s at the longest" in text
    assert "80 contracts stayed, 40$ expecting 1.60$" in text and "own book change in 50%" in text
    assert "2 of the stretches outlasted an order's trip to polymarket_us, 136 ms: through them 80 contracts stayed" in text
    assert any(line.strip().startswith("10-15 min left") for line in lines) and any(line.strip().startswith("5-10 min left") for line in lines)
    database.settle_opportunities(conn, "polymarket_us", "P", "yes")
    lines = []
    report.report_opportunities(database.load_opportunities(conn), "since the first", lines.append)
    assert "  2 settled: bought in full they would have made +55.00$ on their 55$ cost, +100.00%, against +2.50$, +4.55%, expected" in lines
    assert "  at 2c or more for 2.000s in all at the median, every stretch counted, 2.000s at the 90th percentile" in lines


def test_only_a_stretch_outlasting_an_orders_trip_counts_as_one_an_order_could_have_reached():
    lines = []
    report.report_opportunities([opportunity(lasted=0.05), opportunity(start=2000.0, lasted=0.5)], "since the first", lines.append)
    assert "  1 of the stretches outlasted an order's trip to polymarket_us, 136 ms: through them 40 contracts stayed, 20$ expecting 0.80$" in lines


def test_the_status_line_counts_the_opportunities_one_order_could_have_taken_today(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    now = time.time()
    today = max(now - 10, now // 86400 * 86400)          # Since midnight UTC, whenever the test runs.
    database.store_opportunities(conn, [opportunity(start=today), opportunity(start=today + 0.1, take=None),
                                        opportunity(start=now - 2 * 86400)])
    line = run.status(conn, PaperTrader(conn))
    assert line.endswith("; 1 opportunities at their venue's threshold or more, 50$ in one order each expecting +2.00$")


def test_a_time_to_report_from_is_read_in_utc_unless_it_says_otherwise():
    assert report.utc_time("2026-10-14T01:00") == 1791939600
    assert report.utc_time("2026-10-13T21:00-04:00") == 1791939600
