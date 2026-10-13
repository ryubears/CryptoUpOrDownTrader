"""
Tests for the paper trading process's periodic work, the paper orders' storage and settling, and the report, with stand ins
for the venues' reads and streams.
"""

import asyncio
import pytest
import time
from upordowntrader.api import kalshi, polymarket_us
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.model import fair
from upordowntrader.paper import report, run
from upordowntrader.paper.trader import PaperTrader


def order(venue="kalshi", ticker="K", close_time=1000, decided=500.0, side="yes", contracts=10.0, cost=0.4, edge=0.03, **more):
    """
    A paper order as the trader stores it, filled at cost a contract.
    """
    rate = fair.FEE_RATE if venue == "kalshi" else polymarket_us.TAKER_FEE_RATE
    return {"venue": venue, "ticker": ticker, "close_time": close_time, "decided": decided, "side": side, "price": 80_000.0, "model": 0.5,
            "mid": 0.4, "fair": 0.45, "edge": edge, "limit_price": cost, "offered": 50.0, "arrived": decided + 0.02, "contracts": contracts,
            "cost": cost if contracts else None, "best": cost if contracts else None, "fees": contracts * fair.fee(cost, rate),
            "expected": contracts * fair.edge(0.45, cost, rate), **more}


def test_settling_pays_the_side_that_won_less_cost_and_fees_once(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    stored = (order(), order(decided=501.0, side="no", cost=0.55), order(decided=502.0, contracts=0.0),
              order(ticker="OPEN", close_time=9e9, decided=503.0))
    for o in stored:
        database.store_paper_order(conn, o)
    assert database.unsettled_paper_markets(conn, 1000) == [("kalshi", "K")]
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


def test_the_report_gives_each_venue_by_edge_time_left_and_day(tmp_path):
    orders = [{**order(close_time=1000 * k, decided=1000 * k - 400.0, edge=0.02 + 0.01 * k), "result": "yes" if k % 2 else "no"}
              for k in range(1, 5)]
    for o in orders:
        o["profit"] = o["contracts"] * (o["side"] == o["result"]) - o["contracts"] * o["cost"] - o["fees"]
    lines = []
    report.report([*orders, {**order(decided=5000.0), "result": None, "profit": None}], lines.append)
    text = "\n".join(lines)
    assert "5 paper orders from 1970-01-01 00:10 to 1970-01-01 01:23 UTC, 4 settled, 5 filled" in text
    assert "kalshi" in text and "polymarket_us" not in text
    for label in ("all", "edge 2-3c", "edge 3-5c", "edge 5c+", "5-10 min left", "1970-01-01"):
        assert any(line.strip().startswith(label) for line in lines), label


def opportunity(venue="polymarket_us", start=1000.0, take=(100.0, 50.0, 2.0), lasted=0.5, book_changed=1, left=500.0, **more):
    """
    An opportunity as the trader stores it, one order at the threshold or more taking take, (contracts, capital, profit),
    peaking left seconds before its window closes, or none with take None.
    """
    size, capital, profit = take or (None, None, None)
    days = (left + 10) / 86400
    return {"venue": venue, "ticker": "P", "close_time": int(start + left), "side": "yes", "start_time": start, "end_time": start + 1,
            "seconds": 1.0, "peak_time": start, "peak_edge": 0.03, "peak_price": 0.5, "peak_fair": 0.55, "peak_size": 120.0, "peak_capital": 60.0,
            "peak_profit": 2.2, "book_changed": book_changed, "days_held": days, "return_pct": 5.6, "annual_pct": 5.6 * 365 / days,
            "min_edge": 0.02, "min_edge_seconds": lasted if take else None, "min_edge_size": 40.0 if take else None,
            "min_edge_capital": 20.0 if take else None, "min_edge_profit": 0.8 if take else None, "take_size": size, "take_capital": capital,
            "take_profit": profit, **more}


def test_the_opportunities_report_gives_what_one_order_each_could_have_taken_and_how_long_the_edge_lasted(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    rows = [opportunity(start=1000.0), opportunity(start=2000.0, take=(10.0, 5.0, 0.5), lasted=2.0, book_changed=0, left=800.0),
            opportunity(start=3000.0, take=None)]
    database.store_opportunities(conn, rows)
    assert [o["start_time"] for o in database.load_opportunities(conn, 1500.0)] == [2000.0, 3000.0]
    lines = []
    report.report_opportunities(database.load_opportunities(conn), "since the first", lines.append)
    text = "\n".join(lines)
    assert "3 opportunities at a positive edge after the fee, since the first, 2 reaching the threshold" in text
    assert "polymarket_us opportunities (one order at 2c or more)" in text and "kalshi opportunities" not in text
    assert "2 episodes could have taken 55$ in one order each, expecting 2.50$" in text
    yearly = 100 * (2.0 * 365 / rows[0]["days_held"] + 0.5 * 365 / rows[1]["days_held"]) / 55
    assert f"{100 * 2.5 / 55:,.1f}% on capital, {yearly:,.1f}% a year" in text
    assert "at 2c or more for 2.000s at the median, 2.000s at the 90th percentile, 2.000s at the longest" in text
    assert "80 contracts stayed, 40$ expecting 1.60$" in text and "own book change in 50%" in text
    assert "2 of the stretches outlasted an order's trip to polymarket_us, 136 ms: through them 80 contracts stayed" in text
    assert any(line.strip().startswith("10-15 min left") for line in lines) and any(line.strip().startswith("5-10 min left") for line in lines)


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
    assert line.endswith("; 1 opportunities at 2c or more, 50$ in one order each expecting +2.00$")
