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
