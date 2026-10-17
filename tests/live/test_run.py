"""
Tests for the live trading process: its orders sent from a thread and answered back in the loop, and its settling, with
stand-ins for Hyperliquid.
"""

import asyncio
import pytest
import time
from types import SimpleNamespace
from upordowntrader.api import hyperliquid
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.live import run
from upordowntrader.live.executor import Executor

CLOSE = 1_791_897_300
ORDER = {"venue": "hyperliquid", "ticker": "#1230", "close_time": CLOSE, "decided": CLOSE - 200.0, "side": "yes", "chance": 0.53,
         "edge": 0.05, "size": 50, "limit_price": 0.48, "cause": "hyperliquid", "book_at": CLOSE - 200.3}


def stand_ins():
    answers, watched = [], []
    trader = SimpleNamespace(on_answer=lambda order, filled: answers.append((order["ticker"], filled)))
    brakes = SimpleNamespace(watch=watched.append)
    return trader, brakes, answers, watched


def send(sender, order):
    """
    Send an order and wait in the loop for its answer to come back.
    """
    async def go():
        sender.loop = asyncio.get_running_loop()
        sender.send(order)
        for _ in range(200):
            await asyncio.sleep(0.01)
            if sender.trader.answered():
                return
    asyncio.run(go())


def test_an_order_goes_out_from_its_thread_and_its_answer_comes_back_to_the_trader_and_the_brakes(tmp_path):
    trader, brakes, answers, watched = stand_ins()
    trader.answered = lambda: answers
    filled = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": "30", "avgPx": "0.475", "oid": 4}}]}}}
    sender = run.Sender(tmp_path / "history.sqlite", trader, brakes, None, lambda line: None, place=lambda *args: filled)
    send(sender, ORDER)
    sender.stop()
    assert answers == [("#1230", 30.0)] and watched[0].status == "partial"
    (trade,) = database.load_trades(database.connect(tmp_path / "history.sqlite"))
    assert (trade["fair"], trade["quantity"], trade["limit_price"], trade["signal_ts"]) == (0.53, 50, 0.48, CLOSE - 200.0)
    assert trade["cause"] == "hyperliquid"


def test_an_order_whose_thread_failed_counts_in_full_against_the_cap_and_a_dry_run_sends_and_stores_nothing(tmp_path):
    trader, brakes, answers, watched = stand_ins()
    trader.answered = lambda: answers

    def place(*args):
        raise AssertionError("not sent on a dry run")

    sender = run.Sender(tmp_path / "missing" / "no.sqlite", trader, brakes, None, lambda line: None, place=place)
    sender.executor = SimpleNamespace(trade=lambda decision: 1 / 0)
    send(sender, ORDER)
    assert answers == [("#1230", 50)] and watched[0].status == "error" and "ZeroDivisionError" in watched[0].note
    answers.clear()
    dry = run.Sender(tmp_path / "history.sqlite", trader, brakes, None, lambda line: None, dry_run=True, place=place)
    send(dry, ORDER)
    assert answers == [("#1230", 0.0)] and database.load_trades(database.connect(tmp_path / "history.sqlite")) == []


def test_settling_pays_the_trades_and_opportunities_hyperliquid_settled_then_reads_the_balance_and_checks_the_brakes(tmp_path, monkeypatch):
    conn = database.connect(tmp_path / "history.sqlite")
    answer = {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": "10", "avgPx": "0.4", "oid": 4}}]}}}
    Executor(conn, lambda *args: answer, lambda line: None).trade({"ticker": "#1230", "close_time": 1000, "side": "yes", "signal_ts": 800.0,
                                                                   "fair": 0.5, "edge": 0.05, "quantity": 10.0, "limit_price": 0.4})
    database.store_opportunities(conn, [{**{c: None for c in database.OPPORTUNITY_COLUMNS}, "venue": "hyperliquid", "ticker": "#1230",
                                         "close_time": 1000, "side": "yes", "start_time": 800.0, "end_time": 801.0, "seconds": 1.0,
                                         "peak_time": 800.0, "peak_edge": 0.05, "peak_price": 0.4, "peak_fair": 0.45, "peak_size": 10.0,
                                         "peak_capital": 4.0, "peak_profit": 0.5, "book_changed": 1, "days_held": 0.002, "return_pct": 12.5,
                                         "min_edge": 0.02, "take_size": 10.0, "take_capital": 4.0, "take_profit": 0.5, "mode": "live"}])
    monkeypatch.setattr(hyperliquid, "settlements", lambda: [Market("hyperliquid", "#1230", "x", "out-BTC-15m", 100, 1000, 80_000.0, None,
                                                                    "yes", 1008.0)])
    monkeypatch.setattr(run, "read_balance", lambda: 104.0)
    monkeypatch.setattr(run, "read_fills", lambda since: [{"oid": 4, "time": 1_000_500}, {"oid": 4, "time": 1_000_200}, {"oid": 5, "time": 9}])
    database.update(conn, "orders", 1, {"sent_at": time.time() - 100})
    checked = []
    brakes = SimpleNamespace(balance=None, check_results=lambda: checked.append(brakes.balance))
    asyncio.run(run.settle(conn, Executor(conn, log=lambda line: None), brakes))
    (s,) = database.load_settlements(conn)
    assert s["profit"] == pytest.approx(10 * (1 - hyperliquid.SETTLE_FEE_RATE) - 4.0) and checked == [104.0]
    assert database.load_opportunities(conn, mode="live")[0]["actual_profit"] == pytest.approx(10 * (1 - hyperliquid.SETTLE_FEE_RATE) - 4.0)
    assert database.load_opportunities(conn, mode="paper") == []
    assert database.load_orders(conn, 1)[0]["venue_at"] == 1000.2        # Hyperliquid's time for the order's first fill.


def test_live_trading_will_not_start_on_a_key_file_that_holds_an_address_and_says_so_without_showing_it(tmp_path, monkeypatch):
    for name in ("ACCOUNT_FILE", "AGENT_KEY_FILE"):
        monkeypatch.setattr(hyperliquid, name, tmp_path / f"{name.lower()}.txt")
    for name in ("KEY_ID_FILE", "PRIVATE_KEY_FILE"):
        monkeypatch.setattr(run.kalshi, name, tmp_path / f"kalshi_{name.lower()}")
        (tmp_path / f"kalshi_{name.lower()}").write_text("x")
    assert run.startup_problems(dry_run=True) == []
    assert run.startup_problems() == [f"no {hyperliquid.ACCOUNT_FILE}", f"no {hyperliquid.AGENT_KEY_FILE}"]
    address = "0x430c2808faccf318adcabb7d74ef507b49480d2d"
    hyperliquid.ACCOUNT_FILE.write_text(address)
    hyperliquid.AGENT_KEY_FILE.write_text(address + "\n")
    (problem,) = run.startup_problems()
    assert "an address's length" in problem and address[2:] not in problem
    hyperliquid.AGENT_KEY_FILE.write_text("0x0123456789012345678901234567890123456789012345678901234567890123\n")    # The SDK's test key.
    assert run.startup_problems() == []
