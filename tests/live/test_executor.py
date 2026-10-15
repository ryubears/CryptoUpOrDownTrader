"""
Tests for the live executor's record of trades, orders, and settlements, with a stand-in for Hyperliquid's order call.
"""

import pytest
from dataclasses import replace
from upordowntrader.api import hyperliquid
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.live.executor import PAYOUT_SECONDS, Executor

CLOSE = 1_791_897_300
DECISION = {"ticker": "#1277300", "close_time": CLOSE, "side": "no", "signal_ts": CLOSE - 400.0, "fair": 0.6, "edge": 0.05,
            "quantity": 10.0, "limit_price": 0.55, "book_at": CLOSE - 400.4, "book_ts": CLOSE - 400.01}


def filled(size, price):
    return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": str(size), "avgPx": str(price), "oid": 9}}]}}}


def test_a_trade_stores_its_order_before_sending_and_the_answer_after(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    seen, clock = [], iter([100.0, 100.25])

    def place(coin, is_buy, price, size, tif, cloid):
        seen.append((coin, is_buy, price, size, tif, database.load_orders(conn, 1)[0]["status"]))
        return filled(7, 0.54)

    trade_id = Executor(conn, place, lambda line: None, lambda: next(clock)).trade(DECISION)
    assert seen == [("#1277301", True, 0.55, 10.0, "Ioc", "sent")]           # No is bought by its own coin, stored before it went.
    (trade,) = database.load_trades(conn)
    assert (trade["id"], trade["mode"], trade["status"], trade["filled"], trade["cost"]) == (trade_id, "live", "partial", 7.0, pytest.approx(3.78))
    assert trade["pays_at"] == CLOSE + PAYOUT_SECONDS
    (order,) = database.load_orders(conn, trade_id)
    assert (order["status"], order["filled"], order["venue_order_id"], order["latency_ms"], order["tif"]) == ("partial", 7.0, "9", 250, "Ioc")
    assert order["client_id"].startswith("0x") and len(order["client_id"]) == 34 and order["book_at"] == DECISION["book_at"]


def test_a_lost_answer_leaves_the_order_to_be_found_by_its_id_and_the_trade_failed(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")

    def place(*args):
        raise TimeoutError("read timed out")

    trade_id = Executor(conn, place, lambda line: None).trade(DECISION)
    (order,) = database.load_orders(conn, trade_id)
    assert order["status"] == "error" and "TimeoutError" in order["note"] and database.load_trades(conn)[0]["status"] == "failed"


def test_a_refused_order_fails_its_trade_and_one_that_met_nothing_is_unfilled(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    answers = iter([{"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Insufficient balance."}]}}},
                    {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Order could not immediately match"}]}}}])
    executor = Executor(conn, lambda *args: next(answers), lambda line: None)
    executor.trade(DECISION)
    executor.trade(DECISION)
    assert [t["status"] for t in database.load_trades(conn)] == ["failed", "unfilled"]
    assert database.load_orders(conn, 1)[0]["note"] == "Insufficient balance."


def test_a_trade_settles_once_paying_a_dollar_a_contract_when_its_side_won_less_the_fee(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    executor = Executor(conn, lambda *args: filled(10, 0.55), lambda line: None)
    executor.trade(DECISION)
    executor.trade({**DECISION, "side": "yes", "limit_price": 0.47})
    market = Market("hyperliquid", "#1277300", "out-BTC-15m-x", "out-BTC-15m", CLOSE - 900, CLOSE, 80_000.0, None, "no", CLOSE + 8.0)
    assert executor.settle([market, replace(market, ticker="#999", result="yes")]) == 2
    assert executor.settle([market]) == 0
    no, yes = database.load_settlements(conn)
    assert (no["result"], no["payout"], no["settled_at"]) == ("no", 10.0, CLOSE + 8.0)
    assert no["fees"] == pytest.approx(10 * hyperliquid.SETTLE_FEE_RATE) and no["profit"] == pytest.approx(10 - no["fees"] - 5.5)
    assert (yes["payout"], yes["fees"], yes["profit"]) == (0.0, 0.0, pytest.approx(-5.5))
