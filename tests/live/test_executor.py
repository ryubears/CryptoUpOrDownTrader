"""
Tests for the live executor's record of trades, orders, and settlements, with a stand-in for Hyperliquid's order call.
"""

import pytest
from dataclasses import replace
from upordowntrader.api import hyperliquid
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.live import executor as live
from upordowntrader.live.executor import PAYOUT_SECONDS, Executor

CLOSE = 1_791_897_300
DECISION = {"ticker": "#1277300", "close_time": CLOSE, "side": "no", "signal_ts": CLOSE - 400.0, "fair": 0.6, "edge": 0.05,
            "quantity": 10.0, "limit_price": 0.55, "book_at": CLOSE - 400.4, "book_ts": CLOSE - 400.01, "cause": "hyperliquid"}


def filled(size, price):
    return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"totalSz": str(size), "avgPx": str(price), "oid": 9}}]}}}


def executor(conn, place, find=None):
    return Executor(conn, place, lambda line: None, find=find or (lambda *args: None), wait=lambda seconds: None)


def test_a_trade_stores_its_order_before_sending_and_the_answer_after(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    seen, clock = [], iter([100.0, 100.25])

    def place(coin, is_buy, price, size, tif, cloid):
        seen.append((coin, is_buy, price, size, tif, database.load_orders(conn, 1)[0]["status"]))
        return filled(7, 0.54)

    placed = Executor(conn, place, lambda line: None, lambda: next(clock)).trade(DECISION)
    assert seen == [("#1277301", True, 0.55, 10.0, "Ioc", "sent")]           # No is bought by its own coin, stored before it went.
    assert (placed.status, placed.filled, placed.price) == ("partial", 7.0, 0.54)
    (trade,) = database.load_trades(conn)
    assert (trade["id"], trade["mode"], trade["status"], trade["filled"]) == (placed.trade_id, "live", "partial", 7.0)
    assert trade["cost"] == pytest.approx(3.78)
    assert trade["pays_at"] == CLOSE + PAYOUT_SECONDS and trade["cause"] == "hyperliquid"
    (order,) = database.load_orders(conn, placed.trade_id)
    assert (order["status"], order["filled"], order["venue_order_id"], order["latency_ms"], order["tif"]) == ("partial", 7.0, "9", 250, "Ioc")
    assert order["client_id"].startswith("0x") and len(order["client_id"]) == 34 and order["book_at"] == DECISION["book_at"]


def test_a_lost_answer_is_found_by_its_client_id_or_left_unknown(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    asked = []

    def place(*args):
        raise TimeoutError("read timed out")

    def find(cloid, sent, quantity):
        asked.append(cloid)
        return "partial", 4.0, 0.5, 12, "found by its client id"

    placed = executor(conn, place, find).trade(DECISION)
    (order,) = database.load_orders(conn, placed.trade_id)
    assert asked == [order["client_id"]] and (placed.status, placed.filled) == ("partial", 4.0)
    assert order["status"] == "partial" and "TimeoutError" in order["note"] and database.load_trades(conn)[0]["cost"] == pytest.approx(2.0)
    unknown = executor(conn, place).trade(DECISION)                    # Hyperliquid cannot say:
    assert unknown.status == "error" and database.load_trades(conn)[1]["status"] == "failed"      # the outcome stays unknown.


def test_recovering_reads_what_the_order_filled_from_the_accounts_fills(monkeypatch):
    answers = iter([{"status": "unknownOid"},
                    {"status": "order", "order": {"order": {"oid": 7, "origSz": "10", "sz": "0"}, "status": "filled"}},
                    {"status": "order", "order": {"order": {"oid": 8, "origSz": "10", "sz": "10"}, "status": "canceled"}}])
    monkeypatch.setattr(hyperliquid, "order_status", lambda user, cloid: next(answers))
    monkeypatch.setattr(hyperliquid, "user_fills", lambda user, start: [{"oid": 7, "px": "0.5", "sz": "6"}, {"oid": 7, "px": "0.6", "sz": "4"},
                                                                        {"oid": 5, "px": "0.9", "sz": "3"}])
    assert live.recover("0xme", "0x1", 100.0, 10.0)[:3] == ("unfilled", 0.0, None)          # Never reached Hyperliquid.
    assert live.recover("0xme", "0x1", 100.0, 10.0)[:4] == ("filled", 10.0, pytest.approx(0.54), 7)
    assert live.recover("0xme", "0x1", 100.0, 10.0)[:2] == ("unfilled", 0.0)


def test_a_refused_order_fails_its_trade_and_one_that_met_nothing_is_unfilled(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    answers = iter([{"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Insufficient balance."}]}}},
                    {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"error": "Order could not immediately match"}]}}}])
    run = executor(conn, lambda *args: next(answers))
    assert [run.trade(DECISION).status for _ in range(2)] == ["rejected", "unfilled"]
    assert [t["status"] for t in database.load_trades(conn)] == ["failed", "unfilled"]
    assert database.load_orders(conn, 1)[0]["note"] == "Insufficient balance."


def test_a_trade_settles_once_paying_a_dollar_a_contract_when_its_side_won_less_the_fee(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    answers = iter([filled(10, 0.55), filled(10, 0.47), {"status": "ok", "response": {"type": "order", "data": {"statuses": [
        {"error": "Order could not immediately match"}]}}}])
    run = executor(conn, lambda *args: next(answers))
    run.trade(DECISION)
    run.trade({**DECISION, "side": "yes", "limit_price": 0.47})
    run.trade(DECISION)                                         # Fills nothing, and settles at nothing.
    market = Market("hyperliquid", "#1277300", "out-BTC-15m-x", "out-BTC-15m", CLOSE - 900, CLOSE, 80_000.0, None, "no", CLOSE + 8.0)
    assert run.settle([market, replace(market, ticker="#999", result="yes")]) == 2
    assert run.settle([market]) == 0
    no, yes, nothing = database.load_settlements(conn)
    assert (no["result"], no["payout"], no["settled_at"]) == ("no", 10.0, CLOSE + 8.0)
    assert no["fees"] == pytest.approx(10 * hyperliquid.SETTLE_FEE_RATE) and no["profit"] == pytest.approx(10 - no["fees"] - 5.5)
    assert (yes["payout"], yes["fees"], yes["profit"]) == (0.0, 0.0, pytest.approx(-4.7))
    assert (nothing["payout"], nothing["profit"]) == (0.0, 0.0) and database.unsettled_trades(conn) == []
