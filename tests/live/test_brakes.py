"""
Tests for when live trading halts and how a human is told, on a made up clock, with a stand-in for the mail server.
"""

import json
import pytest
from upordowntrader.db import database
from upordowntrader.live import brakes as halting
from upordowntrader.live import notify
from upordowntrader.live.brakes import Brakes
from upordowntrader.live.executor import Placed

NOW = 1_792_300_000.0
HOUR = 3600


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def files(tmp_path, monkeypatch):
    """
    The halt file and the email settings in a folder of the test's own, and the emails sent.
    """
    monkeypatch.setattr(halting, "HALT_FILE", tmp_path / "live_halt.txt")
    monkeypatch.setattr(notify, "EMAIL_FILE", tmp_path / "email.json")
    (tmp_path / "email.json").write_text(json.dumps({"host": "smtp.example.com", "from": "me@example.com", "to": ["me@example.com"]}))
    return tmp_path


def make(tmp_path, clock=None, balance=100.0):
    clock = clock or Clock(NOW)
    conn = database.connect(tmp_path / "history.sqlite")
    emails, logs = [], []
    b = Brakes(conn, logs.append, notify.Notifier(conn, logs.append, lambda settings, subject, body: emails.append((subject, body)), clock), clock)
    b.balance = balance
    return b, emails, logs


def settled(conn, close, profit, at, cost=5.0):
    """
    A live trade of a window closing at close that cost cost and made profit, settled at at.
    """
    trade_id = database.store_trade(conn, {"mode": "live", "venue": "hyperliquid", "ticker": f"#{close}", "close_time": close, "side": "yes",
                                           "signal_ts": close - 100, "fair": 0.5, "edge": 0.03, "quantity": 10.0, "limit_price": 0.5, "filled": 10.0,
                                           "cost": cost, "fees": 0.0, "status": "filled", "pays_at": close + 10, "cause": "hyperliquid"})
    database.store_settlement(conn, {"trade_id": trade_id, "mode": "live", "settled_at": at, "result": "yes", "payout": cost + profit,
                                     "fees": 0.0, "profit": profit})


def test_a_loss_of_30_percent_of_the_money_over_12_hours_halts_writes_why_and_emails(files):
    b, emails, _ = make(files)
    settled(b.conn, int(NOW - 13 * HOUR), -500.0, NOW - 13 * HOUR)          # Too long ago to count.
    settled(b.conn, int(NOW - 2 * HOUR), -20.0, NOW - 2 * HOUR)
    b.check_results()
    assert b.halted is None                                     # 20 of the 120 before: 17%.
    settled(b.conn, int(NOW - HOUR), -23.0, NOW - HOUR)
    b.balance = 57.0
    b.check_results()                                           # 43 of the 100 before: 43%.
    assert "lost 43.00$, 43% of the 100.00$ live money before them" in b.halted
    assert halting.HALT_FILE.read_text().startswith("2026-10-") and "halted: the live trades settled" in halting.HALT_FILE.read_text()
    ((subject, body),) = emails
    assert subject == halting.SUBJECT and "remove" in body and "57.00 USDC" in body
    (alert,) = database.load_alerts(b.conn)
    assert (alert["kind"], alert["sent_at"], alert["error"]) == ("halt", NOW, None)


def test_ten_losing_windows_in_a_row_halt_and_a_winning_one_starts_the_count_again(files):
    b, _, _ = make(files, balance=10_000.0)
    for k in range(9):
        settled(b.conn, 1000 * k, -1.0, NOW - 1000 + k)
    settled(b.conn, 9000, +3.0, NOW - 990)
    for k in range(10, 19):
        settled(b.conn, 1000 * k, -1.0, NOW - 980 + k)
    b.check_results()
    assert b.halted is None                                     # Nine in a row since the winner.
    settled(b.conn, 19_000, -0.5, NOW - 900)
    settled(b.conn, 19_000, +0.2, NOW - 900)                    # A window's trades count together.
    b.check_results()
    assert "lost money in each of the last 10 windows" in b.halted


def test_a_halt_lasts_until_the_file_is_removed_across_restarts_and_counts_nothing_before_it_again(files):
    clock = Clock(NOW)
    b, emails, _ = make(files, clock)
    for k in range(10):
        settled(b.conn, 1000 * k, -1.0, NOW - 100 + k)
    b.check_results()
    assert b.halted and len(emails) == 1
    again, _, logs = make(files, clock)                         # A restart keeps the halt,
    assert again.halted.startswith("live_halt.txt is there") and again.since == NOW
    halting.HALT_FILE.unlink()
    clock.now += 60
    again.refresh()
    again.check_results()
    assert again.halted is None and logs[-1].endswith("was removed")       # and the same losses do not halt again.
    halting.HALT_FILE.write_text("")
    again.refresh()
    assert again.halted == "live_halt.txt is there, remove it to resume: made by hand"


def test_an_order_of_unknown_outcome_halts_at_once_and_three_refusals_in_a_row_do(files):
    b, emails, _ = make(files)

    def order(status):
        trade_id = database.store_trade(b.conn, {"mode": "live", "venue": "hyperliquid", "ticker": "#1", "close_time": 1, "side": "yes",
                                                 "signal_ts": NOW, "fair": 0.5, "edge": 0.03, "quantity": 10.0, "limit_price": 0.5, "filled": 0.0,
                                                 "cost": 0.0, "fees": 0.0, "status": "failed", "pays_at": 11, "cause": None})
        database.store_order(b.conn, {"trade_id": trade_id, "venue": "hyperliquid", "ticker": "#1", "coin": "#1", "action": "buy",
                                      "outcome": "yes", "quantity": 10.0, "limit_price": 0.5, "tif": "Ioc", "client_id": "0x1", "sent_at": NOW,
                                      "status": status, "venue_order_id": None, "answered_at": NOW, "latency_ms": 100, "filled": 0.0,
                                      "dollars": 0.0, "fees": 0.0, "note": None, "response": None, "book_at": None, "book_ts": None})
        b.watch(Placed(trade_id, status, 0.0, None, "Insufficient balance."))

    for status in ("rejected", "rejected", "unfilled", "rejected", "rejected"):
        order(status)
    assert b.halted is None
    order("rejected")
    assert b.halted == "Hyperliquid refused 3 orders in a row, the last with: Insufficient balance."
    b.halted = None
    halting.HALT_FILE.unlink()
    b.watch(Placed(7, "error", 0.0, None, "TimeoutError: read timed out"))
    assert b.halted == "an order's outcome is unknown, trade 7: TimeoutError: read timed out" and len(emails) == 2


def test_an_alert_without_email_settings_is_still_stored_with_why(files):
    (files / "email.json").unlink()
    conn = database.connect(files / "history.sqlite")
    logs = []
    alert_id = notify.Notifier(conn, logs.append, clock=lambda: NOW).send("halt", "subject", "body")
    (alert,) = database.load_alerts(conn)
    assert alert["id"] == alert_id and alert["sent_at"] is None and alert["error"] == "no email settings in email.json"
    assert logs == ["alert: subject", f"alert {alert_id} was not emailed: no email settings in email.json"]
