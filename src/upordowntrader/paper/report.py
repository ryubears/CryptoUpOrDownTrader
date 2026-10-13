"""
Report what the paper orders made, see paper/trader.py, as the replay reports its trades: by venue, with two standard
errors that count each window once, as its orders share its outcome, then by the edge they were decided at, by the
minutes left, and day by day.

    expected    the fair price's edge after the fee on the contracts filled, what they make on average if it is right
    realized    what they made, from how the windows settled
    one each    what one contract an order, the best price filled, made: the edge without the depth

Run from src/ with:
    python3 -m upordowntrader.paper.report
"""

import argparse
import numpy as np
from datetime import UTC, datetime
from upordowntrader.backtest.replay import window_mean
from upordowntrader.db import database
from upordowntrader.model import fair
from upordowntrader.paper.trader import FEE_RATES, VENUES

EDGES = ((0.02, 0.03), (0.03, 0.05), (0.05, 1.0))       # Edges at the decision the report is broken into, in dollars a contract.
SPANS = ((600, 900), (300, 600))                        # Seconds left it is broken into.


def one_each(order):
    """
    What one contract at the best price an order filled made after the fee.
    """
    return fair.edge(float(order["side"] == order["result"]), order["best"], FEE_RATES[order["venue"]])


def row(label, orders):
    """
    One line of the report for settled orders: how many, how many filled, in how many windows, the contracts, and the
    profits a contract, with two standard errors.
    """
    filled = [o for o in orders if o["contracts"] > 0]
    if not filled:
        return f"{label:>16}{len(orders):>8}{0:>8}"
    window = [o["close_time"] for o in filled]
    contracts = np.array([o["contracts"] for o in filled])
    realized, error = window_mean([o["profit"] / o["contracts"] for o in filled], contracts, window)
    one, one_error = window_mean([one_each(o) for o in filled], np.ones(len(filled)), window)
    expected = sum(o["expected"] for o in filled) / contracts.sum()
    profit = sum(o["profit"] for o in filled)
    return (f"{label:>16}{len(orders):>8}{len(filled):>8}{len(set(window)):>9}{contracts.sum():>11,.0f}{100 * expected:>+10.1f}c"
            f"{100 * realized:>+9.1f} ±{100 * error:>4.1f}c{100 * one:>+9.1f} ±{100 * one_error:>4.1f}c{profit:>+10.2f}$")


def report(orders, log=print):
    """
    Print the report of the paper orders, as dicts of the paper_orders columns.
    """
    if not orders:
        log("no paper orders yet, run upordowntrader.paper.run for a while")
        return
    settled = [o for o in orders if o["result"]]
    first, last = (datetime.fromtimestamp(orders[i]["decided"], UTC) for i in (0, -1))
    log(f"\n{len(orders)} paper orders from {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC, {len(settled)} settled, "
        f"{sum(o['contracts'] > 0 for o in orders)} filled")
    header = (f"{'':>16}{'orders':>8}{'filled':>8}{'windows':>9}{'contracts':>11}{'expected':>11}{'realized':>16}{'one each':>16}{'profit':>11}\n"
              f"{'':>52}{'a contract':>11}{'a contract':>16}{'realized':>16}")
    for venue in VENUES:
        mine = [o for o in settled if o["venue"] == venue]
        if not mine:
            continue
        log(f"\n{venue}\n{header}")
        log(row("all", mine))
        for low, high in EDGES:
            label = f"edge {100 * low:.0f}-{100 * high:.0f}c" if high < 1 else f"edge {100 * low:.0f}c+"
            log(row(label, [o for o in mine if low <= o["edge"] < high]))
        for low, high in SPANS:
            log(row(f"{low // 60}-{high // 60} min left", [o for o in mine if low < o["close_time"] - o["decided"] <= high]))
        for day in sorted({datetime.fromtimestamp(o["decided"], UTC).date() for o in mine}):
            log(row(f"{day}", [o for o in mine if datetime.fromtimestamp(o["decided"], UTC).date() == day]))
    log("\nAll after each venue's taker fee, two standard errors counting each window once. Expected: the fair price's edge on the")
    log("contracts filled. Realized: what they made. One each: one contract an order at the best price filled.")


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Report what the paper orders made.")
    ap.add_argument("--db", help="the database the orders are in, by default data/history.sqlite")
    report(database.load_paper_orders(database.connect(ap.parse_args().db)))
