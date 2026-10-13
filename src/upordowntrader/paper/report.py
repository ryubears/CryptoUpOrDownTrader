"""
Report what the paper orders made, see paper/trader.py, as the replay reports its trades: by venue, with two standard
errors that count each window once, as its orders share its outcome, then by the edge they were decided at, by the
minutes left, and day by day.

    expected    the fair price's edge after the fee on the contracts filled, what they make on average if it is right
    realized    what they made, from how the windows settled
    one each    what one contract an order, the best price filled, made: the edge without the depth

Then the opportunities the trader saw, as PredictionArbitrageTrader's summary shows its own: each venue's that one order
could have taken at the threshold or more, what those orders would have cost with the fees, the profit they were expected
to make, its return on that capital and its rate a year, both weighted by capital, until the windows paid, and how long
the edge stayed at the threshold or more. Then the contracts that stayed at it through each one's longest stretch, which
an order sent at any moment of it could have had, of all of them and of those whose stretch outlasted an order's trip to
the venue, see trader.ORDER_SECONDS, which an order could have reached, how often the peak came with the venue's own book
change rather than the fair price's, by minutes left, and the largest. The profits are the fair price's expectation, not
locked in.

Run from src/ with:
    python3 -m upordowntrader.paper.report
    python3 -m upordowntrader.paper.report --hours 24
"""

import argparse
import numpy as np
import time
from datetime import UTC, datetime
from upordowntrader.backtest.replay import window_mean
from upordowntrader.common.stats import quantile
from upordowntrader.db import database
from upordowntrader.model import fair
from upordowntrader.paper.trader import FEE_RATES, ORDER_SECONDS, VENUES

EDGES = ((0.02, 0.03), (0.03, 0.05), (0.05, 1.0))       # Edges at the decision the report is broken into, in dollars a contract.
SPANS = ((600, 900), (300, 600))                        # Seconds left it is broken into.
LARGEST = 5             # The largest opportunities listed.


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


# OPPORTUNITIES

def percent(value):
    """
    A percent for a table cell, or a dash when it cannot be told.
    """
    return "-" if value is None else f"{value:,.1f}"


def seconds(value):
    """
    A duration in seconds to the thousandth, as '0.001s', for a table cell.
    """
    return f"{value or 0:,.3f}s"


def totals(group):
    """
    What the opportunities' one orders would have cost, (capital, expected profit, return %, annual %, minutes held), the
    returns and minutes weighted by capital, each over the days until its window paid.
    """
    capital, profit = sum(o["take_capital"] for o in group), sum(o["take_profit"] for o in group)
    if capital <= 0:
        return capital, profit, None, None, None
    yearly = sum(o["take_profit"] * 365 / o["days_held"] for o in group)
    minutes = sum(o["take_capital"] * o["days_held"] for o in group) / capital * 1440
    return capital, profit, 100 * profit / capital, 100 * yearly / capital, minutes


def print_table(title, header, body, log=print, left=1):
    """
    Print a small aligned table with a title line, its first left columns aligned left and the rest right.
    """
    log(f"\n{title}")
    widths = [max(len(str(r[i])) for r in [header, *body]) for i in range(len(header))]
    for r in [header, *body]:
        log("  " + "  ".join(str(v).ljust(w) if i < left else str(v).rjust(w) for i, (v, w) in enumerate(zip(r, widths, strict=True))))


def report_opportunities(opportunities, label, log=print):
    """
    Print the opportunities, as dicts of the opportunities columns, that one order could have taken at the threshold or
    more, see the module's docstring. label says over what time.
    """
    taken = [o for o in opportunities if o["take_size"]]
    log(f"\n{len(opportunities):,} opportunities at a positive edge after the fee, {label}, {len(taken):,} reaching the threshold")
    for venue in VENUES:
        rows = sorted((o for o in taken if o["venue"] == venue), key=lambda o: o["take_profit"], reverse=True)
        if not rows:
            continue
        cents = f"{100 * rows[0]['min_edge']:.0f}c"
        capital, profit, ret, annual, minutes = totals(rows)
        lasted = [o["min_edge_seconds"] for o in rows]
        through = [o for o in rows if o["min_edge_size"]]
        log(f"\n{venue} opportunities (one order at {cents} or more)")
        log(f"  {len(rows):,} episodes could have taken {capital:,.0f}$ in one order each, expecting {profit:,.2f}$")
        log(f"  {percent(ret)}% on capital, {percent(annual)}% a year, held {minutes or 0:,.1f} minutes on average")
        log(f"  at {cents} or more for {seconds(quantile(lasted, 0.5))} at the median, {seconds(quantile(lasted, 0.9))} at the 90th percentile, "
            f"{seconds(max(lasted))} at the longest")
        log(f"  through each one's longest stretch at {cents} or more, {sum(o['min_edge_size'] for o in through):,.0f} contracts stayed, "
            f"{sum(o['min_edge_capital'] for o in through):,.0f}$ expecting {sum(o['min_edge_profit'] for o in through):,.2f}$")
        reached = [o for o in through if o["min_edge_seconds"] >= ORDER_SECONDS[venue]]
        log(f"  {len(reached):,} of the stretches outlasted an order's trip to {venue}, {1000 * ORDER_SECONDS[venue]:.0f} ms: through them "
            f"{sum(o['min_edge_size'] for o in reached):,.0f} contracts stayed, {sum(o['min_edge_capital'] for o in reached):,.0f}$ expecting "
            f"{sum(o['min_edge_profit'] for o in reached):,.2f}$")
        log(f"  the peak came with the venue's own book change in {100 * np.mean([o['book_changed'] for o in rows]):.0f}%, the fair price's "
            f"move in the rest")
        body = []
        for low, high in SPANS:
            group = [o for o in rows if low < o["close_time"] - o["peak_time"] <= high]
            if group:
                c, p, r, a, m = totals(group)
                lasted = [o["min_edge_seconds"] for o in group]
                body.append((f"{low // 60}-{high // 60} min left", f"{len(group):,}", seconds(quantile(lasted, 0.5)), seconds(max(lasted)),
                             f"{c:,.0f}", f"{p:,.2f}", percent(r), percent(a), f"{m:,.1f}"))
        print_table(f"{venue} opportunities by time left", ("left", "episodes", f"median {cents}+", f"longest {cents}+", "capital $", "profit $",
                                                             "return %", "annual %", "avg min"), body, log)
        largest = []
        for o in rows[:LARGEST]:
            _, _, r, a, m = totals([o])
            largest.append((f"{o['ticker'][-22:]} {o['side']}", f"{100 * o['peak_edge']:.1f}", seconds(o["min_edge_seconds"]),
                            f"{o['take_size']:,.2f}".rstrip("0").rstrip("."), f"{o['take_capital']:,.0f}", f"{o['take_profit']:,.2f}",
                            percent(a), f"{m:,.1f}"))
        print_table(f"{venue} largest opportunities", ("market", "peak c", f"{cents}+ for", "size", "capital $", "profit $", "annual %",
                                                        "minutes"), largest, log)


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Report what the paper orders made, and the opportunities.")
    ap.add_argument("--db", help="the database the orders are in, by default data/history.sqlite")
    ap.add_argument("--hours", type=float, help="only the last this many hours, by default everything")
    args = ap.parse_args()
    conn = database.connect(args.db)
    since = time.time() - 3600 * args.hours if args.hours else 0.0
    report([o for o in database.load_paper_orders(conn) if o["decided"] >= since])
    report_opportunities(database.load_opportunities(conn, since), f"the last {args.hours:g} hours" if args.hours else "since the first")
