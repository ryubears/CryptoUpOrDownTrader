"""
Report what the paper orders made, see paper/trader.py, as the replay reports its trades: by venue, then by the edge they
were decided at, by the minutes left, by what brought the decision, see cause(), and day by day, and what they spent a
window. In cents a contract, after each venue's fees, see LEGEND:

    expected    the edge the fair price saw in each contract filled, the chance it pays less its price and fee: what it
                makes on average if the fair price is right
    made        what each did make once its window settled, with two standard errors that count each window once, as a
                window's orders win or lose together
    top of book what each order would have made had it bought one contract alone, at the best price it got: the edge
                without the deeper levels a larger order walks into

Then the opportunities the trader saw, as PredictionArbitrageTrader's summary shows its own: each venue's edges at the
threshold or more, how many came a window, what buying each in full in one order at its best moment would have cost with
the fees, in all and a window, the profit it was expected to make and that over the cost, how long it was held until the
window paid, and how long the edge stayed at the threshold or more. Then the contracts that stayed at it through each
one's longest stretch, which an order sent at any moment of it could have had, of all of them and of those whose stretch
outlasted an order's trip to the venue, see trader.ORDER_SECONDS, how often the peak came with the venue's own book change
rather than the fair price's, by minutes left, and the largest. The profits are the fair price's expectation, not locked
in, until the windows settle: then what each one actually made beside it, and that over its cost.

Paper and live trading are reported apart, each with its own opportunities, see opportunities.mode in schema.sql: the
paper orders, and the live trades, read as the paper orders are, see live_order(), their contracts at the average price
Hyperliquid filled them at, top of book included. --mode picks one.

Run from src/ with:
    python3 -m upordowntrader.paper.report
    python3 -m upordowntrader.paper.report --hours 24
    python3 -m upordowntrader.paper.report --since 2026-10-14T01:00
    python3 -m upordowntrader.paper.report --mode live
"""

import argparse
import numpy as np
import time
from datetime import UTC, datetime
from upordowntrader.backtest.replay import window_mean
from upordowntrader.common.stats import quantile
from upordowntrader.db import database
from upordowntrader.paper.trader import ORDER_SECONDS, VENUES, venue_edge

EDGES = ((0.02, 0.03), (0.03, 0.05), (0.05, 1.0))       # Edges at the decision the report is broken into, in dollars a contract.
SPANS = ((600, 900), (300, 600), (60, 300))             # Seconds left it is broken into: Hyperliquid trades into the last 5 minutes.
CAUSES = ("on its book", "on Kalshi's book", "on another book", "on an exchange", "on the clock", "not recorded")  # As cause() gives.
LARGEST = 5             # The largest opportunities listed.
LEGEND = ("",
          "How to read it, in cents a contract, after each venue's fees:",
          "  expected     the edge the fair price saw in each contract filled, the chance it pays less its price and fee: what it",
          "               makes on average if the fair price is right",
          "  made         what each did make once its window settled, +/- two standard errors that count each window once, as a",
          "               window's orders win or lose together: a gap to expected smaller than that is noise",
          "  top of book  what each order would have made had it bought one contract alone, at the best price it got: the edge",
          "               without the deeper levels a larger order walks into. A live order's is at the average price it got",
          "  profit       dollars made in all",
          "  on its book, on Kalshi's book, on an exchange, on the clock: what brought the decision, a change of the venue's own",
          "               book, of Kalshi's, an exchange's quote moving the model, or time alone moving it")


def one_each(order):
    """
    What one contract at the best price an order filled made after the venue's fees: the report's top of book.
    """
    return venue_edge(order["venue"], float(order["side"] == order["result"]), order["best"])


def cause(order):
    """
    What brought an order's decision, as the report groups it, see paper_orders.cause in schema.sql: a change of its venue's
    own book, of Kalshi's, of another market's, an exchange's quote, the once a second pricing, or not recorded, as before
    2026-10-15.
    """
    c = order["cause"]
    if c is None:
        return "not recorded"
    if c == order["venue"]:
        return "on its book"
    if c == "kalshi":
        return "on Kalshi's book"
    if c in VENUES or c == "other":
        return "on another book"
    return "on the clock" if c == "time" else "on an exchange"


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


def capital(order):
    """
    What an order's contracts cost with the fees, nothing when none filled.
    """
    return order["contracts"] * (order["cost"] or 0.0) + order["fees"]


def live_order(trade, settlement):
    """
    A live trade, a dict of the trades columns, with its settlement, one of the settlements or None before it, as the
    report reads a paper order: its contracts filled at their average price, what the chance it was decided on expected
    of them, and, once settled, what they made.
    """
    filled = trade["filled"]
    price = (trade["cost"] - trade["fees"]) / filled if filled else None
    return {"venue": trade["venue"], "ticker": trade["ticker"], "close_time": trade["close_time"], "decided": trade["signal_ts"],
            "side": trade["side"], "edge": trade["edge"], "contracts": filled, "cost": price, "best": price, "fees": trade["fees"],
            "expected": filled * venue_edge(trade["venue"], trade["fair"], price) if filled else 0.0, "cause": trade["cause"],
            "result": settlement["result"] if settlement else None, "profit": settlement["profit"] if settlement else None}


def live_orders(conn, since=0.0):
    """
    The live trades decided from since, in seconds since 1970, as the report reads paper orders, oldest first.
    """
    settled = {s["trade_id"]: s for s in database.load_settlements(conn)}
    return [live_order(t, settled.get(t["id"])) for t in database.load_trades(conn, "mode = 'live' AND signal_ts >= ?", (since,))]


def report(orders, log=print, windows=None, kind="paper"):
    """
    Print the report of the orders of a kind, 'paper' or 'live', as dicts of the paper_orders columns, see live_order().
    windows is how many windows the trader priced over the time, see priced(), by default those the orders were in.
    """
    if not orders:
        log("no paper orders yet, run upordowntrader.paper.run for a while" if kind == "paper" else
            "no live orders yet, see upordowntrader.live.run")
        return
    settled = [o for o in orders if o["result"]]
    first, last = (datetime.fromtimestamp(orders[i]["decided"], UTC) for i in (0, -1))
    log(f"\n{len(orders)} {kind} orders from {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC, {len(settled)} settled, "
        f"{sum(o['contracts'] > 0 for o in orders)} filled")
    header = (f"{'':>16}{'orders':>8}{'filled':>8}{'windows':>9}{'contracts':>11}{'expected':>11}{'made':>16}{'top of book':>16}{'profit':>11}\n"
              f"{'':>52}{'c/contract':>11}{'c/contract':>16}{'c/contract':>16}{'$':>11}")
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
            group = [o for o in mine if low < o["close_time"] - o["decided"] <= high]
            if group:
                log(row(f"{low // 60}-{high // 60} min left", group))
        for label in CAUSES:
            group = [o for o in mine if cause(o) == label]
            if group:
                log(row(label, group))
        for day in sorted({datetime.fromtimestamp(o["decided"], UTC).date() for o in mine}):
            log(row(f"{day}", [o for o in mine if datetime.fromtimestamp(o["decided"], UTC).date() == day]))
        filled = [o for o in orders if o["venue"] == venue and o["contracts"] > 0]
        used = {}
        for o in filled:
            used[o["close_time"]] = used.get(o["close_time"], 0.0) + capital(o)
        n = windows or len({o["close_time"] for o in orders})
        if used:
            log(f"  spent a window, fees included: {sum(used.values()) / n:,.2f}$ on average over the {n:,} windows priced, "
                f"{sum(used.values()) / len(used):,.2f}$ in each of the {len(used):,} it bought in, {max(used.values()):,.2f}$ at most; "
                f"{len(filled) / n:.1f} filled orders a window")
    for line in LEGEND:
        log(line)


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
    What buying the opportunities in full, one order each, would have cost with the fees, and expected: (cost, expected
    profit, that as a percent of the cost, minutes held until the windows paid, weighted by cost).
    """
    capital, profit = sum(o["take_capital"] for o in group), sum(o["take_profit"] for o in group)
    if capital <= 0:
        return capital, profit, None, None
    minutes = sum(o["take_capital"] * o["days_held"] for o in group) / capital * 1440
    return capital, profit, 100 * profit / capital, minutes


def actual(group):
    """
    What the settled ones of a group of opportunities actually made, for a table cell, or a dash before any settled.
    """
    settled = [o["actual_profit"] for o in group if o["actual_profit"] is not None]
    return f"{sum(settled):+,.2f}" if settled else "-"


def print_table(title, header, body, log=print, left=1):
    """
    Print a small aligned table with a title line, its first left columns aligned left and the rest right.
    """
    log(f"\n{title}")
    widths = [max(len(str(r[i])) for r in [header, *body]) for i in range(len(header))]
    for r in [header, *body]:
        log("  " + "  ".join(str(v).ljust(w) if i < left else str(v).rjust(w) for i, (v, w) in enumerate(zip(r, widths, strict=True))))


def priced(opportunities):
    """
    How many windows the trader priced, from the opportunities: every window priced has some, on some venue, at a positive
    edge.
    """
    return len({o["close_time"] for o in opportunities})


def report_opportunities(opportunities, label, log=print):
    """
    Print the opportunities, as dicts of the opportunities columns, that one order could have taken at the threshold or
    more, see the module's docstring. label says over what time.
    """
    taken = [o for o in opportunities if o["take_size"]]
    n = priced(opportunities)
    log(f"\n{len(opportunities):,} opportunities at a positive edge after the fee, {label}, over {n:,} windows priced, {len(taken):,} reaching "
        f"the threshold")
    for venue in VENUES:
        rows = sorted((o for o in taken if o["venue"] == venue), key=lambda o: o["take_profit"], reverse=True)
        if not rows:
            continue
        cents = f"{100 * rows[0]['min_edge']:.0f}c"
        capital, profit, ret, minutes = totals(rows)
        lasted = [o["min_edge_seconds"] for o in rows]
        through = [o for o in rows if o["min_edge_size"]]
        per = {}
        for o in rows:
            per[o["close_time"]] = per.get(o["close_time"], 0) + 1
        positive = sum(o["venue"] == venue for o in opportunities)
        others = f", of {positive / n:.1f} at any positive edge" if positive > len(rows) else ""
        log(f"\n{venue} opportunities (edges of {cents} or more)")
        log(f"  {len(rows):,} edges: buying each in full, in one order at its best moment, would have cost {capital:,.0f}$ with the fees, "
            f"expecting {profit:,.2f}$, {percent(ret)}% of the cost, held {minutes or 0:,.1f} minutes on average until the window paid")
        log(f"  a window: {len(rows) / n:.1f} edges on average, in {len(per):,} of the {n:,} windows priced, {max(per.values())} at most{others}; "
            f"buying them would have cost {capital / n:,.0f}$ a window, expecting {profit / n:,.2f}$")
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
        totals_at = [o["min_edge_total"] for o in rows if o["min_edge_total"] is not None]
        if totals_at:
            log(f"  at {cents} or more for {seconds(quantile(totals_at, 0.5))} in all at the median, every stretch counted, "
                f"{seconds(quantile(totals_at, 0.9))} at the 90th percentile")
        settled = [o for o in rows if o["actual_profit"] is not None]
        if settled:
            c, p, _, _ = totals(settled)
            made = sum(o["actual_profit"] for o in settled)
            log(f"  {len(settled):,} settled: bought in full they would have made {made:+,.2f}$ on their {c:,.0f}$ cost, {100 * made / c:+.2f}%, "
                f"against {p:+,.2f}$, {100 * p / c:+.2f}%, expected")
        body = []
        for low, high in SPANS:
            group = [o for o in rows if low < o["close_time"] - o["peak_time"] <= high]
            if group:
                c, p, r, m = totals(group)
                lasted = [o["min_edge_seconds"] for o in group]
                body.append((f"{low // 60}-{high // 60} min left", f"{len(group):,}", seconds(quantile(lasted, 0.5)), seconds(max(lasted)),
                             f"{c:,.0f}", f"{p:,.2f}", percent(r), f"{m:,.1f}", actual(group)))
        print_table(f"{venue} opportunities by time left", ("left", "edges", f"median {cents}+", f"longest {cents}+", "cost $", "expected $",
                                                             "expected %", "avg min", "made $"), body, log)
        largest = []
        for o in rows[:LARGEST]:
            _, _, r, m = totals([o])
            largest.append((f"{o['ticker'][-22:]} {o['side']}", f"{100 * o['peak_edge']:.1f}", seconds(o["min_edge_seconds"]),
                            f"{o['take_size']:,.2f}".rstrip("0").rstrip("."), f"{o['take_capital']:,.0f}", f"{o['take_profit']:,.2f}",
                            percent(r), f"{m:,.1f}", actual([o]), percent(o["actual_return_pct"])))
        print_table(f"{venue} largest opportunities", ("market", "peak c", f"{cents}+ for", "contracts", "cost $", "expected $", "expected %",
                                                        "minutes", "made $", "made %"), largest, log)


def utc_time(text):
    """
    Seconds since 1970 for an ISO 8601 time, in UTC unless it says otherwise.
    """
    when = datetime.fromisoformat(text)
    return (when if when.tzinfo else when.replace(tzinfo=UTC)).timestamp()


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Report what the paper orders made, and the opportunities.")
    ap.add_argument("--db", help="the database the orders are in, by default data/history.sqlite")
    ap.add_argument("--hours", type=float, help="only the last this many hours, by default everything")
    ap.add_argument("--since", type=utc_time, help="only from this time, in UTC, as 2026-10-14T01:00, such as since a change of the rule")
    ap.add_argument("--mode", choices=("paper", "live", "all"), default="all", help="paper trading, live trading, or both, by default both")
    args = ap.parse_args()
    conn = database.connect(args.db)
    since = max(time.time() - 3600 * args.hours if args.hours else 0.0, args.since or 0.0)
    label = f"since {datetime.fromtimestamp(since, UTC):%Y-%m-%d %H:%M} UTC" if since else "since the first"
    for kind in ("paper", "live") if args.mode == "all" else (args.mode,):
        opportunities = database.load_opportunities(conn, since, kind)
        orders = [o for o in database.load_paper_orders(conn) if o["decided"] >= since] if kind == "paper" else live_orders(conn, since)
        if args.mode == "all" and kind == "live" and not orders and not opportunities:
            continue                                # No live trading yet.
        print(f"\n==== {kind} trading")
        report(orders, windows=priced(opportunities) or None, kind=kind)
        report_opportunities(opportunities, label)
