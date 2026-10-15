"""
Compare Hyperliquid's up or down windows with Kalshi's, and with the model, from what the recorder stored, see
record/run.py and api/hyperliquid.py.

    RULES   how the 15 minute series sets its strikes and settles, which its rules leave vague: each strike against the
            perp's mark when the window was listed, read from the explorer, and each result against the marks around
            the close under a few rules, see SETTLE_RULES.
    VENUES  Kalshi's and Hyperliquid's BTC windows closing together: their strikes, how often they settled alike, and
            their mid prices second by second.
    LEAD    which moves first, Hyperliquid's BTC perp, Coinbase, or Kalshi's book, by the correlation of their 0.1 second
            changes at each lag, each by its venue's own time.
    MODEL   the model's chance of each Hyperliquid window, from the perp's mark and its 1 minute candles, against the
            window's mid, by log loss. The model is fitted to BRTI and Kalshi, see model/fair.py, so this is its first
            look at a mark.

The marks carry our clock's time, see record/marks.py, and Hyperliquid keeps only the last CANDLE_LIMIT candles and the
explorer the last 300 actions of each account, so each part reaches only as far back as the recorder has run, or less.

Run from src/ with:
    python3 -m upordowntrader.backtest.hyperliquid
"""

import numpy as np
import time
from datetime import UTC, datetime, timedelta
from upordowntrader.api import hyperliquid
from upordowntrader.backtest import metrics, replay
from upordowntrader.common.log import log
from upordowntrader.db import database
from upordowntrader.history import load
from upordowntrader.model import fair, pricing

SETTLE_RULES = ("mark at the close", "last mark before", "minute before, averaged", "minute around, averaged", "oracle at the close")
MAX_SPREAD = 0.20           # Widest spread a book's mid counts at. Hyperliquid's books are thin, so wider than Kalshi's.
STEP, LAGS = 0.1, range(-10, 11)      # The lead and lag grid, in seconds and steps.
LISTING = 30                # Seconds after a window opens before it is priced: the series lists it 15 to 25 seconds in.
SPANS = ((600, 900), (300, 600), (120, 300), (60, 120), (0, 60), (0, 900))


# MARKS

def in_force(rows, t, column=1):
    """
    The value in a column of marks, (received, mark, oracle) rows oldest first, in force at each of the times t: the
    last received by then. NaN before the first.
    """
    i = np.searchsorted(rows[:, 0], np.asarray(t, dtype=float), side="right") - 1
    return np.where(i >= 0, rows[np.maximum(i, 0), column], np.nan) if len(rows) else np.full(np.shape(t), np.nan)


def interpolated(rows, t):
    """
    The mark at t interpolated between those received just before and after it, NaN without both.
    """
    i = np.searchsorted(rows[:, 0], t, side="right")
    if i == 0 or i == len(rows):
        return np.nan
    (t0, m0), (t1, m1) = rows[i - 1, :2], rows[i, :2]
    return m0 + (m1 - m0) * (t - t0) / (t1 - t0)


def averaged(rows, start, end):
    """
    The mark's average over [start, end), each mark weighted by the time it was in force, NaN without one at start.
    """
    inside = (rows[:, 0] > start) & (rows[:, 0] < end)
    times = np.concatenate([[start], rows[inside, 0], [end]])
    values = np.concatenate([in_force(rows, [start]), rows[inside, 1]])
    return float(np.sum(values * np.diff(times)) / (end - start))


def settled_by(rows, close, rule):
    """
    The settlement price one of SETTLE_RULES gives a window closing at close, from the marks around it.
    """
    if rule == "mark at the close":
        return interpolated(rows, close)
    if rule == "last mark before":
        return float(in_force(rows, [close])[0])
    if rule == "minute before, averaged":
        return averaged(rows, close - 60, close)
    if rule == "minute around, averaged":
        return averaged(rows, close - 30, close + 30)
    return float(in_force(rows, [close], column=2)[0])


def perp_of(m):
    """
    The perp a Hyperliquid window is on, from its series, 'out-BTC-15m'.
    """
    return m.series.split("-")[1]


# RULES

def rules(conn, windows):
    """
    Print each strike against the mark when its window was listed, and how often each of SETTLE_RULES agrees with the
    results, with the closest calls.
    """
    print("\n== RULES: the 15 minute series' strikes and settlements against the perps' marks")
    listings = hyperliquid.listings()
    strikes, agree, closest = [], {rule: [] for rule in SETTLE_RULES}, []
    for m in windows:
        rows = database.read_marks(conn, "hyperliquid", perp_of(m), m.close_time - 960, m.close_time + 90)
        if len(rows) < 2:
            continue
        listed_at = listings.get((m.series, m.close_time, m.strike))
        if listed_at and rows[0, 0] < listed_at < rows[-1, 0]:
            strikes.append((perp_of(m), m.strike - interpolated(rows, listed_at), listed_at - (m.close_time - 900)))
        if not rows[0, 0] < m.close_time - 60 or not m.close_time + 30 < rows[-1, 0]:
            continue
        prices = {rule: settled_by(rows, m.close_time, rule) for rule in SETTLE_RULES}
        for rule, price in prices.items():
            agree[rule].append((price >= m.strike) == (m.result == "yes"))
        closest.append((abs(prices["mark at the close"] - m.strike) / m.strike, m, prices))
    for perp in hyperliquid.UNDERLYINGS:
        gaps = np.array([g for p, g, _ in strikes if p == perp])
        if len(gaps):
            print(f"{perp:<5} strike less the mark when listed, {len(gaps)} windows: median {np.median(gaps):+.4g}, |max| {np.abs(gaps).max():.4g}; "
                  f"listed {np.median([s for p, _, s in strikes if p == perp]):.0f} s after the quarter hour, median")
    n = len(closest)
    print(f"{n} settled windows with marks around the close; how often each rule gives the official result:")
    for rule in SETTLE_RULES:
        print(f"  {rule:<26}{np.mean(agree[rule]) if n else np.nan:>7.1%}")
    print("closest calls, settlement price less the strike under each rule:")
    for _, m, prices in sorted(closest, key=lambda c: c[0])[:6]:
        cells = "  ".join(f"{p - m.strike:+.4g}" for p in prices.values())
        print(f"  {m.series:<13}{datetime.fromtimestamp(m.close_time, UTC):%m-%d %H:%M}  {m.result:<4} {cells}")


# VENUES

def mids(book, t, max_spread=MAX_SPREAD):
    """
    A Book's mid at each of the times t, see replay.book_changes(), NaN where a side was empty or the spread too wide.
    """
    at = np.searchsorted(book.times, t, side="right") - 1
    bid = np.where(at >= 0, book.bid[np.maximum(at, 0)], np.nan)
    ask = np.where(at >= 0, book.ask[np.maximum(at, 0)], np.nan)
    return np.where(ask - bid <= max_spread, (bid + ask) / 2, np.nan)


def recorded(conn, venue):
    """
    The tickers of a venue's windows with books recorded.
    """
    return {row[0] for row in conn.execute("SELECT DISTINCT ticker FROM books WHERE venue = ?", (venue,))}


def venues(conn, windows):
    """
    Print Kalshi's and Hyperliquid's BTC windows closing together: strikes, results, and mids.
    """
    print("\n== VENUES: Kalshi's and Hyperliquid's BTC windows closing together")
    kalshi = {m.close_time: m for m in database.load_markets(conn, "kalshi") if m.result}
    pairs = [(kalshi[m.close_time], m) for m in windows if m.series == "out-BTC-15m" and m.close_time in kalshi]
    if not pairs:
        print("none yet")
        return
    gaps = np.array([h.strike - k.strike for k, h in pairs])
    alike = np.array([k.result == h.result for k, h in pairs])
    print(f"{len(pairs)} windows: Hyperliquid's strike less Kalshi's median {np.median(gaps):+.2f}$, 10% {np.percentile(gaps, 10):+.2f}$, "
          f"90% {np.percentile(gaps, 90):+.2f}$; settled alike {alike.mean():.1%}")
    for k, h in [p for p, same in zip(pairs, alike, strict=True) if not same][:8]:
        print(f"  {datetime.fromtimestamp(k.close_time, UTC):%m-%d %H:%M}  Kalshi {k.result} ({k.strike:,.2f} to {k.settlement:,.2f}), "
              f"Hyperliquid {h.result} (strike {h.strike:,.0f})")
    books_k, books_h = recorded(conn, "kalshi"), recorded(conn, "hyperliquid")
    both = [(k, h) for k, h in pairs if k.ticker in books_k and h.ticker in books_h]
    if not both:
        return
    rows = []
    for k, h in both:
        t = np.arange(h.close_time - 900 + LISTING, h.close_time, 1.0)
        bk, bh = replay.book_changes(conn, k.ticker), replay.book_changes(conn, h.ticker, "hyperliquid")
        at = np.searchsorted(bh.times, t, side="right") - 1
        spread = np.where(at >= 0, bh.ask[np.maximum(at, 0)] - bh.bid[np.maximum(at, 0)], np.nan)
        rows.append(np.column_stack([h.close_time - t, mids(bk, t, 0.10), mids(bh, t), spread]))
    d = np.vstack(rows)
    print(f"{len(both)} windows with both books, second by second: Hyperliquid quoted both sides within {MAX_SPREAD:.0%} "
          f"{np.mean(np.isfinite(d[:, 2])):.0%} of seconds, spread median {100 * np.nanmedian(d[:, 3]):.1f}c")
    print(f"{'left':>11}{'seconds':>9}{'Hyperliquid - Kalshi mid':>26}{'|x| 90%':>9}")
    for low, high in SPANS:
        s = (d[:, 0] > low) & (d[:, 0] <= high) & np.isfinite(d[:, 1]) & np.isfinite(d[:, 2])
        if s.any():
            x = 100 * (d[s, 2] - d[s, 1])
            print(f"{span(low, high):>11}{s.sum():>9,}{np.median(x):>+25.1f}c{np.percentile(np.abs(x), 90):>8.1f}c")


def span(low, high):
    return "all" if (low, high) == (0, 900) else f"{low // 60}-{high // 60} min"


# LEAD

def lead(conn):
    """
    Print the correlation of 0.1 second changes of Hyperliquid's BTC perp's mid with Coinbase's and with Kalshi's Yes mid,
    over the Kalshi windows recorded while the perp was.
    """
    print("\n== LEAD: Hyperliquid's BTC perp against Coinbase and Kalshi, 0.1 s changes; positive lag means the other moves after")
    first = conn.execute("SELECT MIN(received), MAX(received) FROM quotes WHERE source = 'hyperliquid'").fetchone()
    if first[0] is None:
        print("no quotes of the perp yet")
        return
    saved, replay.QUOTE_DELAY = replay.QUOTE_DELAY, 0.0
    perp, coinbase = (replay.quote_feed(conn, s, first[0], first[1]) for s in ("hyperliquid", "coinbase"))
    kalshi = [m for m in replay.recorded_windows(conn) if m.open_time > first[0] + 60 and m.close_time < first[1]]
    pairs = {"coinbase": {lag: ([], []) for lag in LAGS}, "kalshi": {lag: ([], []) for lag in LAGS}}
    for m in kalshi:
        g = np.arange(m.open_time + 30, m.close_time - 90, STEP)
        dp = np.diff(replay.price_at(perp, g))
        book = replay.book_changes(conn, m.ticker)
        at = np.searchsorted(book.times, g, side="right") - 1
        others = {"coinbase": np.diff(replay.price_at(coinbase, g)), "kalshi": np.diff((book.bid[at] + book.ask[at]) / 2)}
        for name, do in others.items():
            for lag in LAGS:
                a, b = dp[max(0, -lag):len(dp) - max(0, lag)], do[max(0, lag):len(do) - max(0, -lag)]
                ok = np.isfinite(a) & np.isfinite(b)
                pairs[name][lag][0].append(a[ok])
                pairs[name][lag][1].append(b[ok])
    replay.QUOTE_DELAY = saved
    print(f"{len(kalshi)} Kalshi windows")
    for name, by_lag in pairs.items():
        if not by_lag[0][0]:
            continue
        corr = {lag: np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1] for lag, (a, b) in by_lag.items()}
        best = max(corr, key=corr.get)
        before, after = sum(corr[lag] for lag in LAGS if lag < 0), sum(corr[lag] for lag in LAGS if lag > 0)
        print(f"  {name:<9} peak {best * STEP:+.1f}s ({corr[best]:.3f}), {name} first {before:.3f}, after {after:.3f}   "
              + " ".join(f"{lag * STEP:+.1f}:{corr[lag]:.2f}" for lag in LAGS if -3 <= lag <= 3))


# MODEL

def candle_history(perp, end):
    """
    A History of a perp's 1 minute candles, as many as Hyperliquid keeps up to end, from the first whole day, and the
    start of that day, in seconds since 1970.
    """
    rows = np.array(hyperliquid.minute_candles(perp, end - 60 * hyperliquid.CANDLE_LIMIT, end), dtype=float)[:, :6]
    first_day = datetime.fromtimestamp(rows[0, 0], UTC).date() + timedelta(days=1)
    start = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp())
    days, before = [], rows[rows[:, 0] < start, 4][-1] if (rows[:, 0] < start).any() else np.nan
    for day_start in range(start, int(end), 86400):
        days.append(load.from_minutes(rows[(rows[:, 0] >= day_start) & (rows[:, 0] < day_start + 86400)], day_start, before))
        before = days[-1][1]["close"][-1]
    return load.assemble(f"hyperliquid {perp}", first_day, days), start


def model(conn, windows, readings):
    """
    Print the model's log loss on the Hyperliquid windows with books recorded, against the windows' mids, by time left.
    The settlement takes readings of the mark, 1 for the mark at the close, 60 for the minute's average.
    """
    print(f"\n== MODEL: the model's chance against Hyperliquid's mids, the settlement read as {readings} reading(s) of the mark")
    books = recorded(conn, "hyperliquid")
    priced = [m for m in windows if m.ticker in books and m.series.startswith("out-")]
    if not priced:
        print("no settled 15 minute window with its book recorded yet")
        return
    histories = {perp: candle_history(perp, time.time()) for perp in hyperliquid.UNDERLYINGS}
    variances = {perp: fair.variance(h) for perp, (h, _) in histories.items()}
    rows = []
    for m in priced:
        perp = perp_of(m)
        marks = database.read_marks(conn, "hyperliquid", perp, m.close_time - 960, m.close_time + 5)
        _, start = histories[perp]
        t = np.arange(m.close_time - 900 + LISTING, m.close_time, 1.0)
        price = in_force(marks, t)
        taken = np.clip(np.floor(t).astype(int) - (m.close_time - readings), 0, readings)
        reads = in_force(marks, m.close_time - readings + np.arange(1, readings + 1))
        observed = np.concatenate([[0.0], np.cumsum(np.nan_to_num(reads))])[taken]
        seconds_left = np.where(taken > 0, readings - taken, np.maximum(m.close_time - t, readings))
        minute = (np.floor(t).astype(int) - start) // 60 - 1
        ok = np.isfinite(price) & (minute >= 0) & (minute < len(variances[perp]))
        z = pricing.z_scores(price[ok], observed[ok], m.strike, seconds_left[ok], np.maximum(variances[perp][minute[ok]], fair.TINY), readings)
        mid = mids(replay.book_changes(conn, m.ticker, "hyperliquid"), t[ok])
        rows.append(np.column_stack([m.close_time - t[ok], fair.chance(z, seconds_left[ok]), mid, np.full(ok.sum(), m.result == "yes")]))
    d = np.vstack(rows)
    print(f"{len(priced)} windows, {sum(m.result == 'yes' for m in priced)} up")
    print(f"{'left':>11}{'seconds':>9}{'model LL':>10}{'mid LL':>9}{'model - mid':>13}")
    for low, high in SPANS:
        s = (d[:, 0] > low) & (d[:, 0] <= high) & np.isfinite(d[:, 1]) & np.isfinite(d[:, 2])
        if s.any():
            print(f"{span(low, high):>11}{s.sum():>9,}{metrics.log_loss(d[s, 1], d[s, 3]):>10.3f}{metrics.log_loss(d[s, 2], d[s, 3]):>9.3f}"
                  f"{100 * np.median(d[s, 1] - d[s, 2]):>+12.1f}c")
    print("Log loss: lower is better, a coin flip scores 0.693. Only seconds when the book quoted both sides count.")


# MAIN

if __name__ == "__main__":
    conn = database.connect()
    # The windows of the perps still recorded: those of others stored before, as ETH's and HYPE's to 2026-10-12, stop short.
    windows = [m for m in database.load_markets(conn, "hyperliquid") if m.result in ("yes", "no") and perp_of(m) in hyperliquid.UNDERLYINGS]
    log(f"{len(windows)} settled Hyperliquid windows on {', '.join(hyperliquid.UNDERLYINGS)}, "
        f"{sum(m.series.startswith('out-') for m in windows)} of the 15 minute series")
    fifteen = [m for m in windows if m.series.startswith("out-")]
    rules(conn, fifteen)
    venues(conn, fifteen)
    lead(conn)
    for readings in (1, 60):
        model(conn, fifteen, readings)
