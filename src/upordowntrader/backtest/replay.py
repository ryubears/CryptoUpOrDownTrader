"""
Replay the recorded Kalshi books against the model: how its fair price compared with the market's, and what taking
every edge would have made.

Every window recorded from its open and settled is replayed. The model prices Up at every book change and every second
from what was known then: Kalshi's exact strike, bitcoin's price from a stand-in for BRTI, see PRICE_SOURCES, moved onto
BRTI's level by the gap between the two over the minute before the window opened, and a variance forecast from Binance's
minutes before, with the settings the calibration chose, see MODEL. Binance's price of a second is only known once the
second is over, so with Binance the model reads the last whole one, up to a second late, which on 2026-10-11 cost it:
Kalshi's book moved within tens of milliseconds of bitcoin, and the model took those moves for edges. An exchange's quote
is known QUOTE_DELAY after the exchange's time for it, so with the exchanges the model is as current as a trader near them.

The trading rule, to revisit before trading live: take an edge each time it reaches the threshold after Kalshi's taker
fee, on either side, buying Yes below the model's price or No above it, but once per episode: a side's edge must fall
below the threshold before it is taken again. The order takes every contract offered at a price whose edge still
reaches the threshold: an immediate or cancel limit at the worst such price, which reaches Kalshi LATENCY seconds after
the change that opened the edge and fills against the book Kalshi had then, at that price or better. A tenth of a
second is what SportsArb measured from AWS us-east-1, near Kalshi: a signed round trip of about 35 ms, and its copy of
the book about 12 ms behind Kalshi's. From farther away it is slower.

Each order says how many contracts were offered within the threshold as it was sent, and how many it filled, with two
profits: the expected, the model's edge on each contract, which is what the trades make on average if the model's
price is right, and the realized, from how the windows settled, which over few windows is mostly luck. The books keep
BOOK_LEVELS levels a side, so an order whose every level kept was worth taking may have been offered more.

Times are the venues' own, not this machine's, whose clock may be off: each book change at its matching engine time,
each Coinbase quote at Coinbase's time, and Binance's seconds at Binance's. A book snapshot carries no time, so it takes
its arrival shifted by the median gap between arrival and matching engine time, and so does every Kraken quote, as
Kraken's stamps are not always the quote's time, see quote_feed().

Run from src/ with:
    python3 -m upordowntrader.backtest.replay
    python3 -m upordowntrader.backtest.replay --price exchanges --threshold 0.05 --latency 0.1
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from upordowntrader.api import binance
from upordowntrader.backtest import metrics
from upordowntrader.common.log import log
from upordowntrader.common.sources import SOURCES
from upordowntrader.db import database
from upordowntrader.history import load
from upordowntrader.model import pricing, volatility
from upordowntrader.record.recorder import BOOK_LEVELS

PRICE_SOURCES = ("binance", "coinbase", "kraken", "exchanges")   # Exchanges is Coinbase's and Kraken's mids averaged.
MODEL = ("close", 30, "scaled", False)  # The calibration's best variance forecast: input, half life, weighting, season.
DOF = 8                 # The Student t's degrees of freedom, as the calibration's best distribution.
THRESHOLD = 0.05        # Edge after the fee, in dollars a contract, an order needs.
LATENCY = 0.1           # Seconds from the change that opens an edge to the order reaching Kalshi.
QUOTE_DELAY = 0.05      # Seconds from an exchange's time for a quote to a trader near it having it.
STAMPED = ("coinbase",) # Exchanges whose stamp on a quote is the quote's time, see quote_feed().
FEE_RATE = 0.07         # Kalshi's taker fee is FEE_RATE * price * (1 - price) a contract, for KXBTC15M's multiplier of 1.
HISTORY_DAYS = 2        # Days of Binance's minutes read before the first window, for the variance forecast.
DAY = 86400


# PRICES

def binance_history(conn, first_day, end_day, log=print):
    """
    Binance's History from first_day up to end_day, from the stored archive, and from the REST API for days not stored
    yet, as the archive publishes a day only the day after.
    """
    symbol, seconds = SOURCES["binance"]
    stored = database.stored_days(conn, database.series_id(conn, "binance", symbol, seconds))
    series = database.series_id(conn, "binance", symbol, seconds)
    days, before = [], np.nan
    for i in range((end_day - first_day).days):
        day = first_day + timedelta(days=i)
        day_start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
        if day in stored:
            rows = database.read_bars(conn, series, day_start, day_start + DAY)
        else:
            log(f"reading {day} from Binance's REST API, not stored yet")
            end = min(datetime.fromtimestamp(day_start + DAY, UTC), datetime.now(UTC))
            found = list(binance.klines(symbol, "1s", datetime.fromtimestamp(day_start, UTC), end))
            rows = np.array([[k.open_time.timestamp(), k.open, k.high, k.low, k.close, k.volume] for k in found]).reshape(-1, 6)
        days.append(load.from_seconds(rows, day_start, before))
        before = days[-1][1]["close"][-1]
    return load.assemble("binance", first_day, days)


@dataclass
class Feed:
    """
    Bitcoin's price from one source, for price_at(): every second's close from start for Binance, or each exchange's
    quotes, as times by the exchange's clock and mids.
    """
    start: int
    seconds: np.ndarray | None = None   # Binance's close of each second from start.
    times: tuple = ()                   # Each exchange's quote times, oldest first.
    mids: tuple = ()                    # And the mid of each quote.


def quote_feed(conn, source, start, end):
    """
    The Feed of an exchange's quotes, or of both Coinbase's and Kraken's for 'exchanges', reaching from a minute before
    start to a minute after end, in seconds since 1970. A quote of a STAMPED exchange is timed by its stamp, and any
    other by its arrival shifted by the median gap between the exchange's stamps and arrival: Kraken's ticker stamps
    some quotes with the time of an earlier change, on 2026-10-11 a fifth of them, 1 in 20 over a second stale.
    """
    sources = ("coinbase", "kraken") if source == "exchanges" else (source,)
    times, mids = [], []
    for name in sources:
        rows = np.array(conn.execute("SELECT received, sent, (bid + ask) / 2 FROM quotes WHERE source = ? AND received >= ? AND received < ? "
                                     "AND bid IS NOT NULL AND ask IS NOT NULL ORDER BY received",
                                     (name, float(start) - 60, float(end) + 60)).fetchall(), dtype=float).reshape(-1, 3)
        received, sent = rows[:, 0], rows[:, 1]
        timed = ~np.isnan(sent)
        shift = np.median(sent[timed] - received[timed]) if timed.any() else 0.0
        when = np.where(timed & (name in STAMPED), sent, received + shift)
        order = np.argsort(when, kind="stable")
        times.append(when[order])
        mids.append(rows[order, 2])
    return Feed(start, times=tuple(times), mids=tuple(mids))


def price_at(feed, t):
    """
    The source's price known at each of the times t: Binance's close of the last whole second, or the mid of each
    exchange's newest quote QUOTE_DELAY before, averaged over the exchanges that have one. NaN where none is known.
    """
    t = np.asarray(t, dtype=float)
    if feed.seconds is not None:
        index = np.floor(t).astype(int) - 1 - feed.start
        inside = (index >= 0) & (index < len(feed.seconds))
        return np.where(inside, feed.seconds[np.clip(index, 0, len(feed.seconds) - 1)], np.nan)
    prices = []
    for times, mids in zip(feed.times, feed.mids, strict=True):
        i = np.searchsorted(times, t - QUOTE_DELAY, side="right") - 1
        prices.append(np.where(i >= 0, mids[np.maximum(i, 0)] if len(mids) else np.nan, np.nan))
    stacked = np.vstack(prices)
    known = np.isfinite(stacked).sum(axis=0)
    return np.where(known > 0, np.nansum(stacked, axis=0) / np.maximum(known, 1), np.nan)


def window_chances(feed, variance, start, market, t):
    """
    The model's chance of Up at each of the times t inside a window, from what was known then, see the module's
    docstring. variance is the forecast per second at the end of each minute from start. BRTI's settlement averages the
    readings at the end of each of the last 60 seconds, so the price known at those moments stands in for them, and those
    taken by a time count as they were.
    """
    t = np.asarray(t, dtype=float)
    readings = np.arange(1, 61) - 60                            # The ends of the 60 seconds before a moment.
    offset = np.nanmean(price_at(feed, market.open_time + readings)) - market.strike    # Over BRTI at the strike, known at the open.
    price = price_at(feed, t) - offset
    taken = np.clip(np.floor(t).astype(int) - (market.close_time - 60), 0, 60)          # Settlement readings already taken.
    settled = price_at(feed, market.close_time + readings) - offset
    observed = np.concatenate([[0.0], np.cumsum(np.nan_to_num(settled))])[taken]
    # Seconds left as the settlement formulas count them: the readings still to come once one is taken, and before then at
    # least the whole minute, since a moment inside the minute's first second has no reading taken and all 60 to come.
    seconds_left = np.where(taken > 0, 60 - taken, np.maximum(market.close_time - t, 60))
    forecast = np.maximum(variance[(np.floor(t).astype(int) - start) // 60 - 1], 1e-14)
    z = pricing.z_scores(price, observed, market.strike, seconds_left, forecast)
    return pricing.up_student_t(z, DOF)


# BOOKS

@dataclass
class Book:
    """
    A recorded market's book at each change, by Kalshi's matching engine time, oldest first.
    """
    times: np.ndarray
    bid: np.ndarray             # The best bid and ask, NaN for an empty side.
    ask: np.ndarray
    bids: list                  # The levels kept, [[price, contracts], ...] best first, at each change.
    asks: list


def best_prices(sides):
    """
    The best price of each side in a list of them, NaN for an empty one.
    """
    return np.array([levels[0][0] if levels else np.nan for levels in sides], dtype=float)


def book_changes(conn, ticker):
    """
    A recorded market's Book. A snapshot, which carries no time, takes its arrival shifted by the median gap between
    arrival and Kalshi's time.
    """
    rows = conn.execute("SELECT received, sent, bids, asks FROM books WHERE ticker = ? ORDER BY received", (ticker,)).fetchall()
    received = np.array([r[0] for r in rows], dtype=float)
    sent = np.array([np.nan if r[1] is None else r[1] for r in rows], dtype=float)
    timed = ~np.isnan(sent)
    shift = np.median(sent[timed] - received[timed]) if timed.any() else 0.0
    times = np.where(timed, sent, received + shift)
    order = np.argsort(times, kind="stable")
    bids, asks = [json.loads(rows[i][2]) for i in order], [json.loads(rows[i][3]) for i in order]
    return Book(times[order], best_prices(bids), best_prices(asks), bids, asks)


def fee(price):
    """
    Kalshi's taker fee a contract at a price.
    """
    return FEE_RATE * price * (1 - price)


def side_levels(book, at, side):
    """
    The levels an order on a side takes from, at change at, as [price to pay, contracts], best first: the asks for Yes,
    and the bids restated as No prices for No.
    """
    return [[p, c] for p, c in book.asks[at]] if side == "yes" else [[1 - p, c] for p, c in book.bids[at]]


def edge(chance, price):
    """
    The model's edge after the fee on a contract bought at price, given the model's chance it pays.
    """
    return chance - price - fee(price)


def take_edges(times, chances, book, close_time, up, threshold=THRESHOLD, latency=LATENCY):
    """
    The orders of one window under the trading rule, see the module's docstring. times and chances are the model's
    chance of Up at each event, book the window's Book, and up how it settled. Returns a dict per order: when, the side,
    the best edge seen, the contracts offered within the threshold and their expected profit as sent, and the contracts
    filled, their average price, and their expected and realized profits, in dollars.
    """
    at = np.searchsorted(book.times, times, side="right") - 1
    seen_ask = np.where(at >= 0, book.ask[np.maximum(at, 0)], np.nan)
    seen_bid = np.where(at >= 0, book.bid[np.maximum(at, 0)], np.nan)
    best_edges = {"yes": edge(chances, seen_ask), "no": edge(1 - chances, 1 - seen_bid)}
    armed, orders = {"yes": True, "no": True}, []
    for i, t in enumerate(times):
        for side in ("yes", "no"):
            if not best_edges[side][i] >= threshold:    # Below it, or the side is empty: the next reach is a new episode.
                armed[side] = True
                continue
            if not armed[side] or t + latency >= close_time:
                continue
            armed[side] = False
            chance = chances[i] if side == "yes" else 1 - chances[i]
            offered = [(p, c) for p, c in side_levels(book, at[i], side) if edge(chance, p) >= threshold]
            limit = max(p for p, _ in offered)
            landed = np.searchsorted(book.times, t + latency, side="right") - 1
            filled = [(p, c) for p, c in side_levels(book, landed, side) if p <= limit + 1e-9]
            won = up if side == "yes" else not up
            contracts = sum(c for _, c in filled)
            orders.append({
                "time": t, "side": side, "seconds_left": close_time - t, "edge": best_edges[side][i],
                "offered": sum(c for _, c in offered), "offered_expected": sum(c * edge(chance, p) for p, c in offered),
                "every_level": len(offered) == BOOK_LEVELS, "contracts": contracts,
                "price": sum(p * c for p, c in filled) / contracts if contracts else None,
                "expected": sum(c * edge(chance, p) for p, c in filled), "realized": sum(c * (float(won) - p - fee(p)) for p, c in filled),
                "one_contract": float(won) - filled[0][0] - fee(filled[0][0]) if filled else None, "won": won,
            })
    return orders


# REPLAY

def recorded_windows(conn):
    """
    The settled Kalshi windows recorded from their open, oldest first.
    """
    first = dict(conn.execute("SELECT ticker, MIN(received) FROM books GROUP BY ticker").fetchall())
    return [m for m in database.load_markets(conn, "kalshi")
            if m.result and m.strike is not None and first.get(m.ticker) is not None and first[m.ticker] <= m.open_time + 5]


def replay(conn, price_source, thresholds, latencies, log=print):
    """
    The model against the market at every second of every recorded window, and the trades of each threshold and latency.
    Returns (windows, per second arrays, {(threshold, latency): trades}).
    """
    windows = recorded_windows(conn)
    if not windows:
        raise SystemExit("no settled window recorded from its open yet, run the recorder for a while")
    first_day = datetime.fromtimestamp(windows[0].open_time, UTC).date() - timedelta(days=HISTORY_DAYS)
    end_day = datetime.fromtimestamp(windows[-1].close_time, UTC).date() + timedelta(days=1)
    history = binance_history(conn, first_day, end_day, log)
    start = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp())
    variance, _ = volatility.forecasts(history, *MODEL)
    if price_source == "binance":
        feed = Feed(start, seconds=history.second_close)
    else:
        feed = quote_feed(conn, price_source, windows[0].open_time, windows[-1].close_time)
    columns = {name: [] for name in ("window", "left", "chance", "bid", "ask", "up")}
    trades = {(threshold, latency): [] for threshold in thresholds for latency in latencies}
    for w, market in enumerate(windows):
        t = np.arange(market.open_time + 1, market.close_time)
        if np.isnan(price_at(feed, [market.open_time - 59, market.open_time + 1])).any():
            continue                                            # The source has no price for this window yet.
        chances = window_chances(feed, variance, start, market, t)
        book = book_changes(conn, market.ticker)
        at = np.searchsorted(book.times, t, side="right") - 1
        ok = (at >= 0) & ~np.isnan(chances)
        up = market.result == "yes"
        for name, values in (("window", np.full(ok.sum(), w)), ("left", market.close_time - t[ok]), ("chance", chances[ok]),
                             ("bid", book.bid[at[ok]]), ("ask", book.ask[at[ok]]), ("up", np.full(ok.sum(), up))):
            columns[name].append(values)
        # Events: every book change inside the window, and every second, when the model's price may change.
        inside = book.times[(book.times > market.open_time) & (book.times < market.close_time)]
        events = np.unique(np.concatenate([inside, t.astype(float)]))
        event_chances = window_chances(feed, variance, start, market, events)
        valid = ~np.isnan(event_chances)
        for key in trades:
            trades[key].extend({**trade, "window": w} for trade in take_edges(events[valid], event_chances[valid], book, market.close_time,
                                                                              up, *key))
    return windows, {name: np.concatenate(values) for name, values in columns.items()}, trades


def report(windows, per_second, trades, price_source):
    """
    Print the model against the market by time left, and the trades of each threshold and latency.
    """
    d = per_second
    two_sided = ~np.isnan(d["bid"]) & ~np.isnan(d["ask"])
    mid = (d["bid"] + d["ask"]) / 2
    print(f"\n{len(windows)} windows from {datetime.fromtimestamp(windows[0].open_time, UTC):%Y-%m-%d %H:%M} to "
          f"{datetime.fromtimestamp(windows[-1].close_time, UTC):%Y-%m-%d %H:%M} UTC, {sum(m.result == 'yes' for m in windows)} up, "
          f"priced from {price_source}; {len(d['up']):,} seconds, {two_sided.mean():.0%} with both sides quoted")
    print(f"\n{'left':>11}{'seconds':>9}{'spread':>8}{'model - mid':>13}{'|gap| 90%':>11}{'model LL':>10}{'market LL':>11}")
    for low, high in ((600, 900), (300, 600), (120, 300), (60, 120), (0, 60), (0, 900)):
        s = two_sided & (d["left"] > low) & (d["left"] <= high)
        if not s.any():
            continue
        gap = 100 * (d["chance"][s] - mid[s])
        label = "all" if (low, high) == (0, 900) else f"{low // 60}-{high // 60} min"
        spread = 100 * np.mean(d["ask"][s] - d["bid"][s])
        print(f"{label:>11}{s.sum():>9,}{spread:>7.1f}c{np.median(gap):>+12.1f}c{np.percentile(np.abs(gap), 90):>10.1f}c"
              f"{metrics.log_loss(d['chance'][s], d['up'][s]):>10.3f}{metrics.log_loss(mid[s], d['up'][s]):>11.3f}")
    print("Log loss: lower is better, a coin flip scores 0.693.")
    print(f"\n{'threshold':>10}{'latency':>9}{'orders':>8}{'filled':>8}{'offered':>10}{'filled':>10}{'expected':>11}{'realized':>11}"
          f"{'per contract':>14}{'one each':>10}")
    print(f"{'':>35}{'contracts':>10}{'contracts':>10}{'profit':>11}{'profit':>11}{'expected':>14}{'realized':>10}")
    for (threshold, latency), found in trades.items():
        filled = [o for o in found if o["contracts"]]
        contracts = sum(o["contracts"] for o in filled)
        expected, realized = sum(o["expected"] for o in filled), sum(o["realized"] for o in filled)
        one = [o["one_contract"] for o in filled]
        print(f"{threshold:>9.0%} {latency:>8.2f}s{len(found):>8}{len(filled):>8}{sum(o['offered'] for o in found):>10,.0f}{contracts:>10,.0f}"
              f"{expected:>+10.0f}${realized:>+10.0f}${100 * expected / contracts if contracts else 0:>+12.1f}c"
              f"{100 * np.mean(one) if one else 0:>+9.1f}c")
    print("Offered: contracts within the threshold as each order was sent. Expected profit: the model's edge on each contract filled, what")
    print("the trades make on average if its price is right. Realized: from how the windows settled. One each: the average realized")
    print("profit of taking a single contract per order, the best one. All after Kalshi's taker fee.")
    key = (THRESHOLD, LATENCY) if (THRESHOLD, LATENCY) in trades else next(iter(trades))
    found = trades[key]
    if found:
        offered = np.array([o["offered"] for o in found])
        print(f"\nAt {key[0]:.0%} and {key[1]:.2f}s: contracts offered an order, median {np.median(offered):,.0f}, 90th percentile "
              f"{np.percentile(offered, 90):,.0f}; {sum(o['every_level'] for o in found)} of {len(found)} orders found every level kept worth "
              f"taking, so were offered more. Expected profit an order, median {np.median([o['offered_expected'] for o in found]):+.2f}$.")
        print(f"{'left':>11}{'orders':>8}{'filled':>8}{'contracts':>11}{'expected':>10}{'realized':>10}")
        for low, high in ((600, 900), (300, 600), (120, 300), (60, 120), (0, 60)):
            part = [o for o in found if low < o["seconds_left"] <= high]
            got = [o for o in part if o["contracts"]]
            print(f"{low // 60:>4}-{high // 60:<2} min{len(part):>8}{len(got):>8}{sum(o['contracts'] for o in got):>11,.0f}"
                  f"{sum(o['expected'] for o in got):>+9.0f}${sum(o['realized'] for o in got):>+9.0f}$")


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay the recorded Kalshi books against the model and the trading rule.")
    ap.add_argument("--price", default="binance", choices=PRICE_SOURCES, help="bitcoin's price the model reads")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="edge after the fee an order needs, in dollars a contract")
    ap.add_argument("--latency", type=float, default=LATENCY, help="seconds from an edge opening to the order reaching Kalshi")
    args = ap.parse_args()
    conn = database.connect()
    thresholds = sorted({args.threshold, 0.02, 0.10})
    latencies = sorted({args.latency, 1.0})
    windows, per_second, trades = replay(conn, args.price, thresholds, latencies, log)
    report(windows, per_second, trades, args.price)
