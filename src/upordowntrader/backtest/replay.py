"""
Replay the recorded Kalshi books against the model: how its fair price compared with the market's, and what taking
every edge would have made.

Every window recorded from its open and settled is replayed. The model prices Up at every book change and every second
from what was known then: Kalshi's exact strike, bitcoin's price from the exchanges BRTI is built from, see PRICE_SOURCES,
moved onto BRTI's level by the gap between the two over the minute before the window opened, and the variance forecast of
model/fair.py from Coinbase's minutes before. That stand-in misses BRTI by a few dollars, which matters only near the
close, so the model's spread includes how far it missed the settlements of the windows before, see settlement_error() and
fair.tracking_variances(). An exchange's quote is known QUOTE_DELAY after the exchange's time for it,
so the model is as current as a trader near the exchanges. That matters: on 2026-10-11 Kalshi's book moved within tens
of milliseconds of bitcoin, and a model reading bitcoin a second late took the market's moves for edges.

The fair price traded on is one of FAIR_PRICES: the model's own chance, recalibrated on Kalshi's settled windows, or that
blended with the market's mid price, see model/fair.py, both fitted on the windows of the benchmark's TRAIN_DAYS before
the first one replayed, see backtest/benchmark.py, so the replay uses nothing from after its windows open.

The trading rule, to revisit before trading live: take an edge each time it reaches the threshold after Kalshi's taker
fee, on either side, buying Yes below the fair price or No above it, but once per episode: a side's edge must fall
below the threshold before it is taken again. The order takes every contract offered at a price whose edge still
reaches the threshold: an immediate or cancel limit at the worst such price, which reaches Kalshi LATENCY seconds after
the change that opened the edge and fills against the book Kalshi had then, at that price or better. A tenth of a
second is what SportsArb measured from AWS us-east-1, near Kalshi: a signed round trip of about 35 ms, and its copy of
the book about 12 ms behind Kalshi's. From farther away it is slower. A side sends no order while its last is on the
way, and the recorded books go on showing the contracts an order took, so later orders leave them be until the book
shows they are gone, see still_resting().

Each order says how many contracts were offered within the threshold as it was sent, and how many it filled, with two
profits: the expected, the fair price's edge on each contract, which is what the trades make on average if the fair
price is right, and the realized, from how the windows settled, which over few windows is mostly luck: the report gives
it with two standard errors that count each window once, as a window's orders share its outcome. The books keep
BOOK_LEVELS levels a side, so an order whose every level kept was worth taking may have been offered more.

Times are the venues' own, not this machine's, whose clock may be off: each book change at its matching engine time,
and each quote at its exchange's time, see quote_feed(). A book snapshot carries no time, so it takes its arrival shifted
by the median gap between arrival and matching engine time, and so does every Kraken quote, as Kraken's stamps are not
always the quote's time.

Run from src/ with:
    python3 -m upordowntrader.backtest.replay
    python3 -m upordowntrader.backtest.replay --price coinbase --fair recalibrated --threshold 0.05 --latency 0.1
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from upordowntrader.backtest import benchmark, metrics
from upordowntrader.common.log import log
from upordowntrader.db import database
from upordowntrader.history import download
from upordowntrader.model import fair, pricing
from upordowntrader.record.recorder import BOOK_LEVELS

EXCHANGES = ("coinbase", "kraken", "bitstamp", "gemini", "bullish", "cryptocom")   # Those recorded, see record/run.py.
PRICE_SOURCES = ("exchanges", *EXCHANGES)  # Exchanges is the mids of all of them averaged.
FAIR_PRICES = ("model", "recalibrated", "blend")
THRESHOLD = 0.05        # Edge after the fee, in dollars a contract, an order needs.
LATENCY = 0.1           # Seconds from the change that opens an edge to the order reaching Kalshi.
QUOTE_DELAY = 0.05      # Seconds from an exchange's time for a quote to a trader near it having it.
STAMPED = ("coinbase", "bitstamp", "gemini", "bullish", "cryptocom")  # Exchanges whose stamp on a quote is its time, see quote_feed().
HISTORY_DAYS = benchmark.HISTORY_DAYS   # Days of Coinbase's minutes read before the first window, for the variance forecast.


# PRICES

@dataclass
class Feed:
    """
    Bitcoin's price from the exchanges' quotes, for price_at(): each exchange's quote times, by the exchange's clock, and
    mids.
    """
    times: tuple                        # Each exchange's quote times, oldest first.
    mids: tuple                         # And the mid of each quote.


def quote_feed(conn, source, start, end):
    """
    The Feed of an exchange's quotes, or of all EXCHANGES' for 'exchanges', reaching from a minute before
    start to a minute after end, in seconds since 1970. A quote of a STAMPED exchange is timed by its stamp, and any
    other by its arrival shifted by the median gap between the exchange's stamps and arrival: Kraken's ticker stamps
    some quotes with the time of an earlier change, on 2026-10-11 a fifth of them, 1 in 20 over a second stale.
    """
    sources = EXCHANGES if source == "exchanges" else (source,)
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
    return Feed(tuple(times), tuple(mids))


def price_at(feed, t):
    """
    The price known at each of the times t: the mid of each exchange's newest quote QUOTE_DELAY before, averaged over the
    exchanges that have one. NaN where none is known. Equally: on the 78 windows recorded to 2026-10-11 23:30, weighting
    the mids by each exchange's traded volume, or by the sizes at its best levels, or each one's micro price, priced
    no better, and the equal average of Coinbase, Bitstamp, and Crypto.com's minute candles tracked 2026's settlements
    better than one weighted by volume.
    """
    t = np.asarray(t, dtype=float)
    prices = []
    for times, mids in zip(feed.times, feed.mids, strict=True):
        i = np.searchsorted(times, t - QUOTE_DELAY, side="right") - 1
        prices.append(np.where(i >= 0, mids[np.maximum(i, 0)] if len(mids) else np.nan, np.nan))
    stacked = np.vstack(prices)
    known = np.isfinite(stacked).sum(axis=0)
    return np.where(known > 0, np.nansum(stacked, axis=0) / np.maximum(known, 1), np.nan)


def window_chances(feed, variance, start, market, t, fit=None, tracking=0.0):
    """
    The model's chance of Up at each of the times t inside a window, from what was known then, see the module's
    docstring, recalibrated by a fair.Fit when one is given. variance is the forecast per second at the end of each minute
    from start, and tracking the window's tracking variance, see fair.tracking_variances(). BRTI's settlement averages the
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
    forecast = np.maximum(variance[(np.floor(t).astype(int) - start) // 60 - 1], fair.TINY)
    z = pricing.z_scores(price, observed, market.strike, seconds_left, forecast, tracking=tracking)
    return fair.chance(z, seconds_left, fit)


def settlement_error(feed, market):
    """
    The log of the stand-in's settlement over a window's own: the average of the price known at its 60 readings, moved
    onto BRTI's level by the gap at the open as window_chances() moves it. NaN unless the window has settled and the price
    was known at every reading.
    """
    if market.settlement is None:
        return np.nan
    readings = np.arange(1, 61) - 60
    offset = np.mean(price_at(feed, market.open_time + readings)) - market.strike
    return float(np.log((np.mean(price_at(feed, market.close_time + readings)) - offset) / market.settlement))


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


def side_levels(book, at, side):
    """
    The levels an order on a side takes from, at change at, as [price to pay, contracts], best first: the asks for Yes,
    and the bids restated as No prices for No.
    """
    return [[p, c] for p, c in book.asks[at]] if side == "yes" else [[1 - p, c] for p, c in book.bids[at]]


def blend_with_book(chances, book, times, close_time, fit):
    """
    The model's chances at the times blended with the market's mid then, see fair.blended(), and left as they are where
    the book lacked a side.
    """
    at = np.searchsorted(book.times, times, side="right") - 1
    mid = np.where(at >= 0, (book.bid[np.maximum(at, 0)] + book.ask[np.maximum(at, 0)]) / 2, np.nan)
    return np.where(np.isnan(mid), chances, fair.blended(mid, chances, close_time - times, fit))


def resting(book, at, side, price):
    """
    The contracts the levels kept at change at show at a price for an order on a side, see side_levels(): 0 when the
    price lies among them but none is there, and None when it lies beyond the levels kept, where the book says nothing.
    """
    levels = side_levels(book, at, side)
    for p, c in levels:
        if round(p, 6) == price:
            return c
    return None if len(levels) == BOOK_LEVELS and price > levels[-1][0] else 0.0


def still_resting(taken, book, start, end, side):
    """
    Lower the contracts earlier orders on a side took at each price, taken as {price: contracts}, to the fewest the book
    showed there at any change after start up to end. The recorded book goes on showing the orders a fill took, as the
    fill is ours alone, until they leave it, and no more of them can still be there than the level held.
    """
    for j in range(start + 1, end + 1):
        if not taken:
            return
        for price in list(taken):
            c = resting(book, j, side, price)
            if c is not None and c < taken[price]:
                taken[price] = c
                if c <= 0:
                    del taken[price]


def untaken(levels, taken):
    """
    The levels, as [price, contracts], less what earlier orders took at each price, those left empty dropped.
    """
    left = [(p, c - taken.get(round(p, 6), 0.0)) for p, c in levels]
    return [(p, c) for p, c in left if c > 0]


def take_edges(times, chances, book, close_time, up, threshold=THRESHOLD, latency=LATENCY):
    """
    The orders of one window under the trading rule, see the module's docstring. times and chances are the fair price
    of Up at each event, book the window's Book, and up how it settled. Returns a dict per order: when, the side,
    the best edge seen, the contracts offered within the threshold and their expected profit as sent, and the contracts
    filled, their average price, and their expected and realized profits, in dollars. An edge whose every contract an
    earlier order took sends none.
    """
    at = np.searchsorted(book.times, times, side="right") - 1
    seen_ask = np.where(at >= 0, book.ask[np.maximum(at, 0)], np.nan)
    seen_bid = np.where(at >= 0, book.bid[np.maximum(at, 0)], np.nan)
    best_edges = {"yes": fair.edge(chances, seen_ask), "no": fair.edge(1 - chances, 1 - seen_bid)}
    armed, orders = {"yes": True, "no": True}, []
    lands = {"yes": -np.inf, "no": -np.inf}     # When each side's last order lands.
    taken = {"yes": {}, "no": {}}               # The contracts each side's orders took at each price, see still_resting().
    checked = {"yes": 0, "no": 0}               # The book change taken was last brought up to.
    for i, t in enumerate(times):
        for side in ("yes", "no"):
            if not best_edges[side][i] >= threshold:    # Below it, or the side is empty: the next reach is a new episode.
                armed[side] = True
                continue
            if not armed[side] or t < lands[side] or t + latency >= close_time:
                continue
            chance = chances[i] if side == "yes" else 1 - chances[i]
            still_resting(taken[side], book, checked[side], at[i], side)
            checked[side] = at[i]
            worth = [(p, c) for p, c in side_levels(book, at[i], side) if fair.edge(chance, p) >= threshold]
            offered = untaken(worth, taken[side])
            if not offered:
                continue
            armed[side], lands[side] = False, t + latency
            limit = max(p for p, _ in offered)
            landed = np.searchsorted(book.times, t + latency, side="right") - 1
            still_resting(taken[side], book, checked[side], landed, side)
            checked[side] = landed
            filled = untaken([(p, c) for p, c in side_levels(book, landed, side) if p <= limit + 1e-9], taken[side])
            for p, c in filled:
                taken[side][round(p, 6)] = taken[side].get(round(p, 6), 0.0) + c
            won = up if side == "yes" else not up
            contracts = sum(c for _, c in filled)
            orders.append({
                "time": t, "side": side, "seconds_left": close_time - t, "edge": best_edges[side][i],
                "offered": sum(c for _, c in offered), "offered_expected": sum(c * fair.edge(chance, p) for p, c in offered),
                "every_level": len(worth) == BOOK_LEVELS, "contracts": contracts,
                "price": sum(p * c for p, c in filled) / contracts if contracts else None,
                "expected": sum(c * fair.edge(chance, p) for p, c in filled), "realized": sum(c * fair.edge(float(won), p) for p, c in filled),
                "one_contract": fair.edge(float(won), filled[0][0]) if filled else None, "won": won,
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


def replay(conn, price_source, thresholds, latencies, fair_price="blend", log=print):
    """
    The fair price, one of FAIR_PRICES, against the market at every second of every recorded window the price source
    covers, and the trades of each threshold and latency. Returns (the windows replayed, per second arrays,
    {(threshold, latency): trades}).
    """
    windows = recorded_windows(conn)
    if not windows:
        raise SystemExit("no settled window recorded from its open yet, run the recorder for a while")
    fit = benchmark.fit_before(conn, windows[0].open_time, log) if fair_price != "model" else None
    if fit is None and fair_price != "model":
        raise SystemExit("no Kalshi windows with minutes before the first replayed, download them with upordowntrader.history.markets")
    first_day = datetime.fromtimestamp(windows[0].open_time, UTC).date() - timedelta(days=HISTORY_DAYS)
    end_day = datetime.fromtimestamp(windows[-1].close_time, UTC).date() + timedelta(days=1)
    history = download.recent_history(conn, first_day, end_day, log)
    start = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp())
    variance = fair.variance(history)
    feed = quote_feed(conn, price_source, windows[0].open_time, windows[-1].close_time)
    tracking = fair.tracking_variances([settlement_error(feed, m) for m in windows], [m.open_time for m in windows],
                                       [m.close_time for m in windows])
    columns = {name: [] for name in ("window", "left", "chance", "bid", "ask", "up")}
    trades = {(threshold, latency): [] for threshold in thresholds for latency in latencies}
    replayed = []
    for market, track in zip(windows, tracking, strict=True):
        t = np.arange(market.open_time + 1, market.close_time)
        if np.isnan(price_at(feed, [market.open_time - 59, market.open_time + 1])).any():
            continue                                            # The source has no price for this window yet.
        w = len(replayed)
        replayed.append(market)
        chances = window_chances(feed, variance, start, market, t, fit, track)
        book = book_changes(conn, market.ticker)
        if fair_price == "blend":
            chances = blend_with_book(chances, book, t.astype(float), market.close_time, fit)
        at = np.searchsorted(book.times, t, side="right") - 1
        ok = (at >= 0) & ~np.isnan(chances)
        up = market.result == "yes"
        for name, values in (("window", np.full(ok.sum(), w)), ("left", market.close_time - t[ok]), ("chance", chances[ok]),
                             ("bid", book.bid[at[ok]]), ("ask", book.ask[at[ok]]), ("up", np.full(ok.sum(), up))):
            columns[name].append(values)
        # Events: every book change inside the window, and every second, when the model's price may change.
        inside = book.times[(book.times > market.open_time) & (book.times < market.close_time)]
        events = np.unique(np.concatenate([inside, t.astype(float)]))
        event_chances = window_chances(feed, variance, start, market, events, fit, track)
        if fair_price == "blend":
            event_chances = blend_with_book(event_chances, book, events, market.close_time, fit)
        valid = ~np.isnan(event_chances)
        for key in trades:
            trades[key].extend({**trade, "window": w} for trade in take_edges(events[valid], event_chances[valid], book, market.close_time,
                                                                              up, *key))
    if not replayed:
        raise SystemExit(f"no recorded window has {price_source} quotes from its open yet, run the recorder for a while")
    return replayed, {name: np.concatenate(values) for name, values in columns.items()}, trades


def window_mean(values, weights, window):
    """
    The weighted mean of values and two standard errors that count each window once, as its orders share its outcome.
    """
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    mean = np.sum(weights * values) / np.sum(weights)
    _, by_window = np.unique(window, return_inverse=True)
    sums = np.bincount(by_window, weights=weights * (values - mean))
    return mean, 2 * np.sqrt(np.sum(sums ** 2)) / np.sum(weights)


def report(windows, per_second, trades, price_source, fair_price):
    """
    Print the fair price against the market by time left, and the trades of each threshold and latency.
    """
    d = per_second
    two_sided = ~np.isnan(d["bid"]) & ~np.isnan(d["ask"])
    mid = (d["bid"] + d["ask"]) / 2
    print(f"\n{len(windows)} windows from {datetime.fromtimestamp(windows[0].open_time, UTC):%Y-%m-%d %H:%M} to "
          f"{datetime.fromtimestamp(windows[-1].close_time, UTC):%Y-%m-%d %H:%M} UTC, {sum(m.result == 'yes' for m in windows)} up, "
          f"priced from {price_source}, fair price {fair_price}; {len(d['up']):,} seconds, {two_sided.mean():.0%} with both sides quoted")
    print(f"\n{'left':>11}{'seconds':>9}{'spread':>8}{'fair - mid':>13}{'|gap| 90%':>11}{'fair LL':>10}{'market LL':>11}")
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
    print(f"\n{'threshold':>10}{'latency':>9}{'orders':>8}{'filled':>8}{'windows':>9}{'contracts':>11}{'expected':>11}{'realized':>11}"
          f"{'a contract':>27}{'one each':>16}")
    print(f"{'':>45}{'profit':>11}{'profit':>11}{'expected':>11}{'realized':>16}{'realized':>16}")
    for (threshold, latency), found in trades.items():
        filled = [o for o in found if o["contracts"]]
        if not filled:
            print(f"{threshold:>9.0%} {latency:>8.2f}s{len(found):>8}{0:>8}")
            continue
        window = [o["window"] for o in filled]
        contracts = np.array([o["contracts"] for o in filled])
        expected, realized = sum(o["expected"] for o in filled), sum(o["realized"] for o in filled)
        per_contract, error = window_mean([o["realized"] / o["contracts"] for o in filled], contracts, window)
        one, one_error = window_mean([o["one_contract"] for o in filled], np.ones(len(filled)), window)
        print(f"{threshold:>9.0%} {latency:>8.2f}s{len(found):>8}{len(filled):>8}{len(set(window)):>9}{contracts.sum():>11,.0f}"
              f"{expected:>+10.0f}${realized:>+10.0f}${100 * expected / contracts.sum():>+10.1f}c"
              f"{100 * per_contract:>+9.1f} ±{100 * error:>4.1f}c{100 * one:>+9.1f} ±{100 * one_error:>4.1f}c")
    print("Expected profit: the fair price's edge on each contract filled, what the trades make on average if it is right. Realized:")
    print("from how the windows settled, with two standard errors that count each window once. One each: the realized profit of")
    print("taking a single contract an order, the best one. All after Kalshi's taker fee.")
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
    ap.add_argument("--price", default="exchanges", choices=PRICE_SOURCES, help="bitcoin's price the model reads")
    ap.add_argument("--fair", default="blend", choices=FAIR_PRICES, help="the fair price traded on")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="edge after the fee an order needs, in dollars a contract")
    ap.add_argument("--latency", type=float, default=LATENCY, help="seconds from an edge opening to the order reaching Kalshi")
    args = ap.parse_args()
    conn = database.connect()
    thresholds = sorted({args.threshold, 0.02, 0.10})
    latencies = sorted({args.latency, 1.0})
    windows, per_second, trades = replay(conn, args.price, thresholds, latencies, args.fair, log)
    report(windows, per_second, trades, args.price, args.fair)
