"""
Which bitcoin market moves first, and does the one that leads tell the model what Kalshi's market knows near the close?

The quotes recorded, see record/run.py, are those of the six exchanges BRTI is built from, whose average the model reads,
see backtest/replay.py, and of bitcoin perpetuals, where price moves may start. On 2026-10-13 Kalshi's mid moved 0.1 to
0.5 seconds before the exchanges' average, and with 1 to 2 minutes left anticipated about half of the average's move to
the close, more than its own head start explains; neither BRTI, momentum, nor the exchanges' order book imbalance
accounted for the rest. This asks whether a perpetual does.

    LEAD    each source's changes over STEP seconds against those of the exchanges' average, of the other five for one
            of the six, and against Kalshi's mid, at lags up to LAG_SECONDS each way, all by the venues' own times: the
            lag the correlation peaks at, and the share of it on the side where the source moves first. And its catch up:
            how much of the source's move away from the average, its gap less the gap's mean over the last BASIS_SECONDS,
            see ahead(), the average then makes over the next 1 and 5 seconds; 1 when it follows in full.
    MODEL   the model with that move added to the average, beta times it, beta fitted on the earlier half of the windows
            and scored on the later half, by log loss at 1 to 5 minutes left, against the model alone and Kalshi's mid.
            Here a source's quotes count from when they reached the AWS instance, as a trader there would have them,
            while the exchanges' count as the replay counts them.

Run from src/ with:
    python3 -m upordowntrader.backtest.lead
"""

import numpy as np
from datetime import UTC, datetime, timedelta
from upordowntrader.backtest import metrics, replay
from upordowntrader.common.log import log
from upordowntrader.db import database
from upordowntrader.history import download
from upordowntrader.model import fair, pricing
from upordowntrader.record.run import QUOTE_STREAMS

STEP = 0.1                  # Seconds each change is measured over.
LAG_SECONDS = 3             # The furthest lag tried each way.
BASIS_SECONDS = 60          # Seconds a source's gap to the average is averaged over, see ahead().
MAX_SPREAD = 0.10           # The widest Kalshi spread a mid counts at.
SPANS = ((60, 120), (120, 300), (60, 300))      # Seconds left the model is scored over.
BETAS = np.round(np.arange(-0.5, 2.01, 0.05), 2)
MIN_WINDOWS = 8             # Windows each half needs before a source's model is fitted.


# FEEDS

def arrival_feed(conn, source, start, end):
    """
    A source's mids as replay.price_at() reads a Feed, each known from when it reached the recorder: its times are set
    replay.QUOTE_DELAY early, as price_at() reads each quote that much after its time.
    """
    rows = np.array(conn.execute("SELECT received, (bid + ask) / 2 FROM quotes WHERE source = ? AND received >= ? AND received < ? "
                                 "AND bid IS NOT NULL AND ask IS NOT NULL ORDER BY received",
                                 (source, float(start), float(end))).fetchall(), dtype=float).reshape(-1, 2)
    return replay.Feed((rows[:, 0] - replay.QUOTE_DELAY,), (rows[:, 1],))


def joined(feeds):
    """
    One Feed of several, whose price is their mids averaged, see replay.price_at().
    """
    return replay.Feed(tuple(t for f in feeds for t in f.times), tuple(m for f in feeds for m in f.mids))


def at_once(feed, t):
    """
    The price of a Feed at each of the times t by the venues' own times, without replay.QUOTE_DELAY: its times are set
    that much early, as price_at() reads each quote that much after its time.
    """
    return replay.price_at(replay.Feed(tuple(times - replay.QUOTE_DELAY for times in feed.times), feed.mids), t)


def kalshi_mid(book, t):
    """
    Kalshi's mid at each of the times t, NaN where a side was empty or the spread wider than MAX_SPREAD.
    """
    at = np.searchsorted(book.times, t, side="right") - 1
    bid, ask = (np.where(at >= 0, x[np.maximum(at, 0)], np.nan) for x in (book.bid, book.ask))
    return np.where(ask - bid <= MAX_SPREAD, (bid + ask) / 2, np.nan)


def ahead(source, average, step_seconds, basis_seconds=BASIS_SECONDS):
    """
    How far a source has moved ahead of the average at each point of a grid step_seconds apart: its gap to the average
    less the gap's mean over the basis_seconds before, which takes out the slow gap of a perp's funding or tether's price.
    NaN until basis_seconds have passed.
    """
    gap = np.asarray(source, dtype=float) - np.asarray(average, dtype=float)
    n = round(basis_seconds / step_seconds)
    sums = np.concatenate([[0.0], np.cumsum(np.nan_to_num(gap))])
    counts = np.concatenate([[0], np.cumsum(np.isfinite(gap))])
    out = np.full(len(gap), np.nan)
    if len(gap) > n:
        i = np.arange(n, len(gap))
        out[n:] = gap[n:] - (sums[i] - sums[i - n]) / np.maximum(counts[i] - counts[i - n], 1)
    return out


# LEAD

def add_pairs(sums, a, b):
    """
    Add the pairs of changes a and b at each lag to their running sums, {lag: [n, Σx, Σy, Σx², Σy², Σxy]}: at lag k,
    a's change at each step with b's k steps later, so at a positive lag a moves first.
    """
    for lag, s in sums.items():
        x, y = a[max(0, -lag):len(a) - max(0, lag)], b[max(0, lag):len(b) - max(0, -lag)]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        s += [len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]


def correlations(sums):
    """
    The correlation at each lag from running sums, see add_pairs(), NaN without pairs or variance.
    """
    out = {}
    for lag, (n, sx, sy, sxx, syy, sxy) in sums.items():
        var_x, var_y = sxx / n - (sx / n) ** 2 if n else 0, syy / n - (sy / n) ** 2 if n else 0
        out[lag] = (sxy / n - sx * sy / n ** 2) / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan
    return out


def peak(corr):
    """
    The lag in seconds the correlation peaks at, the peak, and the share of the positive correlation at positive lags,
    where the source moves first.
    """
    lag = max(corr, key=lambda k: -np.inf if np.isnan(corr[k]) else corr[k])
    first = sum(max(c, 0) for k, c in corr.items() if k > 0)
    after = sum(max(c, 0) for k, c in corr.items() if k < 0)
    return lag * STEP, corr[lag], first / (first + after) if first + after > 0 else np.nan


def add_catch_up(sums, moved, average):
    """
    Add to the running sums of the slope of the average's move over the next h seconds on how far the source had moved
    ahead of it, {h: [Σxy, Σx²]}, both a second apart.
    """
    for h, s in sums.items():
        x, y = moved[:-h], average[h:] - average[:-h]
        ok = np.isfinite(x) & np.isfinite(y)
        s[0] += (x[ok] * y[ok]).sum()
        s[1] += (x[ok] ** 2).sum()


def lead(conn, windows, sources):
    """
    Print each source's lead over the exchanges' average and over Kalshi's mid, and the average's catch up, over the
    recorded windows the source quoted through.
    """
    print(f"\n== LEAD: changes over {STEP} s, by the venues' own times; 'leads by' is the lag of the peak correlation, positive when the "
          f"source moves first,\n   'first' the share of correlation on that side, and catch up the share of its move ahead the average makes")
    start, end = windows[0].open_time - BASIS_SECONDS - 60, windows[-1].close_time + 10
    feeds = {s: replay.quote_feed(conn, s, start, end) for s in sources}
    lags = range(-round(LAG_SECONDS / STEP), round(LAG_SECONDS / STEP) + 1)
    sums = {s: ({lag: np.zeros(6) for lag in lags}, {lag: np.zeros(6) for lag in lags}) for s in sources}
    catch = {s: {h: [0.0, 0.0] for h in (1, 5)} for s in sources}
    spans = {s: 0 for s in sources}
    for m in windows:
        g = np.arange(m.open_time, m.close_time, STEP)
        dm = np.diff(kalshi_mid(replay.book_changes(conn, m.ticker), g))
        for s in sources:
            src = at_once(feeds[s], g)
            if np.isnan(src[0]):
                continue
            others = joined([feeds[e] for e in replay.EXCHANGES if e != s and e in feeds])
            average = at_once(others, g)
            add_pairs(sums[s][0], np.diff(src), np.diff(average))
            add_pairs(sums[s][1], np.diff(src), dm)
            each = slice(None, None, round(1 / STEP))                          # Once a second.
            add_catch_up(catch[s], ahead(src, average, STEP)[each], average[each])
            spans[s] += 1
    print(f"{'source':<15}{'windows':>8}{'vs average: leads by':>22}{'peak':>7}{'first':>7}{'vs Kalshi: leads by':>21}{'peak':>7}"
          f"{'first':>7}{'catch up 1 s':>14}{'5 s':>6}")
    for s in sources:
        if not spans[s]:
            continue
        (la, ca, fa), (lk, ck, fk) = peak(correlations(sums[s][0])), peak(correlations(sums[s][1]))
        up = [catch[s][h][0] / catch[s][h][1] if catch[s][h][1] else np.nan for h in (1, 5)]
        print(f"{s:<15}{spans[s]:>8}{la:>+21.1f}s{ca:>7.3f}{fa:>7.0%}{lk:>+20.1f}s{ck:>7.3f}{fk:>7.0%}{up[0]:>14.2f}{up[1]:>6.2f}")
    print("Each of the six exchanges is set against the average of the other five.")


# MODEL

def model_rows(conn, windows, sources, history, start):
    """
    For each window the sources quoted through, the seconds at 1 to 5 minutes left: the model's log price of the expected
    settlement over the strike and its spread, Kalshi's mid, whether Up won, and how far ahead of the average each source
    was, in log, as a trader on the AWS instance would have them, see ahead(). Returns a dict of arrays with each
    window's index, oldest first.
    """
    variance = fair.variance(history)
    feed = replay.quote_feed(conn, "exchanges", windows[0].open_time - BASIS_SECONDS - 60, windows[-1].close_time)
    tracking = fair.tracking_variances([replay.settlement_error(feed, m) for m in windows], [m.open_time for m in windows],
                                       [m.close_time for m in windows])
    arrivals = {s: arrival_feed(conn, s, windows[0].open_time - BASIS_SECONDS - 60, windows[-1].close_time) for s in sources}
    readings = np.arange(1, 61) - 60
    rows = {c: [] for c in ("w", "left", "num", "spread", "mid", "up", *sources)}
    for w, (m, track) in enumerate(zip(windows, tracking, strict=True)):
        t = np.arange(m.close_time - 300, m.close_time - 60, 1.0)
        basis = np.arange(t[0] - BASIS_SECONDS, t[-1] + 1, 1.0)
        if np.isnan(replay.price_at(feed, [m.open_time - 59, basis[0]])).any():
            continue
        offset = np.nanmean(replay.price_at(feed, m.open_time + readings)) - m.strike
        average = replay.price_at(feed, basis)
        est = average[BASIS_SECONDS:] - offset
        fcast = np.maximum(variance[(t.astype(int) - start) // 60 - 1], fair.TINY)
        for s in sources:
            rows[s].append(ahead(replay.price_at(arrivals[s], basis), average, 1.0)[BASIS_SECONDS:] / est)
        for c, v in (("w", np.full(len(t), w)), ("left", m.close_time - t), ("num", np.log(est / m.strike)),
                     ("spread", pricing.spreads(fcast, m.close_time - t, tracking=track)),
                     ("mid", kalshi_mid(replay.book_changes(conn, m.ticker), t)), ("up", np.full(len(t), m.result == "yes"))):
            rows[c].append(v)
    return {c: np.concatenate(v) if v else np.array([]) for c, v in rows.items()}


def model(conn, windows, sources, log=print):
    """
    Print the model with each source's move ahead added, fitted on the earlier half of the windows and scored on the
    later, and with the sources' moves averaged, against the model alone and Kalshi's mid.
    """
    print("\n== MODEL: log loss at 1 to 5 minutes left, beta fitted on the earlier half of the windows, scored on the later")
    if not windows:
        print("no window with every source quoting yet")
        return
    first_day = datetime.fromtimestamp(windows[0].open_time, UTC).date() - timedelta(days=replay.HISTORY_DAYS)
    end_day = datetime.fromtimestamp(windows[-1].close_time, UTC).date() + timedelta(days=1)
    history = download.recent_history(conn, first_day, end_day, log)
    start = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp())
    d = model_rows(conn, windows, sources, history, start)
    if not len(d["w"]):
        print("no window with every source quoting yet")
        return
    d["all"] = np.nanmean(np.column_stack([d[s] for s in sources]), axis=1)
    ok = np.isfinite(d["mid"]) & np.isfinite(d["num"]) & np.isfinite(d["all"])
    later = d["w"] >= np.median(np.unique(d["w"]))
    up = d["up"].astype(bool)
    print(f"{len(np.unique(d['w'][ok & ~later]))} windows to fit on, {len(np.unique(d['w'][ok & later]))} to score")
    if min(len(np.unique(d["w"][ok & ~later])), len(np.unique(d["w"][ok & later]))) < MIN_WINDOWS:
        print(f"too few: {MIN_WINDOWS} a half needed")
        return

    def chance(moved, beta):
        return pricing.up_student_t((d["num"] + beta * np.nan_to_num(moved)) / d["spread"], fair.DOF)

    print(f"{'with':<15}{'beta':>6}" + "".join(f"{f'{low // 60}-{high // 60} min: market':>20}{'model':>7}{'+ source':>10}" for low, high in SPANS))
    for s in (*sources, "all"):
        fit = ok & ~later
        beta = BETAS[int(np.argmin([metrics.log_loss(chance(d[s], b)[fit], up[fit]) for b in BETAS]))]
        cells = []
        for low, high in SPANS:
            x = ok & later & (d["left"] > low) & (d["left"] <= high)
            cells.append(f"{metrics.log_loss(d['mid'][x], up[x]):>20.3f}{metrics.log_loss(chance(d[s], 0.0)[x], up[x]):>7.3f}"
                         f"{metrics.log_loss(chance(d[s], beta)[x], up[x]):>10.3f}")
        print(f"{s:<15}{beta:>6.2f}" + "".join(cells))
    print("'all' adds the sources' moves averaged. A beta of 1 moves the model's price by the whole of the source's move ahead.")


# MAIN

if __name__ == "__main__":
    conn = database.connect()
    windows = replay.recorded_windows(conn)
    if not windows:
        raise SystemExit("no settled window recorded from its open yet, run the recorder for a while")
    sources = list(QUOTE_STREAMS)
    lead(conn, windows, sources)
    perps = [s for s in sources if s not in replay.EXCHANGES]
    recorded = [m for m in windows if all(conn.execute("SELECT 1 FROM quotes WHERE source = ? AND received <= ? LIMIT 1",
                                                       (s, m.open_time - BASIS_SECONDS - 60)).fetchone() for s in perps)]
    model(conn, recorded, perps, log)
