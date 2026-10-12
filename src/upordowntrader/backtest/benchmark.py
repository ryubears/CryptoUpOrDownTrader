"""
Benchmark the fair price model against Kalshi's market, minute by minute, over every settled window Kalshi kept.

Kalshi's windows carry the exact BRTI averages they settle on, and its candlesticks the market's best Yes bid and ask as
each minute ended, see history/markets.py. So at the end of each of a window's first 14 minutes, its marks, the model's
chance of Up can be scored against how the window settled and against the market's mid price, over months of windows
rather than the days recorded. The model reads Coinbase's close of the minute, moved onto BRTI's level by the gap between
the two at the opens before, see brti_gaps(), with the variance forecast of model/fair.py from the minutes before.

Coinbase's minutes are a rough stand-in for BRTI: a minute's close is its last trade, and an open's average is taken as
(open + high + low + close) / 4, whose gap to BRTI's moved by about $13 from one open to the next in 2026. So late in a
window, when a few dollars decide it, the model does worse here than it would with the exchanges recorded live, see
backtest/replay.py, and the early marks are the fairer test of what it knows. Windows Coinbase or Kalshi got wrong are
left out, see scorable().

Everything is out of sample: the recalibration and the blend, see model/fair.py, are fitted every STEP_DAYS on the
TRAIN_DAYS of marks before, and only marks after the first TRAIN_DAYS are scored. The replay fits them the same way on
the TRAIN_DAYS before the windows it replays, see fit_before(). The report:

    log loss       by time left: the market's mid, the model before 2026-10-12 (the calibration's forecast, the gap at
                   the open alone), the model now, see model/fair.py, recalibrated, and blended with the market
    calibration    the market's own: how often its favorite won against what it said
    disagreements  when the recalibrated model and the market differ by THRESHOLD or more, which came true
    trades         one contract at the minute's closing ask or bid whenever a fair price's edge after Kalshi's taker fee
                   reaches a threshold: the realized profit a contract, with two standard errors that count each window
                   once, as its marks share an outcome. No latency, depth, or queue: each order fills at the quote.
    months         the scores and trades month by month, to see whether they hold

Run from src/ with:
    python3 -m upordowntrader.backtest.benchmark
"""

import argparse
import numpy as np
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from scipy.signal import lfilter
from upordowntrader.backtest import metrics
from upordowntrader.common.log import log
from upordowntrader.db import database
from upordowntrader.history import download
from upordowntrader.model import fair, pricing, volatility

SERIES = "KXBTC15M"
WINDOW_SECONDS = 900
MARKS = 14                  # Minute marks scored in a window, the ends of its first 14 minutes.
MAX_SPREAD = 0.10           # The widest bid to ask a mark's mid price is taken from.
MAX_GAP = 0.02              # The furthest a window's strike may be from Coinbase's average then, as a share, to be scored.
GAP_HALF_LIFE = 4           # Opens. The half life of the average of the gap between Coinbase and BRTI, see brti_gaps().
TRAIN_DAYS = 60             # Days of marks each fit is made on.
STEP_DAYS = 7               # Days between fits.
HISTORY_DAYS = 7            # Days of Coinbase's minutes read before the first window, for the day's variance forecast.
THRESHOLD = 0.05            # Disagreement, and edge after the fee, in dollars a contract, the report centers on.
THRESHOLDS = (0.02, 0.05)   # Edges the trades are tried at.
SPANS = ((600, 900), (300, 600), (120, 300), (0, 120))  # Seconds left the report is broken into.
DAY = 86400


@dataclass
class Marks:
    """
    The minute marks scored, one per window and mark, as flat arrays.
    """
    windows: list               # The windows, as Markets, oldest first.
    window: np.ndarray          # Which window each mark is in, an index into windows.
    seconds_left: np.ndarray
    bid: np.ndarray             # The market's best Yes bid and ask as the minute ended, NaN for an empty side.
    ask: np.ndarray
    mid: np.ndarray             # Their average, NaN unless both are there within MAX_SPREAD.
    price: np.ndarray           # Coinbase's close of the minute ending at the mark.
    strike: np.ndarray
    up: np.ndarray              # Whether Up won.
    minute: np.ndarray          # The minute ending at the mark, as an index into the history's minutes.
    day: np.ndarray             # Days from the first window's day.


# MARKS

def settled_windows(conn, end_time):
    """
    The series' settled windows with their exact strike and settlement and their minutes stored, closing by end_time, in
    seconds since 1970, oldest first, and the minutes, see database.read_minutes().
    """
    minutes = database.read_minutes(conn, "kalshi")
    windows = [m for m in database.load_markets(conn, "kalshi")
               if m.series == SERIES and m.result in ("yes", "no") and m.strike is not None and m.settlement is not None
               and m.close_time - m.open_time == WINDOW_SECONDS and m.ticker in minutes and m.close_time <= end_time]
    return windows, minutes


def marks(windows, minutes, history, start):
    """
    The Marks of the windows. start is the history's start, in seconds since 1970.
    """
    rows = []
    for w, m in enumerate(windows):
        quotes = {int(t): (bid, ask) for t, bid, ask in minutes[m.ticker]}
        for k in range(1, MARKS + 1):
            rows.append((w, m.open_time + 60 * k, *quotes.get(m.open_time + 60 * k, (np.nan, np.nan))))
    r = np.array(rows, dtype=float).reshape(-1, 4)
    window, time = r[:, 0].astype(int), r[:, 1].astype(int)
    bid, ask = r[:, 2], r[:, 3]
    quoted = (ask > bid) & (ask - bid <= MAX_SPREAD)
    minute = (time - start) // 60 - 1
    opens = np.array([m.open_time for m in windows])
    return Marks(windows=windows, window=window, seconds_left=np.array([m.close_time for m in windows])[window] - time, bid=bid, ask=ask,
                 mid=np.where(quoted, (bid + ask) / 2, np.nan), price=history.minute_close[minute],
                 strike=np.array([m.strike for m in windows])[window], up=np.array([m.result == "yes" for m in windows])[window],
                 minute=minute, day=(opens[window] - opens[0] // DAY * DAY) // DAY)


def scorable(windows, history, start):
    """
    Whether each window can be scored: Coinbase has a bar in every minute from the one before its open to its last mark,
    and its strike lies within MAX_GAP of Coinbase's average then. Coinbase was down for hours on 2026-05-08, and Kalshi
    set a few strikes wrong, 7.24 for bitcoin at 72,447 on 2026-04-13.
    """
    first = (np.array([w.open_time for w in windows]) - start) // 60 - 1
    missing = np.concatenate([[0], np.cumsum(history.minute_missing)])
    complete = missing[first + MARKS + 1] - missing[first] == 0
    return complete & (np.abs(history.minute_average[first] / np.array([w.strike for w in windows]) - 1) <= MAX_GAP)


def brti_gaps(windows, history, start, half_life=GAP_HALF_LIFE):
    """
    How far Coinbase ran above BRTI at each window, as the model takes it: at each open, the gap between Coinbase's
    average of the minute before and the strike, BRTI's exact average of it, averaged over the opens up to the window's
    own, each one's weight halving every half_life opens, since a single open's is mostly the noise of the stand-in.
    """
    opens = np.array([m.open_time for m in windows])
    gaps = history.minute_average[(opens - start) // 60 - 1] - np.array([m.strike for m in windows])
    decay = 0.5 ** (1 / half_life)
    averaged, _ = lfilter([1 - decay], [1, -decay], gaps, zi=[decay * gaps[0]])
    return gaps, averaged


def settlement_errors(windows, history, start, gap):
    """
    The log of Coinbase's stand-in for each window's settlement over BRTI's: its average of the minute before the close
    less the gap to BRTI the model takes at the open, see brti_gaps(). NaN where Coinbase has no bar in that minute.
    """
    last = (np.array([m.close_time for m in windows]) - start) // 60 - 1
    average = np.where(history.minute_missing[last] > 0, np.nan, history.minute_average[last])
    return np.log((average - gap) / np.array([m.settlement for m in windows]))


def z_scores(conn, windows, minutes, log=print):
    """
    The Marks of the scorable windows, see scorable(), the gaps at their opens, see brti_gaps(), and at each mark the
    model's z score as it was before 2026-10-12, from the calibration's forecast and the gap at the open alone, and as it
    is now, see model/fair.py, with the tracking variance of Coinbase's stand-in, see settlement_errors(). Coinbase's days
    not stored yet are downloaded.
    """
    first_day = datetime.fromtimestamp(windows[0].open_time, UTC).date() - timedelta(days=HISTORY_DAYS)
    end_day = datetime.fromtimestamp(windows[-1].close_time, UTC).date() + timedelta(days=1)
    history = download.recent_history(conn, first_day, end_day, log)
    start = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp())
    windows = [w for w, ok in zip(windows, scorable(windows, history, start), strict=True) if ok]
    m = marks(windows, minutes, history, start)
    gaps, gap = brti_gaps(windows, history, start)
    tracking = fair.tracking_variances(settlement_errors(windows, history, start, gap), [w.open_time for w in windows],
                                       [w.close_time for w in windows])
    before, _ = volatility.forecasts(history, *fair.FORECAST)
    z_before = pricing.z_scores(m.price - gaps[m.window], 0.0, m.strike, m.seconds_left, np.maximum(before[m.minute], fair.TINY))
    z = pricing.z_scores(m.price - gap[m.window], 0.0, m.strike, m.seconds_left, np.maximum(fair.variance(history)[m.minute], fair.TINY),
                         tracking=tracking[m.window])
    return m, gaps, z_before, z


# SCORING

def walk_forward(z, marks, train_days=TRAIN_DAYS, step_days=STEP_DAYS):
    """
    The recalibrated chance of Up and the blend at every mark, each from a fair.Fit on the train_days of marks before its
    step_days. NaN for the marks of the first train_days, which have no past to fit on.
    """
    recalibrated, blend = np.full(len(z), np.nan), np.full(len(z), np.nan)
    for first in range(train_days, int(marks.day.max()) + 1, step_days):
        past = (marks.day >= first - train_days) & (marks.day < first)
        now = (marks.day >= first) & (marks.day < first + step_days)
        if now.any() and past.any():
            fit = fair.fit(z[past], marks.mid[past], marks.up[past], marks.seconds_left[past])
            recalibrated[now] = fair.chance(z[now], marks.seconds_left[now], fit)
            blend[now] = fair.blended(marks.mid[now], recalibrated[now], marks.seconds_left[now], fit)
    return recalibrated, blend


def fit_before(conn, before, log=print):
    """
    The fair.Fit on the settled windows of the TRAIN_DAYS before a moment, in seconds since 1970, as walk_forward() makes
    them, for the replay. None when no window with minutes is stored for those days, see history/markets.py.
    """
    windows, minutes = settled_windows(conn, before)
    windows = [w for w in windows if w.open_time >= before - TRAIN_DAYS * DAY]
    if not windows:
        return None
    m, _, _, z = z_scores(conn, windows, minutes, log)
    return fair.fit(z, m.mid, m.up, m.seconds_left)


def trades(price, marks, threshold, inside):
    """
    One contract wherever a fair price's edge after Kalshi's taker fee reaches threshold, among the marks inside, at the
    minute's closing ask for Yes or bid for No: how many, the realized profit a contract, and its standard error, counting
    each window once.
    """
    yes = inside & (fair.edge(price, marks.ask) >= threshold)
    no = inside & (fair.edge(1 - price, 1 - marks.bid) >= threshold)
    up = marks.up.astype(float)
    profit = np.concatenate([fair.edge(up[yes], marks.ask[yes]), fair.edge(1 - up[no], 1 - marks.bid[no])])
    if not len(profit):
        return 0, np.nan, np.nan
    _, by_window = np.unique(np.concatenate([marks.window[yes], marks.window[no]]), return_inverse=True)
    sums = np.bincount(by_window, weights=profit - profit.mean())
    return len(profit), float(profit.mean()), float(np.sqrt(np.sum(sums ** 2)) / len(profit))


# REPORT

def span_label(low, high):
    """
    A span of seconds left as minutes, 10-15 min.
    """
    return f"{low // 60}-{high // 60} min"


def trade_cells(price, marks, threshold, inside):
    """
    The trades of a fair price in each of SPANS, as report cells.
    """
    cells = []
    for low, high in SPANS:
        count, mean, error = trades(price, marks, threshold, inside & (marks.seconds_left > low) & (marks.seconds_left <= high))
        cells.append(f"{100 * mean:>+6.1f} +-{200 * error:>4.1f}c {count:>6,}" if count else f"{'':>22}")
    return cells


def report(marks, gaps, chances, log=print):
    """
    Print the benchmark, see the module's docstring. chances maps each fair price's name to its chance of Up at every mark.
    """
    m = marks
    scored = ~np.isnan(m.mid) & ~np.isnan(chances["blend"])
    first, last = m.windows[0], m.windows[-1]
    log(f"\n{len(m.windows):,} settled windows of {SERIES}, {datetime.fromtimestamp(first.open_time, UTC):%Y-%m-%d} to "
        f"{datetime.fromtimestamp(last.close_time, UTC):%Y-%m-%d}, {np.mean(~np.isnan(m.mid)):.0%} of marks quoted within {100 * MAX_SPREAD:.0f}c. "
        f"Scored out of sample: {len(np.unique(m.window[scored])):,} windows, {scored.sum():,} marks, after the first {TRAIN_DAYS} days.")
    log(f"Coinbase minus BRTI at the opens: median {np.median(gaps):+.2f}$, varying from one open to the next by {np.std(np.diff(gaps)):.2f}$.")

    names = ["market", *chances]
    log("\nLog loss by time left, lower is better, a coin flip scores 0.693:")
    log(f"{'left':<12}" + "".join(f"{name:>14}" for name in names))
    for low, high in (*SPANS, (0, WINDOW_SECONDS)):
        s = scored & (m.seconds_left > low) & (m.seconds_left <= high)
        label = "all" if (low, high) == (0, WINDOW_SECONDS) else span_label(low, high)
        scores = [metrics.log_loss(m.mid[s], m.up[s]), *(metrics.log_loss(p[s], m.up[s]) for p in chances.values())]
        log(f"{label:<12}" + "".join(f"{score:>14.4f}" for score in scores))

    log("\nThe market's calibration, by its favorite's price:")
    log(f"{'favorite':<12}{'marks':>9}{'windows':>9}{'said':>8}{'won':>8}{'gap':>8}{'error':>8}")
    for row in metrics.favorites(m.mid[scored], m.up[scored], m.window[scored]):
        log(f"{row['low']:>4.0%} to {row['high']:<4.0%}{row['evaluations']:>9,}{row['windows']:>9,}{row['said']:>8.1%}{row['won']:>8.1%}"
            f"{100 * (row['won'] - row['said']):>+7.1f}c{100 * row['error']:>7.1f}c")

    p = chances["recalibrated"]
    log(f"\nWhen the recalibrated model and the market disagree by {100 * THRESHOLD:.0f}c or more, which came true:")
    log(f"{'left':<12}{'model':<8}{'marks':>8}{'windows':>9}{'market':>9}{'model':>8}{'won':>8}")
    for low, high in SPANS:
        for side, sign in (("above", 1), ("below", -1)):
            s = scored & (sign * (p - m.mid) >= THRESHOLD) & (m.seconds_left > low) & (m.seconds_left <= high)
            if s.any():
                log(f"{span_label(low, high):<12}{side:<8}{s.sum():>8,}{len(np.unique(m.window[s])):>9,}{m.mid[s].mean():>9.3f}{p[s].mean():>8.3f}"
                    f"{m.up[s].mean():>8.3f}")

    log("\nOne contract at the minute's closing quote when the edge after the fee reaches the threshold: realized profit a contract,")
    log("+- two standard errors, and trades.")
    log(f"{'fair price':<16}{'edge':>5}" + "".join(f"{span_label(low, high):>23}" for low, high in SPANS))
    for name, price in chances.items():
        for threshold in THRESHOLDS:
            log(f"{name:<16}{100 * threshold:>4.0f}c" + "".join(f"{cell:>23}" for cell in trade_cells(price, m, threshold, scored)))

    log(f"\nBy month: log loss of the market, recalibrated, and blend, and the recalibrated model's trades at {100 * THRESHOLD:.0f}c, at any time.")
    month = np.array([datetime.fromtimestamp(w.open_time, UTC).strftime("%Y-%m") for w in m.windows])[m.window]
    log(f"{'month':<9}{'windows':>8}{'market':>9}{'recal':>9}{'blend':>9}{'trades':>8}{'realized':>16}")
    for name in sorted(set(month[scored])):
        s = scored & (month == name)
        count, mean, error = trades(p, m, THRESHOLD, s)
        log(f"{name:<9}{len(np.unique(m.window[s])):>8,}{metrics.log_loss(m.mid[s], m.up[s]):>9.4f}{metrics.log_loss(p[s], m.up[s]):>9.4f}"
            f"{metrics.log_loss(chances['blend'][s], m.up[s]):>9.4f}{count:>8,}" + (f"{100 * mean:>+9.1f} +-{200 * error:.1f}c" if count else ""))


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score the fair price model against Kalshi's market and settlements, minute by minute.")
    ap.add_argument("--start", type=date.fromisoformat, help="first UTC day of windows, YYYY-MM-DD, by default the first stored")
    args = ap.parse_args()
    conn = database.connect()
    windows, minutes = settled_windows(conn, datetime.now(UTC).timestamp())
    if args.start:
        windows = [w for w in windows if w.open_time >= datetime(args.start.year, args.start.month, args.start.day, tzinfo=UTC).timestamp()]
    if not windows:
        raise SystemExit("no settled windows with minutes, download them first with upordowntrader.history.markets")
    m, gaps, z_before, z = z_scores(conn, windows, minutes, log)
    log(f"{len(m.window):,} marks of {len(windows):,} windows; fitting every {STEP_DAYS} days")
    recalibrated, blend = walk_forward(z, m)
    report(m, gaps, {"before": fair.chance(z_before, m.seconds_left), "model": fair.chance(z, m.seconds_left), "recalibrated": recalibrated,
                     "blend": blend}, log=log)
