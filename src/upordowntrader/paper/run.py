"""
Paper trade Kalshi's and Polymarket US's Bitcoin 15 minute windows live, see paper/trader.py, beside the recorder, on
streams of its own, so either can stop or change without the other.

The process streams Kalshi's books of the window trading and the next, Polymarket US's of the same windows when its API
key is in data/, and the best bid and ask of the exchanges BRTI is built from. It reads Kalshi's windows just after each
quarter hour and every REFRESH_SECONDS, and every STRIKE_SECONDS while the window trading has no strike yet. Once a window
opens it reads the window's tracking variance from the recorded quotes, and, when the trader started after the window
opened, its offset, see trader.recorded_offset(). Every VARIANCE_SECONDS it reads Coinbase's minutes again for the
variance forecast, and every FIT_SECONDS it downloads Kalshi's windows settled since, see history/markets.py, and fits
the recalibration and the blend again on the TRAIN_DAYS before, see benchmark.fit_before(). Nothing is traded until the
first fit is made, a minute or two after the start. Every SETTLE_SECONDS it reads how the windows with paper orders
settled, on each order's own venue, and stores what they made. The trader stores the opportunities it sees as they end,
and those still open as the process stops. A status line comes every STATUS_SECONDS. The work on the database outside
the trader's own writes runs in threads on connections of their own.

It only decides and stores pretend orders, nothing is sent. See paper/report.py for what they made and the opportunities.

Run from src/ with:
    python3 -m upordowntrader.paper.run
    python3 -m upordowntrader.paper.run --seconds 600 --db /tmp/paper.sqlite
"""

import argparse
import asyncio
import math
import signal
import time
from datetime import UTC, datetime, timedelta
from upordowntrader.api import kalshi, polymarket_us
from upordowntrader.backtest import benchmark, replay
from upordowntrader.common.log import log, on_failure
from upordowntrader.common.periodic import Periodic
from upordowntrader.db import database
from upordowntrader.history import download, markets
from upordowntrader.model import fair
from upordowntrader.paper.trader import PaperTrader, recorded_offset, tracking_variance
from upordowntrader.record.run import QUOTE_STREAMS, code_version

SERIES = "KXBTC15M"
WINDOW_SECONDS = 900
OPEN_DELAY = 2              # Seconds after a window opens before it is read again, so Kalshi has set its strike.
REFRESH_SECONDS = 60
STRIKE_SECONDS = 2          # Seconds between reads while the window trading has no strike yet.
LOOKAHEAD_SECONDS = 1800    # How far ahead the windows read reach: past the one trading now to the next.
VARIANCE_SECONDS = 60
FIT_SECONDS = 86400
SETTLE_SECONDS = 30
STATUS_SECONDS = 300
HISTORY_DAYS = benchmark.HISTORY_DAYS   # Days of Coinbase's minutes the variance forecast reads, as the replay does.


def quiet(message):
    """
    A log that drops what it is given, for the downloads' progress lines, which would come every minute.
    """


# WORK IN THREADS, each on a connection of its own

def read_tracking(db_path, market, errors):
    """
    A window's tracking variance, see trader.tracking_variance().
    """
    return tracking_variance(database.connect(db_path), market, errors)


def read_offset(db_path, market):
    """
    A window's offset from the recorder's quotes, see trader.recorded_offset().
    """
    return recorded_offset(database.connect(db_path), market)


def read_variance(db_path):
    """
    The variance forecast at the end of each minute from HISTORY_DAYS ago to the end of today, and the first minute's
    start, from Coinbase's minutes, see fair.variance(): the days stored, and today's read from Coinbase.
    """
    today = datetime.now(UTC).date()
    first = today - timedelta(days=HISTORY_DAYS)
    history = download.recent_history(database.connect(db_path), first, today + timedelta(days=1), quiet)
    return fair.variance(history), int(datetime(first.year, first.month, first.day, tzinfo=UTC).timestamp())


def make_fit(db_path):
    """
    The fair.Fit on the TRAIN_DAYS of Kalshi's windows settled before now, after storing those settled since the last
    download, see history/markets.py.
    """
    conn = database.connect(db_path)
    markets.download(conn, log=quiet)
    return benchmark.fit_before(conn, time.time(), quiet)


# PERIODIC WORK

async def refresh(trader, streams, db_path):
    """
    Price Kalshi's windows closing from now to LOOKAHEAD_SECONDS ahead, stream their books on each venue traded, and read
    the tracking variance of each opened with its strike, and its offset when the trader has no readings from before it
    opened. The books of windows closed are dropped from the streams.
    """
    now = time.time()
    found = [m for m in await asyncio.to_thread(kalshi.windows, SERIES, now - WINDOW_SECONDS, now + LOOKAHEAD_SECONDS) if m.close_time > now]
    trader.set_windows(found)
    for venue, stream in streams.items():
        names = {trader.windows[m.close_time].names[venue] for m in found}
        stream.add(names)
        stream.remove(stream.wanted - names)
    for w in trader.windows.values():
        if w.tracking is None and w.market.strike is not None and w.market.open_time <= now < w.market.close_time:
            w.tracking = await asyncio.to_thread(read_tracking, db_path, w.market, trader.errors)
            dollars = math.sqrt(w.tracking) * w.market.strike
            trader.log(f"paper: {w.market.ticker} opened at a strike of {w.market.strike:,.2f}, tracking error ${dollars:.2f}")
        if w.offset is None and w.market.strike is not None and w.market.open_time + 1 <= now < w.market.close_time and trader.trading(now) is None:
            w.offset = await asyncio.to_thread(read_offset, db_path, w.market)
            trader.log(f"paper: {w.market.ticker} opened before the trader started, its price's offset read from the recorder's quotes: "
                       f"{w.offset:+.2f}")


async def refresh_variance(trader, db_path):
    trader.variance = await asyncio.to_thread(read_variance, db_path)


async def refit(trader, db_path):
    started = time.time()
    fit = await asyncio.to_thread(make_fit, db_path)
    if fit is None:
        trader.log("paper: no Kalshi windows with minutes in the days before, so no fit and no trading")
        return
    trader.fit = fit
    trader.log(f"paper: fitted in {time.time() - started:.0f}s; the blend's weights on the market and the model, 14 to 1 minutes "
               f"left: {' '.join(f'{m:.2f}/{p:.2f}' for _, m, p in fit.blend[::-1])}")


async def settle(conn, log):
    """
    Read how the windows with paper orders closed and not settled yet went, each on its own venue, and settle the orders.
    """
    pending = database.unsettled_paper_markets(conn, time.time())
    if not pending:
        return
    tickers = [ticker for venue, ticker in pending if venue == "kalshi"]
    names = [ticker for venue, ticker in pending if venue == "polymarket_us"]
    found = await asyncio.to_thread(lambda: [*(kalshi.markets(tickers) if tickers else []), *(polymarket_us.window(name) for name in names)])
    for m in found:
        if m is not None and m.result in ("yes", "no") and database.settle_paper_orders(conn, m.venue, m.ticker, m.result):
            made = conn.execute("SELECT COUNT(*), SUM(contracts), SUM(profit) FROM paper_orders WHERE venue = ? AND ticker = ?",
                                (m.venue, m.ticker)).fetchone()
            log(f"paper {m.venue} {m.ticker} settled {m.result}: {made[0]} orders, {made[1]:g} contracts, {made[2]:+.2f}$")


def status(conn, trader):
    """
    One line on the window trading, what the paper orders made since midnight UTC, and the opportunities one order could
    have taken at the threshold or more since then: how many, what that would have cost, and what it was expected to make.
    """
    now = time.time()
    midnight = now // 86400 * 86400
    orders, contracts, profit, unsettled = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(contracts), 0), COALESCE(SUM(profit), 0), SUM(result IS NULL) FROM paper_orders WHERE decided >= ?",
        (midnight,)).fetchone()
    found, capital, expected = conn.execute("SELECT COUNT(*), COALESCE(SUM(take_capital), 0), COALESCE(SUM(take_profit), 0) "
                                            "FROM opportunities WHERE start_time >= ? AND take_size > 0", (midnight,)).fetchone()
    w = trader.trading(now)
    line = "paper: no window priced"
    if w is not None:
        line = (f"paper: {w.market.ticker} {(w.market.close_time - now) / 60:.1f} min left, price {trader.stand_in() - w.offset:,.2f} "
                f"against {w.market.strike:,.2f}")
    ready = "" if trader.fit is not None and trader.variance is not None else ", waiting for the fit and the variance forecast"
    return (f"{line}{ready}; today {orders} orders, {contracts:g} contracts, {profit:+.2f}$ settled, {unsettled or 0} orders to settle; "
            f"{found} opportunities at {100 * trader.threshold:.0f}c or more, {capital:,.0f}$ in one order each expecting {expected:+,.2f}$")


# MAIN LOOP

async def trade(db_path=None, seconds=None, log=log):
    """
    Paper trade until seconds have passed, or for good without.
    """
    conn = database.connect(db_path)
    trader = PaperTrader(conn, log)
    venue_streams = {"kalshi": kalshi.KalshiBookStream([], lambda t, b, a, s: trader.on_book("kalshi", t, b, a, s),
                                                       lambda start, end, ids: trader.on_gap("kalshi", ids), log)}
    if polymarket_us.has_key():
        venue_streams["polymarket_us"] = polymarket_us.PolymarketUSBookStream(
            [], lambda t, b, a, s: trader.on_book("polymarket_us", t, b, a, s), lambda start, end, ids: trader.on_gap("polymarket_us", ids), log)
    else:
        log(f"no Polymarket US API key in {polymarket_us.KEY_ID_FILE.parent}, so only Kalshi is traded")
    trader.venues = tuple(venue_streams)
    for venue, stream in venue_streams.items():
        stream.depth = trader.depth
        stream.on_state = lambda ticker, why, venue=venue: trader.on_state(venue, ticker, why)
    quote_streams = {}
    for source in replay.EXCHANGES:
        stream_class, product = QUOTE_STREAMS[source]
        quote_streams[source] = stream_class([product], lambda product, bids, asks, sent, source=source: trader.on_quote(source, bids, asks),
                                             None, log)
    trader.feeds = {**venue_streams, **quote_streams}
    feeds = [asyncio.create_task(s.run()) for s in trader.feeds.values()]
    for s, feed in zip(trader.feeds.values(), feeds, strict=True):
        feed.add_done_callback(on_failure(log, f"the paper {s.name} stream"))
    refresher = Periodic(REFRESH_SECONDS, log, "paper: reading the windows")
    variance = Periodic(VARIANCE_SECONDS, log, "paper: reading the variance forecast")
    fitter = Periodic(FIT_SECONDS, log, "paper: fitting")
    settler = Periodic(SETTLE_SECONDS, log, "paper: settling")
    started = last_status = time.time()
    window = None
    log(f"paper trading {SERIES} on {', '.join(trader.venues)} at an edge of {100 * trader.threshold:.0f}c, up to {trader.contracts} "
        f"contracts an order, with code {code_version()}")
    try:
        while seconds is None or time.time() - started < seconds:
            await asyncio.sleep(1)
            now = time.time()
            trader.tick(now)
            if int((now - OPEN_DELAY) // WINDOW_SECONDS) != window:      # A window opened, so read it and the next at once.
                window = int((now - OPEN_DELAY) // WINDOW_SECONDS)
                refresher.again()
            w = trader.opened(now)
            if w and (w.market.strike is None or w.tracking is None or w.offset is None) and now - (refresher.last or 0) >= STRIKE_SECONDS:
                refresher.again()
            refresher.tick(now, lambda: refresh(trader, venue_streams, db_path))
            variance.tick(now, lambda: refresh_variance(trader, db_path))
            fitter.tick(now, lambda: refit(trader, db_path))
            settler.tick(now, lambda: settle(conn, log))
            if now - last_status >= STATUS_SECONDS:
                log(status(conn, trader))
                last_status = now
    finally:
        for feed in feeds:
            feed.cancel()
        trader.stop(time.time())


async def until_stopped(seconds, db_path=None):
    """
    Paper trade into the database at db_path, by default database.DB_PATH, stopping on a kill as on Ctrl-C.
    """
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await trade(db_path, seconds)


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paper trade Kalshi's and Polymarket US's Bitcoin 15 minute windows live.")
    ap.add_argument("--seconds", type=float, help="stop after this long, by default run until stopped")
    ap.add_argument("--db", help="the database to store the orders in and read the history from, by default data/history.sqlite")
    args = ap.parse_args()
    missing = [path for path in (kalshi.KEY_ID_FILE, kalshi.PRIVATE_KEY_FILE) if not path.exists()]
    if missing:
        raise SystemExit(f"The Kalshi API key is missing: {', '.join(str(p) for p in missing)}.")
    try:
        asyncio.run(until_stopped(args.seconds, args.db))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log("stopped")
