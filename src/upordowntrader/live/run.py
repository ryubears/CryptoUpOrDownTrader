"""
Trade Hyperliquid's Bitcoin 15 minute windows with real money, by the rule its paper orders were decided by from
2026-10-16 16:15 UTC, see paper/trader.py, in the last 5 minutes of each window alone, where they made money: Kalshi's
mid as the fair price, both its sides quoted within 10 cents; an order on a change of Hyperliquid's own book while the
edge after its fee is 2 cents or more, once an episode; an immediate or cancel order of up to 50 contracts at the worst
price within the edge, and no more than 250 contracts of a window, both sides together; from 300 to 60 seconds before
the close; and none worth under Hyperliquid's $10 at its limit.

The process streams Kalshi's books of the window trading and the next, for their mid alone, and Hyperliquid's of the same
windows as it lists them, on streams of its own, beside the recorder and the paper trader, so it can stop or change
without them. It reads the windows as paper/run.py does, without the model, sends each order from a thread of its own,
see Sender and live/executor.py, reads the account's USDC every BALANCE_SECONDS, and every SETTLE_SECONDS settles the
trades of the windows Hyperliquid has settled, with their opportunities, then sees whether live trading should halt,
see live/brakes.py, which emails a human when it does. A status line comes every STATUS_SECONDS.

It needs the account's address and an approved API wallet's key in data/, see api/hyperliquid.py, Kalshi's API key for
its books, and USDC on the account's spot side. Without them, or with a key file that holds no key, it exits at once
with EXIT_MISSING, which its service does not restart on, see startup_problems(). With --dry-run nothing is sent: each
order is logged and answered as filling nothing, and nothing is stored, neither trades nor opportunities.

Run from src/ with:
    python3 -m upordowntrader.live.run --dry-run --seconds 900
    python3 -m upordowntrader.live.run
"""

import argparse
import asyncio
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from upordowntrader.api import hyperliquid, kalshi
from upordowntrader.common.log import log, on_failure
from upordowntrader.common.periodic import Periodic
from upordowntrader.db import database
from upordowntrader.live import notify
from upordowntrader.live.brakes import HALT_FILE, LOSING_WINDOWS, LOSS_HOURS, LOSS_SHARE, Brakes
from upordowntrader.live.executor import PAYOUT_SECONDS, Executor, Placed
from upordowntrader.paper import run as paper_run
from upordowntrader.paper.trader import LIVE_MAX_LEFT, MIN_LEFT, PaperTrader
from upordowntrader.record.run import code_version

VENUE = "hyperliquid"
WINDOW_SECONDS = paper_run.WINDOW_SECONDS
OPEN_DELAY = paper_run.OPEN_DELAY
REFRESH_SECONDS = paper_run.REFRESH_SECONDS
STRIKE_SECONDS = paper_run.STRIKE_SECONDS
LISTING_SECONDS = paper_run.LISTING_SECONDS
BALANCE_SECONDS = 60
SETTLE_SECONDS = 30
STATUS_SECONDS = 300
FILLS_SECONDS = 3600        # How far back the orders filled are timed by the account's fills, see settle().
EXIT_MISSING = 2            # The exit status without the Hyperliquid files, which the service does not restart on.


class Sender:
    """
    Sends the trader's live orders from a thread of its own, one at a time, through an Executor on a connection of its
    own, and passes each answer back to the trader and the brakes in the event loop. An order whose outcome stays unknown
    counts at its full size against the window's cap. dry_run sends nothing.
    """

    def __init__(self, db_path, trader, brakes, loop, log=log, dry_run=False, place=None):
        self.db_path = db_path
        self.trader = trader
        self.brakes = brakes
        self.loop = loop
        self.log = log
        self.dry_run = dry_run
        self.place = place
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-orders")
        self.executor = None        # Made in the thread, on its first order, so its connection is the thread's.

    def send(self, order):
        """
        Send an order the trader decided, see trader.consider(), in the thread.
        """
        decision = {"ticker": order["ticker"], "close_time": order["close_time"], "side": order["side"], "signal_ts": order["decided"],
                    "fair": order["chance"], "edge": order["edge"], "quantity": order["size"], "limit_price": order["limit_price"],
                    "cause": order["cause"], "book_at": order["book_at"], "book_ts": order["decided"]}
        future = self.pool.submit(self.work, decision)
        future.add_done_callback(lambda f: self.loop.call_soon_threadsafe(self.done, order, f))

    def work(self, decision):
        """
        In the thread: send the order and store it, or, on a dry run, log it.
        """
        if self.dry_run:
            self.log(f"live {VENUE} {decision['side']} {decision['ticker']} with {(decision['close_time'] - decision['signal_ts']) / 60:.1f} min "
                     f"left, dry run, not sent: {decision['quantity']:g} to {decision['limit_price']:.4f}, edge {100 * decision['edge']:.1f}c")
            return Placed(None, "unfilled", 0.0, None, "dry run, not sent")
        if self.executor is None:
            self.executor = Executor(database.connect(self.db_path), self.place, self.log)
        return self.executor.trade(decision)

    def done(self, order, future):
        """
        In the loop: pass an order's answer on.
        """
        try:
            placed = future.result()
        except Exception as e:      # Before its answer was stored, so what it did is unknown.
            placed = Placed(None, "error", 0.0, None, f"the order's thread failed, {type(e).__name__}: {e}")
        self.trader.on_answer(order, order["size"] if placed.status == "error" else placed.filled)
        self.brakes.watch(placed)

    def stop(self):
        """
        Wait for the order in flight, if any, to be answered and stored.
        """
        self.pool.shutdown(wait=True)


def read_balance():
    """
    The account's USDC on its spot side.
    """
    return hyperliquid.usdc(hyperliquid.balances(hyperliquid.ACCOUNT_FILE.read_text().strip()))


def read_fills(since):
    """
    The account's fills from since, in seconds since 1970.
    """
    return hyperliquid.user_fills(hyperliquid.ACCOUNT_FILE.read_text().strip(), since)


async def refresh_balance(brakes):
    brakes.balance = await asyncio.to_thread(read_balance)


async def settle(conn, executor, brakes, dry_run=False):
    """
    Settle the live trades and opportunities of the windows closed that Hyperliquid has settled, then, when trades that
    held contracts settled, read the balance with their payouts and see whether live trading should halt, and read the
    account's fills for when Hyperliquid filled each order, see executor.time_fills(), off the orders' own path.
    """
    due = time.time() - PAYOUT_SECONDS
    trades = database.unsettled_trades(conn, now=due)
    markets = {ticker for venue, ticker in database.unsettled_markets(conn, due) if venue == VENUE} | {t["ticker"] for t in trades}
    if not markets:
        return
    settled = [m for m in await asyncio.to_thread(hyperliquid.settlements) if m.ticker in markets and m.result in ("yes", "no")]
    for m in settled:
        database.settle_opportunities(conn, m.venue, m.ticker, m.result, hyperliquid.SETTLE_FEE_RATE)
    if executor.settle(settled) and not dry_run:
        await refresh_balance(brakes)
        brakes.check_results()
        since = time.time() - FILLS_SECONDS
        if database.untimed_orders(conn, since):
            executor.time_fills(await asyncio.to_thread(read_fills, since - 60), since)


def status(conn, trader, brakes):
    """
    One line on the window trading, what the live orders did since midnight UTC, the balance, and whether trading.
    """
    now = time.time()
    midnight = now // 86400 * 86400
    orders, filled, contracts = conn.execute("SELECT COUNT(*), COALESCE(SUM(filled > 0), 0), COALESCE(SUM(filled), 0) FROM trades "
                                             "WHERE mode = 'live' AND signal_ts >= ?", (midnight,)).fetchone()
    profit = conn.execute("SELECT COALESCE(SUM(s.profit), 0) FROM settlements s JOIN trades t ON t.id = s.trade_id "
                          "WHERE t.mode = 'live' AND t.signal_ts >= ?", (midnight,)).fetchone()[0]
    w = trader.opened(now)
    line = "live: no window"
    if w is not None:
        line = f"live: {w.names.get(VENUE, 'not listed yet')} of {w.market.ticker}, {(w.market.close_time - now) / 60:.1f} min left"
    balance = "balance unknown" if brakes.balance is None else f"{brakes.balance:,.2f} USDC"
    state = f"HALTED, {brakes.halted}" if brakes.halted else "trading"
    return f"{line}; today {orders} orders, {filled} filled, {contracts:g} contracts, {profit:+.2f}$ settled; {balance}; {state}"


# MAIN LOOP

async def trade(db_path=None, seconds=None, dry_run=False, log=log):
    """
    Trade live until seconds have passed, or for good without.
    """
    conn = database.connect(db_path)
    trader = PaperTrader(conn, log)
    trader.traded, trader.live = {VENUE}, {VENUE}
    brakes = Brakes(conn, log, notify.Notifier(conn, log))
    trader.live_halted = lambda: brakes.halted
    streams = {"kalshi": kalshi.KalshiBookStream([], lambda t, b, a, s: trader.on_book("kalshi", t, b, a, s),
                                                 lambda start, end, ids: trader.on_gap("kalshi", ids), log),
               VENUE: hyperliquid.HyperliquidBookStream(list(hyperliquid.UNDERLYINGS), lambda t, b, a, s: trader.on_book(VENUE, t, b, a, s),
                                                        lambda start, end, ids: trader.on_gap(VENUE, ids), log)}
    trader.venues = tuple(streams)
    for venue, stream in streams.items():
        stream.depth = trader.depth
        stream.on_state = lambda ticker, why, venue=venue: trader.on_state(venue, ticker, why)
    trader.feeds = streams
    sender = Sender(db_path, trader, brakes, asyncio.get_running_loop(), log, dry_run)
    trader.send = sender.send
    if dry_run:
        trader.store_opportunities = trader.finished.clear     # A dry run stores nothing.
    feeds = [asyncio.create_task(s.run()) for s in streams.values()]
    for s, feed in zip(streams.values(), feeds, strict=True):
        feed.add_done_callback(on_failure(log, f"the live {s.name} stream"))
    executor = Executor(conn, log=log)
    refresher = Periodic(REFRESH_SECONDS, log, "live: reading the windows")
    balance = Periodic(BALANCE_SECONDS, log, "live: reading the balance")
    settler = Periodic(SETTLE_SECONDS, log, "live: settling")
    started = last_status = time.time()
    window = None
    loss = f"a {LOSS_SHARE:.0%} loss over {LOSS_HOURS} hours, " if LOSS_SHARE else ""
    log(f"live trading {VENUE}{', a dry run, nothing sent' if dry_run else ''}: {100 * trader.thresholds[VENUE]:.0f}c edges against Kalshi's "
        f"mid, from {LIVE_MAX_LEFT[VENUE]} to {MIN_LEFT[VENUE]} seconds before the close, up to {trader.contracts} contracts an order and "
        f"{trader.window_contracts} a window; halting at {loss}{LOSING_WINDOWS} losing windows in a row, an order of unknown outcome, or "
        f"refusals, and while {HALT_FILE} is there; code {code_version()}")
    try:
        while seconds is None or time.time() - started < seconds:
            await asyncio.sleep(1)
            now = time.time()
            trader.tick(now)
            brakes.refresh()
            if int((now - OPEN_DELAY) // WINDOW_SECONDS) != window:      # A window opened, so read it and the next at once.
                window = int((now - OPEN_DELAY) // WINDOW_SECONDS)
                refresher.again()
            w = trader.opened(now)
            unlisted = w is not None and now - w.market.open_time < LISTING_SECONDS and VENUE not in w.names
            if unlisted and now - (refresher.last or 0) >= STRIKE_SECONDS:
                refresher.again()
            refresher.tick(now, lambda: paper_run.refresh(trader, streams, db_path))
            if hyperliquid.ACCOUNT_FILE.exists():
                balance.tick(now, lambda: refresh_balance(brakes))
            settler.tick(now, lambda: settle(conn, executor, brakes, dry_run))
            if now - last_status >= STATUS_SECONDS:
                log(status(conn, trader, brakes))
                last_status = now
    finally:
        for feed in feeds:
            feed.cancel()
        sender.stop()
        trader.stop(time.time())


async def until_stopped(seconds, db_path=None, dry_run=False):
    """
    Trade live into the database at db_path, by default database.DB_PATH, stopping on a kill as on Ctrl-C.
    """
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await trade(db_path, seconds, dry_run)


def startup_problems(dry_run=False):
    """
    What keeps live trading from starting, in words, none when it can: Kalshi's key files, and unless a dry run, the
    Hyperliquid account's address and an API wallet key that is one, see hyperliquid.key_problem(). Without this a bad
    key would fail only at the first order, which would halt live trading as an order of unknown outcome.
    """
    problems = [f"no {path}" for path in (kalshi.KEY_ID_FILE, kalshi.PRIVATE_KEY_FILE) if not path.exists()]
    if not dry_run:
        if not hyperliquid.ACCOUNT_FILE.exists():
            problems.append(f"no {hyperliquid.ACCOUNT_FILE}")
        problem = hyperliquid.key_problem()
        if problem:
            problems.append(problem)
    return problems


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trade Hyperliquid's Bitcoin 15 minute windows with real money, in their last 5 minutes.")
    ap.add_argument("--seconds", type=float, help="stop after this long, by default run until stopped")
    ap.add_argument("--db", help="the database to store the trades in, by default data/history.sqlite")
    ap.add_argument("--dry-run", action="store_true", help="send nothing: log each order instead, and store nothing")
    args = ap.parse_args()
    problems = startup_problems(args.dry_run)
    if problems:
        log(f"live trading cannot start: {'; '.join(problems)}")
        sys.exit(EXIT_MISSING)
    try:
        asyncio.run(until_stopped(args.seconds, args.db, args.dry_run))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log("stopped")
