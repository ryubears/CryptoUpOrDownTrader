"""
Record Kalshi's Bitcoin 15 minute windows: their order books as they change, and how each one settled, and beside them
the best bid and ask of Coinbase, Kraken, Bitstamp, Gemini, Bullish, and Crypto.com, six of the seven exchanges BRTI is
built from, and of Hyperliquid's BTC perp, as they move. And the same of Hyperliquid's up or down windows on BTC, with
the perp's mark, which they settle on.

A window's market is listed a day ahead and trades only during its 15
minutes. Just after each quarter hour, and every REFRESH_SECONDS besides,
the process reads the series' markets closing from a window before its
last read to LOOKAHEAD_SECONDS ahead, stores them, and streams the books of
those not yet closed: the window trading now, and the next, so its first
book is not missed. A closed window's stream is dropped. Every
SETTLE_SECONDS it reads how the closed windows went: Kalshi's result, and
the exact BRTI averages, the strike set as each window opened and the
settlement as it closed, which the calibration's stand-ins can be checked
against. The quote streams, see QUOTE_STREAMS, run all the while, and
their quotes are stored by record/quotes.py. A status line comes every
STATUS_SECONDS.

While this machine sleeps the process is paused with every other, and its
connection may die without saying so. The once a second tick notices on
waking, when more than SLEEP_SECONDS have passed since the last: the
stretch is stored as a gap for every stream, each connects again at once,
and the windows are read again, back to before the sleep, so even those that
opened and closed while it slept are stored with their strike and
settlement. Their books are lost.

Hyperliquid's windows, see api/hyperliquid.py, are read every REFRESH_SECONDS and HYPERLIQUID_DELAY after each quarter
hour, once the 15 minute series has listed its next, and their books recorded as Kalshi's are, under the venue
'hyperliquid', on a connection that also carries the perps' marks, stored by record/marks.py. Every SETTLE_SECONDS the
closed ones are settled: the 15 minute series' as its deployer settled them, read from Hyperliquid's explorer, and the
daily series' from the marks recorded around the close, see settle_hyperliquid().

It only listens and reads, nothing trades. The stream needs the Kalshi API
key in data/, see api/kalshi.py. Ctrl-C or kill stops it after writing what
is waiting.

Run from src/ with:
    python3 -m upordowntrader.record.run
    python3 -m upordowntrader.record.run --seconds 120 --db /tmp/check.sqlite
"""

import argparse
import asyncio
import signal
import subprocess
import time
from dataclasses import replace
from upordowntrader.api import binance, bitstamp, bullish, bybit, coinbase, cryptocom, deribit, gemini, hyperliquid, kalshi, kraken, okx
from upordowntrader.common.log import log, on_failure
from upordowntrader.common.paths import ROOT
from upordowntrader.common.periodic import Periodic
from upordowntrader.db import database
from upordowntrader.record.marks import MarkRecorder
from upordowntrader.record.quotes import QuoteRecorder
from upordowntrader.record.recorder import BOOK_LEVELS, Recorder

SERIES = "KXBTC15M"         # Kalshi's Bitcoin 15 minute up or down windows.
WINDOW_SECONDS = 900
OPEN_DELAY = 2              # Seconds after a window opens before it is read again, so Kalshi has set its strike.
REFRESH_SECONDS = 60
LOOKAHEAD_SECONDS = 1800    # How far ahead the windows read reach: past the one trading now to the next.
SETTLE_SECONDS = 30         # Kalshi settles a window a few seconds after it closes.
STATUS_SECONDS = 60
SLEEP_SECONDS = 10          # A tick this long after the last means the machine slept, or the process stalled.
HYPERLIQUID_DELAY = 30      # Seconds after a quarter hour before Hyperliquid's windows are read: its 15 minute series lists
                            # the next window 15 to 25 seconds after.
SETTLE_REACH = 6 * 3600     # Seconds after its close a Hyperliquid window is still looked for in the explorer, which keeps
                            # about 12 hours of the 15 minute series' settlements.
MARK_REACH = 30             # Seconds around a daily window's close its settlement reads marks from.
# Each exchange whose best bid and ask are recorded, with its quote stream and its name for bitcoin in dollars. The first
# six are among the exchanges BRTI is built from, as CF Benchmarks listed them on 2026-08-24; LMAX Digital, the other, has
# no public feed to read without an account. The rest are bitcoin perpetuals, where price moves may start, recorded to
# see which leads, see backtest/lead.py: Hyperliquid's, the busiest four elsewhere that answer the AWS instance, three in
# tether, and Coinbase Derivatives' US one.
QUOTE_STREAMS = {"coinbase": (coinbase.CoinbaseQuoteStream, "BTC-USD"), "kraken": (kraken.KrakenQuoteStream, "BTC/USD"),
                 "bitstamp": (bitstamp.BitstampQuoteStream, "btcusd"), "gemini": (gemini.GeminiQuoteStream, "BTCUSD"),
                 "bullish": (bullish.BullishQuoteStream, "BTCUSD"), "cryptocom": (cryptocom.CryptoComQuoteStream, "BTC_USD"),
                 "hyperliquid": (hyperliquid.HyperliquidQuoteStream, "BTC"), "binance perp": (binance.BinanceQuoteStream, "BTCUSDT"),
                 "bybit perp": (bybit.BybitQuoteStream, "BTCUSDT"), "okx perp": (okx.OkxQuoteStream, "BTC-USDT-SWAP"),
                 "deribit perp": (deribit.DeribitQuoteStream, "BTC-PERPETUAL"),
                 "coinbase perp": (coinbase.CoinbasePerpQuoteStream, "BIP-20DEC30-CDE")}


async def refresh(conn, stream, recorder, reads):
    """
    Store the windows closing from a window before the last read, or before now on the first, to LOOKAHEAD_SECONDS
    ahead, and stream those not yet closed. The books of those closed are dropped from the stream and the recorder both.
    reads holds when the last read was, under "at", so a read after a sleep reaches back to before it.
    """
    now = time.time()
    found = await asyncio.to_thread(kalshi.windows, SERIES, reads.get("at", now) - WINDOW_SECONDS, now + LOOKAHEAD_SECONDS)
    database.upsert_markets(conn, found)
    trading = {m.ticker for m in found if m.close_time > now}
    closed = stream.wanted - trading
    stream.add(trading)
    stream.remove(closed)
    recorder.forget(closed)
    reads["at"] = now


async def settle(conn, log):
    """
    Read how the closed windows not settled yet went, store it, and log each one settled.
    """
    tickers = database.unsettled(conn, "kalshi", time.time())
    if not tickers:
        return
    database.upsert_markets(conn, await asyncio.to_thread(kalshi.markets, tickers))
    for m in database.load_markets(conn, "kalshi"):     # As stored, so with the strike read as the window opened.
        if m.ticker in tickers and m.result and m.settlement is not None and m.strike is not None:
            log(f"kalshi {m.ticker} settled {m.result}, {m.settlement:,.2f} against a strike of {m.strike:,.2f}")


async def refresh_hyperliquid(conn, stream, recorder):
    """
    Store Hyperliquid's up or down windows listed now, and stream the books of those not yet closed. The books of those
    closed, or no longer listed, are dropped from the stream and the recorder both. The perps the stream carries for
    their marks stay.
    """
    now = time.time()
    found = await asyncio.to_thread(hyperliquid.windows)
    database.upsert_markets(conn, found)
    trading = {m.ticker for m in found if m.close_time > now}
    closed = {c for c in stream.wanted if c.startswith("#")} - trading
    stream.add(trading)
    stream.remove(closed)
    recorder.forget(closed)


def interpolated_mark(conn, coin, t):
    """
    A perp's mark at t, in seconds since 1970, interpolated between the marks in force just before and after, as
    Hyperliquid settles its daily windows, or None without a mark recorded within MARK_REACH on each side. Our clock
    times the marks, so a settlement this close to the strike may have gone the other way.
    """
    rows = database.read_marks(conn, "hyperliquid", coin, t - MARK_REACH, t + MARK_REACH)
    before, after = rows[rows[:, 0] <= t], rows[rows[:, 0] > t]
    if not len(before) or not len(after):
        return None
    (t0, m0), (t1, m1) = before[-1, :2], after[0, :2]
    return float(m0 + (m1 - m0) * (t - t0) / (t1 - t0))


async def settle_hyperliquid(conn, log):
    """
    Settle Hyperliquid's windows closed by now: the 15 minute series' as its deployer did, read from the explorer with
    every other window it settled lately, so those never seen listed are stored too, and the daily series' by the mark
    interpolated to the close, see interpolated_mark(). A window closed over SETTLE_REACH ago is left as it is, and so is
    one on a perp no longer among hyperliquid.UNDERLYINGS, whose marks are no longer recorded.
    """
    now = time.time()
    closed = [m for m in database.load_markets(conn, "hyperliquid") if m.result is None and now - SETTLE_REACH < m.close_time <= now
              and m.series.split("-")[1] in hyperliquid.UNDERLYINGS]
    official = {}
    if any(m.series.startswith(hyperliquid.FIFTEEN_DEPLOYER) for m in closed):
        official = {m.ticker: m for m in await asyncio.to_thread(hyperliquid.settlements)}
    done = []
    for m in closed:
        if m.ticker in official:
            done.append(replace(m, result=official[m.ticker].result, settled_at=official[m.ticker].settled_at))
        elif m.series.startswith("recurring") and m.strike is not None:
            price = interpolated_mark(conn, m.series.split("-")[1], m.close_time)
            if price is not None:
                done.append(replace(m, settlement=price, result="yes" if price >= m.strike else "no", settled_at=now))
    database.upsert_markets(conn, [*official.values(), *done])
    for m in done:
        worked_out = f", the mark {m.settlement:,.2f}" if m.settlement is not None else ""
        log(f"hyperliquid {m.series} {m.ticker} settled {m.result} against a strike of {m.strike:,.6g}{worked_out}")


def code_version():
    """
    The commit the process runs, with '-dirty' when files differ from it, or 'unknown' outside a git checkout, so the log
    says what code recorded what follows it.
    """
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def woke(streams, last_tick, now, log):
    """
    Whether the machine slept, or the process stalled, between two ticks. If so, store the stretch as a gap for each
    stream, through its own on_gap, which drops what it carried, and have each connect again at once.
    """
    if now - last_tick <= SLEEP_SECONDS:
        return False
    log(f"no tick for {now - last_tick:.0f}s, this machine slept or the process stalled, storing a gap and reconnecting")
    for stream in streams:
        stream.on_gap(last_tick, now, sorted(stream.wanted))
        stream.reconnect()
    return True


def quote_stream(source, quotes, log):
    """
    The quote stream of one of QUOTE_STREAMS, passing its quotes and gaps to the quote recorder under the source's name.
    """
    stream_class, product = QUOTE_STREAMS[source]
    return stream_class([product], lambda product, bids, asks, sent: quotes.on_quote(source, bids, asks, sent),
                        lambda start, end, products: quotes.on_gap(source, start, end), log)


async def record(conn, seconds=None, log=log):
    """
    Record until seconds have passed, or for good without, then write what is waiting.
    """
    recorder = Recorder(conn, "kalshi", log)
    quotes = QuoteRecorder(conn, log)
    stream = kalshi.KalshiBookStream([], recorder.on_book, recorder.on_gap, log)
    stream.depth = BOOK_LEVELS
    stream.on_state = recorder.on_state
    hl_recorder = Recorder(conn, "hyperliquid", log)
    marks = MarkRecorder(conn, "hyperliquid", log)
    hl_stream = hyperliquid.HyperliquidBookStream(hyperliquid.UNDERLYINGS, hl_recorder.on_book, hl_recorder.on_gap, log)
    hl_stream.depth = BOOK_LEVELS
    hl_stream.on_mark = marks.on_mark
    streams = [stream, hl_stream, *(quote_stream(source, quotes, log) for source in QUOTE_STREAMS)]
    feeds = [asyncio.create_task(s.run()) for s in streams]
    for s, feed in zip(streams, feeds, strict=True):
        feed.add_done_callback(on_failure(log, f"the {s.name} stream"))
    refresher = Periodic(REFRESH_SECONDS, log, "reading the windows")
    settler = Periodic(SETTLE_SECONDS, log, "reading the settlements")
    hl_refresher = Periodic(REFRESH_SECONDS, log, "reading Hyperliquid's windows")
    hl_settler = Periodic(SETTLE_SECONDS, log, "reading Hyperliquid's settlements")
    reads = {}
    started = last_status = last_tick = time.time()
    window = hl_window = None
    log(f"recording {SERIES}, {BOOK_LEVELS} levels a side, the quotes of {', '.join(QUOTE_STREAMS)}, and Hyperliquid's up or down "
        f"windows on {', '.join(hyperliquid.UNDERLYINGS)}, with code {code_version()}")
    try:
        while seconds is None or time.time() - started < seconds:
            await asyncio.sleep(1)
            now = time.time()
            if woke(streams, last_tick, now, log):
                refresher.again()
                hl_refresher.again()
            last_tick = now
            if int((now - OPEN_DELAY) // WINDOW_SECONDS) != window:      # A window opened, so read it and the next at once.
                window = int((now - OPEN_DELAY) // WINDOW_SECONDS)
                refresher.again()
            if int((now - HYPERLIQUID_DELAY) // WINDOW_SECONDS) != hl_window:     # Hyperliquid has listed its next window.
                hl_window = int((now - HYPERLIQUID_DELAY) // WINDOW_SECONDS)
                hl_refresher.again()
            refresher.tick(now, lambda: refresh(conn, stream, recorder, reads))
            settler.tick(now, lambda: settle(conn, log))
            hl_refresher.tick(now, lambda: refresh_hyperliquid(conn, hl_stream, hl_recorder))
            hl_settler.tick(now, lambda: settle_hyperliquid(conn, log))
            for writer in (recorder, quotes, hl_recorder, marks):
                writer.flush()
            if now - last_status >= STATUS_SECONDS:
                for writer in (recorder, quotes, hl_recorder, marks):
                    log(writer.status())
                last_status = now
    finally:
        for feed in feeds:
            feed.cancel()
        for writer in (recorder, quotes, hl_recorder, marks):
            writer.flush()
            log(writer.status())


async def until_stopped(seconds, db_path=None):
    """
    Record into the database at db_path, by default database.DB_PATH, stopping on a kill as on Ctrl-C, by cancelling, so
    what is waiting is written first either way.
    """
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await record(database.connect(db_path), seconds)


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Record Kalshi's Bitcoin 15 minute windows' order books into SQLite.")
    ap.add_argument("--seconds", type=float, help="stop after this long, by default run until stopped")
    ap.add_argument("--db", help="the database to record into, by default data/history.sqlite, another for a trial run")
    args = ap.parse_args()
    missing = [path for path in (kalshi.KEY_ID_FILE, kalshi.PRIVATE_KEY_FILE) if not path.exists()]
    if missing:
        raise SystemExit(f"The Kalshi API key is missing: {', '.join(str(p) for p in missing)}. Copy SportsArb's key files there.")
    try:
        asyncio.run(until_stopped(args.seconds, args.db))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log("stopped")
