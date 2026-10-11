"""
Record Kalshi's Bitcoin 15 minute windows: their order books as they change, and how each one settled, and beside them
the best bid and ask of Coinbase, Kraken, Bitstamp, Gemini, Bullish, and Crypto.com, six of the seven exchanges BRTI is
built from, as they move.

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

It only listens and reads, nothing trades. The stream needs the Kalshi API
key in data/, see api/kalshi.py. Ctrl-C or kill stops it after writing what
is waiting.

Run from src/ with:
    python3 -m upordowntrader.record.run
    python3 -m upordowntrader.record.run --seconds 120
"""

import argparse
import asyncio
import signal
import time
from upordowntrader.api import bitstamp, bullish, coinbase, cryptocom, gemini, kalshi, kraken
from upordowntrader.common.log import log, on_failure
from upordowntrader.common.periodic import Periodic
from upordowntrader.db import database
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
# Each exchange whose best bid and ask are recorded, with its quote stream and its name for bitcoin in dollars. All are
# among the exchanges BRTI is built from, as CF Benchmarks listed them on 2026-08-24. LMAX Digital, the other, has no public
# feed to read without an account.
QUOTE_STREAMS = {"coinbase": (coinbase.CoinbaseQuoteStream, "BTC-USD"), "kraken": (kraken.KrakenQuoteStream, "BTC/USD"),
                 "bitstamp": (bitstamp.BitstampQuoteStream, "btcusd"), "gemini": (gemini.GeminiQuoteStream, "BTCUSD"),
                 "bullish": (bullish.BullishQuoteStream, "BTCUSD"), "cryptocom": (cryptocom.CryptoComQuoteStream, "BTC_USD")}


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
    streams = [stream, *(quote_stream(source, quotes, log) for source in QUOTE_STREAMS)]
    feeds = [asyncio.create_task(s.run()) for s in streams]
    for s, feed in zip(streams, feeds, strict=True):
        feed.add_done_callback(on_failure(log, f"the {s.name} stream"))
    refresher = Periodic(REFRESH_SECONDS, log, "reading the windows")
    settler = Periodic(SETTLE_SECONDS, log, "reading the settlements")
    reads = {}
    started = last_status = last_tick = time.time()
    window = None
    log(f"recording {SERIES}, {BOOK_LEVELS} levels a side, and the quotes of {', '.join(QUOTE_STREAMS)}")
    try:
        while seconds is None or time.time() - started < seconds:
            await asyncio.sleep(1)
            now = time.time()
            if woke(streams, last_tick, now, log):
                refresher.again()
            last_tick = now
            if int((now - OPEN_DELAY) // WINDOW_SECONDS) != window:      # A window opened, so read it and the next at once.
                window = int((now - OPEN_DELAY) // WINDOW_SECONDS)
                refresher.again()
            refresher.tick(now, lambda: refresh(conn, stream, recorder, reads))
            settler.tick(now, lambda: settle(conn, log))
            recorder.flush()
            quotes.flush()
            if now - last_status >= STATUS_SECONDS:
                log(recorder.status())
                log(quotes.status())
                last_status = now
    finally:
        for feed in feeds:
            feed.cancel()
        recorder.flush()
        quotes.flush()
        log(recorder.status())
        log(quotes.status())


async def until_stopped(seconds):
    """
    Record, stopping on a kill as on Ctrl-C, by cancelling, so what is waiting is written first either way.
    """
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    await record(database.connect(), seconds)


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Record Kalshi's Bitcoin 15 minute windows' order books into SQLite.")
    ap.add_argument("--seconds", type=float, help="stop after this long, by default run until stopped")
    args = ap.parse_args()
    missing = [path for path in (kalshi.KEY_ID_FILE, kalshi.PRIVATE_KEY_FILE) if not path.exists()]
    if missing:
        raise SystemExit(f"The Kalshi API key is missing: {', '.join(str(p) for p in missing)}. Copy SportsArb's key files there.")
    try:
        asyncio.run(until_stopped(args.seconds))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log("stopped")
