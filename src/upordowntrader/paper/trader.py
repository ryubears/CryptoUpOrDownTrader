"""
Paper trade the windows live: price Up as the replay does, from what reaches us as it comes, and decide pretend orders on
Kalshi and Polymarket US under a rule fixed before the first one, each filled against the book its venue had when the
order would have reached it.

The rule, set on 2026-10-13 from the benchmark and the replay, where only the blend kept an edge after the fee, and only
before the last 5 minutes, whose moves the market sees coming and the model does not:

    fair price  the model recalibrated and blended with Kalshi's mid, see model/fair.py, fitted on the TRAIN_DAYS of settled
                windows before, see benchmark.fit_before(), and fitted again daily by paper/run.py
    when        from the open until MIN_LEFT seconds before the close
    edge        THRESHOLD a contract or more after the venue's taker fee at the best price, on either side, buying Yes below
                the fair price or No above it, on Kalshi or Polymarket US, which trade the same bet
    episodes    once per venue and side until its edge falls below the threshold, as the replay takes them
    size        an immediate or cancel limit for up to CONTRACTS contracts at the worst price whose edge reaches the threshold

An order reaches its venue ORDER_SECONDS after it is decided, the 90th percentile of PredictionArbitrageTrader's live
orders from the same AWS region, and fills against the newest book the venue made by then: the trader waits for a book
the venue made later, which shows every change before has reached us, or FEED_SECONDS. A side sends no order while its
last waits so. A paper order leaves the venue's book as it was, so the contracts it took are kept off later orders until
the book shows fewer, as the replay does, see replay.still_resting().

The model reads, as each reaches us:
    price      the average of the newest mids of the exchanges BRTI is built from, see replay.EXCHANGES, those whose stream is
               up, moved onto BRTI's level at the window's open by the average gap between the two over the minute before it,
               or, for a window that opened before the trader started, the gap as the recorder's quotes show it, see
               recorded_offset()
    variance   fair.variance() of Coinbase's minutes, read again every minute by paper/run.py
    tracking   the stand-in's tracking variance from the windows settled before, see tracking_variance()
The replay times each exchange's quote by its own clock, a moment after it was made. Live, a quote counts when it arrives,
which is what a trader here has.
"""

import math
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from upordowntrader.api import polymarket_us
from upordowntrader.backtest import replay
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.model import fair, pricing

THRESHOLD = 0.02            # Edge after the fee, in dollars a contract, an order needs: the benchmark's best before the last 5 minutes.
CONTRACTS = 10              # The most contracts an order takes.
MIN_LEFT = 300              # Seconds before the close the last order may be decided.
DEPTH = 10                  # Levels a side the trader keeps of each book.
VENUES = ("kalshi", "polymarket_us")
FEE_RATES = {"kalshi": fair.FEE_RATE, "polymarket_us": polymarket_us.TAKER_FEE_RATE}
ORDER_SECONDS = {"kalshi": 0.016, "polymarket_us": 0.136}   # From an order's decision to its venue's book.
FEED_SECONDS = {"kalshi": 0.1, "polymarket_us": 0.3}        # The longest a venue's book takes to reach us, at the 90th percentile.
TAPE_SECONDS = 10           # Seconds of each book's changes kept, for the book an order meets.
READING_SECONDS = 1800      # Seconds of the price's readings kept, back past the open of the window trading.
CATCH_UP = 10               # The most seconds without a price a reading is filled back over. Longer, the process stalled.
WINDOW_KEEP = 60            # Seconds after its close a window is forgotten, with its books.


@dataclass
class Window:
    """
    A window the trader prices, from Kalshi's market.
    """
    market: Market              # Kalshi's market, its strike None until it opens.
    names: dict                 # Each venue's market of the window, by venue.
    tracking: float | None = None   # The tracking variance, see tracking_variance(), None until read.
    offset: float | None = None     # The price over BRTI's level at the open, None until known, NaN when it cannot be.


# BOOKS

def side_levels(book, side):
    """
    The levels an order on a side takes from, as [price to pay, contracts], best first: the asks for Yes, and the bids
    restated as No prices for No.
    """
    bids, asks = book
    return [[p, c] for p, c in asks] if side == "yes" else [[round(1 - p, 6), c] for p, c in bids]


def book_mid(book):
    """
    A book's mid, an empty side counted as a bid of 0 or an ask of 1, as replay.blend_with_book() counts it. None when
    both sides are empty.
    """
    bids, asks = book
    if not bids and not asks:
        return None
    return ((bids[0][0] if bids else 0.0) + (asks[0][0] if asks else 1.0)) / 2


def shown(levels, price, depth):
    """
    The contracts levels show at a price: 0 when the price lies among them but none is there, and None when it lies past
    the last of depth levels, where the book says nothing.
    """
    for p, c in levels:
        if round(p, 6) == price:
            return c
    return None if len(levels) >= depth and price > levels[-1][0] else 0.0


# TRACKING

def settlement_error(conn, market):
    """
    A settled window's tracking error, see replay.settlement_error(), from the recorder's quotes of the minutes before its
    open and its close alone.
    """
    open_feed, close_feed = (replay.quote_feed(conn, "exchanges", t - 60, t) for t in (market.open_time, market.close_time))
    return replay.settlement_error(open_feed, market, close_feed)


def recorded_offset(conn, market):
    """
    A window's offset, see Window, from the recorder's quotes of the minute before it opened, as the replay reads them, or
    NaN without any.
    """
    readings = market.open_time + np.arange(1, 61) - 60
    prices = replay.price_at(replay.quote_feed(conn, "exchanges", market.open_time - 60, market.open_time), readings)
    return float(np.nanmean(prices) - market.strike) if np.isfinite(prices).any() else math.nan


def tracking_variance(conn, market, errors):
    """
    A Kalshi window's tracking variance, see fair.tracking_variances(), from the stand-in's errors on the windows settled
    before it opened, read from the recorded quotes. errors holds each window's, by ticker, so it is read once.
    """
    settled = [m for m in database.load_markets(conn, "kalshi") if m.result and m.strike is not None and m.settlement is not None
               and m.close_time < market.open_time][-fair.TRACKING_WINDOWS:]
    for m in settled:
        if m.ticker not in errors:
            errors[m.ticker] = settlement_error(conn, m)
    return float(fair.tracking_variances([*(errors[m.ticker] for m in settled), np.nan], [*(m.open_time for m in settled), market.open_time],
                                         [*(m.close_time for m in settled), market.close_time])[-1])


# TRADER

class PaperTrader:
    """
    Prices the window trading from what the streams pass on, and decides, fills, and stores paper orders, see the module's
    docstring. paper/run.py feeds it the windows, the variance forecast, and the fit, and settles its orders.
    """

    def __init__(self, conn, log=print, clock=time.time, threshold=THRESHOLD, contracts=CONTRACTS, depth=DEPTH):
        self.conn = conn
        self.log = log
        self.clock = clock
        self.threshold = threshold
        self.contracts = contracts
        self.depth = depth
        self.feeds = {}             # The streams by name, a venue or an exchange, whose ws is None while down. One without counts as up.
        self.venues = VENUES        # The venues traded.
        self.mids = {}              # Each exchange's newest mid.
        self.readings = {}          # The price known as each whole second began, by the second.
        self.second = None          # The last whole second readings were stored to.
        self.windows = {}           # Windows by close time.
        self.books = {}             # (venue, ticker) maps to its newest (bids, asks).
        self.halted = set()         # The (venue, ticker) of markets their venue says are not trading.
        self.tapes = {}             # (venue, ticker) maps to its changes, (venue's time, bids, asks), the last TAPE_SECONDS.
        self.fit = None             # The fair.Fit, None until paper/run.py has made one.
        self.variance = None        # (forecast per second at the end of each minute, the first minute's start), see fair.variance().
        self.armed = {}             # (venue, ticker, side) maps to whether its next edge starts an episode.
        self.busy = {}              # (venue, ticker, side) maps to when its order in flight is filled.
        self.taken = {}             # (venue, ticker, side) maps to {price: contracts its orders took}.
        self.pending = []           # Orders decided and not filled yet.
        self.orders = 0             # Orders stored since the start.
        self.errors = {}            # Each settled Kalshi window's tracking error, by ticker, see tracking_variance().

    def up(self, name):
        """
        Whether a venue's or an exchange's stream is connected, or has none to say.
        """
        stream = self.feeds.get(name)
        return stream is None or stream.ws is not None

    # WHAT THE STREAMS PASS ON

    def stand_in(self):
        """
        The newest mids of the exchanges whose streams are up, averaged, or NaN without any.
        """
        mids = [mid for name, mid in self.mids.items() if self.up(name)]
        return sum(mids) / len(mids) if mids else math.nan

    def mark(self, now):
        """
        Store the price as each whole second since the last stored began, up to now: the reading a settlement would take
        then. Called before anything changes the price.
        """
        second = int(now)
        if self.second is not None and second > self.second:
            price = self.stand_in()
            for s in range(max(self.second + 1, second - CATCH_UP), second + 1):
                self.readings[s] = price
        self.second = second if self.second is None else max(self.second, second)

    def on_quote(self, source, bids, asks):
        """
        Note an exchange's best bid and ask, as [[price, size]] lists.
        """
        now = self.clock()
        self.mark(now)
        if bids and asks:
            self.mids[source] = (bids[0][0] + asks[0][0]) / 2
        else:
            self.mids.pop(source, None)
        self.evaluate(now)

    def on_book(self, venue, ticker, bids, asks, sent=None):
        """
        Note a venue's book of a market, its best levels a side, and when the venue made it, or None when it did not say. A
        book the same as the last, as Kalshi passes on after a change deeper than its levels, changes nothing.
        """
        now = self.clock()
        key = (venue, ticker)
        if self.books.get(key) == (bids, asks):
            return
        self.books[key] = (bids, asks)
        tape = self.tapes.setdefault(key, deque())
        tape.append((sent if sent is not None else now, bids, asks))
        while len(tape) > 1 and tape[1][0] <= now - TAPE_SECONDS:
            tape.popleft()
        self.lower_taken(venue, ticker, (bids, asks))
        self.evaluate(now)

    def on_gap(self, venue, tickers):
        """
        Forget a venue's books of markets whose stream was down, until it sends them again.
        """
        for ticker in tickers:
            self.books.pop((venue, ticker), None)
            self.tapes.pop((venue, ticker), None)

    def on_state(self, venue, ticker, why):
        """
        Note what a venue says of a market's trading: why it stopped, in a word, or None when it trades again.
        """
        if why is None:
            self.halted.discard((venue, ticker))
        else:
            self.halted.add((venue, ticker))

    def lower_taken(self, venue, ticker, book):
        """
        Lower what orders took at each price to what the newest book shows there, see replay.still_resting().
        """
        for side in ("yes", "no"):
            ours = self.taken.get((venue, ticker, side))
            if not ours:
                continue
            levels = side_levels(book, side)
            for price in list(ours):
                c = shown(levels, price, self.depth)
                if c is not None and c < ours[price]:
                    if c > 0:
                        ours[price] = c
                    else:
                        del ours[price]

    # WINDOWS

    def set_windows(self, markets):
        """
        Price Kalshi's windows, Markets, and the same windows on the other venues: new ones added, known ones given their
        strike once it is set.
        """
        for m in markets:
            w = self.windows.get(m.close_time)
            if w is None:
                names = {"kalshi": m.ticker, "polymarket_us": polymarket_us.slug(m.open_time)}
                self.windows[m.close_time] = Window(m, {venue: names[venue] for venue in self.venues})
            elif m.strike is not None:
                w.market = m

    def opened(self, now):
        """
        The window trading at now, or None.
        """
        return next((w for w in self.windows.values() if w.market.open_time <= now < w.market.close_time), None)

    def trading(self, now):
        """
        The window trading at now with its strike, tracking variance, and offset known, or None. The offset is known a
        second after the open, from the readings of the minute before, see the module's docstring. A window without any,
        opened before the trader started, waits for paper/run.py to read it from the recorder's quotes.
        """
        w = self.opened(now)
        if w is None or w.market.strike is None or w.tracking is None:
            return None
        if w.offset is None and now >= w.market.open_time + 1:
            known = [r for r in (self.readings.get(s, math.nan) for s in range(w.market.open_time - 59, w.market.open_time + 1)) if math.isfinite(r)]
            if known:
                w.offset = sum(known) / len(known) - w.market.strike
        return w if w.offset is not None and math.isfinite(w.offset) else None

    def model_chance(self, w, price, now):
        """
        The model's chance of Up at now, recalibrated, with the window's price before its averaging minute, or NaN without
        a variance forecast for the minute.
        """
        forecast, start = self.variance
        i = (int(now) - start) // 60 - 1
        if not 0 <= i < len(forecast) or not math.isfinite(price):
            return math.nan
        left = w.market.close_time - now
        z = pricing.z_scores(price, 0.0, w.market.strike, left, max(forecast[i], fair.TINY), tracking=w.tracking)
        return float(fair.chance(z, left, self.fit))

    # ORDERS

    def evaluate(self, now):
        """
        Fill the orders whose books are known, then price the window trading and decide the orders its edges call for.
        """
        self.fill_due(now)
        w = self.trading(now)
        if w is None or self.fit is None or self.variance is None or w.market.close_time - now <= MIN_LEFT:
            return
        kalshi = self.books.get(("kalshi", w.names["kalshi"]))
        mid = book_mid(kalshi) if kalshi is not None and self.up("kalshi") else None
        if mid is None:
            return
        price = self.stand_in() - w.offset
        model = self.model_chance(w, price, now)
        if not math.isfinite(model):
            return
        fair_up = float(fair.blended(mid, model, w.market.close_time - now, self.fit))
        seen = {"price": price, "model": model, "mid": mid, "fair": fair_up}
        for venue, ticker in w.names.items():
            book = self.books.get((venue, ticker))
            if book is not None and self.up(venue) and (venue, ticker) not in self.halted:
                for side in ("yes", "no"):
                    self.consider(now, w, venue, ticker, book, side, seen)

    def consider(self, now, w, venue, ticker, book, side, seen):
        """
        Decide an order on one side of a venue's market when its edge calls for one, see the module's docstring.
        """
        key = (venue, ticker, side)
        chance = seen["fair"] if side == "yes" else 1 - seen["fair"]
        rate = FEE_RATES[venue]
        levels = side_levels(book, side)
        if not levels or fair.edge(chance, levels[0][0], rate) < self.threshold:
            self.armed[key] = True          # The next reach is a new episode.
            return
        if not self.armed.get(key, True) or now < self.busy.get(key, -math.inf):
            return
        offered = replay.untaken([(p, c) for p, c in levels if fair.edge(chance, p, rate) >= self.threshold], self.taken.get(key, {}))
        if not offered:
            return
        self.armed[key] = False
        arrived = now + ORDER_SECONDS[venue]
        self.busy[key] = arrived + FEED_SECONDS[venue]
        self.pending.append({"venue": venue, "ticker": ticker, "close_time": w.market.close_time, "decided": now, "side": side, **seen,
                             "edge": fair.edge(chance, levels[0][0], rate), "limit_price": max(p for p, _ in offered),
                             "offered": sum(c for _, c in offered), "arrived": arrived, "chance": chance})

    def fill_due(self, now):
        """
        Fill each order whose venue's book at its arrival is known: a later book has come, or FEED_SECONDS have passed.
        """
        for order in [o for o in self.pending if self.known(o, now)]:
            self.pending.remove(order)
            self.fill(order)

    def known(self, order, now):
        """
        Whether the book an order met is known, see fill_due().
        """
        tape = self.tapes.get((order["venue"], order["ticker"]))
        return now >= order["arrived"] + FEED_SECONDS[order["venue"]] or bool(tape and tape[-1][0] > order["arrived"])

    def fill(self, order):
        """
        Fill an order against the newest book its venue made by its arrival, at its limit or better, less what earlier orders
        took, up to the order's size, then store and log it.
        """
        venue, side, chance = order["venue"], order["side"], order.pop("chance")
        key = (venue, order["ticker"], side)
        before = [change for change in self.tapes.get(key[:2], ()) if change[0] <= order["arrived"]]
        levels = side_levels(before[-1][1:], side) if before else []
        ours = self.taken.setdefault(key, {})
        filled, wanted = [], self.contracts
        for p, c in replay.untaken([(p, c) for p, c in levels if p <= order["limit_price"] + 1e-9], ours):
            if wanted <= 0:
                break
            filled.append((p, min(c, wanted)))
            wanted -= filled[-1][1]
        for p, c in filled:
            ours[round(p, 6)] = ours.get(round(p, 6), 0.0) + c
        contracts = sum(c for _, c in filled)
        rate = FEE_RATES[venue]
        row = {**order, "contracts": contracts, "cost": sum(p * c for p, c in filled) / contracts if contracts else None,
               "best": filled[0][0] if filled else None, "fees": sum(c * fair.fee(p, rate) for p, c in filled),
               "expected": sum(c * fair.edge(chance, p, rate) for p, c in filled)}
        database.store_paper_order(self.conn, row)
        self.orders += 1
        took = f"filled {contracts:g} at {row['cost']:.3f}" if contracts else "filled none"
        self.log(f"paper {venue} {side} {order['ticker']} with {(order['close_time'] - order['decided']) / 60:.1f} min left: edge "
                 f"{100 * order['edge']:.1f}c, fair {order['fair']:.1%} (model {order['model']:.1%}, Kalshi mid {order['mid']:.1%}), "
                 f"{order['offered']:g} offered to {order['limit_price']:.2f}, {took}")

    # TIME

    def tick(self, now):
        """
        Once a second: store the readings, fill and decide as time alone moves the price, and forget what is past.
        """
        self.mark(now)
        self.evaluate(now)
        for s in [s for s in self.readings if s < now - READING_SECONDS]:
            del self.readings[s]
        for close in [c for c in self.windows if c < now - WINDOW_KEEP]:
            markets = set(self.windows.pop(close).names.items())
            for held in (self.books, self.tapes, self.taken, self.armed, self.busy):
                for key in [k for k in held if k[:2] in markets]:
                    del held[key]
            self.halted -= markets
