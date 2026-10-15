"""
Paper trade the windows live: price Up as the replay does, from what reaches us as it comes, and decide pretend orders on
Kalshi, Polymarket US, and Hyperliquid under rules fixed before the first one, each filled against the book its venue had
when the order would have reached it.

The rule, set on 2026-10-13 from the benchmark and the replay, where only the blend kept an edge after the fee, and only
before the last 5 minutes, whose moves the market sees coming and the model does not:

    fair price  the model recalibrated and blended with Kalshi's mid, see model/fair.py, fitted on the TRAIN_DAYS of settled
                windows before, see benchmark.fit_before(), and fitted again daily by paper/run.py
    when        from the open until MIN_LEFT seconds before the close
    edge        THRESHOLDS a contract or more after the venue's taker fee at the best price, on either side, buying Yes below
                the fair price or No above it, on Kalshi or Polymarket US, which trade the same bet
    episodes    once per venue and side until its edge falls below the threshold, as the replay takes them
    size        an immediate or cancel limit for up to CONTRACTS contracts at the worst price whose edge reaches the threshold,
                and from 2026-10-14 no more than WINDOW_CONTRACTS contracts of a window on a venue, both sides together, the
                orders on their way counted at their size: without it, a model at odds with the market for a whole window
                bought the same side again each time its edge came back, 1,346 contracts of the 22:15 window of 2026-10-13.
                10 and 50 until 2026-10-15, 50 and 250 since, to see what the edges hold at a size worth trading

Hyperliquid's 15 minute windows, the same bet on its perp's mark, from 2026-10-15 under a rule of their own, to see
whether its quotes lag Kalshi's: the fair price is Kalshi's mid alone, both sides quoted within MAX_SPREAD, without the
model; orders from its listing, 15 to 25 seconds into the window, until MIN_LEFT seconds before the close; an edge of
THRESHOLDS, from 2026-10-15 2 cents, as on the other venues; the rest as above. A replay of its recorded books had found
+3.6 +/- 2.3 cents a contract at 3 cents, but those books were the 5.4 second snapshots of Hyperliquid's l2Book feed alone,
whose quotes had often moved already, see api/hyperliquid.py, so the paper orders are the test. Its window's market is found in its list, see
name_windows(). Opening a position there is free, and the fee it may take at settlement, see SETTLE_FEES, a fraction of a
cent, is left out of the edge and taken from the profit as the window settles.

An order reaches its venue ORDER_SECONDS after it is decided, the 90th percentile of PredictionArbitrageTrader's live
orders from the same AWS region, and fills against the newest book the venue made by then: the trader waits for a book
the venue made later, which shows every change before has reached us, or FEED_SECONDS. A side sends no order while its
last waits so. A paper order leaves the venue's book as it was, so the contracts it took are kept off later orders until
the book shows fewer, as the replay does, see replay.still_resting(). Each order notes what brought its decision: a change
of its venue's own book, of Kalshi's, whose mid the fair price is blended with, an exchange's quote, or the once a second
pricing as time alone moves the model, see paper_orders.cause in schema.sql.

Beside the orders, the trader records every opportunity, as PredictionArbitrageTrader does: each stretch where buying one
side of a venue's market had a positive edge after the fee, at the best price by the fair price, while the window could
be priced, see Opportunity. Each says how long it lasted, its peak edge and the contracts at a positive edge then, walking
the book's levels, what they cost with the fees and their expected profit, the return on that capital and its rate a
year until the window pays, the longest unbroken stretch at the threshold or more and the contracts at it throughout, and
what one order could have had at the threshold or more at the best moment. The books are the venues' own, without what
paper orders took, and the profits expected ones, the fair price's edge, not locked in as an arbitrage's.

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
from upordowntrader.api import hyperliquid, polymarket_us
from upordowntrader.backtest import replay
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.model import fair, pricing

# Edge after the fee, in dollars a contract, an order needs: the benchmark's best before the last 5 minutes on Kalshi and
# Polymarket US, and the same on Hyperliquid.
THRESHOLDS = {"kalshi": 0.02, "polymarket_us": 0.02, "hyperliquid": 0.02}
CONTRACTS = 50              # The most contracts an order takes.
WINDOW_CONTRACTS = 250      # The most contracts the trader buys of one window on one venue, both sides together.
MIN_LEFT = {"kalshi": 300, "polymarket_us": 300, "hyperliquid": 60}     # Seconds before the close the last order may be decided.
MAX_SPREAD = 0.10           # The widest Kalshi spread whose mid Hyperliquid is traded against.
DEPTH = 10                  # Levels a side the trader keeps of each book.
VENUES = ("kalshi", "polymarket_us", "hyperliquid")
FEE_RATES = {"kalshi": fair.FEE_RATE, "polymarket_us": polymarket_us.TAKER_FEE_RATE, "hyperliquid": 0.0}     # Of p(1 - p), on buying.
SETTLE_FEES = {"hyperliquid": hyperliquid.SETTLE_FEE_RATE}  # Of a winning contract's dollar, as it settles.
ORDER_SECONDS = {"kalshi": 0.016, "polymarket_us": 0.136, "hyperliquid": 0.4}   # From an order's decision to its venue's book, by
                                                                                # its clock: Hyperliquid's as its replay took it.
FEED_SECONDS = {"kalshi": 0.1, "polymarket_us": 0.3, "hyperliquid": 0.6}    # The longest a venue's book takes to reach us, at the
                                                                            # 90th percentile.
TAPE_SECONDS = 10           # Seconds of each book's changes kept, for the book an order meets.
READING_SECONDS = 1800      # Seconds of the price's readings kept, back past the open of the window trading.
CATCH_UP = 10               # The most seconds without a price a reading is filled back over. Longer, the process stalled.
WINDOW_KEEP = 60            # Seconds after its close a window is forgotten, with its books.
PAYOUT_SECONDS = {"kalshi": 5, "polymarket_us": 10, "hyperliquid": 10}     # Seconds after its close a window pays: Kalshi settled
                                                                        # one 3.7 s after on 2026-10-10, Polymarket US one 10 s
                                                                        # after on 2026-10-13, and Hyperliquid settles about 8 s
                                                                        # after.
LOG_PROFIT_DOLLARS = 10     # Opportunities one order could have taken this much expected profit from are logged as they end.
DAY = 86400


@dataclass
class Window:
    """
    A window the trader prices, from Kalshi's market.
    """
    market: Market              # Kalshi's market, its strike None until it opens.
    names: dict                 # Each venue's market of the window, by venue.
    tracking: float | None = None   # The tracking variance, see tracking_variance(), None until read.
    offset: float | None = None     # The price over BRTI's level at the open, None until known, NaN when it cannot be.


@dataclass
class Opportunity:
    """
    A venue's market and side while its edge after the fee stays positive, followed moment by moment, see
    PaperTrader.follow(), and stored as a row of the opportunities table as it ends, see row().
    """
    venue: str
    ticker: str
    close_time: int
    side: str
    start: float                    # When the edge went positive.
    peak: dict                      # The best moment so far: its time, edge, price, fair price, the contracts at a positive
                                    # edge, their capital and expected profit, and whether the venue's own book changed then.
    take: tuple | None = None       # (contracts, capital, profit) one order could have had at the threshold or more at the best
                                    # moment so far, or None before any.
    since: float | None = None      # When the current stretch at the threshold or more began, or None outside one.
    least: tuple = (0.0, 0.0, 0.0)  # The fewest contracts at the threshold or more so far in the stretch, as take.
    longest: tuple | None = None    # The longest stretch so far, (seconds, contracts, capital, profit), or None before any.
    total: float | None = None      # Seconds at the threshold or more in all, over every stretch ended, or None before any.

    def end_stretch(self, now):
        """
        Close the current stretch at the threshold or more at now, keeping it when it is the longest, and adding it to the
        total.
        """
        if self.since is None:
            return
        if self.longest is None or now - self.since > self.longest[0]:
            self.longest = (now - self.since, *self.least)
        self.total = (self.total or 0.0) + now - self.since
        self.since = None

    def see(self, moment, at_threshold, now):
        """
        Follow one moment of the opportunity: moment as peak is, and at_threshold the (contracts, capital, profit) at the
        threshold or more, or None when the best price's edge is under it.
        """
        if moment["edge"] > self.peak["edge"]:
            self.peak = moment
        if at_threshold is None:
            self.end_stretch(now)
            return
        if self.since is None:
            self.since, self.least = now, at_threshold
        elif at_threshold[0] < self.least[0]:
            self.least = at_threshold
        if self.take is None or at_threshold[2] > self.take[2]:
            self.take = at_threshold

    def row(self, end, min_edge):
        """
        The opportunity as a row of the opportunities table, ended at end, measured at min_edge, the threshold.
        """
        self.end_stretch(end)
        peak = self.peak
        days = (self.close_time + PAYOUT_SECONDS[self.venue] - peak["time"]) / DAY
        return_pct = 100 * peak["edge"] / (peak["price"] + fair.fee(peak["price"], FEE_RATES[self.venue]))
        seconds, size, capital, profit = self.longest or (None, None, None, None)
        take_size, take_capital, take_profit = self.take or (None, None, None)
        return {"venue": self.venue, "ticker": self.ticker, "close_time": self.close_time, "side": self.side, "start_time": self.start,
                "end_time": end, "seconds": end - self.start, "peak_time": peak["time"], "peak_edge": peak["edge"], "peak_price": peak["price"],
                "peak_fair": peak["fair"], "peak_size": peak["size"], "peak_capital": peak["capital"], "peak_profit": peak["profit"],
                "book_changed": int(peak["book_changed"]), "days_held": days, "return_pct": return_pct, "annual_pct": return_pct * 365 / days,
                "min_edge": min_edge, "min_edge_seconds": seconds, "min_edge_size": size, "min_edge_capital": capital,
                "min_edge_profit": profit, "take_size": take_size, "take_capital": take_capital, "take_profit": take_profit,
                "take_return_pct": 100 * take_profit / take_capital if take_capital else None, "min_edge_total": self.total}


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


def depth(levels, chance, rate, least):
    """
    The contracts on levels, [price, contracts] best first, whose edge after the fee at rate is least or more, and more
    than nothing, given the chance their side pays: how many, what they cost with the fees, and their expected profit.
    """
    size = capital = profit = 0.0
    for p, c in levels:
        e = fair.edge(chance, p, rate)
        if e < least or e <= 0:
            break
        size, capital, profit = size + c, capital + c * (p + fair.fee(p, rate)), profit + c * e
    return size, capital, profit


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

    def __init__(self, conn, log=print, clock=time.time, thresholds=THRESHOLDS, contracts=CONTRACTS, window_contracts=WINDOW_CONTRACTS,
                 depth=DEPTH):
        self.conn = conn
        self.log = log
        self.clock = clock
        self.thresholds = dict(thresholds)      # Each venue's edge an order needs.
        self.contracts = contracts
        self.window_contracts = window_contracts
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
        self.held = {}              # (venue, ticker) maps to the contracts orders filled, both sides together.
        self.pending = []           # Orders decided and not filled yet.
        self.orders = 0             # Orders stored since the start.
        self.errors = {}            # Each settled Kalshi window's tracking error, by ticker, see tracking_variance().
        self.opportunities = {}     # (venue, ticker, side) maps to its open Opportunity.
        self.finished = []          # Rows of the opportunities ended since they were last stored.

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
        self.evaluate(now, source=source)

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
        self.evaluate(now, key)

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
                self.windows[m.close_time] = Window(m, {venue: names[venue] for venue in self.venues if venue in names})
            elif m.strike is not None:
                w.market = m

    def name_windows(self, venue, markets):
        """
        Give the windows priced their market on a venue whose names the time does not tell, Hyperliquid, from the
        markets it lists, Markets, each by its close time.
        """
        if venue not in self.venues:
            return
        for m in markets:
            w = self.windows.get(m.close_time)
            if w is not None:
                w.names[venue] = m.ticker

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

    # PRICING

    def price(self, now):
        """
        The window trading and what its fair price was made from at now, {price, model, mid, fair}, or None when it cannot
        be priced: no window with all it needs, no fit or variance forecast yet, no more left than any venue trades to, see
        MIN_LEFT, or no Kalshi book. kalshi_mid is Kalshi's mid when both its sides are quoted within MAX_SPREAD, else None.
        """
        w = self.trading(now)
        if w is None or self.fit is None or self.variance is None or w.market.close_time - now <= min(MIN_LEFT[v] for v in self.venues):
            return None
        kalshi = self.books.get(("kalshi", w.names["kalshi"]))
        mid = book_mid(kalshi) if kalshi is not None and self.up("kalshi") else None
        if mid is None:
            return None
        price = self.stand_in() - w.offset
        model = self.model_chance(w, price, now)
        if not math.isfinite(model):
            return None
        bids, asks = kalshi
        strict = (bids[0][0] + asks[0][0]) / 2 if bids and asks and asks[0][0] - bids[0][0] <= MAX_SPREAD else None
        return w, {"price": price, "model": model, "mid": mid, "fair": float(fair.blended(mid, model, w.market.close_time - now, self.fit)),
                   "kalshi_mid": strict}

    def up_chance(self, venue, seen):
        """
        The chance of Up a venue is traded against, from what price() saw: the fair price, or Kalshi's mid alone on
        Hyperliquid, None when there is none.
        """
        return seen["kalshi_mid"] if venue == "hyperliquid" else seen["fair"]

    def evaluate(self, now, changed=None, source=None):
        """
        Fill the orders whose books are known, then price the window trading, follow its opportunities, and decide the
        orders its edges call for. changed is the (venue, ticker) whose book change brought the pricing, source the exchange
        whose quote did, or neither for the once a second pricing.
        """
        self.fill_due(now)
        priced = self.price(now)
        if priced is None:
            self.end_opportunities(now)
            return
        w, seen = priced
        cause = source or "time"
        if changed is not None:
            cause = changed[0] if changed in w.names.items() else "other"
        for venue, ticker in w.names.items():
            book = self.books.get((venue, ticker))
            up = self.up_chance(venue, seen)
            if book is None or up is None or not self.up(venue) or (venue, ticker) in self.halted or w.market.close_time - now <= MIN_LEFT[venue]:
                self.end_opportunities(now, (venue, ticker))
                continue
            for side in ("yes", "no"):
                self.follow(now, w, venue, ticker, book, side, up if side == "yes" else 1 - up, changed == (venue, ticker))
                self.consider(now, w, venue, ticker, book, side, seen, up, cause)

    # OPPORTUNITIES

    def follow(self, now, w, venue, ticker, book, side, chance, changed):
        """
        Open, extend, or end the opportunity on one side of a venue's market, whose side pays with chance, from its book at
        now. changed says whether the venue's own book change brought this pricing.
        """
        key = (venue, ticker, side)
        rate = FEE_RATES[venue]
        levels = side_levels(book, side)
        edge = fair.edge(chance, levels[0][0], rate) if levels else -math.inf
        if edge <= 0:
            self.end_opportunity(key, now)
            return
        size, capital, profit = depth(levels, chance, rate, 0.0)
        moment = {"time": now, "edge": edge, "price": levels[0][0], "fair": chance, "size": size, "capital": capital, "profit": profit,
                  "book_changed": changed}
        o = self.opportunities.get(key)
        if o is None:
            o = self.opportunities[key] = Opportunity(venue, ticker, w.market.close_time, side, now, moment)
        least = self.thresholds[venue]
        o.see(moment, depth(levels, chance, rate, least) if edge >= least else None, now)

    def end_opportunity(self, key, now):
        """
        End the open opportunity of a (venue, ticker, side), if any, at now: keep its row to store, and log it when one
        order could have taken LOG_PROFIT_DOLLARS or more from it.
        """
        o = self.opportunities.pop(key, None)
        if o is None:
            return
        row = o.row(now, self.thresholds[o.venue])
        self.finished.append(row)
        if (row["take_profit"] or 0) >= LOG_PROFIT_DOLLARS:
            self.log(f"opportunity {o.venue} {o.side} {o.ticker}: {100 * row['peak_edge']:.1f}c x {row['peak_size']:,.0f} = "
                     f"{row['peak_profit']:,.2f}$ expected, lasted {row['seconds']:.1f}s, {100 * row['min_edge']:.0f}c or more for "
                     f"{row['min_edge_seconds']:.1f}s with {row['min_edge_size']:,.0f} contracts throughout; one order could have taken "
                     f"{row['take_size']:,.0f} for {row['take_capital']:,.2f}$, expecting {row['take_profit']:,.2f}$")

    def end_opportunities(self, now, market=None):
        """
        End every open opportunity at now, or those of one (venue, ticker).
        """
        for key in [k for k in self.opportunities if market is None or k[:2] == market]:
            self.end_opportunity(key, now)

    def store_opportunities(self):
        """
        Store the opportunities ended since they were last stored.
        """
        if self.finished:
            database.store_opportunities(self.conn, self.finished)
            self.finished = []

    # ORDERS

    def consider(self, now, w, venue, ticker, book, side, seen, up, cause):
        """
        Decide an order on one side of a venue's market when its edge calls for one, see the module's docstring: up is the
        chance of Up the venue is traded against, see up_chance(), and cause says what brought the pricing, see evaluate().
        """
        key = (venue, ticker, side)
        chance = up if side == "yes" else 1 - up
        rate, least = FEE_RATES[venue], self.thresholds[venue]
        levels = side_levels(book, side)
        if not levels or fair.edge(chance, levels[0][0], rate) < least:
            self.armed[key] = True          # The next reach is a new episode.
            return
        if not self.armed.get(key, True) or now < self.busy.get(key, -math.inf):
            return
        room = self.room(venue, ticker)
        offered = replay.untaken([(p, c) for p, c in levels if fair.edge(chance, p, rate) >= least], self.taken.get(key, {}))
        if room <= 1e-9 or not offered:
            return
        self.armed[key] = False
        arrived = now + ORDER_SECONDS[venue]
        self.busy[key] = arrived + FEED_SECONDS[venue]
        self.pending.append({"venue": venue, "ticker": ticker, "close_time": w.market.close_time, "decided": now, "side": side, **seen, "fair": up,
                             "edge": fair.edge(chance, levels[0][0], rate), "limit_price": max(p for p, _ in offered),
                             "offered": sum(c for _, c in offered), "arrived": arrived, "cause": cause, "chance": chance,
                             "size": min(self.contracts, room)})

    def room(self, venue, ticker):
        """
        The contracts the window cap leaves an order on a venue's market: WINDOW_CONTRACTS less what orders filled there
        and what those on their way may fill.
        """
        coming = sum(o["size"] for o in self.pending if (o["venue"], o["ticker"]) == (venue, ticker))
        return self.window_contracts - self.held.get((venue, ticker), 0.0) - coming

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
        took, up to the order's size, then store and log it, and say when it fills the window's cap.
        """
        venue, side, chance, wanted = order["venue"], order["side"], order.pop("chance"), order.pop("size")
        key = (venue, order["ticker"], side)
        before = [change for change in self.tapes.get(key[:2], ()) if change[0] <= order["arrived"]]
        levels = side_levels(before[-1][1:], side) if before else []
        ours = self.taken.setdefault(key, {})
        filled = []
        for p, c in replay.untaken([(p, c) for p, c in levels if p <= order["limit_price"] + 1e-9], ours):
            if wanted <= 0:
                break
            filled.append((p, min(c, wanted)))
            wanted -= filled[-1][1]
        for p, c in filled:
            ours[round(p, 6)] = ours.get(round(p, 6), 0.0) + c
        contracts = sum(c for _, c in filled)
        held = self.held[key[:2]] = self.held.get(key[:2], 0.0) + contracts
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
        if contracts and held >= self.window_contracts - 1e-9:
            self.log(f"paper {venue} {order['ticker']} holds {held:g} contracts, the window's cap, so it takes no more")

    # TIME

    def tick(self, now):
        """
        Once a second: store the readings, fill and decide as time alone moves the price, store the opportunities ended,
        and forget what is past.
        """
        self.mark(now)
        self.evaluate(now)
        self.store_opportunities()
        for s in [s for s in self.readings if s < now - READING_SECONDS]:
            del self.readings[s]
        for close in [c for c in self.windows if c < now - WINDOW_KEEP]:
            markets = set(self.windows.pop(close).names.items())
            for kept in (self.books, self.tapes, self.taken, self.armed, self.busy, self.held):
                for key in [k for k in kept if k[:2] in markets]:
                    del kept[key]
            self.halted -= markets

    def stop(self, now):
        """
        End every open opportunity at now and store them, as the process stops.
        """
        self.end_opportunities(now)
        self.store_opportunities()
