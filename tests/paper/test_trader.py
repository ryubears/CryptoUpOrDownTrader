"""
Tests for the paper trader: the price it reads, the orders its rule decides, and the books they fill against, on a made up
clock, with no network.
"""

import math
import numpy as np
import pytest
from types import SimpleNamespace
from upordowntrader.api import polymarket_us
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.model import fair
from upordowntrader.paper import trader as paper

OPEN = 1_791_896_400                    # 2026-10-13 13:00 UTC.
CLOSE = OPEN + 900
STRIKE = 80_000.0
MODEL_ONLY = fair.Fit(np.ones(fair.MINUTES), np.full(fair.MINUTES, 8), np.tile([0.0, 0.0, 1.0], (fair.MINUTES, 1)))   # The fair price is the model's.
BIDS = [[0.48, 100.0]]                  # A book around a chance of a half, with no edge at the strike.
ASKS = [[0.52, 30.0], [0.54, 20.0], [0.95, 50.0]]


class Clock:
    """
    A clock the test moves.
    """

    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def setup(tmp_path, venues=("kalshi",)):
    """
    A trader on a clock a minute and a bit before the window opens, with the window, a fit whose fair price is the model's,
    a variance forecast, and Coinbase quoting 50 above BRTI's level, stepped through the open. Orders of up to 10 contracts
    and 50 a window, as the small books here are made for, the rule's own sizes aside, see the test of them.
    """
    clock = Clock(OPEN - 70.0)
    logs = []
    t = paper.PaperTrader(database.connect(tmp_path / "history.sqlite"), logs.append, clock, contracts=10, window_contracts=50)
    t.venues = venues
    t.fit, t.variance = MODEL_ONLY, (np.full(200, 1e-9), OPEN - 3600)
    t.set_windows([Market("kalshi", "K", "K", "KXBTC15M", OPEN, CLOSE, STRIKE, None, None, None)])
    t.windows[CLOSE].tracking = 0.0
    t.on_quote("coinbase", [[80_049.5, 1.0]], [[80_050.5, 1.0]])
    step(t, clock, OPEN + 60.0)
    return t, clock, logs


def step(t, clock, to):
    """
    Move the clock to a time, ticking the trader each second on the way.
    """
    while clock.now + 1 <= to:
        clock.now += 1
        t.tick(clock.now)
    clock.now = to
    t.tick(to)


def quote(t, clock, price):
    """
    Coinbase quotes a mid of price now.
    """
    t.on_quote("coinbase", [[price - 0.5, 1.0]], [[price + 0.5, 1.0]])


def orders(t):
    return database.load_paper_orders(t.conn)


def test_the_price_reads_on_brtis_level_from_the_minute_before_the_open(tmp_path):
    t, clock, _ = setup(tmp_path)
    w = t.trading(clock.now)
    assert w.offset == pytest.approx(50.0) and t.stand_in() - w.offset == pytest.approx(STRIKE)
    assert t.model_chance(w, STRIKE, clock.now) == pytest.approx(0.5)


def test_a_window_whose_open_was_missed_waits_for_its_offset_from_the_recorded_quotes(tmp_path):
    clock = Clock(OPEN + 30.0)
    conn = database.connect(tmp_path / "history.sqlite")
    t = paper.PaperTrader(conn, print, clock)
    market = Market("kalshi", "K", "K", "KXBTC15M", OPEN, CLOSE, STRIKE, None, None, None)
    t.set_windows([market])
    t.windows[CLOSE].tracking = 0.0
    quote(t, clock, 80_050.0)
    step(t, clock, OPEN + 32.0)
    assert t.trading(clock.now) is None
    database.store_quotes(conn, [("coinbase", s, s, STRIKE + 39.5 + (s > OPEN - 30), 1.0, STRIKE + 40.5 + (s > OPEN - 30), 1.0)
                                 for s in np.arange(OPEN - 90.0, OPEN + 1)])    # 40 above, then 41 for the last 30 seconds.
    t.windows[CLOSE].offset = paper.recorded_offset(conn, market)
    # A reading a moment after a quote reads the one a second before, so the last 29 of the 60 read 41.
    assert t.windows[CLOSE].offset == pytest.approx(40 + 29 / 60) and t.trading(clock.now) is t.windows[CLOSE]


def test_an_edge_sends_an_order_that_fills_against_the_book_when_it_arrives(tmp_path):
    t, clock, logs = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    quote(t, clock, 80_110.0)                                   # 60 above the strike, so Up is likely: Yes at 52 is cheap.
    (order,) = t.pending
    decided = clock.now
    assert (order["side"], order["limit_price"], order["offered"]) == ("yes", 0.54, 50.0)     # Not 95 cents, too dear.
    clock.now += 0.005
    t.on_book("kalshi", "K", BIDS, ASKS[1:], decided + 0.01)    # 52 cents taken before it arrived,
    clock.now += 0.02
    t.on_book("kalshi", "K", [[0.49, 5.0], *BIDS], ASKS[1:], decided + 0.03)   # and a later book shows it all reached us.
    (stored,) = orders(t)
    chance = order["fair"]
    assert (stored["contracts"], stored["cost"], stored["best"]) == (10.0, 0.54, 0.54)
    assert stored["arrived"] == pytest.approx(decided + paper.ORDER_SECONDS["kalshi"])
    assert stored["fees"] == pytest.approx(10 * fair.fee(0.54)) and stored["expected"] == pytest.approx(10 * fair.edge(chance, 0.54))
    assert logs[-1].startswith("paper kalshi yes K with 14.0 min left: edge ") and logs[-1].endswith("50 offered to 0.54, filled 10 at 0.540")


def test_an_order_with_no_later_book_fills_once_its_feed_time_has_passed(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    quote(t, clock, 80_110.0)
    assert not orders(t)
    step(t, clock, clock.now + paper.ORDER_SECONDS["kalshi"] + paper.FEED_SECONDS["kalshi"])
    assert [(o["contracts"], o["cost"]) for o in orders(t)] == [(10.0, 0.52)]


def test_an_edge_is_taken_once_until_it_falls_below_the_threshold_and_then_less_what_was_taken(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    quote(t, clock, 80_110.0)
    step(t, clock, clock.now + 1)
    quote(t, clock, 80_111.0)                                   # The same episode.
    step(t, clock, clock.now + 1)
    assert len(orders(t)) == 1
    quote(t, clock, 80_050.0)                                   # At the strike, no edge: the episode ends,
    quote(t, clock, 80_110.0)                                   # and a new one starts, with the 10 taken at 52 still shown.
    step(t, clock, clock.now + 1)
    assert [(o["offered"], o["contracts"]) for o in orders(t)] == [(50.0, 10.0), (40.0, 10.0)]
    t.on_book("kalshi", "K", BIDS, [[0.52, 5.0], *ASKS[1:]], clock.now)     # The level shrank below the 20 taken,
    quote(t, clock, 80_050.0)
    quote(t, clock, 80_110.0)
    step(t, clock, clock.now + 1)
    assert orders(t)[-1]["offered"] == 20.0                     # so 52 cents is gone for us, and 54 cents was not taken.


def test_a_window_takes_no_more_than_its_cap_on_a_venue_counting_the_orders_on_their_way(tmp_path):
    t, clock, logs = setup(tmp_path)
    t.window_contracts = 25
    t.on_book("kalshi", "K", BIDS, [[0.52, 1000.0], [0.54, 1000.0]], clock.now)
    sizes = []
    for _ in range(4):                                          # Four episodes of the same edge.
        quote(t, clock, 80_110.0)
        sizes.append(t.pending[0]["size"] if t.pending else None)
        if t.pending:
            assert t.room("kalshi", "K") == 25 - sum(o["contracts"] for o in orders(t)) - t.pending[0]["size"]
        step(t, clock, clock.now + 1)
        quote(t, clock, 80_050.0)
    assert sizes == [10, 10, 5, None] and [o["contracts"] for o in orders(t)] == [10.0, 10.0, 5.0]
    assert logs.count("paper kalshi K holds 25 contracts, the window's cap, so it takes no more") == 1


def test_the_rule_buys_up_to_50_contracts_an_order_and_250_a_window_on_a_venue(tmp_path):
    t, clock, logs = setup(tmp_path)
    t.contracts, t.window_contracts = paper.CONTRACTS, paper.WINDOW_CONTRACTS
    t.on_book("kalshi", "K", BIDS, [[0.52, 10_000.0]], clock.now)
    for _ in range(6):                                          # Six episodes of the same edge.
        quote(t, clock, 80_110.0)
        step(t, clock, clock.now + 1)
        quote(t, clock, 80_050.0)
    assert [o["contracts"] for o in orders(t)] == [50.0] * 5
    assert logs.count("paper kalshi K holds 250 contracts, the window's cap, so it takes no more") == 1


def test_no_order_in_the_last_five_minutes_or_on_a_book_whose_stream_is_down(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    t.feeds = {"kalshi": SimpleNamespace(ws=None)}
    quote(t, clock, 80_110.0)
    quote(t, clock, 80_050.0)
    t.feeds = {}
    step(t, clock, CLOSE - paper.MIN_LEFT["kalshi"])
    quote(t, clock, 80_110.0)
    assert not t.pending and not orders(t)


def test_no_order_on_a_market_its_venue_says_is_not_trading(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    t.on_state("kalshi", "K", "paused")
    quote(t, clock, 80_110.0)
    assert not t.pending
    t.on_state("kalshi", "K", None)
    quote(t, clock, 80_111.0)
    assert len(t.pending) == 1


def test_an_exchange_whose_stream_is_down_is_left_out_of_the_price(tmp_path):
    t, _, _ = setup(tmp_path)
    t.on_quote("kraken", [[80_000.0, 1.0]], [[80_010.0, 1.0]])
    assert t.stand_in() == pytest.approx((80_050.0 + 80_005.0) / 2)
    t.feeds = {"kraken": SimpleNamespace(ws=None)}
    assert t.stand_in() == pytest.approx(80_050.0)


def test_polymarket_is_traded_on_its_own_book_with_its_own_fee_at_the_fair_price_kalshi_moves(tmp_path):
    t, clock, _ = setup(tmp_path, venues=("kalshi", "polymarket_us"))
    name = polymarket_us.slug(OPEN)
    quote(t, clock, 80_110.0)                                                   # Up is likely.
    t.on_book("kalshi", "K", [[0.75, 10.0]], [[0.85, 10.0]], clock.now)         # Kalshi's book has no edge,
    t.on_book("polymarket_us", name, [[0.48, 10.0]], [[0.52, 5.0]], clock.now)  # Polymarket US's has on Yes.
    step(t, clock, clock.now + 1)
    (order,) = orders(t)
    assert (order["venue"], order["ticker"], order["side"], order["contracts"]) == ("polymarket_us", name, "yes", 5.0)
    assert order["fees"] == pytest.approx(5 * fair.fee(0.52, polymarket_us.TAKER_FEE_RATE))
    assert order["mid"] == pytest.approx(0.80)


def test_each_order_notes_what_brought_its_decision(tmp_path):
    t, clock, _ = setup(tmp_path, venues=("kalshi", "polymarket_us"))
    name = polymarket_us.slug(OPEN)
    quote(t, clock, 80_110.0)                                                   # Up is likely,
    t.on_book("polymarket_us", name, [[0.48, 10.0]], [[0.52, 5.0]], clock.now)  # but nothing is priced without Kalshi's book,
    t.on_book("kalshi", "K", [[0.75, 10.0]], [[0.85, 10.0]], clock.now)         # so its coming brings Polymarket US's order.
    clock.now += 1
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)                             # Kalshi's own book brings an edge on Yes.
    step(t, clock, clock.now + 1)
    quote(t, clock, 80_050.0)                                   # At the strike the episodes end,
    quote(t, clock, 80_110.0)                                   # and Coinbase's quote brings Kalshi's back.
    step(t, clock, clock.now + 1)
    quote(t, clock, 80_050.0)
    t.fit = None
    quote(t, clock, 80_110.0)                                   # The edge comes back while the fair price cannot be made,
    t.fit = MODEL_ONLY
    step(t, clock, clock.now + 2)                               # so the next second's pricing brings the order.
    assert [(o["venue"], o["cause"]) for o in orders(t)] == [("polymarket_us", "kalshi"), ("kalshi", "kalshi"), ("kalshi", "coinbase"),
                                                             ("kalshi", "time")]


def listed(t, ticker="#1230"):
    """
    Hyperliquid lists the window, as paper/run.py passes its market list on.
    """
    t.name_windows("hyperliquid", [Market("hyperliquid", ticker, "out-BTC-15m-x", "out-BTC-15m", OPEN, CLOSE, STRIKE + 60, None, None, None)])


def test_hyperliquid_is_traded_against_kalshis_mid_alone_at_its_own_threshold(tmp_path):
    t, clock, _ = setup(tmp_path, venues=("kalshi", "hyperliquid"))
    listed(t)
    t.on_book("kalshi", "K", [[0.49, 10.0]], [[0.51, 10.0]], clock.now)     # Kalshi's mid is 50c, and so is the model at the strike.
    t.on_book("hyperliquid", "#1230", [[0.40, 20.0]], [[0.49, 50.0]], clock.now)
    assert not t.pending                                        # Yes at 49c is 1c under Kalshi's mid: not Hyperliquid's 2c.
    quote(t, clock, 80_110.0)                                   # The model says Up is likely, which Hyperliquid's fair price leaves out,
    assert [o["venue"] for o in t.pending] == ["kalshi"]        # though Kalshi's own rule trades on it.
    t.on_book("hyperliquid", "#1230", [[0.40, 20.0]], [[0.46, 5.0], [0.48, 30.0], [0.49, 50.0]], clock.now)
    (order,) = [o for o in t.pending if o["venue"] == "hyperliquid"]
    assert (order["side"], order["limit_price"], order["offered"], order["fair"]) == ("yes", 0.48, 35.0, 0.5)
    assert order["edge"] == pytest.approx(0.04) and order["cause"] == "hyperliquid"
    step(t, clock, clock.now + 1)
    (stored,) = [o for o in orders(t) if o["venue"] == "hyperliquid"]
    assert (stored["contracts"], stored["cost"], stored["fees"]) == (10.0, pytest.approx(0.47), 0.0)        # 5 at 46c, 5 at 48c.
    assert stored["arrived"] == pytest.approx(order["decided"] + paper.ORDER_SECONDS["hyperliquid"])


def test_hyperliquid_trades_until_its_last_minute_after_kalshi_has_stopped_and_not_without_a_tight_kalshi_book(tmp_path):
    t, clock, _ = setup(tmp_path, venues=("kalshi", "hyperliquid"))
    listed(t)
    step(t, clock, CLOSE - 200)                                 # Inside Kalshi's last 5 minutes.
    t.on_book("kalshi", "K", [[0.44, 10.0]], [[0.45, 10.0]], clock.now)     # Yes at 45c would be an edge for Kalshi's rule.
    t.on_book("hyperliquid", "#1230", [[0.30, 10.0]], [[0.40, 10.0]], clock.now)
    assert [o["venue"] for o in t.pending] == ["hyperliquid"]
    t.on_book("kalshi", "K", [[0.30, 10.0]], [[0.45, 10.0]], clock.now)     # Kalshi's spread past MAX_SPREAD: no fair price for Hyperliquid,
    assert ("hyperliquid", "#1230", "yes") not in t.opportunities          # so its opportunity ends.
    step(t, clock, CLOSE - paper.MIN_LEFT["hyperliquid"] + 1)
    t.on_book("kalshi", "K", [[0.44, 10.0]], [[0.45, 10.0]], clock.now)
    t.on_book("hyperliquid", "#1230", [[0.30, 10.0]], [[0.39, 10.0]], clock.now)
    assert len([o for o in orders(t) if o["venue"] == "hyperliquid"]) == 1 and not t.pending     # None in its last minute.


def test_an_opportunity_counts_every_stretch_at_the_threshold_and_the_takes_expected_return():
    moment = {"time": 100.0, "edge": 0.03, "price": 0.5, "fair": 0.53, "size": 10.0, "capital": 5.0, "profit": 0.3, "book_changed": False}
    o = paper.Opportunity("hyperliquid", "#1230", CLOSE, "yes", 100.0, moment)
    o.see(moment, (10.0, 5.0, 0.3), 100.0)
    o.see({**moment, "edge": 0.01}, None, 101.5)                # Under the threshold after 1.5 s,
    o.see(moment, (4.0, 2.0, 0.12), 103.0)                      # at it again for 1 s more.
    row = o.row(104.0, 0.03)
    assert (row["min_edge_seconds"], row["min_edge_total"], row["take_size"]) == (1.5, 2.5, 10.0)
    assert row["take_return_pct"] == pytest.approx(6.0) and row["days_held"] == pytest.approx((CLOSE + 10 - 100.0) / 86400)


def test_a_closed_window_is_forgotten_with_its_books(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    step(t, clock, CLOSE + paper.WINDOW_KEEP + 1)
    assert not t.windows and not t.books and not t.tapes


def test_the_tracking_variance_comes_from_the_stand_ins_errors_on_the_windows_settled_before(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    windows, rows, errors = [], [], []
    for k in range(10):
        open_time, error = OPEN - 1800 * (10 - k), 2.0 * (k - 4.5)     # A window apart, so their quotes do not mix.
        settlement = STRIKE + 10 * k
        windows.append(Market("kalshi", f"T{k}", f"T{k}", "KXBTC15M", open_time, open_time + 900, STRIKE, settlement, "yes", None))
        # Coinbase runs 50 above BRTI at the open, and misses the settlement by error more.
        rows += [("coinbase", t, t, STRIKE + 49.5, 1.0, STRIKE + 50.5, 1.0) for t in np.arange(open_time - 61, open_time + 1, 0.5)]
        rows += [("coinbase", t, t, settlement + 49.5 + error, 1.0, settlement + 50.5 + error, 1.0)
                 for t in np.arange(open_time + 839, open_time + 901, 0.5)]
        errors.append(math.log((settlement + error) / settlement))
    # And one whose close no exchange quoted, which is left out rather than read from the quotes of its open.
    windows.append(Market("kalshi", "QUIET", "QUIET", "KXBTC15M", OPEN - 1200, OPEN - 300, STRIKE, STRIKE + 500, "yes", None))
    rows += [("coinbase", t, t, STRIKE + 49.5, 1.0, STRIKE + 50.5, 1.0) for t in np.arange(OPEN - 1261, OPEN - 1199, 0.5)]
    database.upsert_markets(conn, windows)
    database.store_quotes(conn, rows)
    window = Market("kalshi", "NOW", "NOW", "KXBTC15M", OPEN, CLOSE, None, None, None, None)
    cache = {}
    assert paper.tracking_variance(conn, window, cache) == pytest.approx(np.mean(np.square(errors)))
    assert set(cache) == {f"T{k}" for k in range(10)} | {"QUIET"} and np.isnan(cache["QUIET"])


def opportunities(t):
    return database.load_opportunities(t.conn)


def test_an_opportunity_keeps_its_peak_its_longest_stretch_at_the_threshold_and_what_one_order_could_have_had(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", BIDS, ASKS, clock.now)
    start = clock.now
    quote(t, clock, 80_110.0)                                   # Yes at 52 and 54 cents reach the threshold, 95 is dear.
    clock.now += 0.5
    quote(t, clock, 80_100.0)                                   # Up a touch less likely, so the peak stays at the start,
    t.on_book("kalshi", "K", BIDS, [[0.52, 10.0], *ASKS[1:]], clock.now)     # and 20 at 52 cents go.
    clock.now += 1.5
    end = clock.now
    quote(t, clock, 80_050.0)                                   # Back at the strike: no edge.
    t.tick(clock.now)
    (o,) = opportunities(t)
    assert (o["venue"], o["ticker"], o["side"], o["close_time"]) == ("kalshi", "K", "yes", CLOSE)
    assert (o["start_time"], o["end_time"], o["seconds"], o["peak_time"], o["book_changed"]) == (start, end, 2.0, start, 0)
    assert (o["peak_price"], o["peak_size"], o["min_edge"]) == (0.52, 50.0, paper.THRESHOLDS["kalshi"])
    fee = fair.fee(0.52)
    assert o["peak_capital"] == pytest.approx(30 * (0.52 + fee) + 20 * (0.54 + fair.fee(0.54)))
    assert o["peak_profit"] == pytest.approx(30 * o["peak_edge"] + 20 * fair.edge(o["peak_fair"], 0.54))
    assert o["days_held"] == pytest.approx((CLOSE + paper.PAYOUT_SECONDS["kalshi"] - start) / 86400)
    assert o["return_pct"] == pytest.approx(100 * o["peak_edge"] / (0.52 + fee))
    assert o["annual_pct"] == pytest.approx(o["return_pct"] * 365 / o["days_held"])
    assert (o["min_edge_seconds"], o["min_edge_size"], o["take_size"]) == (2.0, 30.0, 50.0)    # The fewest throughout, the best moment.
    assert o["min_edge_capital"] == pytest.approx(10 * (0.52 + fee) + 20 * (0.54 + fair.fee(0.54)))
    assert 0 < o["min_edge_profit"] < o["take_profit"] and o["take_capital"] == pytest.approx(o["peak_capital"])


def test_an_edge_under_the_threshold_is_an_opportunity_without_a_take_and_ends_when_the_window_cannot_be_priced(tmp_path):
    t, clock, _ = setup(tmp_path)
    t.on_book("kalshi", "K", [[0.40, 10.0]], [[0.48, 40.0]], clock.now)    # At a chance of a half, Yes at 48 cents makes 0.25c.
    assert not t.pending
    step(t, clock, CLOSE - paper.MIN_LEFT["kalshi"])
    (o,) = opportunities(t)
    assert o["end_time"] == CLOSE - paper.MIN_LEFT["kalshi"] and o["book_changed"] == 1     # The peak came with Kalshi's own book.
    assert o["peak_size"] == 40.0 and o["peak_profit"] == pytest.approx(40 * o["peak_edge"]) and 0 < o["peak_edge"] < paper.THRESHOLDS["kalshi"]
    assert (o["min_edge_seconds"], o["min_edge_size"], o["take_size"], o["take_profit"]) == (None, None, None, None)


def test_the_open_opportunities_are_stored_as_the_trader_stops_and_a_halted_market_ends_its_own(tmp_path):
    t, clock, _ = setup(tmp_path, venues=("kalshi", "polymarket_us"))
    name = polymarket_us.slug(OPEN)
    t.on_book("kalshi", "K", [[0.40, 10.0]], [[0.48, 40.0]], clock.now)
    t.on_book("polymarket_us", name, [[0.40, 10.0]], [[0.48, 5.0]], clock.now)
    clock.now += 1
    t.on_state("polymarket_us", name, "suspended")
    t.tick(clock.now)
    assert [(o["venue"], o["end_time"]) for o in opportunities(t)] == [("polymarket_us", clock.now)]
    t.stop(clock.now + 1)
    assert sorted((o["venue"], o["end_time"]) for o in opportunities(t)) == [("kalshi", clock.now + 1), ("polymarket_us", clock.now)]
