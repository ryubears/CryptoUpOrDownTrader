"""
Tests for the replay: the model's chances from the price before, the stand-in prices, and the trading rule's orders.
"""

import numpy as np
import pytest
from dataclasses import replace
from upordowntrader.backtest import replay
from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.model import fair

START = 1_000_000                       # The first second of the made up history.
OPEN = START + 3600


def book(*changes):
    """
    A Book from (time, bids, asks) changes, the levels best first.
    """
    times = np.array([c[0] for c in changes], dtype=float)
    bids, asks = [c[1] for c in changes], [c[2] for c in changes]
    return replay.Book(times, replay.best_prices(bids), replay.best_prices(asks), bids, asks)


ASKS = [[0.60, 100.0], [0.62, 50.0], [0.66, 20.0]]   # At a chance of 70%, the first two clear 5 cents after the fee.
BIDS = [[0.58, 80.0]]                                  # No at 42 cents is no bargain at 30%.


def test_an_order_takes_every_level_within_the_threshold_that_is_still_there_when_it_lands():
    b = book((5.0, BIDS, ASKS), (10.05, BIDS, [[0.61, 30.0], [0.62, 40.0], [0.63, 10.0]]))
    (order,) = replay.take_edges(np.array([10.0]), np.array([0.70]), b, close_time=900, up=True)
    assert (order["side"], order["offered"], order["contracts"]) == ("yes", 150.0, 70.0)      # Up to the limit of 62 cents.
    assert order["price"] == pytest.approx((0.61 * 30 + 0.62 * 40) / 70)
    assert order["expected"] == pytest.approx(30 * fair.edge(0.70, 0.61) + 40 * fair.edge(0.70, 0.62))
    assert order["realized"] == pytest.approx(30 * fair.edge(1.0, 0.61) + 40 * fair.edge(1.0, 0.62))
    assert order["one_contract"] == pytest.approx(fair.edge(1.0, 0.61))


def test_an_order_fills_nothing_when_the_price_moved_away_before_it_landed():
    b = book((5.0, BIDS, ASKS), (10.05, BIDS, [[0.65, 500.0]]))
    (order,) = replay.take_edges(np.array([10.0]), np.array([0.70]), b, close_time=900, up=False)
    assert (order["offered"], order["contracts"], order["expected"], order["realized"]) == (150.0, 0, 0, 0)
    assert order["one_contract"] is None


def test_an_edge_is_taken_once_until_it_falls_below_the_threshold():
    b = book((5.0, BIDS, ASKS), (12.0, BIDS, [[0.69, 10.0]]), (13.0, BIDS, ASKS))
    orders = replay.take_edges(np.array([10.0, 11.0, 12.0, 13.0]), np.full(4, 0.70), b, close_time=900, up=True)
    assert [o["time"] for o in orders] == [10.0, 13.0]         # 11 is the same episode; it ended at 12, and 13 is a new one.


def test_an_order_takes_only_what_the_last_left_and_none_goes_while_the_last_is_on_its_way():
    b = book((5.0, BIDS, ASKS), (11.0, BIDS, [[0.60, 130.0], [0.62, 50.0]]))      # 30 more at 60 cents at 11.
    times = np.array([10.0, 10.02, 10.04, 10.5, 11.0])
    orders = replay.take_edges(times, np.array([0.70, 0.60, 0.70, 0.70, 0.70]), b, close_time=900, up=True)
    # 10.02 ends the episode, but at 10.04 the first order is on its way, and at 10.5 the book shows only what it took.
    assert [(o["time"], o["offered"], o["contracts"]) for o in orders] == [(10.0, 150.0, 150.0), (11.0, 30.0, 30.0)]


def test_what_an_order_took_is_there_again_once_the_level_has_emptied():
    b = book((5.0, BIDS, ASKS), (10.5, BIDS, ASKS[1:]), (10.7, BIDS, ASKS))       # 60 cents emptied, then 100 more came.
    orders = replay.take_edges(np.array([10.0, 10.6, 11.0]), np.array([0.70, 0.60, 0.70]), b, close_time=900, up=True)
    assert [(o["time"], o["offered"]) for o in orders] == [(10.0, 150.0), (11.0, 100.0)]    # 62 cents never emptied.


def test_no_order_goes_when_it_would_land_after_the_close():
    b = book((5.0, BIDS, ASKS))
    assert replay.take_edges(np.array([899.95]), np.array([0.70]), b, close_time=900, up=True) == []


def flat_feed(price, jumps=()):
    """
    A Feed of one exchange quoting price from START, and each (time, price) of jumps after.
    """
    times = np.array([START, *(t for t, _ in jumps)], dtype=float)
    return replay.Feed((times,), (np.array([price, *(p for _, p in jumps)]),))


def test_the_model_reads_the_exchanges_on_brtis_level_and_a_jump_once_it_is_known():
    market = Market("kalshi", "T", "E", "KXBTC15M", OPEN, OPEN + 900, 80_000.0, None, "yes", None)
    variance = np.full(2 * 60, 1e-9)
    t = np.array([OPEN + 100.0, OPEN + 100.04, OPEN + 100.06])
    chances = replay.window_chances(flat_feed(80_050.0), variance, START, market, t)    # The stand-in runs 50 above BRTI.
    assert np.allclose(chances, 0.5, atol=0.01)                                         # At the strike, after the offset.
    later = replay.window_chances(flat_feed(80_050.0, [(OPEN + 100.0, 80_150.0)]), variance, START, market, t)
    assert later[:2].tolist() == chances[:2].tolist() and later[2] > 0.9                 # Known QUOTE_DELAY after it was sent.


def test_with_the_exchanges_the_model_reads_the_newest_quote_a_moment_after_it_was_sent():
    times, mids = np.array([START + 10.0, START + 20.0]), np.array([100.0, 110.0])
    feed = replay.Feed((times,), (mids,))
    prices = replay.price_at(feed, [START + 5.0, START + 20.02, START + 20.06, START + 30.0])
    assert np.isnan(prices[0]) and prices[1:].tolist() == [100.0, 110.0, 110.0]   # The second quote is known QUOTE_DELAY after.


def test_the_exchanges_feed_averages_the_two_and_reads_by_the_exchanges_time(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_quotes(conn, [("coinbase", START + 1.9, START + 1.5, 100.0, 1.0, 100.2, 1.0),
                                 ("coinbase", START + 4.2, START + 3.9, 101.0, 1.0, 101.2, 1.0),
                                 ("kraken", START + 2.1, START + 2.05, 99.9, 1.0, 100.1, 1.0)])
    feed = replay.quote_feed(conn, "exchanges", START, START + 10)
    prices = replay.price_at(feed, [START + 1.0, START + 2.0, START + 3.0, START + 5.0])
    assert np.isnan(prices[0]) and prices[1:].tolist() == pytest.approx([100.1, (100.1 + 100.0) / 2, (101.1 + 100.0) / 2])


def test_a_kraken_quote_with_a_stale_stamp_is_timed_by_its_arrival(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_quotes(conn, [("kraken", START + 2.1, START + 2.05, 99.0, 1.0, 101.0, 1.0),
                                 ("kraken", START + 3.1, START + 3.05, 101.0, 1.0, 103.0, 1.0),
                                 ("kraken", START + 4.1, START + 1.0, 103.0, 1.0, 105.0, 1.0)])    # Stamped as an older quote.
    feed = replay.quote_feed(conn, "kraken", START, START + 10)
    assert feed.times[0].tolist() == pytest.approx([START + 2.05, START + 3.05, START + 4.05])          # Arrival, 0.05 earlier.
    assert replay.price_at(feed, [START + 2.5, START + 4.5]).tolist() == [100.0, 104.0]


def test_a_moment_inside_the_averaging_minutes_first_second_counts_no_reading_taken():
    market = Market("kalshi", "T", "E", "KXBTC15M", OPEN, OPEN + 900, 80_000.0, None, "yes", None)
    feed = flat_feed(80_010.0)
    variance = np.full(2 * 60, 1e-9)
    close = market.close_time
    before, inside, first = replay.window_chances(feed, variance, START, market, np.array([close - 60.0, close - 59.6, close - 59.0]))
    assert inside == pytest.approx(before, abs=0.01) and first == pytest.approx(before, abs=0.01)    # No jump into the minute.



def test_the_settlement_error_is_the_stand_ins_average_on_brtis_level_over_kalshis():
    market = Market("kalshi", "T", "E", "KXBTC15M", OPEN, OPEN + 900, 80_000.0, 80_010.0, "yes", None)
    feed = flat_feed(80_050.0, [(OPEN + 800.0, 80_070.0)])                             # Runs 50 above BRTI at the open.
    assert replay.settlement_error(feed, market) == pytest.approx(np.log(80_020.0 / 80_010.0))
    assert np.isnan(replay.settlement_error(feed, replace(market, settlement=None)))


def test_the_tracking_variance_keeps_the_model_from_certainty_in_a_close_call():
    market = Market("kalshi", "T", "E", "KXBTC15M", OPEN, OPEN + 900, 80_000.0, None, "yes", None)
    feed = flat_feed(80_000.0, [(OPEN + 100.0, 80_002.0)])                             # $2 up through the whole minute.
    variance, t = np.full(2 * 60, 1e-9), np.array([OPEN + 899.5])
    exact = replay.window_chances(feed, variance, START, market, t)
    tracked = replay.window_chances(feed, variance, START, market, t, tracking=(3.5 / 80_000) ** 2)
    assert exact[0] > 0.99 and 0.6 < tracked[0] < 0.8
