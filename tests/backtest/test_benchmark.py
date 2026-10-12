"""
Tests for the benchmark against Kalshi's market: the marks read from its minutes, the gap to BRTI, the walk forward, and
the trades at the minute's quotes.
"""

import numpy as np
import pytest
from synthetic import FIRST_DAY, day_start, random_walk_history
from upordowntrader.backtest import benchmark
from upordowntrader.db.models import Market
from upordowntrader.model import fair

START = day_start(FIRST_DAY, 0)


def window(i, strike, result="yes"):
    """
    The i-th window of the made up history's second day.
    """
    open_time = START + 86400 + 900 * i
    return Market("kalshi", f"T{i}", f"E{i}", "KXBTC15M", open_time, open_time + 900, strike, strike + 1, result, None)


def test_marks_take_the_minutes_closing_quotes_and_coinbases_close_then():
    history = random_walk_history(2, 1e-4)
    w = window(0, 80_000.0)
    minutes = {"T0": np.array([[w.open_time + 60, 0.40, 0.42], [w.open_time + 120, 0.40, 0.60], [w.open_time + 180, np.nan, 0.5]])}
    m = benchmark.marks([w], minutes, history, START)
    assert len(m.window) == benchmark.MARKS and m.seconds_left[:3].tolist() == [840, 780, 720]
    assert m.mid[0] == pytest.approx(0.41) and np.isnan(m.mid[1:]).all()        # Too wide a spread, a side missing, no minute.
    assert m.price[0] == history.minute_close[(w.open_time + 60 - START) // 60 - 1] and m.up.all() and (m.day == 0).all()


def test_a_window_is_scored_only_with_every_coinbase_minute_and_a_believable_strike():
    history = random_walk_history(2, 1e-4)
    windows = [window(0, 80_000.0), window(1, 7.24), window(2, 80_000.0)]       # Kalshi set a strike as 7.24 once.
    history.minute_missing[(windows[2].open_time - START) // 60 + 5] = 60         # And Coinbase lost a minute in the third.
    assert benchmark.scorable(windows, history, START).tolist() == [True, False, False]


def test_the_gap_to_brti_is_averaged_over_the_opens_before():
    history = random_walk_history(2, 1e-4)
    windows = [window(i, 80_000.0) for i in range(3)]
    gaps, averaged = benchmark.brti_gaps(windows, history, START, half_life=1)
    assert gaps[0] == pytest.approx(history.minute_average[(windows[0].open_time - START) // 60 - 1] - 80_000.0)
    assert averaged[0] == pytest.approx(gaps[0]) and averaged[1] == pytest.approx((gaps[0] + gaps[1]) / 2)


def made_up_marks(days, per_day=20):
    """
    Marks of one window a mark, every made up window on its own day, priced by a market that is right, and up whenever the
    market says over a half.
    """
    n = days * per_day
    rng = np.random.default_rng(0)
    mid = rng.uniform(0.2, 0.8, n)
    return benchmark.Marks(windows=[None] * n, window=np.arange(n), seconds_left=np.full(n, 600), bid=mid - 0.01, ask=mid + 0.01, mid=mid,
                           price=np.ones(n), strike=np.ones(n), up=mid > 0.5, minute=np.zeros(n, dtype=int), day=np.arange(n) // per_day)


def test_the_walk_forward_scores_nothing_before_its_first_fit():
    m = made_up_marks(days=benchmark.TRAIN_DAYS + 10)
    recalibrated, blend = benchmark.walk_forward(np.zeros(len(m.window)), m)
    first = m.day < benchmark.TRAIN_DAYS
    assert np.isnan(recalibrated[first]).all() and not np.isnan(recalibrated[~first]).any() and not np.isnan(blend[~first]).any()


def test_trades_take_the_quote_when_the_edge_after_the_fee_reaches_the_threshold():
    m = made_up_marks(days=1, per_day=2)
    m.bid[:], m.ask[:], m.up[:] = 0.49, 0.51, [True, True]
    count, mean, _ = benchmark.trades(np.array([0.60, 0.40]), m, 0.05, np.ones(2, dtype=bool))
    assert count == 2 and mean == pytest.approx(fair.edge(1.0, 0.51) * 0.5 + fair.edge(0.0, 0.51) * 0.5)   # Yes won, No lost.
    assert benchmark.trades(np.array([0.53, 0.47]), m, 0.05, np.ones(2, dtype=bool))[0] == 0


def test_coinbases_settlement_is_its_average_of_the_last_minute_less_the_gap_at_the_open():
    history = random_walk_history(2, 1e-4)
    windows = [window(0, 80_000.0), window(1, 80_000.0)]
    last = (windows[1].close_time - START) // 60 - 1
    history.minute_missing[last] = 60                                             # Coinbase lost the second's last minute.
    errors = benchmark.settlement_errors(windows, history, START, np.array([5.0, 6.0]))
    average = history.minute_average[(windows[0].close_time - START) // 60 - 1]
    assert errors[0] == pytest.approx(np.log((average - 5.0) / windows[0].settlement)) and np.isnan(errors[1])
