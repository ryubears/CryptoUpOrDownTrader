"""
Tests for which market moves first: the lead found from changes at each lag, how far a source has moved ahead of the
average, the average's catch up, and the feeds read by arrival and by the venues' times.
"""

import numpy as np
import pytest
from upordowntrader.backtest import lead, replay
from upordowntrader.db import database


def test_the_correlation_peaks_at_the_lag_the_source_moves_first_by():
    a = np.random.default_rng(0).normal(0, 1, 20_000)
    b = np.concatenate([np.zeros(3), a[:-3]]) + np.random.default_rng(1).normal(0, 0.5, len(a))    # b follows 3 steps later.
    sums = {lag: np.zeros(6) for lag in range(-5, 6)}
    lead.add_pairs(sums, a[:10_000], b[:10_000])
    lead.add_pairs(sums, a[10_000:], b[10_000:])                                # Added a stretch at a time.
    seconds, corr, first = lead.peak(lead.correlations(sums))
    assert seconds == pytest.approx(3 * lead.STEP) and corr > 0.85 and first > 0.95


def test_a_sources_move_ahead_leaves_out_its_slow_gap():
    average = np.full(200, 100.0)
    source = np.where(np.arange(200) < 100, 150.0, 160.0)                       # 50 above, then 10 more from the 100th second.
    moved = lead.ahead(source, average, 1.0)
    assert np.isnan(moved[:60]).all() and moved[60:100] == pytest.approx(0.0)
    assert moved[[100, 130, 160]] == pytest.approx([10.0, 5.0, 0.0])            # As the gap's mean over 60 s catches up.


def test_the_catch_up_is_whole_once_the_average_has_followed():
    source = np.cumsum(np.random.default_rng(2).normal(0, 1, 50_000))
    average = np.concatenate([np.full(2, source[0]), source[:-2]])              # The average follows 2 seconds later.
    sums = {1: [0.0, 0.0], 5: [0.0, 0.0]}
    lead.add_catch_up(sums, lead.ahead(source, average, 1.0), average)
    assert sums[5][0] / sums[5][1] == pytest.approx(1.0, abs=0.1) and sums[1][0] / sums[1][1] == pytest.approx(0.5, abs=0.1)


def test_a_quote_counts_from_when_it_arrived_or_from_its_venue_time(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_quotes(conn, [("binance perp", 100.08, 100.0, 99.0, 1.0, 101.0, 1.0), ("binance perp", 110.08, 110.0, 101.0, 1.0, 103.0, 1.0)])
    arrived = lead.arrival_feed(conn, "binance perp", 0, 200)
    assert replay.price_at(arrived, [100.07, 100.08, 110.07, 110.08]).tolist()[1:] == [100.0, 100.0, 102.0]
    assert np.isnan(replay.price_at(arrived, [100.07])[0])
    stamped = replay.quote_feed(conn, "binance perp", 0, 200)
    assert lead.at_once(stamped, [99.99, 100.0, 110.0]).tolist()[1:] == [100.0, 102.0]       # By the venue's time, at once.


def test_kalshis_mid_counts_only_both_sides_within_the_widest_spread():
    book = replay.Book(np.array([1.0, 2.0, 3.0]), np.array([0.40, 0.30, np.nan]), np.array([0.42, 0.45, 0.50]), [], [])
    mids = lead.kalshi_mid(book, [0.5, 1.5, 2.5, 3.5])
    assert np.isnan(mids[[0, 2, 3]]).all() and mids[1] == pytest.approx(0.41)
