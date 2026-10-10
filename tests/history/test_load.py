"""
Tests for turning stored bars into a History: minute bars, missing data, and reading from the database.
"""

import numpy as np
import pytest
import synthetic
from datetime import timedelta
from upordowntrader.db import database
from upordowntrader.history import load

START = synthetic.day_start(synthetic.FIRST_DAY, 0)


def test_forward_fill_carries_the_last_number_and_starts_from_before():
    filled = load.forward_fill(np.array([np.nan, 2.0, np.nan, np.nan, 5.0]), 1.0)
    assert filled.tolist() == [1.0, 2.0, 2.0, 2.0, 5.0]


def test_a_day_of_seconds_becomes_minute_bars_with_its_gaps_filled_and_counted():
    rows = synthetic.random_walk_rows(1, 1e-4)[0]
    kept = np.ones(load.DAY, dtype=bool)
    kept[65:70] = False                                     # Five seconds missing in the second minute.
    close, minute = load.from_seconds(rows[kept], START, rows[0, 1])
    assert close[65:70].tolist() == [rows[64, 4]] * 5      # The price before the gap carries through it.
    assert minute["missing"][:3].tolist() == [0, 5, 0]
    assert minute["open"][0] == rows[0, 1] and minute["close"][0] == rows[59, 4]
    assert minute["high"][0] == rows[:60, 2].max() and minute["low"][0] == rows[:60, 3].min()
    assert minute["average"][0] == pytest.approx(rows[:60, 4].mean())
    assert minute["volume"][1] == pytest.approx(rows[60:120][kept[60:120], 5].sum())
    returns = np.diff(np.log(rows[:, 4]), prepend=np.log(rows[0, 1]))
    assert minute["realized"][0] == pytest.approx((returns[:60] ** 2).sum())


def test_load_reads_minute_bars_and_counts_a_day_not_stored_as_missing(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    series = database.series_id(conn, "coinbase", "BTC-USD", 60)
    for day in (0, 2):                                      # The day between is not stored.
        start = synthetic.day_start(synthetic.FIRST_DAY, day)
        bars = [(start + 60 * m, 100.0 + m, 101.0 + m, 99.0 + m, 100.5 + m, 2.0, None) for m in range(load.MINUTES)]
        database.store_day(conn, series, synthetic.FIRST_DAY + timedelta(days=day), bars)
    history = load.load(conn, "coinbase", log=lambda message: None)
    assert history.days() == 3 and history.second_close is None and history.minute_realized is None
    assert history.minute_missing.reshape(3, -1).sum(axis=1).tolist() == [0, 60 * load.MINUTES, 0]
    assert history.minute_close[load.MINUTES] == history.minute_close[load.MINUTES - 1]     # Carried over the missing day.
    assert history.minute_average[0] == pytest.approx((100 + 101 + 99 + 100.5) / 4)
