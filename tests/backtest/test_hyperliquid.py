"""
Tests for the comparison of Hyperliquid's windows: the marks read at a moment and over a stretch, and the settlement
rules tried.
"""

import numpy as np
import pytest
from upordowntrader.backtest import hyperliquid as compare

MARKS = np.array([[100.0, 10.0, 11.0], [102.0, 20.0, 21.0], [104.0, 30.0, np.nan]])     # (received, mark, oracle).


def test_a_mark_is_in_force_from_when_it_came_and_interpolated_between_two():
    assert np.isnan(compare.in_force(MARKS, [99.0])[0]) and compare.in_force(MARKS, [100.0, 103.9]).tolist() == [10.0, 20.0]
    assert compare.interpolated(MARKS, 101.0) == pytest.approx(15.0) and np.isnan(compare.interpolated(MARKS, 105.0))
    assert compare.averaged(MARKS, 100, 104) == pytest.approx(15.0)        # Half the stretch at each.


def test_each_settlement_rule_reads_the_marks_its_own_way():
    rows = np.column_stack([np.arange(0.0, 200.0), np.arange(0.0, 200.0), np.full(200, 7.0)])
    close = 120
    assert [compare.settled_by(rows, close, rule) for rule in compare.SETTLE_RULES] == pytest.approx([120.0, 120.0, 89.5, 119.5, 7.0], abs=0.06)
