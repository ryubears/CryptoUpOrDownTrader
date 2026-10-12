"""
Tests for the recorder of perps' mark prices.
"""

from upordowntrader.db import database
from upordowntrader.record.marks import MarkRecorder


def test_each_change_of_a_perps_prices_is_stored_and_a_repeat_is_not(tmp_path):
    conn = database.connect(tmp_path / "history.sqlite")
    marks = MarkRecorder(conn, "hyperliquid")
    for coin, mark, oracle in [("BTC", 83802.3, 83840.4), ("BTC", 83802.3, 83840.4), ("BTC", 83803.0, 83840.4), ("ETH", 2555.2, None)]:
        marks.on_mark(coin, mark, oracle)
    marks.flush()
    rows = conn.execute("SELECT venue, coin, mark, oracle FROM marks ORDER BY rowid").fetchall()
    assert rows == [("hyperliquid", "BTC", 83802.3, 83840.4), ("hyperliquid", "BTC", 83803.0, 83840.4), ("hyperliquid", "ETH", 2555.2, None)]
    assert database.read_marks(conn, "hyperliquid", "BTC", 0, 2e9)[:, 1].tolist() == [83802.3, 83803.0]
    assert marks.status() == "hyperliquid BTC 3 marks, 2 stored, ETH 1 marks, 1 stored"
