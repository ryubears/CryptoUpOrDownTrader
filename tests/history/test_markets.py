"""
Tests for the download of Kalshi's settled windows and their minutes, with a stand in for Kalshi.
"""

from upordowntrader.db import database
from upordowntrader.db.models import Market
from upordowntrader.history import markets

OPEN = 1_786_490_100        # A quarter hour, in seconds since 1970.


def window(i, result="yes"):
    """
    The i-th window from OPEN.
    """
    return Market("kalshi", f"KXBTC15M-{i}", f"KXBTC15M-E{i}", "KXBTC15M", OPEN + 900 * i, OPEN + 900 * (i + 1), 80_000.0, 80_001.0,
                  result, None)


def test_a_run_holds_as_many_windows_as_their_minutes_allow():
    found = markets.runs([window(i) for i in range(10)], limit=100)
    # Two windows span 31 minutes, 62 counted; three span 46, 138 counted, too many.
    assert [[m.ticker for m in run] for run in found] == [[f"KXBTC15M-{i}", f"KXBTC15M-{i + 1}"] for i in range(0, 10, 2)]
    assert markets.runs([]) == []


def test_only_settled_windows_without_minutes_are_fetched(tmp_path, monkeypatch):
    conn = database.connect(tmp_path / "history.sqlite")
    database.store_minutes(conn, "kalshi", {"KXBTC15M-0": [(OPEN + 60, 0.5, 0.51, 0.5, 10.0)]})
    asked = []

    def minute_quotes(tickers, start, end):
        asked.append((tickers, start, end))
        return {t: [(start + 60, 0.4, 0.42, None, 0.0)] for t in tickers if t != "KXBTC15M-1"}     # Settled too long ago.

    monkeypatch.setattr(markets.kalshi, "historical_windows", lambda series: [window(1), window(0)])
    monkeypatch.setattr(markets.kalshi, "windows", lambda series, start, end: [window(2), window(3, result=None)])
    monkeypatch.setattr(markets.kalshi, "minute_quotes", minute_quotes)
    monkeypatch.setattr(markets.kalshi, "historical_minute_quotes",
                        lambda ticker, start, end: asked.append(ticker) or [(start + 60, 0.3, 0.32, None, 1.0)])
    assert markets.download(conn, log=lambda message: None) == 2
    assert asked == [(["KXBTC15M-1", "KXBTC15M-2"], OPEN + 900, OPEN + 2700), "KXBTC15M-1"]    # Not 0, with minutes, nor 3, unsettled.
    assert len(database.load_markets(conn, "kalshi")) == 4
    assert sorted(database.read_minutes(conn, "kalshi")) == ["KXBTC15M-0", "KXBTC15M-1", "KXBTC15M-2"]
