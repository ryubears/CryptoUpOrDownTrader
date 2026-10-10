"""
Tests for the history download, with stand ins for the exchanges.
"""

from datetime import UTC, date, datetime, timedelta
from upordowntrader.api.http import RequestFailed
from upordowntrader.db import database
from upordowntrader.history import download


def test_only_the_days_not_stored_are_fetched_and_an_unpublished_one_is_left(tmp_path, monkeypatch):
    fetched = []

    def fetch(day):
        fetched.append(day)
        if day == date(2026, 10, 4):
            raise RequestFailed(404, "<Error><Code>NoSuchKey</Code></Error>", "https://data.binance.vision/x")
        return [(int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()), 1.0, 1.0, 1.0, 1.0, 1.0, 1)]

    monkeypatch.setattr(download, "FETCHERS", {"binance": fetch})
    conn = database.connect(tmp_path / "history.sqlite")
    series = database.series_id(conn, "binance", "BTCUSDT", 1)
    database.store_day(conn, series, date(2026, 10, 2), [])
    stored = download.download(conn, "binance", date(2026, 10, 1), date(2026, 10, 5), log=lambda message: None)
    assert sorted(fetched) == [date(2026, 10, 1), date(2026, 10, 3), date(2026, 10, 4)]
    assert stored == 2 and set(database.stored_days(conn, series)) == {date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)}


def test_today_is_never_downloaded(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(download, "FETCHERS", {"binance": lambda day: fetched.append(day) or []})
    today = datetime.now(UTC).date()
    download.download(database.connect(tmp_path / "history.sqlite"), "binance", today - timedelta(days=1), today + timedelta(days=2),
                      log=lambda message: None)
    assert fetched == [today - timedelta(days=1)]
