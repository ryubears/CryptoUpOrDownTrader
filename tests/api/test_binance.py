"""
Tests for the Binance client, with stand ins for its REST API and its archive.
"""

import hashlib
import httpx
import io
import pytest
import zipfile
from datetime import UTC, date, datetime, timedelta
from upordowntrader.api import binance, http

T0 = datetime(2026, 10, 1, tzinfo=UTC)
T0_MS = int(T0.timestamp() * 1000)


def kline_row(open_time, length=1000):
    """
    A REST or archive row for a kline opening at open_time and lasting length, both in milliseconds or both in microseconds.
    """
    return [open_time, "100.0", "101.5", "99.5", "100.5", "1.25", open_time + length - 1, "125.6", 7, "0.5", "50.2", "0"]


def klines_api(first, count, length):
    """
    A stand in for the klines endpoint, holding count klines from first, each length milliseconds long.
    """
    def answer(request):
        params = request.url.params
        start, end, limit = int(params["startTime"]), int(params["endTime"]), int(params["limit"])
        rows = [kline_row(t, length) for t in range(first, first + count * length, length) if start <= t <= end]
        return httpx.Response(200, json=rows[:limit])

    return answer


def zipped(name, rows):
    """
    An archive file: a zip holding one CSV of the rows, named as Binance names it.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"{name}.csv", "".join(",".join(str(v) for v in row) + "\n" for row in rows))
    return buffer.getvalue()


def archive_server(files):
    """
    A stand in for the archive, serving files by path and answering 404 for any other, as its S3 bucket does.
    """
    def answer(request):
        if request.url.path in files:
            return httpx.Response(200, content=files[request.url.path])
        return httpx.Response(404, text="<Error><Code>NoSuchKey</Code></Error>")

    return answer


def published(path, payload):
    """
    The files Binance publishes for an archive file: the zip and its checksum, keyed by their URL paths.
    """
    checksum = f"{hashlib.sha256(payload).hexdigest()}  {path.rsplit('/', 1)[-1]}\n".encode()
    return {path: payload, f"{path}.CHECKSUM": checksum}


def test_a_row_parses_into_a_kline():
    assert binance.parse_kline(kline_row(T0_MS)) == binance.Kline(
        open_time=T0, open=100.0, high=101.5, low=99.5, close=100.5, volume=1.25, quote_volume=125.6, trade_count=7,
        taker_buy_volume=0.5, taker_buy_quote_volume=50.2)


def test_klines_pages_through_the_whole_range(serve):
    sent = serve(klines_api(T0_MS, 2500, 1000))
    klines = list(binance.klines("BTCUSDT", "1s", T0, T0 + timedelta(seconds=2500)))
    assert [k.open_time for k in klines] == [T0 + timedelta(seconds=i) for i in range(2500)]
    assert len(sent) == 3                                   # 1000, 1000, then 500.
    assert sent[0].url.path == "/api/v3/klines" and sent[0].url.params["symbol"] == "BTCUSDT"


def test_klines_leave_out_the_one_opening_at_the_end(serve):
    serve(klines_api(T0_MS, 100, 60_000))
    klines = list(binance.klines("BTCUSDT", "1m", T0, T0 + timedelta(minutes=10)))
    assert len(klines) == 10 and klines[-1].open_time == T0 + timedelta(minutes=9)


def test_an_interval_binance_does_not_offer_is_refused():
    with pytest.raises(ValueError, match="unsupported interval"):
        binance.fetch_klines("BTCUSDT", "2s")


def test_a_daily_archive_is_checked_then_parsed_with_its_microsecond_timestamps(serve):
    path = "/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2026-10-01.zip"
    rows = [kline_row(T0_MS * 1000 + i * 1_000_000, 1_000_000) for i in range(3)]      # Files from 2025 on count microseconds.
    sent = serve(archive_server(published(path, zipped("BTCUSDT-1s-2026-10-01", rows))))
    klines = list(binance.daily_archive("BTCUSDT", "1s", date(2026, 10, 1)))
    assert [k.open_time for k in klines] == [T0 + timedelta(seconds=i) for i in range(3)]
    assert sent[0].url.host == "data.binance.vision"


def test_a_monthly_archive_is_parsed_with_its_millisecond_timestamps(serve):
    path = "/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-03.zip"
    march = datetime(2024, 3, 1, tzinfo=UTC)
    serve(archive_server(published(path, zipped("BTCUSDT-1m-2024-03", [kline_row(int(march.timestamp() * 1000), 60_000)]))))
    (kline,) = binance.monthly_archive("BTCUSDT", "1m", 2024, 3)
    assert kline.open_time == march


def test_an_archive_with_the_wrong_checksum_is_refused(serve):
    path = "/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-2026-10-01.zip"
    serve(archive_server({path: zipped("BTCUSDT-1s-2026-10-01", [kline_row(T0_MS)]), f"{path}.CHECKSUM": b"deadbeef  x.zip"}))
    with pytest.raises(binance.ChecksumMismatch):
        binance.daily_archive("BTCUSDT", "1s", date(2026, 10, 1))


def test_an_archive_not_yet_published_raises_404(serve):
    serve(archive_server({}))
    with pytest.raises(http.RequestFailed) as failed:
        binance.daily_archive("BTCUSDT", "1s", date(2026, 10, 9))
    assert failed.value.status == 404
