"""
Tests for the shared GET requests' retries and rate limiting, with a stand in for the server.
"""

import httpx
import pytest
from datetime import UTC, datetime, timedelta, timezone
from upordowntrader.api import http

URL = "https://api.example.com/v1/klines"


def test_the_rate_limiter_lets_a_burst_through_then_spaces_requests(monkeypatch):
    clock, waits = [0.0], []

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(http.time, "sleep", sleep)
    limiter = http.RateLimiter(2, burst=2)
    for _ in range(4):
        limiter.wait()
    assert waits == [0.5, 0.5]          # The first two go at once, then one every half second.


def test_a_429_is_retried_after_the_wait_the_server_asks_for(serve, slept):
    answers = iter([httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, json={"ok": 1})])
    serve(lambda request: next(answers))
    assert http.get_json(URL) == {"ok": 1}
    assert slept == [3.0]


def test_a_post_sends_its_body_as_json(serve):
    sent = serve(lambda request: httpx.Response(200, json={"ok": 1}))
    assert http.post_json(URL, {"type": "outcomeMeta"}) == {"ok": 1}
    assert sent[0].method == "POST" and sent[0].headers["content-type"] == "application/json" and sent[0].content == b'{"type":"outcomeMeta"}'


def test_network_errors_are_retried_until_an_answer_comes(serve, slept):
    answers = [httpx.ConnectError("refused"), httpx.ConnectError("refused"), httpx.Response(200, json=[1])]

    def answer(request):
        next_answer = answers.pop(0)
        if isinstance(next_answer, Exception):
            raise next_answer
        return next_answer

    serve(answer)
    assert http.get_json(URL) == [1]
    assert len(slept) == 2


def test_an_error_status_raises_at_once_with_the_body(serve, slept):
    sent = serve(lambda request: httpx.Response(400, json={"code": -1120, "msg": "Invalid interval."}))
    with pytest.raises(http.RequestFailed) as failed:
        http.get_json(URL, {"interval": "2s"})
    assert failed.value.status == 400 and "Invalid interval." in failed.value.body
    assert failed.value.url == URL + "?interval=2s"
    assert len(sent) == 1 and slept == []      # Never retried.


def test_a_server_error_that_outlasts_the_retries_raises(serve, slept):
    sent = serve(lambda request: httpx.Response(503))
    with pytest.raises(http.RequestFailed) as failed:
        http.get_json(URL)
    assert failed.value.status == 503
    assert len(sent) == http.RETRIES and len(slept) == http.RETRIES - 1


def test_a_wait_longer_than_max_retry_after_raises_rather_than_waits(serve, slept):
    serve(lambda request: httpx.Response(418, headers={"Retry-After": "7200"}))      # Binance's IP ban, for two hours.
    with pytest.raises(http.RequestFailed):
        http.get_json(URL)
    assert slept == []


def test_utc_converts_a_datetime_with_a_time_zone_and_refuses_a_naive_one():
    eastern = timezone(timedelta(hours=-4))
    assert http.utc(datetime(2026, 1, 1, tzinfo=eastern)) == datetime(2026, 1, 1, 4, tzinfo=UTC)
    with pytest.raises(ValueError):
        http.utc(datetime(2026, 1, 1))
