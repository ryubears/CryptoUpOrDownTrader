"""
HTTP requests shared by the exchange clients.

request() sends a GET through a host's rate limiter, and retries network errors, rate limit answers, and server errors,
waiting as long as the server asks when it says, so a client can page through a year of candles without minding either.
Any other error status raises RequestFailed with what the server sent. One httpx client keeps connections open between
calls, since paging through history makes thousands of calls to the same host.
"""

import httpx
import random
import threading
import time
from datetime import UTC

USER_AGENT = "CryptoUpOrDownTrader research (github.com/ryubears/CryptoUpOrDownTrader)"
TIMEOUT = 30            # Seconds a request may wait to connect, or between pieces of the answer.
RETRIES = 5             # Tries per request before giving up.
BACKOFF = 0.5           # Seconds before the first retry, doubling with each one after, up to MAX_BACKOFF.
MAX_BACKOFF = 30
MAX_RETRY_AFTER = 120   # The longest wait a server may ask for. Asked to wait longer, as a Binance IP ban does, the request fails.
RETRY_STATUSES = {418, 429, 500, 502, 503, 504}     # 418 is Binance's ban for ignoring its 429s.
CLIENT = httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})


class RequestFailed(Exception):
    """
    A request the server answered with an error status, with the status and the body it sent.
    """

    def __init__(self, status, body, url):
        super().__init__(f"HTTP {status} from {url}: {body[:300]}")
        self.status = status
        self.body = body
        self.url = url


class RateLimiter:
    """
    A token bucket for one host: lets a burst of requests through at once, then one every 1 / rate seconds. Safe to use
    from threads.
    """

    def __init__(self, rate, burst=None):
        self.rate = rate                            # Requests a second, on average.
        self.burst = burst or max(1, int(rate))     # Requests that may go at once after a quiet spell.
        self.tokens = float(self.burst)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        """
        Block until a request may be sent.
        """
        with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                time.sleep((1 - self.tokens) / self.rate)


def backoff(attempt):
    """
    Seconds to wait after failed try number attempt, counting from 1: BACKOFF doubled with each try up to MAX_BACKOFF,
    less up to half of it at random, so clients that failed together do not all retry together.
    """
    return min(MAX_BACKOFF, BACKOFF * 2 ** (attempt - 1)) * random.uniform(0.5, 1)


def retry_delay(response, attempt):
    """
    Seconds to wait before retrying a response with a status in RETRY_STATUSES: what its Retry-After header asks, or
    backoff() without one. None when the server asks for longer than MAX_RETRY_AFTER.
    """
    try:
        asked = float(response.headers["Retry-After"])
    except (KeyError, ValueError):      # No header, or the HTTP date form, which neither Binance nor Coinbase sends.
        return backoff(attempt)
    return asked if asked <= MAX_RETRY_AFTER else None


def request(url, params=None, limiter=None):
    """
    GET a URL through the limiter and return the response. Network errors and the statuses in RETRY_STATUSES are retried,
    RETRIES tries in all. Any other error status, or a retryable one that outlasts the tries, raises RequestFailed.
    """
    for attempt in range(1, RETRIES + 1):
        if limiter:
            limiter.wait()
        try:
            response = CLIENT.get(url, params=params)
        except httpx.TransportError:
            if attempt == RETRIES:
                raise
            time.sleep(backoff(attempt))
            continue
        if response.is_success:
            return response
        delay = retry_delay(response, attempt) if response.status_code in RETRY_STATUSES and attempt < RETRIES else None
        if delay is None:
            raise RequestFailed(response.status_code, response.text, str(response.request.url))
        time.sleep(delay)


def get_json(url, params=None, limiter=None):
    """
    GET a URL and parse the JSON body, see request().
    """
    return request(url, params, limiter).json()


def get_bytes(url, params=None, limiter=None):
    """
    GET a URL and return the body as bytes, see request().
    """
    return request(url, params, limiter).content


def utc(dt):
    """
    The datetime in UTC. A naive datetime is refused, since its time zone would be a guess.
    """
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime {dt!r}, pass one with a time zone")
    return dt.astimezone(UTC)
