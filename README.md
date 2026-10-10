# CryptoUpOrDownTrader

A trading bot for the Bitcoin up or down markets on Kalshi and Polymarket
US, which ask whether bitcoin will be higher or lower at the end of a 15
minute window than at its start. Both settle on CF Benchmarks' BRTI: the
average of its 60 one second readings in the minute before the window
opens, against the same average in the minute before it closes.

The plan is to price each contract from how much bitcoin has been moving
lately, the way a digital option is priced, to check on years of history
that the prices it gives come true as often as they say, and to trade a
fixed amount once per market when the market's price is off from ours by
more than the fees and the model's own error.

## How it works

Everything lives in `src/upordowntrader/`, a package without `__init__.py`
files, imported as `from upordowntrader.api import binance`. So far that is
the exchange clients in `api/`, which read bitcoin's price history for the
calibration. They are thin: requests, paging, rate limits, and retries, and
no database code.

`api/binance.py` reads Binance's klines, down to 1 second, from its REST
API or its bulk archive. api.binance.com refuses US IPs, so the REST calls
go to data-api.binance.vision, Binance's public mirror of its market data
endpoints. The archive at data.binance.vision keeps one zipped CSV per day
or month, checked against the SHA-256 Binance publishes beside it, and
reads a day of 1 second klines in about 3 seconds, a year in about 18
minutes. Binance prices bitcoin in USDT, which differs from the USD the
settlement index uses by typically under 0.1%, too little to matter for
how much the price moves.

`api/coinbase.py` reads Coinbase's candles, down to 1 minute, 300 to a
call, a year in about 12 minutes. Coinbase is one of the exchanges BRTI is
built from, so its price is the closest free stand-in for the settlement
price.

`api/http.py` is what they share: one connection pool, a rate limiter per
host, and retries of network errors, rate limit answers, and server
errors, waiting as long as the server asks.

The `data/` folder holds the local databases and API keys, and is not
committed.

## Running it

It needs [uv](https://docs.astral.sh/uv/). `uv sync` installs it, `uv run
pytest` runs the tests, `uv run pytest -m network` runs the smoke tests
against the real exchanges, and `uv run ruff check` lints and checks the
imports' order. The code is formatted by hand, so `ruff format` is not
used.
