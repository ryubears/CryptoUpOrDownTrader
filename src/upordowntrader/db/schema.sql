-- The tables. database.connect() runs this on every connection, so a new table appears on its own.

-- A price series: one source's bars of one length for one symbol.
CREATE TABLE IF NOT EXISTS series (
    id       INTEGER PRIMARY KEY,
    source   TEXT NOT NULL,      -- 'coinbase'.
    symbol   TEXT NOT NULL,      -- The source's own name for the pair, 'BTC-USD'.
    seconds  INTEGER NOT NULL,   -- Bar length: 60 for Coinbase's 1 minute candles, 1 for a source with 1 second bars.
    UNIQUE (source, symbol, seconds)
);

-- One row per bar. Prices are in the quote currency, USD, and volume in bitcoin.
CREATE TABLE IF NOT EXISTS bars (
    series_id  INTEGER NOT NULL,
    time       INTEGER NOT NULL,   -- When the bar opens, in seconds since 1970, UTC.
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    trades     INTEGER,            -- How many trades the bar had, where the source says.
    PRIMARY KEY (series_id, time)
) WITHOUT ROWID;

-- The UTC days whose bars are wholly stored, so a rerun of the download skips them.
CREATE TABLE IF NOT EXISTS days (
    series_id  INTEGER NOT NULL,
    day        TEXT NOT NULL,      -- YYYY-MM-DD.
    bars       INTEGER NOT NULL,   -- How many bars the day had. A full day of 1 second bars has 86,400, of 1 minute bars 1,440.
    PRIMARY KEY (series_id, day)
) WITHOUT ROWID;

-- Each window market recorded, from the venue's market reads: when it trades, its strike, and how it settled.
CREATE TABLE IF NOT EXISTS markets (
    venue       TEXT NOT NULL,      -- 'kalshi'.
    ticker      TEXT NOT NULL,      -- 'KXBTC15M-26OCT101515-15'.
    event       TEXT NOT NULL,      -- 'KXBTC15M-26OCT101515', named for when it closes in US Eastern time.
    series      TEXT NOT NULL,      -- 'KXBTC15M'.
    open_time   INTEGER NOT NULL,   -- When it opens and closes, in seconds since 1970.
    close_time  INTEGER NOT NULL,
    strike      REAL,               -- The BRTI average of the 60 seconds before it opened. Null until it opens.
    settlement  REAL,               -- The BRTI average of the 60 seconds before it closed. Null until it settles.
    result      TEXT,               -- 'yes' when the settlement was at least the strike, 'no' otherwise. Null until it settles.
    settled_at  REAL,               -- When the venue settled it, in seconds since 1970.
    PRIMARY KEY (venue, ticker)
);

-- A recorded market's book, seen from the Yes side, each time its best bid or ask price moved, and once a second while
-- anything else in its kept levels changed, see record/recorder.py.
CREATE TABLE IF NOT EXISTS books (
    venue     TEXT NOT NULL,
    ticker    TEXT NOT NULL,
    received  REAL NOT NULL,        -- When the change reached us, in seconds since 1970.
    sent      REAL,                 -- When the venue's matching engine made it, from a delta. Null for a snapshot, which says no time.
    bid       REAL,                 -- The best Yes bid, null when there is none.
    bid_size  REAL,                 -- Contracts at it.
    ask       REAL,                 -- The best Yes ask, the best No bid restated, null when there is none.
    ask_size  REAL,
    bids      TEXT NOT NULL,        -- The best levels as JSON [[price, contracts], ...], best first.
    asks      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_books_ticker ON books (venue, ticker, received);

-- Stretches when a venue's feed was down, so its books could not be trusted.
CREATE TABLE IF NOT EXISTS gaps (
    venue       TEXT NOT NULL,
    start_time  REAL NOT NULL,      -- When the connection was lost, in seconds since 1970.
    end_time    REAL,               -- When a new connection was subscribed. Null if the recorder stopped first.
    PRIMARY KEY (venue, start_time)
);

-- Each exchange's best bid and ask, for a stand-in for BRTI, each time its best bid or ask price moved, and once a second
-- while only a size changed, as record/quotes.py stores them.
CREATE TABLE IF NOT EXISTS quotes (
    source    TEXT NOT NULL,        -- 'coinbase', 'kraken', 'bitstamp', 'gemini', 'bullish', 'cryptocom', or 'hyperliquid'.
    received  REAL NOT NULL,        -- When the quote reached us, in seconds since 1970 by our clock.
    sent      REAL,                 -- When the exchange says it was, where it does.
    bid       REAL,                 -- The best bid, null when there is none.
    bid_size  REAL,                 -- Bitcoin at it.
    ask       REAL,                 -- The best ask, null when there is none.
    ask_size  REAL
);

CREATE INDEX IF NOT EXISTS idx_quotes_source ON quotes (source, received);

-- A venue's perps' mark and oracle prices, each change as it came, which some venues' window markets settle on, as
-- record/marks.py stores them. Hyperliquid's say no time, so they go by ours.
CREATE TABLE IF NOT EXISTS marks (
    venue     TEXT NOT NULL,        -- 'hyperliquid'.
    coin      TEXT NOT NULL,        -- The perp, 'BTC'.
    received  REAL NOT NULL,        -- When the price reached us, in seconds since 1970 by our clock.
    mark      REAL NOT NULL,        -- The venue's mark price.
    oracle    REAL                  -- The venue's index of spot prices, null when it sent none.
);

CREATE INDEX IF NOT EXISTS idx_marks ON marks (venue, coin, received);

-- A window market's best Yes bid and ask minute by minute, from the venue's candlesticks, see history/markets.py.
CREATE TABLE IF NOT EXISTS minutes (
    venue   TEXT NOT NULL,
    ticker  TEXT NOT NULL,
    time    INTEGER NOT NULL,   -- When the minute ended, in seconds since 1970.
    bid     REAL,               -- The best Yes bid as the minute ended, null when there was none.
    ask     REAL,               -- The best Yes ask, null when there was none.
    price   REAL,               -- The last trade's price by then, null before any.
    volume  REAL,               -- Contracts traded in the minute.
    PRIMARY KEY (venue, ticker, time)
) WITHOUT ROWID;
