-- The tables. database.connect() runs this on every connection, so a new table appears on its own.

-- A price series: one source's bars of one length for one symbol.
CREATE TABLE IF NOT EXISTS series (
    id       INTEGER PRIMARY KEY,
    source   TEXT NOT NULL,      -- 'binance' or 'coinbase'.
    symbol   TEXT NOT NULL,      -- The source's own name for the pair, 'BTCUSDT' or 'BTC-USD'.
    seconds  INTEGER NOT NULL,   -- Bar length: 1 for Binance's 1 second klines, 60 for Coinbase's 1 minute candles.
    UNIQUE (source, symbol, seconds)
);

-- One row per bar. Prices are in the quote currency, USDT or USD, and volume in bitcoin.
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
