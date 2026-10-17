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
    source    TEXT NOT NULL,        -- 'coinbase', 'kraken', 'bitstamp', 'gemini', 'bullish', 'cryptocom', 'hyperliquid', or a perp, as
                                    -- 'binance perp', to 2026-10-15.
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

-- The paper trader's orders, see paper/trader.py: what it would have sent and why, what the venue's book filled when the
-- order would have reached it, and, once the window settled, what that made. Times in seconds since 1970.
CREATE TABLE IF NOT EXISTS paper_orders (
    venue        TEXT NOT NULL,      -- 'kalshi', 'polymarket_us', or 'hyperliquid'.
    ticker       TEXT NOT NULL,      -- The venue's market: for Hyperliquid its Yes side's coin, '#1277300'.
    close_time   INTEGER NOT NULL,   -- When the window closes, which the venues' markets of a window share.
    decided      REAL NOT NULL,      -- When the order was decided, by our clock.
    side         TEXT NOT NULL,      -- 'yes' or 'no'.
    price        REAL NOT NULL,      -- Bitcoin's price the model read, the exchanges' average moved onto BRTI's level.
    model        REAL NOT NULL,      -- The model's chance of Up, recalibrated.
    mid          REAL NOT NULL,      -- Kalshi's mid then, an empty side counted as a bid of 0 or an ask of 1.
    fair         REAL NOT NULL,      -- The fair price of Up traded on, the model blended with Kalshi's mid, or Kalshi's mid
                                     -- alone on Hyperliquid, and on Polymarket US from 2026-10-16.
    edge         REAL NOT NULL,      -- The best price's edge after the venue's taker fee, in dollars a contract.
    limit_price  REAL NOT NULL,      -- The worst price, on the side bought, whose edge reached the threshold.
    offered      REAL NOT NULL,      -- Contracts at the limit or better as decided, less what earlier orders took.
    arrived      REAL NOT NULL,      -- When the order would have reached the venue, by the venue's clock.
    contracts    REAL NOT NULL,      -- Contracts the venue's book then filled, up to the order's size.
    cost         REAL,               -- Their average price, null when none filled.
    best         REAL,               -- The best price filled, for what one contract an order would make. Null when none filled.
    fees         REAL NOT NULL,      -- The taker fees on them, in dollars.
    expected     REAL NOT NULL,      -- The fair price's edge after the fee on the contracts filled, in dollars.
    result       TEXT,               -- 'yes' or 'no' once the venue settled the window, null until.
    profit       REAL,               -- What the contracts filled made after the fees, in dollars, once settled, Hyperliquid's
                                     -- taken from the winning contracts' dollars as they settle.
    cause        TEXT,               -- What brought the decision: the venue whose book of the window's market changed,
                                     -- 'kalshi' or 'polymarket_us', the exchange whose quote moved the price, as 'coinbase',
                                     -- 'time' for the once a second pricing, or 'other' for another market's book. Null for
                                     -- the orders before 2026-10-15, and added to the table after its first rows, see
                                     -- database.ADDED_COLUMNS.
    PRIMARY KEY (venue, ticker, decided, side)
);

-- Each stretch where buying one side of a window's market on a venue had a positive edge after the venue's taker fee, by
-- the fair price the paper trader trades on, one row as the stretch ends, as PredictionArbitrageTrader records its
-- opportunities, see paper/trader.py. Profits are expected ones, the fair price's edge, until the window settles and the
-- actual ones are set beside them, see database.settle_opportunities(). On Polymarket US and Hyperliquid from 2026-10-16,
-- as their rule trades, only those whose edge reached the threshold on a change of the venue's own book, and their columns
-- at the threshold over such stretches alone, see trader.Opportunity. Times in seconds since 1970 by our clock, dollars a
-- contract or in all as said.
CREATE TABLE IF NOT EXISTS opportunities (
    venue             TEXT NOT NULL,      -- 'kalshi', 'polymarket_us', or 'hyperliquid'.
    ticker            TEXT NOT NULL,      -- The venue's market.
    close_time        INTEGER NOT NULL,   -- When the window closes, which the venues' markets of a window share.
    side              TEXT NOT NULL,      -- 'yes' or 'no'.
    start_time        REAL NOT NULL,      -- When the edge went positive.
    end_time          REAL NOT NULL,      -- When it went back to zero or less, or the window could no longer be priced.
    seconds           REAL NOT NULL,
    peak_time         REAL NOT NULL,
    peak_edge         REAL NOT NULL,      -- The edge after the fee at the best price, in dollars a contract, at the peak.
    peak_price        REAL NOT NULL,      -- The best price then, on the side bought.
    peak_fair         REAL NOT NULL,      -- The fair price of the side then.
    peak_size         REAL NOT NULL,      -- Contracts at a positive edge, walking the book's levels, at the peak.
    peak_capital      REAL NOT NULL,      -- What they cost with the fees, in dollars.
    peak_profit       REAL NOT NULL,      -- Their expected profit after the fees, in dollars.
    book_changed      INTEGER NOT NULL,   -- 1 when the peak came with a change of the venue's own book, 0 when the fair price moved.
    days_held         REAL NOT NULL,      -- From the peak until the window pays.
    return_pct        REAL NOT NULL,      -- The peak's edge over what a contract ties up, its price and fee, as a percent.
    min_edge          REAL NOT NULL,      -- The edge the paper trader needs, which the next columns measure at.
    min_edge_seconds  REAL,               -- The longest unbroken stretch at min_edge or more, null without one.
    min_edge_size     REAL,               -- Contracts at min_edge or more through all of that stretch.
    min_edge_capital  REAL,               -- What they cost with the fees.
    min_edge_profit   REAL,               -- Their expected profit, at the stretch's thinnest moment.
    take_size         REAL,               -- Contracts one order could have had at min_edge or more, at the best moment, null
                                          -- without one.
    take_capital      REAL,               -- What they cost with the fees.
    take_profit       REAL,               -- Their expected profit.
    take_return_pct   REAL,               -- take_profit over take_capital, as a percent.
    min_edge_total    REAL,               -- Seconds at min_edge or more in all, over every stretch, null without one.
    result            TEXT,               -- How the venue settled the window, 'yes' or 'no', null until it has.
    actual_profit     REAL,               -- What the take made once settled: its winning contracts' dollars, less a venue's fee on
                                          -- them, less take_capital. Null until settled, or without a take.
    actual_return_pct REAL,               -- actual_profit over take_capital, as a percent.
    mode              TEXT NOT NULL DEFAULT 'paper',  -- 'live' when the venue was traded with real money as it was seen, by
                                          -- live/run.py, 'paper' when on paper, by paper/run.py.
    PRIMARY KEY (venue, ticker, side, start_time)
);
-- The last six columns came after the table had rows, see database.ADDED_COLUMNS, and so did the index of those not
-- settled, see database.ADDED_INDEXES. annual_pct, return_pct scaled to a year, which a 15 minute window always makes
-- huge, was dropped on 2026-10-16, see database.DROPPED_COLUMNS.

CREATE INDEX IF NOT EXISTS idx_opportunities_start ON opportunities (start_time);

-- Each decision to buy one side of a window's market on a venue with real money, or in its paper twin, as
-- PredictionArbitrageTrader keeps its trades, see live/executor.py: what was wanted, and in the end what its orders filled.
-- Times in seconds since 1970 by our clock.
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY,
    mode         TEXT NOT NULL,      -- 'live' for real orders, 'paper' for none.
    venue        TEXT NOT NULL,      -- 'hyperliquid', so far.
    ticker       TEXT NOT NULL,      -- The venue's market, for Hyperliquid its Yes side's coin.
    close_time   INTEGER NOT NULL,   -- When the window closes.
    side         TEXT NOT NULL,      -- 'yes' or 'no', the side bought.
    signal_ts    REAL NOT NULL,      -- When it was decided.
    fair         REAL NOT NULL,      -- The chance the side bought pays, as decided on.
    edge         REAL NOT NULL,      -- Dollars a contract at the best price, after the fee, at the signal.
    quantity     REAL NOT NULL,      -- Contracts wanted.
    limit_price  REAL NOT NULL,      -- The worst price a contract accepted.
    filled       REAL NOT NULL,      -- Contracts its orders filled.
    cost         REAL NOT NULL,      -- Dollars paid for them, fees included.
    fees         REAL NOT NULL,      -- Fees paid when they were bought.
    status       TEXT NOT NULL,      -- 'sent' while its order is out, then 'filled', 'partial', 'unfilled', or 'failed'.
    pays_at      REAL NOT NULL,      -- When the window pays.
    cause        TEXT                -- What brought the decision, as paper_orders.cause says. Added after the table was made,
                                     -- see database.ADDED_COLUMNS.
);

CREATE INDEX IF NOT EXISTS idx_trades_market ON trades (venue, ticker);

-- Every real order sent to a venue, stored before it goes out and updated with the venue's answer, as
-- PredictionArbitrageTrader keeps its orders. Times in seconds since 1970 by our clock.
CREATE TABLE IF NOT EXISTS orders (
    id             INTEGER PRIMARY KEY,
    trade_id       INTEGER NOT NULL,   -- The trade it was sent for, see trades.
    venue          TEXT NOT NULL,
    ticker         TEXT NOT NULL,      -- The trade's market.
    coin           TEXT NOT NULL,      -- What the order named: on Hyperliquid the side's coin, '#1277301' for No.
    action         TEXT NOT NULL,      -- 'buy' or 'sell'.
    outcome        TEXT NOT NULL,      -- 'yes' or 'no'.
    quantity       REAL NOT NULL,      -- Contracts asked for.
    limit_price    REAL NOT NULL,      -- The worst price a contract accepted, before fees.
    tif            TEXT NOT NULL,      -- How long it may wait: 'Ioc' fills at once or not at all, 'Alo' only rests, 'Gtc' either.
    client_id      TEXT NOT NULL,      -- Our id for the order, sent with it, so one whose answer was lost can be found.
    sent_at        REAL NOT NULL,
    status         TEXT NOT NULL,      -- 'sent' until the venue answers, then 'filled', 'partial', 'resting', 'unfilled',
                                       -- 'rejected', or 'error'.
    venue_order_id TEXT,               -- The venue's id for the order, once it answered.
    answered_at    REAL,
    latency_ms     INTEGER,            -- From sending the order to its answer.
    filled         REAL NOT NULL,      -- Contracts bought or sold.
    dollars        REAL NOT NULL,      -- Paid for a buy or received for a sale, fees included.
    fees           REAL NOT NULL,
    note           TEXT,               -- Why the venue refused the order, or the error.
    response       TEXT,               -- The venue's answer as JSON, for reconciling.
    book_at        REAL,               -- The venue's time for the market's newest book when the order went out.
    book_ts        REAL,               -- When that book reached us. sent_at less it is how stale the book was.
    venue_at       REAL                -- The venue's time for the order's first fill, so venue_at less sent_at is how long it
                                       -- took to reach the venue's book: on Hyperliquid from the account's fills, read as its
                                       -- window settles. Null for one that filled nothing, or until read. Added after the
                                       -- table was made, see database.ADDED_COLUMNS.
);

CREATE INDEX IF NOT EXISTS idx_orders_trade ON orders (trade_id);

-- How each trade paid out, one row once its window settled, as PredictionArbitrageTrader keeps its settlements, those
-- that filled nothing too, at nothing. A trade with no row here is still open. Times in seconds since 1970.
CREATE TABLE IF NOT EXISTS settlements (
    trade_id       INTEGER PRIMARY KEY,   -- The trade, see trades.
    mode           TEXT NOT NULL,         -- The trade's mode.
    settled_at     REAL NOT NULL,         -- The venue's settlement time.
    result         TEXT NOT NULL,         -- How the market settled, 'yes' or 'no'.
    payout         REAL NOT NULL,         -- Dollars the contracts held paid: one each when the side bought won.
    fees           REAL NOT NULL,         -- What the venue took at settlement.
    profit         REAL NOT NULL          -- payout less fees less the trade's cost.
);

-- Everything live trading told a human, by live/notify.py, whether or not the email went out, as PredictionArbitrageTrader
-- keeps its alerts. Times in seconds since 1970.
CREATE TABLE IF NOT EXISTS alerts (
    id       INTEGER PRIMARY KEY,
    time     REAL NOT NULL,      -- When it was raised.
    kind     TEXT NOT NULL,      -- 'halt' when live trading halted, see live/brakes.py.
    subject  TEXT NOT NULL,
    body     TEXT NOT NULL,
    sent_at  REAL,               -- When the email went out, null until it has.
    error    TEXT                -- Why it could not be sent, as no email settings in data/.
);
