"""
Hyperliquid public market data, read without an account: its up or down outcome markets on BTC, their order books, the
BTC perp's mark price, and its best bid and ask.

Hyperliquid's outcome markets, HIP-4, pay $1 a contract on one side when they settle, like Kalshi's. Two series of up or
down windows run on BTC, see windows(), and the same on ETH and HYPE, left out of UNDERLYINGS for now:

    15 minute   run by the deployer Hyperliquid calls 'out', one window a quarter hour, the same windows as Kalshi's
                KXBTC15M. Each quarter hour it settles the window just closed, about 8 seconds after, then lists the next,
                its strike the perp's mark price then, about 15 to 25 seconds after. It settles on the perp's mark price,
                with a 60 second window its rules describe no further, see backtest/hyperliquid.py.
    daily       run by Hyperliquid itself, settling at 06:00 UTC to the perp's mark price interpolated to that moment,
                its strike set as it opens.

A market leaves the list once it settles, so how it settled is read from Hyperliquid's explorer, which keeps each
account's last 300 actions: those of the 15 minute series' accounts reach back about 12 hours, so the windows settled in
them are read whole, strike and all, even those never seen listed. The daily series has no account, so the recorder
works its settlement out from the marks, by Hyperliquid's rule.

Each market has a book for each side, Yes and No, merged, so a Yes bid at p is a No ask at 1 - p and the Yes side's
book is the whole of it. A side's coin is '#' and 10 times the market's number, plus 1 for No. The books are thin: on
2026-10-12 a few tens of contracts a level. The perp's mark comes about once a second without a time; its best bid and
ask come as they change, with Hyperliquid's time. This is the only file that knows Hyperliquid's field
names and message formats.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from upordowntrader.api.bookstream import BookStream
from upordowntrader.api.http import RateLimiter, post_json
from upordowntrader.db.models import Market

INFO_URL = "https://api.hyperliquid.xyz/info"
EXPLORER_URL = "https://rpc.hyperliquid.xyz/explorer"
WS_URL = "wss://api.hyperliquid.xyz/ws"
LIMITER = RateLimiter(2)        # Hyperliquid allows 1200 weight a minute per IP, 20 for these reads; this keeps far under.
UNDERLYINGS = ("BTC",)      # The perps whose up or down windows are read. ETH and HYPE have the same series.
FIFTEEN_DEPLOYER = "out"        # The deployer of the 15 minute series.
FIFTEEN_SECONDS = 900
PERIODS = {"15m": 900, "1h": 3600, "1d": 86400}     # A recurring series' period, as Hyperliquid writes it, in seconds.
CANDLE_LIMIT = 5000         # The most recent candles Hyperliquid keeps of each length.


# MARKETS

def described(text):
    """
    A market's description, 'perp:BTC|threshold:83474|time:20261012-1115', as a dict.
    """
    return dict(part.split(":", 1) for part in text.split("|") if ":" in part)


def expiry(text):
    """
    A market's time to settle, '20261012-1115' in UTC, in seconds since 1970.
    """
    return int(datetime.strptime(text, "%Y%m%d-%H%M").replace(tzinfo=UTC).timestamp())


def coin(outcome, side=0):
    """
    The coin of a market's side, 0 for Yes and 1 for No, as books and trades name it: '#113760'.
    """
    return f"#{10 * outcome + side}"


def outcome_of(ticker):
    """
    The market number of a Yes side's coin.
    """
    return int(ticker[1:]) // 10


def window(o):
    """
    The Market of one of outcomeMeta's outcomes when it is a window of one of the up or down series on UNDERLYINGS, else
    None. A series is named for who runs it, the underlying, and the period: 'out-BTC-15m', 'recurring-BTC-1d'.
    """
    d = described(o.get("description", ""))
    if o.get("name") == "template:binaryPrice" and o.get("venue") == FIFTEEN_DEPLOYER and d.get("perp") in UNDERLYINGS \
            and d.get("priceDescription") == f"{d['perp']}-USDC perp mark":
        series, close, period, strike = f"{FIFTEEN_DEPLOYER}-{d['perp']}-15m", expiry(d["time"]), FIFTEEN_SECONDS, d.get("threshold")
        if close % FIFTEEN_SECONDS:
            return None                                 # Not on a quarter hour, so not one of the series' windows.
    elif o.get("name") == "Recurring" and d.get("class") == "priceBinary" and d.get("underlying") in UNDERLYINGS and d.get("period") in PERIODS:
        series, close, period, strike = f"recurring-{d['underlying']}-{d['period']}", expiry(d["expiry"]), PERIODS[d["period"]], d.get("targetPrice")
    else:
        return None
    return Market("hyperliquid", coin(o["outcome"]), f"{series}-{d.get('time') or d.get('expiry')}", series, close - period, close,
                  float(strike) if strike else None, None, None, None)


def outcome_meta():
    """
    Hyperliquid's list of outcome markets not settled yet, with who deploys them.
    """
    return post_json(INFO_URL, {"type": "outcomeMeta"}, LIMITER)


def windows():
    """
    The windows of the up or down series listed now, not settled yet, see window().
    """
    return [m for m in (window(o) for o in outcome_meta().get("outcomes", [])) if m]


def accounts(meta, venue, action):
    """
    The accounts that take an action for a deployer, such as 'settleOutcome', from outcomeMeta's deployers.
    """
    return sorted({user for d in meta.get("deployers", []) if d.get("venue") == venue
                   for name, users in d.get("subDeployers", []) if name == action for user in users})


def actions(venue, action):
    """
    The last 300 explorer actions of each account that takes an action for a deployer.
    """
    return [tx for user in accounts(outcome_meta(), venue, action)
            for tx in post_json(EXPLORER_URL, {"type": "userDetails", "user": user}, LIMITER).get("txs", [])]


def result(fraction):
    """
    A market's result from the fraction of $1 its Yes side settled to: 'yes' for 1, 'no' for 0, and any other, which no
    up or down window should settle to, as the fraction written out.
    """
    return {1.0: "yes", 0.0: "no"}.get(float(fraction), f"{float(fraction):g}")


def settled(txs):
    """
    The windows of the up or down series an account settled, see window(), from its explorer actions, each with its
    result and when it settled.
    """
    found = []
    for tx in txs:
        action = tx.get("action") or {}
        settle = (action.get("operation") or {}).get("settleOutcome")
        if tx.get("error") is None and settle and len(settle.get("nameAndDescription") or []) == 2:
            name, description = settle["nameAndDescription"]
            m = window({"outcome": settle["outcome"], "name": name, "description": description, "venue": action.get("venue")})
            if m:
                found.append(replace(m, result=result(settle["settleFraction"]), settled_at=tx["time"] / 1000))
    return found


def settlements(venue=FIFTEEN_DEPLOYER):
    """
    The windows a deployer settled lately, see settled(), from each of its settling accounts' last 300 actions.
    """
    return settled(actions(venue, "settleOutcome"))


def listed(txs):
    """
    When an account listed each window of the up or down series, from its explorer actions: {(series, close time,
    strike): seconds since 1970}.
    """
    found = {}
    for tx in txs:
        action = tx.get("action") or {}
        register = (action.get("operation") or {}).get("registerStandaloneOutcomeFromTemplate")
        if tx.get("error") is None and register:
            description = "|".join(f"{key}:{value}" for key, value in register.get("keywordToValue") or [])
            m = window({"outcome": 0, "name": f"template:{register.get('id')}", "description": description, "venue": action.get("venue")})
            if m:
                found[(m.series, m.close_time, m.strike)] = tx["time"] / 1000
    return found


def listings(venue=FIFTEEN_DEPLOYER):
    """
    When a deployer listed its windows lately, see listed(), from each of its listing accounts' last 300 actions.
    """
    return listed(actions(venue, "registerStandaloneOutcomeFromTemplate"))


def minute_candles(perp, start, end):
    """
    A perp's 1 minute candles opening from start up to end, in seconds since 1970, as (open time, open, high, low, close,
    volume, trades) tuples, oldest first. Hyperliquid keeps only the last CANDLE_LIMIT, about three and a half days.
    """
    rows = post_json(INFO_URL, {"type": "candleSnapshot", "req": {"coin": perp, "interval": "1m", "startTime": int(start * 1000),
                                                                   "endTime": int(end * 1000)}}, LIMITER)
    return sorted((c["t"] // 1000, float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]), float(c["v"]), c.get("n")) for c in rows)


# STREAMING

def levels(side, depth):
    """
    A side of a book message, [{"px": "0.13", "sz": "20.0", "n": 1}, ...] best first, as [[price, size], ...].
    """
    return [[float(level["px"]), float(level["sz"])] for level in (side or [])[:depth]]


def subscription_frame(method, wanted):
    """
    The frame that subscribes to, or unsubscribes from, a coin: an outcome side's book for '#' coins, a perp's prices,
    its mark among them, for any other. method is 'subscribe' or 'unsubscribe'.
    """
    kind = "l2Book" if wanted.startswith("#") else "activeAssetCtx"
    return {"method": method, "subscription": {"type": kind, "coin": wanted}}


class HyperliquidBookStream(BookStream):
    """
    Hyperliquid's outcome books and its perps' marks, on one connection. A wanted '#' coin's book goes to on_book with
    Hyperliquid's time, every time it changes, each message the whole book. Any other wanted coin is a perp whose mark
    and oracle prices go to on_mark(coin, mark, oracle) as they come, about once a second, which keeps the connection
    from going quiet: Hyperliquid closes one it has sent nothing for a minute.
    """

    name = "hyperliquid"

    def __init__(self, contract_ids, on_book, on_gap=None, log=print):
        super().__init__(contract_ids, on_book, on_gap, log)
        self.on_mark = lambda coin, mark, oracle: None

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        for wanted in sorted(self.wanted):
            await ws.send(json.dumps(subscription_frame("subscribe", wanted)))

    async def send_command(self, ws, action, coins):
        for wanted in coins:
            await ws.send(json.dumps(subscription_frame("subscribe" if action == "add" else "unsubscribe", wanted)))

    def handle(self, raw):
        m = json.loads(raw)
        channel, data = m.get("channel"), m.get("data") or {}
        if channel == "error":
            self.log(f"hyperliquid stream error: {str(data)[:200]}")
        if not isinstance(data, dict) or data.get("coin") not in self.wanted:
            return False                                        # A subscription's acknowledgement, or a coin let go.
        if channel == "l2Book":
            bids, asks = data.get("levels") or ([], [])
            self.on_book(data["coin"], levels(bids, self.depth), levels(asks, self.depth), data["time"] / 1000 if data.get("time") else None)
            return True
        if channel == "activeAssetCtx":
            ctx = data.get("ctx") or {}
            self.on_mark(data["coin"], float(ctx["markPx"]), float(ctx["oraclePx"]) if ctx.get("oraclePx") else None)
            return True
        return False


class HyperliquidQuoteStream(BookStream):
    """
    The best bid and ask of Hyperliquid's perps, such as 'BTC', passed on as one level a side, as the book streams pass
    their books, with Hyperliquid's time. Not one of the exchanges BRTI is built from, but among the busiest markets in
    bitcoin, and a perp, which often moves before spot.
    """

    name = "hyperliquid perp"

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        for perp in sorted(self.wanted):
            await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "bbo", "coin": perp}}))

    async def send_command(self, ws, action, perps):
        for perp in perps:
            await ws.send(json.dumps({"method": "subscribe" if action == "add" else "unsubscribe", "subscription": {"type": "bbo", "coin": perp}}))

    def handle(self, raw):
        m = json.loads(raw)
        data = m.get("data") or {}
        if m.get("channel") != "bbo" or data.get("coin") not in self.wanted:
            return False                                        # A subscription's acknowledgement.
        bid, ask = (data.get("bbo") or [None, None])[:2]
        sent = data["time"] / 1000 if data.get("time") else None
        self.on_book(data["coin"], levels([bid] if bid else [], 1), levels([ask] if ask else [], 1), sent)
        return True
