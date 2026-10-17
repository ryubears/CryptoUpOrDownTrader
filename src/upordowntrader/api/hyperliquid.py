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
ask come as they change, with Hyperliquid's time.

Orders are signed, not sent with a key: an API wallet the account approved on Hyperliquid, which can trade but not
withdraw, signs each action, see signature(), and the account, which holds the USDC and the positions, is named by its
address for the reads. Both live in data/, see ACCOUNT_FILE and AGENT_KEY_FILE. An outcome side is ordered by its asset
number, see asset(), and buying No is buying the No side's coin. Opening a position is free; Hyperliquid charges outcome
trades only when a position is closed or settles, see SETTLE_FEE_RATE. Hyperliquid's terms bar US persons and use from
the US, which the AWS instance in us-east-1 is.

This is the only file that knows Hyperliquid's field names and message formats.
"""

import eth_account
import functools
import json
import msgpack
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from eth_account.messages import encode_typed_data
from eth_utils import keccak, to_hex
from upordowntrader.api import http
from upordowntrader.api.bookstream import BookStream
from upordowntrader.api.http import RateLimiter, post_json
from upordowntrader.common.paths import DATA_DIR
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
EXCHANGE_URL = "https://api.hyperliquid.xyz/exchange"
ACCOUNT_FILE = DATA_DIR / "hyperliquid_account.txt"     # The account's address, 0x and 40 hex digits, which holds the money.
AGENT_KEY_FILE = DATA_DIR / "hyperliquid_agent_key.txt"     # The private key of an API wallet the account approved.
OUTCOME_ASSETS = 100_000_000    # An outcome side's asset number is this plus its coin's number, see asset().
SETTLE_FEE_RATE = 0.0007    # What Hyperliquid may take of a winning contract's dollar at settlement: its base spot taker rate,
                            # an upper bound, as its fee page gives no outcome rate. Opening is free. Unconfirmed until an account
                            # reads its own, see user_fees().
ORDER_TIMEOUT = 10          # Seconds an order's answer may take.
SIGNIFICANT = 5             # The most significant figures a price may have.
MIN_ORDER_DOLLARS = 10.0    # The least an order may be worth, contracts times its limit on the side's own coin, as outcome orders
                            # resting on 2026-10-18 showed: 1% of 1,080 were under it either way, remainders of orders partly
                            # filled, against 20% under it by the Yes price alone. Sizes are whole contracts.
DOMAIN = {"chainId": 1337, "name": "Exchange", "verifyingContract": "0x0000000000000000000000000000000000000000", "version": "1"}
AGENT_TYPES = {"Agent": [{"name": "source", "type": "string"}, {"name": "connectionId", "type": "bytes32"}],
               "EIP712Domain": [{"name": "name", "type": "string"}, {"name": "version", "type": "string"},
                                {"name": "chainId", "type": "uint256"}, {"name": "verifyingContract", "type": "address"}]}


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


# ORDERS

@functools.cache
def credentials():
    """
    The account's address and the API wallet's Account, which signs, read from the data folder once.
    """
    return ACCOUNT_FILE.read_text().strip(), eth_account.Account.from_key(AGENT_KEY_FILE.read_text().strip())


def has_key():
    """
    Whether the account's address and the API wallet's key are in the data folder.
    """
    return ACCOUNT_FILE.exists() and AGENT_KEY_FILE.exists()


def key_problem():
    """
    Why the API wallet's key in the data folder cannot sign, in words that never show what the file holds, or None when
    it can: a private key is 32 bytes, 64 hex digits, and an address in its place is 20.
    """
    if not AGENT_KEY_FILE.exists():
        return f"no {AGENT_KEY_FILE}"
    try:
        eth_account.Account.from_key(AGENT_KEY_FILE.read_text().strip())
    except Exception:
        size = len(AGENT_KEY_FILE.read_text().strip().removeprefix("0x"))
        return (f"{AGENT_KEY_FILE.name} holds no private key: {size} characters where a key has 64 hex digits"
                + (", an address's length, so it may be the API wallet's address" if size == 40 else ""))
    return None


def asset(coin):
    """
    The asset number an order names an outcome side by: OUTCOME_ASSETS plus the side's coin's number, 100001230 for '#1230'.
    """
    return OUTCOME_ASSETS + int(coin[1:])


def side_coin(ticker, side):
    """
    The coin to buy for a side of a window, its ticker the Yes side's coin: the ticker for 'yes', its No side for 'no'.
    """
    return ticker if side == "yes" else coin(outcome_of(ticker), 1)


def wire(x):
    """
    A price or size as Hyperliquid hashes it, as its own SDK writes one: up to 8 decimals without trailing zeros, '0.54'
    or '10'. One that would round is refused.
    """
    text = f"{x:.8f}"
    if abs(float(text) - x) >= 1e-12:
        raise ValueError(f"{x!r} has more than 8 decimals")
    return f"{Decimal(text).normalize():f}"


def price_wire(price):
    """
    A price cut to the SIGNIFICANT figures Hyperliquid takes, as wire() writes it.
    """
    return wire(float(f"{price:.{SIGNIFICANT}g}"))


def client_id():
    """
    A new id for an order, sent with it, so an order whose answer was lost can be found: 0x and 32 hex digits.
    """
    return f"0x{uuid.uuid4().hex}"


def order_action(coin, is_buy, price, size, tif="Ioc", cloid=None):
    """
    The action that places one limit order on an outcome side: 'Ioc' fills what it can at once and drops the rest, 'Alo'
    only rests, 'Gtc' does either. The fields' order matters, as the action is hashed as written.
    """
    order = {"a": asset(coin), "b": is_buy, "p": price_wire(price), "s": wire(size), "r": False, "t": {"limit": {"tif": tif}}}
    if cloid:
        order["c"] = cloid
    return {"type": "order", "orders": [order], "grouping": "na"}


def cancel_action(coin, cloid):
    """
    The action that cancels an order by our id for it.
    """
    return {"type": "cancelByCloid", "cancels": [{"asset": asset(coin), "cloid": cloid}]}


def signature(action, nonce, account, expires_after=None):
    """
    An API wallet's signature of an action, as Hyperliquid's own SDK makes it: an EIP-712 'Agent' message whose
    connectionId is the keccak hash of the action packed with msgpack, the nonce, no vault, and the expiry, in
    milliseconds, when there is one.
    """
    data = msgpack.packb(action) + nonce.to_bytes(8, "big") + b"\x00"
    if expires_after is not None:
        data += b"\x00" + expires_after.to_bytes(8, "big")
    message = encode_typed_data(full_message={"domain": DOMAIN, "types": AGENT_TYPES, "primaryType": "Agent",
                                              "message": {"source": "a", "connectionId": keccak(data)}})
    signed = account.sign_message(message)
    return {"r": to_hex(signed.r), "s": to_hex(signed.s), "v": signed.v}


_last_nonce = 0


def nonce():
    """
    The time in milliseconds, made larger than the last one handed out, as Hyperliquid takes each nonce once.
    """
    global _last_nonce
    _last_nonce = max(int(time.time() * 1000), _last_nonce + 1)
    return _last_nonce


def send(action, account=None, expires_after=None):
    """
    Sign an action with the API wallet and send it once, without retrying, as a retry could place an order twice: the
    answer as JSON, its 'status' 'ok' or 'err'. A lost answer is found later by the order's client_id.
    """
    account = account or credentials()[1]
    n = nonce()
    body = {"action": action, "nonce": n, "signature": signature(action, n, account, expires_after), "vaultAddress": None,
            "expiresAfter": expires_after}
    return http.CLIENT.post(EXCHANGE_URL, json=body, timeout=ORDER_TIMEOUT).json()


def place(coin, is_buy, price, size, tif="Ioc", cloid=None, account=None):
    """
    Place one limit order on an outcome side, see order_action(), and return Hyperliquid's answer, see answer().
    """
    return send(order_action(coin, is_buy, price, size, tif, cloid), account)


def cancel(coin, cloid, account=None):
    """
    Cancel an order by our id for it.
    """
    return send(cancel_action(coin, cloid), account)


def answer(response, size):
    """
    What an order's answer says, as (status, contracts filled, their average price, Hyperliquid's id for the order, why
    not): status 'filled' or 'partial' for what an order took at once, 'resting' for one left on the book, 'unfilled'
    for one that met nothing, 'rejected' for one refused, the order or the whole action, as from an API wallet not
    approved, and 'error' for an answer that cannot be read, which leaves the order's outcome unknown.
    """
    if isinstance(response, dict) and response.get("status") == "err":
        return "rejected", 0.0, None, None, str(response.get("response"))[:500]
    if not isinstance(response, dict) or response.get("status") != "ok":
        return "error", 0.0, None, None, str(response)[:500]
    statuses = ((response.get("response") or {}).get("data") or {}).get("statuses") or [{}]
    s = statuses[0]
    if "filled" in s:
        filled = float(s["filled"]["totalSz"])
        return ("filled" if filled >= size - 1e-9 else "partial"), filled, float(s["filled"]["avgPx"]), s["filled"].get("oid"), None
    if "resting" in s:
        return "resting", 0.0, None, s["resting"].get("oid"), None
    why = str(s.get("error", s))
    return ("unfilled" if "could not immediately match" in why else "rejected"), 0.0, None, None, why


def info(body):
    """
    One read of the info endpoint.
    """
    return post_json(INFO_URL, body, LIMITER)


def order_status(user, cloid):
    """
    An order of the account's, found by our id for it: its status and what it filled.
    """
    return info({"type": "orderStatus", "user": user, "oid": cloid})


def user_fills(user, start):
    """
    The account's fills from start, in seconds since 1970, each with its price, size, fee, and order.
    """
    return info({"type": "userFillsByTime", "user": user, "startTime": int(start * 1000)})


def user_fees(user):
    """
    The account's fee rates and its volume, as Hyperliquid charges them.
    """
    return info({"type": "userFees", "user": user})


def balances(user):
    """
    The account's spot balances, USDC among them, from which outcome orders are paid.
    """
    return info({"type": "spotClearinghouseState", "user": user})


def usdc(state):
    """
    The USDC of an account's spot balances, see balances(), in dollars: 0 without any. The 15 minute markets are quoted in
    USDC, outcomeMeta's quoteToken.
    """
    return next((float(b["total"]) for b in state.get("balances") or [] if b.get("coin") == "USDC"), 0.0)


def perp_state(user):
    """
    The account's perp side: its value and what it could withdraw. A deposit lands there, and is moved to the spot side
    on Hyperliquid before outcome orders can use it, unless the account's spot and perp sides are one.
    """
    return info({"type": "clearinghouseState", "user": user})


def agents(user):
    """
    The API wallets the account approved, each with its address, name, and when it lapses, in milliseconds.
    """
    return info({"type": "extraAgents", "user": user})


def open_orders(user):
    """
    The account's orders resting on a book.
    """
    return info({"type": "openOrders", "user": user})


# STREAMING

def levels(side, depth):
    """
    A side of a book message, [{"px": "0.13", "sz": "20.0", "n": 1}, ...] best first, as [[price, size], ...].
    """
    return [[float(level["px"]), float(level["sz"])] for level in (side or [])[:depth]]


def subscription_frames(method, wanted):
    """
    The frames that subscribe to, or unsubscribe from, a coin: an outcome side's book and its best bid and ask for '#'
    coins, a perp's prices, its mark among them, for any other. method is 'subscribe' or 'unsubscribe'.
    """
    kinds = ("l2Book", "bbo") if wanted.startswith("#") else ("activeAssetCtx",)
    return [{"method": method, "subscription": {"type": kind, "coin": wanted}} for kind in kinds]


def merged(levels, best, bid):
    """
    A side of a book with its best level from the bbo feed laid over the levels of the last snapshot: the best, then the
    snapshot's levels behind it, those it has passed dropped. An empty side when best is None.
    """
    if best is None:
        return []
    return [best, *(level for level in levels if (level[0] < best[0] if bid else level[0] > best[0]))]


class HyperliquidBookStream(BookStream):
    """
    Hyperliquid's outcome books and its perps' marks, on one connection. A wanted '#' coin's book goes to on_book with
    Hyperliquid's time, every time it changes. Hyperliquid sends the whole book, l2Book, only every 5.4 seconds or so, for
    a perp as for an outcome, on 2026-10-15, while the best bid and ask, bbo, come as they move, every 0.4 seconds or so
    on a 15 minute window: so the book passed on is the newest snapshot with the newest best levels laid over it, see
    merged(), each time either comes. Any other wanted coin is a perp whose mark and oracle prices go to on_mark(coin,
    mark, oracle) as they come, about once a second, which keeps the connection from going quiet: Hyperliquid closes one
    it has sent nothing for a minute.
    """

    name = "hyperliquid"

    def __init__(self, contract_ids, on_book, on_gap=None, log=print):
        super().__init__(contract_ids, on_book, on_gap, log)
        self.on_mark = lambda coin, mark, oracle: None
        self.snapshots = {}         # Coin maps to its newest whole book, (Hyperliquid's time, bids, asks).
        self.tops = {}              # Coin maps to its newest best levels, (Hyperliquid's time, bid, ask), each None when empty.

    def reset(self):
        self.snapshots, self.tops = {}, {}

    def connect(self):
        return self.open_connection(WS_URL)

    async def subscribe(self, ws):
        for wanted in sorted(self.wanted):
            for frame in subscription_frames("subscribe", wanted):
                await ws.send(json.dumps(frame))

    async def send_command(self, ws, action, coins):
        for wanted in coins:
            for frame in subscription_frames("subscribe" if action == "add" else "unsubscribe", wanted):
                await ws.send(json.dumps(frame))
            if action != "add":
                self.snapshots.pop(wanted, None)
                self.tops.pop(wanted, None)

    def pass_on(self, coin):
        """
        Pass on a coin's book: its snapshot, with the best levels laid over it when they are newer, timed by the newer.
        """
        sent, bids, asks = self.snapshots.get(coin, (None, [], []))
        top = self.tops.get(coin)
        if top is not None and (sent is None or top[0] >= sent):
            sent, bids, asks = top[0], merged(bids, top[1], True)[:self.depth], merged(asks, top[2], False)[:self.depth]
        self.on_book(coin, bids, asks, sent)

    def handle(self, raw):
        m = json.loads(raw)
        channel, data = m.get("channel"), m.get("data") or {}
        if channel == "error":
            self.log(f"hyperliquid stream error: {str(data)[:200]}")
        if not isinstance(data, dict) or data.get("coin") not in self.wanted:
            return False                                        # A subscription's acknowledgement, or a coin let go.
        coin, sent = data["coin"], data["time"] / 1000 if data.get("time") else None
        if channel == "l2Book":
            bids, asks = data.get("levels") or ([], [])
            self.snapshots[coin] = (sent, levels(bids, self.depth), levels(asks, self.depth))
            self.pass_on(coin)
            return True
        if channel == "bbo":
            bid, ask = (data.get("bbo") or [None, None])[:2]
            self.tops[coin] = (sent, *(levels([level], 1)[0] if level else None for level in (bid, ask)))
            self.pass_on(coin)
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
