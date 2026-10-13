"""
Polymarket US API client, for its Bitcoin up or down windows, after PredictionArbitrageTrader's client, less the trading.

Polymarket US lists a 15 minute window on Bitcoin each quarter hour, as Kalshi does, settled the same way: on CF
Benchmarks' BRTI averaged over the 60 seconds before the window opens, its price to beat, and before it closes, Up winning
a tie. On 2026-10-13 its strikes and settlements were Kalshi's to the cent, so the two venues trade the same bet. A
window's market is named for its open, as cpc-btc-updown-15m-2026-10-13-1300z, listed about 12 hours ahead, and trades
from its open, its book empty before.

Two jobs. The query half reads a window's market by its name from the public gateway: when it opens and closes, its
strike, set a few seconds after it opens, and once it resolves, its settlement and how it went. The gateway's books are
cached for 30 seconds, too stale to record. The streaming half opens the signed markets websocket, which sends a
market's whole book, best first from the Yes side, each time it changes, about ten times a second on 2026-10-13, with
the matching engine's time, so the local book is replaced rather than patched. The first book of a market on a
connection is a snapshot, whose time is its last change, maybe long ago, so it is passed on with no time. A connection
takes at most WS_SUBSCRIPTIONS subscription requests at once, so each window is its own request, ended when the window
is dropped, which gives the request back. This is the only file that knows Polymarket US's field names and message
formats.

The websocket needs an API key, the same files as PredictionArbitrageTrader's: the key id in
data/polymarket_us_key_id.txt and the Ed25519 secret, in base64, in data/polymarket_us_secret_key.txt. The market reads
need none.
"""

import base64
import functools
import json
import time
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from upordowntrader.api.bookstream import BookStream, Reconnect
from upordowntrader.api.http import RateLimiter, RequestFailed, get_json
from upordowntrader.common.paths import DATA_DIR
from upordowntrader.common.timeutil import epoch
from upordowntrader.db.models import Market

GATEWAY = "https://gateway.polymarket.us/v1"    # The public market reads.
WS_URL = "wss://api.polymarket.us/v1/ws/markets"
WS_PATH = "/v1/ws/markets"      # WS_URL's path, which the handshake's signature covers.
KEY_ID_FILE = DATA_DIR / "polymarket_us_key_id.txt"
SECRET_KEY_FILE = DATA_DIR / "polymarket_us_secret_key.txt"
LIMITER = RateLimiter(5)        # Cloudflare answered PredictionArbitrageTrader's reads with HTTP 429 at about 6 a second.
SERIES = "cpc-btc-updown-15m"   # The prefix of the 15 minute windows' names.
WINDOW_SECONDS = 900
WS_SUBSCRIPTIONS = 10           # Subscription requests a connection takes at once. The 11th is refused, see WS_FULL.
WS_FULL = "max subscriptions per connection reached"
OPEN_STATE = "MARKET_STATE_OPEN"    # The state a book gives a market that trades. Others are preopen, suspended, halted,
                                    # expired, terminated, and match_and_close_auction.
RESOLVED = "MARKET_STATUS_RESOLVED"
TAKER_FEE_RATE = 0.0695         # The taker fee is TAKER_FEE_RATE * price * (1 - price) a contract, as of 2026-10-13.


# SIGNING

@functools.cache
def credentials():
    """
    The account's key id and Ed25519 private key, read from the data folder once. The secret is the 32 byte seed, or a
    64 byte key whose first half it is.
    """
    secret = base64.b64decode(SECRET_KEY_FILE.read_text().strip())
    return KEY_ID_FILE.read_text().strip(), Ed25519PrivateKey.from_private_bytes(secret[:32])


def signed_headers(method, path):
    """
    The three headers that authenticate a request: the account's Ed25519 key over the timestamp, the method, and the path.
    The websocket handshake uses the same scheme.
    """
    key_id, key = credentials()
    ts = str(int(time.time() * 1000))
    return {"X-PM-Access-Key": key_id, "X-PM-Timestamp": ts, "X-PM-Signature": base64.b64encode(key.sign(f"{ts}{method}{path}".encode())).decode()}


def has_key():
    """
    Whether the API key's files are in the data folder.
    """
    return KEY_ID_FILE.exists() and SECRET_KEY_FILE.exists()


# QUERY

def amount(value):
    """
    One of the API's amounts, {'value': '83249.07', 'currency': 'USD'}, as a float, or None when it is not set.
    """
    return float(value["value"]) if value and value.get("value") not in (None, "") else None


def slug(open_time):
    """
    The name of the 15 minute window opening at open_time, in seconds since 1970.
    """
    return f"{SERIES}-{time.strftime('%Y-%m-%d-%H%M', time.gmtime(open_time))}z"


def market(m):
    """
    A Market from one market of the gateway. A resolved one's long side, Up, is priced 1 when it won and 0 when it lost,
    and its end moves to when it resolved.
    """
    terms = m.get("assetPriceTerms") or {}
    long_side = next((s for s in m.get("marketSides") or [] if s.get("long")), {})
    resolved = m.get("status") == RESOLVED and long_side.get("price") in ("0", "1")
    return Market(
        venue="polymarket_us",
        ticker=m["slug"],
        event=m["slug"],
        series=SERIES,
        open_time=int(epoch(terms["windowStart"])),
        close_time=int(epoch(terms["windowEnd"])),
        strike=amount(terms.get("priceToBeat")),
        settlement=amount(terms.get("settlementPrice")),
        result=("yes" if long_side["price"] == "1" else "no") if resolved else None,
        settled_at=epoch(m.get("endDate")) if resolved else None,
    )


def window(name):
    """
    The window's Market by its name, see slug(), or None when it is not listed.
    """
    try:
        return market(get_json(f"{GATEWAY}/market/slug/{name}", limiter=LIMITER)["market"])
    except RequestFailed as e:
        if e.status == 404:
            return None
        raise


def windows(start, end):
    """
    The 15 minute windows closing from start to end, in seconds since 1970, as Markets, a read a window. Those not listed
    yet are left out.
    """
    first = -(-int(start) // WINDOW_SECONDS) * WINDOW_SECONDS - WINDOW_SECONDS     # The open of the first closing from start.
    found = (window(slug(t)) for t in range(first, int(end) - WINDOW_SECONDS + 1, WINDOW_SECONDS))
    return [m for m in found if m is not None]


# STREAMING

def levels(entries, reverse):
    """
    The feed's price levels as [price, contracts] lists, best first, those empty dropped.
    """
    parsed = [[float(e["px"]["value"]), float(e["qty"])] for e in entries or []]
    return sorted((level for level in parsed if level[1] > 0), key=lambda level: level[0], reverse=reverse)


def not_trading(state):
    """
    Why a market whose book gives this state is not trading, in a word, 'suspended' for MARKET_STATE_SUSPENDED, or None
    when it trades. A book without a state counts as trading.
    """
    return None if not state or state == OPEN_STATE else state.removeprefix("MARKET_STATE_").lower()


def subscribe_frame(name):
    """
    The request that subscribes to one market's whole book, every change sent as it comes, with the market's name as its id.
    """
    return {"subscribe": {"requestId": name, "subscriptionType": "SUBSCRIPTION_TYPE_MARKET_DATA", "marketSlugs": [name],
                          "responsesDebounced": False}}


class PolymarketUSBookStream(BookStream):
    """
    The signed markets websocket, a subscription request per market, see the module's docstring. Each book replaces the
    last, and its state says when the market is not trading, see not_trading(). Heartbeats and acknowledgements do not
    count as data. A request refused as one too many, which the requests given back should prevent, rebuilds the
    connection, whose fresh subscriptions start the count again.
    """

    name = "polymarket_us"

    def reset(self):
        self.seen = set()       # Markets whose first book, a snapshot, has come on this connection.

    def connect(self):
        return self.open_connection(WS_URL, signed_headers("GET", WS_PATH))

    async def subscribe(self, ws):
        for name in sorted(self.wanted):
            await ws.send(json.dumps(subscribe_frame(name)))

    async def send_command(self, ws, action, names):
        for name in names:
            await ws.send(json.dumps(subscribe_frame(name) if action == "add" else {"unsubscribe": {"requestId": name}}))

    def handle(self, raw):
        m = json.loads(raw)
        data = m.get("marketData")
        if not data:
            if m.get("error") and WS_FULL in str(m["error"]):
                raise Reconnect(f"refused {m.get('requestId')} as one subscription too many")
            if m.get("error"):
                self.log(f"polymarket_us stream error {m['error']} on {m.get('requestId')}")
            return False
        name = data.get("marketSlug")
        if name not in self.wanted:
            return True
        bids, asks = levels(data.get("bids"), reverse=True), levels(data.get("offers"), reverse=False)
        self.books[name] = (bids, asks)
        self.set_state(name, not_trading(data.get("state")))
        self.on_book(name, bids[:self.depth], asks[:self.depth], epoch(data.get("transactTime")) if name in self.seen else None)
        self.seen.add(name)
        return True
