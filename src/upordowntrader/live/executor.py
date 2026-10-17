"""
Trade on Hyperliquid with real money, and keep the record PredictionArbitrageTrader keeps of its own: each decision a row
of trades, each order sent a row of orders, stored before it goes out and updated with Hyperliquid's answer, and each
trade's payout a row of settlements once its window settled, see schema.sql.

Each decision buys one side of a window's market with one immediate or cancel order, as the paper trader's orders do: Yes
by its coin, No by the No side's, see hyperliquid.side_coin(). Opening is free on Hyperliquid, and at settlement it may
take SETTLE_FEE_RATE of a winning contract's dollar, see api/hyperliquid.py.

An order whose answer was lost, or could not be read, may have filled or not. Hyperliquid is asked again RECOVER_SECONDS
later, by the order's client id, and what it filled is read from the account's fills, see recover(). One it cannot say
of keeps the status 'error', and halts live trading, see live/brakes.py, so a human looks before more go out.

live/run.py sends the orders, from a thread of its own, see Placed for what each one comes back as. Before one can go out
the account needs USDC on Hyperliquid's spot side and an API wallet whose key is in data/, see api/hyperliquid.py, and
Hyperliquid's terms bar US persons and use from the US, where the AWS instance in us-east-1 is.

Run from src/ with:
    python3 -m upordowntrader.live.executor --check             the account's balances, fee rates, API wallets, and open orders
    python3 -m upordowntrader.live.executor --check --email     and a test email through data/email.json, see live/notify.py
"""

import argparse
import json
import time
from dataclasses import dataclass
from upordowntrader.api import hyperliquid
from upordowntrader.db import database

PAYOUT_SECONDS = 10         # Seconds after its close Hyperliquid pays a window, about 8.
TRADE_STATUS = {"filled": "filled", "partial": "partial", "unfilled": "unfilled"}     # Else 'failed'.
RECOVER_SECONDS = 3         # Seconds after a lost answer that Hyperliquid is asked what became of the order.


@dataclass
class Placed:
    """
    What became of a decision's order: its trade's id, the order's status, as hyperliquid.answer() gives it, the
    contracts it filled, their average price, and why not, or the error.
    """
    trade_id: int | None
    status: str
    filled: float
    price: float | None
    note: str | None


def recover(user, cloid, sent, quantity):
    """
    What became of an order whose answer was lost, found by its client id, as hyperliquid.answer() says it: 'unfilled' for
    one Hyperliquid does not know, which never reached it, else what the account's fills of it since sent show. None when
    Hyperliquid could not say.
    """
    status = hyperliquid.order_status(user, cloid)
    if status.get("status") == "unknownOid":
        return "unfilled", 0.0, None, None, "not on Hyperliquid by its client id"
    order = ((status.get("order") or {}).get("order")) if status.get("status") == "order" else None
    if not order:
        return None
    fills = [f for f in hyperliquid.user_fills(user, sent - 60) if f.get("oid") == order.get("oid")]
    filled = sum(float(f["sz"]) for f in fills)
    if filled <= 0:
        return "unfilled", 0.0, None, order.get("oid"), f"found {status['order'].get('status')}, nothing filled"
    price = sum(float(f["px"]) * float(f["sz"]) for f in fills) / filled
    return ("filled" if filled >= quantity - 1e-9 else "partial"), filled, price, order.get("oid"), "found by its client id"


class Executor:
    """
    Sends the orders of decisions and settles their trades. place is the venue's call, hyperliquid.place, and find what
    recovers an order whose answer was lost, recover() for the account, stand-ins in tests.
    """

    def __init__(self, conn, place=None, log=print, clock=time.time, find=None, wait=time.sleep):
        self.conn = conn
        self.place = place or hyperliquid.place
        self.find = find or (lambda cloid, sent, quantity: recover(hyperliquid.credentials()[0], cloid, sent, quantity))
        self.log = log
        self.clock = clock
        self.wait = wait

    def trade(self, decision):
        """
        Buy a side of a window's market on Hyperliquid for a decision, a dict: ticker, its Yes side's coin, close_time,
        side, signal_ts, fair, the chance the side pays, edge, quantity, limit_price, cause, what brought it, and the book
        decided on, book_at, by Hyperliquid's clock, and book_ts, when it reached us. The trade and its order are stored
        before the order goes out, then updated with the answer. Returns what became of it, a Placed.
        """
        d = decision
        trade_id = database.store_trade(self.conn, {"mode": "live", "venue": "hyperliquid", "ticker": d["ticker"], "close_time": d["close_time"],
                                                    "side": d["side"], "signal_ts": d["signal_ts"], "fair": d["fair"], "edge": d["edge"],
                                                    "quantity": d["quantity"], "limit_price": d["limit_price"], "filled": 0.0, "cost": 0.0,
                                                    "fees": 0.0, "status": "sent", "pays_at": d["close_time"] + PAYOUT_SECONDS,
                                                    "cause": d.get("cause")})
        coin, cloid, sent = hyperliquid.side_coin(d["ticker"], d["side"]), hyperliquid.client_id(), self.clock()
        order_id = database.store_order(self.conn, {"trade_id": trade_id, "venue": "hyperliquid", "ticker": d["ticker"], "coin": coin,
                                                    "action": "buy", "outcome": d["side"], "quantity": d["quantity"],
                                                    "limit_price": d["limit_price"], "tif": "Ioc", "client_id": cloid, "sent_at": sent,
                                                    "status": "sent", "venue_order_id": None, "answered_at": None, "latency_ms": None,
                                                    "filled": 0.0, "dollars": 0.0, "fees": 0.0, "note": None, "response": None,
                                                    "book_at": d.get("book_at"), "book_ts": d.get("book_ts")})
        response = None
        try:
            response = self.place(coin, True, d["limit_price"], d["quantity"], "Ioc", cloid)
            answered = self.clock()
            status, filled, price, oid, why = hyperliquid.answer(response, d["quantity"])
        except Exception as e:      # The answer was lost, so the order may be in or not.
            answered = self.clock()
            status, filled, price, oid, why = "error", 0.0, None, None, f"{type(e).__name__}: {e}"[:500]
        if status == "error":
            self.wait(RECOVER_SECONDS)
            try:
                found = self.find(cloid, sent, d["quantity"])
            except Exception as e:
                found, why = None, f"{why}; not found by its client id, {type(e).__name__}: {e}"[:500]
            if found is not None:
                status, filled, price, oid, how = found
                why = f"{why}; {how}"[:500]
        dollars = filled * price if filled else 0.0
        database.update(self.conn, "orders", order_id, {"status": status, "venue_order_id": None if oid is None else str(oid),
                                                        "answered_at": answered, "latency_ms": round(1000 * (answered - sent)), "filled": filled,
                                                        "dollars": dollars, "note": why,
                                                        "response": None if response is None else json.dumps(response)[:4000]})
        database.update(self.conn, "trades", trade_id, {"filled": filled, "cost": dollars, "status": TRADE_STATUS.get(status, "failed")})
        took = f"filled {filled:g} at {price:.4f}" if filled else f"filled none, {why or status}"
        self.log(f"live hyperliquid {d['side']} {d['ticker']}: {d['quantity']:g} to {d['limit_price']:.4f}, {took}, "
                 f"{round(1000 * (answered - sent))} ms")
        return Placed(trade_id, status, filled, price, why)

    def settle(self, markets):
        """
        Settle the trades not settled yet of the markets settled, Markets with their result: each contract held pays a
        dollar when its side won, less Hyperliquid's fee at settlement, and one that filled nothing pays nothing. Returns
        how many trades that held contracts were settled.
        """
        done = 0
        for m in markets:
            if m.result not in ("yes", "no"):
                continue
            for t in database.unsettled_trades(self.conn, m.venue, m.ticker):
                payout = t["filled"] * (t["side"] == m.result)
                fees = hyperliquid.SETTLE_FEE_RATE * payout
                database.store_settlement(self.conn, {"trade_id": t["id"], "mode": t["mode"], "settled_at": m.settled_at or self.clock(),
                                                      "result": m.result, "payout": payout, "fees": fees, "profit": payout - fees - t["cost"]})
                if t["filled"]:
                    self.log(f"live hyperliquid {t['side']} {t['ticker']} settled {m.result}: {t['filled']:g} contracts, "
                             f"{payout - fees - t['cost']:+.2f}$")
                    done += 1
        return done


def check(log=print, email=False):
    """
    Print what Hyperliquid says of the account: its USDC on the spot side, which outcome orders pay from, its perp side,
    its fee rates, its API wallets, whether the one whose key is in data/ is among them, and its open orders. With email,
    send a test email through data/email.json, see live/notify.py.
    """
    user = hyperliquid.ACCOUNT_FILE.read_text().strip()
    spot = hyperliquid.balances(user)
    log(f"spot: {hyperliquid.usdc(spot):,.2f} USDC, {json.dumps(spot.get('balances'))}")
    perp = hyperliquid.perp_state(user)
    log(f"perp: account value {perp.get('marginSummary', {}).get('accountValue')}, withdrawable {perp.get('withdrawable')}")
    fees = hyperliquid.user_fees(user)
    log(f"fees: {json.dumps({k: v for k, v in fees.items() if k != 'dailyUserVlm'})}")
    approved = hyperliquid.agents(user)
    log(f"API wallets: {json.dumps(approved)}")
    problem = hyperliquid.key_problem()
    if problem is None:
        mine = hyperliquid.credentials()[1].address.lower()
        listed = any(isinstance(a, dict) and str(a.get("address", "")).lower() == mine for a in approved or [])
        log(f"the API wallet in {hyperliquid.AGENT_KEY_FILE.name}, {mine}, {'is' if listed else 'is NOT'} among them")
    else:
        log(f"API wallet key: {problem}")
    log(f"open orders: {json.dumps(hyperliquid.open_orders(user))}")
    if email:
        from upordowntrader.live import notify
        error = notify.deliver("CryptoUpOrDownTrader test email", "Live trading alerts will reach you here.")
        log(f"email: {'sent' if error is None else f'not sent, {error}'}")


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read the Hyperliquid account live trading uses.")
    ap.add_argument("--check", action="store_true", help="print the account's balances, fee rates, API wallets, and open orders")
    ap.add_argument("--email", action="store_true", help="with --check, also send a test email through data/email.json")
    args = ap.parse_args()
    if not hyperliquid.ACCOUNT_FILE.exists():
        raise SystemExit(f"No Hyperliquid account address in {hyperliquid.ACCOUNT_FILE}.")
    if args.check:
        check(email=args.email)
