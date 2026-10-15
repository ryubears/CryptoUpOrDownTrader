"""
Trade on Hyperliquid with real money, and keep the record PredictionArbitrageTrader keeps of its own: each decision a row
of trades, each order sent a row of orders, stored before it goes out and updated with Hyperliquid's answer, and each
trade's payout a row of settlements once its window settled, see schema.sql.

Each decision buys one side of a window's market with one immediate or cancel order, as the paper trader's orders do: Yes
by its coin, No by the No side's, see hyperliquid.side_coin(). Opening is free on Hyperliquid, and at settlement it may
take SETTLE_FEE_RATE of a winning contract's dollar, see api/hyperliquid.py.

Nothing calls the executor yet: paper/run.py sends no real order. Before one can go out the account needs USDC on
Hyperliquid and an API wallet whose key is in data/, see api/hyperliquid.py, and Hyperliquid's terms bar US persons and
use from the US, where the AWS instance in us-east-1 is.

Run from src/ with:
    python3 -m upordowntrader.live.executor --check     the account's balances, fee rates, and open orders
"""

import argparse
import json
import time
from upordowntrader.api import hyperliquid
from upordowntrader.db import database

PAYOUT_SECONDS = 10         # Seconds after its close Hyperliquid pays a window, about 8.
TRADE_STATUS = {"filled": "filled", "partial": "partial", "unfilled": "unfilled"}     # Else 'failed'.


class Executor:
    """
    Sends the orders of decisions and settles their trades. place is the venue's call, hyperliquid.place, a stand-in in
    tests.
    """

    def __init__(self, conn, place=None, log=print, clock=time.time):
        self.conn = conn
        self.place = place or hyperliquid.place
        self.log = log
        self.clock = clock

    def trade(self, decision):
        """
        Buy a side of a window's market on Hyperliquid for a decision, a dict: ticker, its Yes side's coin, close_time,
        side, signal_ts, fair, the chance the side pays, edge, quantity, limit_price, and the book decided on, book_at, by
        Hyperliquid's clock, and book_ts, when it reached us. The trade and its order are stored before the order goes
        out, then updated with the answer. Returns the trade's id.
        """
        d = decision
        trade_id = database.store_trade(self.conn, {"mode": "live", "venue": "hyperliquid", "ticker": d["ticker"], "close_time": d["close_time"],
                                                    "side": d["side"], "signal_ts": d["signal_ts"], "fair": d["fair"], "edge": d["edge"],
                                                    "quantity": d["quantity"], "limit_price": d["limit_price"], "filled": 0.0, "cost": 0.0,
                                                    "fees": 0.0, "status": "sent", "pays_at": d["close_time"] + PAYOUT_SECONDS})
        coin, cloid, sent = hyperliquid.side_coin(d["ticker"], d["side"]), hyperliquid.client_id(), self.clock()
        order_id = database.store_order(self.conn, {"trade_id": trade_id, "venue": "hyperliquid", "ticker": d["ticker"], "coin": coin,
                                                    "action": "buy", "outcome": d["side"], "quantity": d["quantity"],
                                                    "limit_price": d["limit_price"], "tif": "Ioc", "client_id": cloid, "sent_at": sent,
                                                    "status": "sent", "venue_order_id": None, "answered_at": None, "latency_ms": None,
                                                    "filled": 0.0, "dollars": 0.0, "fees": 0.0, "note": None, "response": None,
                                                    "book_at": d.get("book_at"), "book_ts": d.get("book_ts")})
        try:
            response = self.place(coin, True, d["limit_price"], d["quantity"], "Ioc", cloid)
        except Exception as e:      # The answer was lost, so the order may be in or not: it is found later by its client id.
            database.update(self.conn, "orders", order_id, {"status": "error", "note": f"{type(e).__name__}: {e}"[:500], "answered_at": self.clock()})
            database.update(self.conn, "trades", trade_id, {"status": "failed"})
            self.log(f"live hyperliquid {d['side']} {d['ticker']}: no answer, {type(e).__name__}: {e}")
            return trade_id
        answered = self.clock()
        status, filled, price, oid, why = hyperliquid.answer(response, d["quantity"])
        dollars = filled * price if filled else 0.0
        database.update(self.conn, "orders", order_id, {"status": status, "venue_order_id": None if oid is None else str(oid),
                                                        "answered_at": answered, "latency_ms": round(1000 * (answered - sent)), "filled": filled,
                                                        "dollars": dollars, "note": why, "response": json.dumps(response)[:4000]})
        database.update(self.conn, "trades", trade_id, {"filled": filled, "cost": dollars, "status": TRADE_STATUS.get(status, "failed")})
        took = f"filled {filled:g} at {price:.4f}" if filled else f"filled none, {why or status}"
        self.log(f"live hyperliquid {d['side']} {d['ticker']}: {d['quantity']:g} to {d['limit_price']:.4f}, {took}, "
                 f"{round(1000 * (answered - sent))} ms")
        return trade_id

    def settle(self, markets):
        """
        Settle the open trades of the markets settled, Markets with their result: each contract held pays a dollar when
        its side won, less Hyperliquid's fee at settlement. Returns how many trades were settled.
        """
        done = 0
        for m in markets:
            if m.result not in ("yes", "no"):
                continue
            for t in database.open_trades(self.conn, m.venue, m.ticker):
                payout = t["filled"] * (t["side"] == m.result)
                fees = hyperliquid.SETTLE_FEE_RATE * payout
                database.store_settlement(self.conn, {"trade_id": t["id"], "mode": t["mode"], "settled_at": m.settled_at or self.clock(),
                                                      "result": m.result, "payout": payout, "fees": fees, "profit": payout - fees - t["cost"]})
                self.log(f"live hyperliquid {t['side']} {t['ticker']} settled {m.result}: {t['filled']:g} contracts, "
                         f"{payout - fees - t['cost']:+.2f}$")
                done += 1
        return done

    def settle_due(self):
        """
        Settle the trades of windows closed by now that Hyperliquid has settled, from its explorer.
        """
        if not database.open_trades(self.conn, now=self.clock()):
            return 0
        return self.settle(hyperliquid.settlements())


def check(log=print):
    """
    Print what Hyperliquid says of the account: its balances, its fee rates, and its open orders.
    """
    user = hyperliquid.ACCOUNT_FILE.read_text().strip()
    log(f"balances: {json.dumps(hyperliquid.balances(user).get('balances'))}")
    fees = hyperliquid.user_fees(user)
    log(f"fees: {json.dumps({k: v for k, v in fees.items() if k != 'dailyUserVlm'})}")
    log(f"open orders: {json.dumps(hyperliquid.open_orders(user))}")


# MAIN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Trade on Hyperliquid with real money: for now, read the account.")
    ap.add_argument("--check", action="store_true", help="print the account's balances, fee rates, and open orders")
    args = ap.parse_args()
    if not hyperliquid.ACCOUNT_FILE.exists():
        raise SystemExit(f"No Hyperliquid account address in {hyperliquid.ACCOUNT_FILE}.")
    if args.check:
        check()
