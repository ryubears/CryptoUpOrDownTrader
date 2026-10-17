"""
When live trading halts, as PredictionArbitrageTrader's brakes decide its own. Each rule reads the trades, orders, and
settlements tables, so a restart does not reset what it has seen, and each counts only what came after the last halt,
so what led to one does not halt live trading again once a human has cleared it.

Live trading halts, and no new order goes out, when:

- the live trades settled in the last LOSS_HOURS lost, net, LOSS_SHARE or more of the live money before them: the
  account's USDC, read again as they settle, plus what the live trades still open cost, less what the settled ones made.
- each of the last LOSING_WINDOWS windows live trades filled in lost money, net.
- an order's outcome is unknown: its answer was lost or could not be read, and Hyperliquid could not say what became of
  it by its client id, see executor.recover(), so the records may be wrong until a human has looked.
- Hyperliquid refused REJECT_LIMIT orders in a row, as when the API wallet is not approved or the account lacks the USDC.

A halt adds its reason to HALT_FILE, logs it, stores an alert, and emails a human, see live/notify.py. Live trading stays
halted while HALT_FILE is there, across restarts too, and trades again once a human has removed it, which is looked for
every second, so no restart is needed. What is held still settles. Making the file halts live trading by hand.
"""

import time
from datetime import UTC, datetime
from upordowntrader.common.paths import DATA_DIR
from upordowntrader.db import database

HALT_FILE = DATA_DIR / "live_halt.txt"      # Why live trading halted, kept until a human removes it.
LOSS_HOURS = 12             # The sliding window the loss rule looks at.
LOSS_SHARE = 0.30           # Net loss of the live trades settled in it, as a share of the live money before them, that halts.
LOSING_WINDOWS = 10         # Windows in a row whose live trades lost money, net, that halt.
REJECT_LIMIT = 3            # Orders Hyperliquid refuses in a row that halt.
SUBJECT = "CryptoUpOrDownTrader live trading halted"


def utc(seconds):
    """
    A time in seconds since 1970 as '2026-10-18 15:00:00'.
    """
    return f"{datetime.fromtimestamp(seconds, UTC):%Y-%m-%d %H:%M:%S}"


class Brakes:
    """
    Decides when live trading halts, from the trades, orders, and settlements stored. notifier is the Notifier from
    live/notify.py that tells a human. balance is the account's USDC as last read, which live/run.py keeps.
    """

    def __init__(self, conn, log=print, notifier=None, clock=time.time):
        self.conn = conn
        self.log = log
        self.notifier = notifier
        self.clock = clock
        self.balance = None         # The account's USDC, None until read.
        self.halted = None          # Why live trading halted, while it has. No new orders go out.
        self.since = database.last_alert_time(conn, "halt") or 0.0     # When it last halted. The rules count only what came after.
        self.refresh()

    def refresh(self):
        """
        Once a second: halted while HALT_FILE is there, made by a halt or by hand, and trading again once it has gone.
        """
        if HALT_FILE.exists():
            if self.halted is None:
                lines = HALT_FILE.read_text().strip().splitlines()
                self.halted = f"{HALT_FILE.name} is there, remove it to resume: {'; '.join(lines) or 'made by hand'}"
                self.log(f"live trading halted: {self.halted}")
        elif self.halted is not None:
            self.halted = None
            self.log(f"live trading resumes: {HALT_FILE} was removed")

    def halt(self, reason):
        """
        Stop new orders: add the reason to HALT_FILE, log it, and tell a human. Once halted, a later reason is only logged.
        """
        if self.halted:
            self.log(f"live trading, halted already, would halt again: {reason}")
            return
        now = self.clock()
        self.halted, self.since = reason, now
        HALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HALT_FILE.open("a") as f:
            f.write(f"{utc(now)} UTC halted: {reason}\n")
        self.log(f"live trading halted: {reason}")
        if self.notifier:
            balance = "unknown" if self.balance is None else f"{self.balance:,.2f} USDC"
            self.notifier.send("halt", SUBJECT, f"At {utc(now)} UTC: {reason}\n\nNo new live orders go out on Hyperliquid. What is held "
                                                f"still settles. The account's balance: {balance}.\n\nOnce the account is checked on "
                                                f"Hyperliquid, remove {HALT_FILE} to resume. No restart is needed.")

    # ORDERS

    def watch(self, placed):
        """
        After each order's answer, a live/executor.Placed: halt on an unknown outcome, or on Hyperliquid refusing
        REJECT_LIMIT orders in a row.
        """
        if placed.status == "error":
            self.halt(f"an order's outcome is unknown, trade {placed.trade_id}: {placed.note}")
        elif placed.status == "rejected":
            recent = database.recent_orders(self.conn, REJECT_LIMIT, self.since)
            if len(recent) >= REJECT_LIMIT and all(status == "rejected" for _, status in recent):
                self.halt(f"Hyperliquid refused {len(recent)} orders in a row, the last with: {placed.note}")

    # RESULTS

    def check_results(self):
        """
        After live trades settle, with the balance read since: halt on too big a loss over the last LOSS_HOURS, or on
        LOSING_WINDOWS losing windows in a row.
        """
        if self.halted:
            return
        now = self.clock()
        results = database.live_results(self.conn, self.since)
        net = sum(r["profit"] for r in results if r["settled_at"] >= now - LOSS_HOURS * 3600)
        if net < 0 and self.balance is None:
            self.log("live trading: the loss rule waits for the account's balance")
        elif net < 0:
            held = sum(t["cost"] for t in database.open_trades(self.conn) if t["mode"] == "live")
            before = self.balance + held - net          # The live money before these results.
            if before > 0 and -net / before >= LOSS_SHARE:
                self.halt(f"the live trades settled in the last {LOSS_HOURS} hours lost {-net:,.2f}$, {-net / before:.0%} of the "
                          f"{before:,.2f}$ live money before them, at or over the {LOSS_SHARE:.0%} limit")
                return
        windows = {}
        for r in results:
            windows[r["close_time"]] = windows.get(r["close_time"], 0.0) + r["profit"]
        last = [windows[close] for close in sorted(windows)][-LOSING_WINDOWS:]
        if len(last) == LOSING_WINDOWS and all(profit < 0 for profit in last):
            self.halt(f"the live trades lost money in each of the last {LOSING_WINDOWS} windows they filled in, "
                      f"{-sum(last):,.2f}$ in all")
