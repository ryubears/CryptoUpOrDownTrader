"""
Shared logic for the venue book streams, as in SportsArb.

A book stream keeps one websocket connection carrying every wanted
contract, keeps a live order book per contract, and hands each change to
a callback. This base class holds what the venues have in common: the
wanted set, the queue of changes made while the connection runs, and the
connect, subscribe, read, and reconnect loop. A venue subclass supplies
how to connect, what to send to subscribe, how to turn a queued change
into a frame, and how to apply one message.

A venue can also stop a market trading while its book stays up: pause it,
close it, or decide it. A stream that hears so from its venue says it
through on_state, see set_state().
"""

import asyncio
import time
import websockets
from upordowntrader.common.log import with_traceback

# Pause before each attempt after a failure. The first retry is immediate, since most drops are
# one off and every second costs data. Repeated failures back off, and the last value repeats.
RECONNECT_SECONDS = (0, 1, 3, 10)
OPEN_TIMEOUT = 20           # Seconds allowed for the websocket handshake.
# Reconnect after this long without a message that counts as data. Generous, since feeds send nothing while books are idle.
STALE_SECONDS = 300


def reconnect_pause(num_failures):
    """
    Seconds to wait before the next attempt after this many failures in a row.
    """
    return RECONNECT_SECONDS[min(num_failures, len(RECONNECT_SECONDS)) - 1]


class Reconnect(Exception):
    """
    Raised by a subclass while handling a message when the connection must be rebuilt,
    for example after a skipped sequence number. The message is logged.
    """


class BookStream:
    """
    One websocket connection carrying every wanted contract. Runs forever
    once started, reconnecting when the connection drops, goes silent, or
    a subclass asks for it. Every failure starts a gap that ends when the
    next connection is subscribed, reported through on_gap with the
    contracts this connection carries, so the recorder can mark the stretch
    and drop just their books. A connection left with nothing to carry
    stays closed, and reports no gap, until contracts are added to it.
    Each change goes to on_book(contract_id, bids, asks, sent), where sent
    is when the venue says it made the change, in seconds since 1970, or
    None when the message does not say, so how far the feed runs behind
    the venue can be measured. Subclasses set name and implement the venue
    hooks below.
    """

    name = "venue"                  # Used in log lines.
    stale_seconds = STALE_SECONDS
    depth = 5                       # Levels a side passed to on_book. The recorder sets it to what it keeps.

    def __init__(self, contract_ids, on_book, on_gap=None, log=print):
        self.wanted = set(contract_ids)
        self.on_book = on_book
        self.log = log
        self.on_gap = on_gap or (lambda start, end, contract_ids: None)    # Called with the gap's times, seconds since 1970, and the contracts.
        # Called with (contract id, why) when the venue says a market stopped trading, why in a word, or with why None when
        # it says the market trades again, see set_state().
        self.on_state = lambda contract_id, why: None
        self.down_since = None      # When the current gap began, in seconds since 1970, or None while connected.
        self.num_failures = 0       # Failures in a row, reset once a connection is subscribed.
        self.books = {}
        self.states = {}            # Contract id maps to why its market is not trading, as its venue last said. Kept across
                                    # reconnects, since a venue may not say it again.
        self.commands = asyncio.Queue()     # Pending ("add" or "remove", [contract ids]) changes.
        self.ws = None              # The live connection, or None between connections.
        self.closing = None         # The close reconnect() started, kept so it is not lost before it runs.

    def add(self, contract_ids):
        """
        Start streaming more contracts. Takes effect on the live connection.
        """
        new = set(contract_ids) - self.wanted
        self.wanted |= new
        if new:
            self.commands.put_nowait(("add", sorted(new)))

    def remove(self, contract_ids):
        """
        Stop streaming contracts and forget their books.
        """
        gone = set(contract_ids) & self.wanted
        self.wanted -= gone
        for contract_id in gone:
            self.books.pop(contract_id, None)
            self.states.pop(contract_id, None)
        if gone:
            self.commands.put_nowait(("remove", sorted(gone)))

    def reconnect(self):
        """
        Close the live connection, so the loop connects again at once, as after this machine slept, when the connection
        may be dead without saying so until the stale clock runs out.
        """
        if self.ws is not None:
            self.closing = asyncio.create_task(self.ws.close())

    def set_state(self, contract_id, why):
        """
        Note what the venue says of a market: why it is not trading, in a
        word, or None when it trades. Only a change goes on to on_state.
        """
        if self.states.get(contract_id) == why:
            return
        if why is None:
            del self.states[contract_id]
        else:
            self.states[contract_id] = why
        self.on_state(contract_id, why)

    # VENUE HOOKS

    def connect(self):
        """
        Return the websocket connection context manager for this venue, normally from open_connection.
        """
        raise NotImplementedError

    def open_connection(self, url, headers=None):
        """
        The connection every venue uses. No receive queue limit, so a busy
        loop delays our timestamps instead of stalling the socket, which
        some feeds answer by closing the connection as a slow consumer.
        """
        return websockets.connect(url, additional_headers=headers, open_timeout=OPEN_TIMEOUT, max_size=None, max_queue=None)

    async def subscribe(self, ws):
        """
        Send whatever subscribes a fresh connection to every wanted contract.
        """
        raise NotImplementedError

    async def send_command(self, ws, action, contract_ids):
        """
        Send the frame that applies one queued add or remove to the live connection.
        """
        raise NotImplementedError

    def handle(self, raw):
        """
        Apply one raw message. Return True when it counted as data, so the stale
        clock resets, and False for keepalive replies. Raise Reconnect to rebuild
        the connection.
        """
        raise NotImplementedError

    def reset(self):
        """
        Clear any per connection state before a new connection. Books are cleared by the loop.
        """

    # LOOP

    async def send_commands(self, ws):
        """
        Forward queued changes to the live connection, forever.
        """
        while True:
            action, contract_ids = await self.commands.get()
            await self.send_command(ws, action, contract_ids)

    async def read(self, ws):
        """
        Read messages until the connection fails or goes silent.
        """
        last_data = time.time()
        while True:
            remaining = self.stale_seconds - (time.time() - last_data)
            if remaining <= 0:
                raise TimeoutError
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            if self.handle(raw):
                last_data = time.time()

    async def run(self):
        """
        Connect, subscribe to every wanted contract, and process messages until
        the connection fails, then reconnect. Changes queued before the
        subscription, during the handshake too, are dropped, because the fresh
        subscription covers the whole wanted set and sending them again would
        subscribe to their contracts twice.
        """
        while True:
            self.books = {}
            self.ws = None
            self.reset()                    # Before waiting, so a connection left with nothing to carry counts as fresh.
            while not self.wanted:
                self.down_since = None      # Nothing is carried, so no stretch without data is missed.
                await asyncio.sleep(1)
            try:
                async with self.connect() as ws:
                    self.ws = ws
                    while not self.commands.empty():
                        self.commands.get_nowait()
                    await self.subscribe(ws)
                    if self.down_since:
                        self.on_gap(self.down_since, time.time(), sorted(self.wanted))
                        self.down_since = None
                    self.num_failures = 0
                    sender = asyncio.create_task(self.send_commands(ws))
                    try:
                        await self.read(ws)
                    finally:
                        sender.cancel()
                        self.ws = None
            except Reconnect as e:
                self.log(f"{self.name} stream {e}, reconnecting")
            except TimeoutError:
                self.log(f"{self.name} stream silent for {self.stale_seconds}s, reconnecting")
            except asyncio.CancelledError:
                raise
            except (websockets.ConnectionClosed, OSError) as e:
                # For a closed connection the message carries the close code and reason from each side.
                self.log(f"{self.name} stream dropped ({type(e).__name__}: {str(e)[:100]}), reconnecting")
            except Exception as e:
                # A rejected handshake or a bad message must never end the stream for good.
                # It may also be a bug in the code the stream feeds, so the traceback goes with it.
                self.log(with_traceback(f"{self.name} stream failed ({type(e).__name__}: {str(e)[:120]}), reconnecting", e))
            if self.down_since is None:
                self.down_since = time.time()
            self.num_failures += 1
            await asyncio.sleep(reconnect_pause(self.num_failures))
