"""
Tests for the shared stream loop, driven by a scripted fake connection, as in SportsArb.
"""

import asyncio
import contextlib
from upordowntrader.api import bookstream


class FakeConnection:
    """
    A websocket stand in. It hands out scripted messages, records what was
    sent, and raises the scripted exception when the messages run out.
    """

    def __init__(self, messages, then):
        self.messages = list(messages)
        self.then = then
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def send(self, frame):
        self.sent.append(frame)

    async def recv(self):
        if self.messages:
            await asyncio.sleep(0)
            return self.messages.pop(0)
        if self.then == "hang":
            await asyncio.sleep(3600)
        raise self.then


class ParkedConnection:
    """
    Returned once the script runs out. Entering it never completes, which parks the loop.
    """

    async def __aenter__(self):
        await asyncio.sleep(3600)

    async def __aexit__(self, *args):
        return False


class ScriptedStream(bookstream.BookStream):
    """
    A venue with a scripted connection per attempt and trivial frames.
    """

    name = "scripted"
    stale_seconds = 0.05

    def __init__(self, contract_ids, connections):
        self.logs = []
        self.gaps = []
        super().__init__(contract_ids, lambda *a: None, lambda start, end, carried: self.gaps.append((start, end, carried)), self.logs.append)
        self.connections = list(connections)
        self.used = []
        self.resets = 0
        self.handled = []

    def connect(self):
        if not self.connections:
            return ParkedConnection()
        conn = self.connections.pop(0)
        self.used.append(conn)
        return conn

    async def subscribe(self, ws):
        await ws.send(("subscribe", sorted(self.wanted)))

    async def send_command(self, ws, action, contract_ids):
        await ws.send((action, contract_ids))

    def handle(self, raw):
        if raw == "gap":
            raise bookstream.Reconnect("skipped a message")
        self.books[raw] = True
        self.handled.append(raw)
        return True

    def reset(self):
        self.resets += 1


def run_until_connections_used(stream):
    async def scenario():
        task = asyncio.create_task(stream.run())
        while stream.connections:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    asyncio.run(scenario())


def test_the_loop_subscribes_and_reconnects_on_silence_a_skip_and_a_drop(monkeypatch):
    monkeypatch.setattr(bookstream, "RECONNECT_SECONDS", (0,))
    first = FakeConnection(["a"], then="hang")                                 # Goes silent after one message.
    second = FakeConnection(["b", "gap"], then="hang")                         # Asks for a reconnect.
    third = FakeConnection(["c"], then=ConnectionResetError("peer reset"))     # Drops.
    fourth = FakeConnection([], then="hang")
    stream = ScriptedStream(["x", "y"], [first, second, third, fourth])
    run_until_connections_used(stream)
    assert [c.sent[0] for c in stream.used] == [("subscribe", ["x", "y"])] * 4
    assert stream.logs[:3] == ["scripted stream silent for 0.05s, reconnecting",
                               "scripted stream skipped a message, reconnecting",
                               "scripted stream dropped (ConnectionResetError: peer reset), reconnecting"]
    assert stream.handled == ["a", "b", "c"] and stream.books == {}             # Books cleared before the last connection.
    assert len(stream.gaps) == 3 and all(start <= end for start, end, _ in stream.gaps)    # One gap per failure, closed on resubscribe.
    assert all(carried == ["x", "y"] for _, _, carried in stream.gaps)                     # Each names the contracts to drop.


def test_reconnect_pause_is_immediate_first_and_backs_off_after():
    assert [bookstream.reconnect_pause(n) for n in (1, 2, 3, 4, 5, 9)] == [0, 1, 3, 10, 10, 10]


def test_changes_made_while_running_are_sent_and_pending_ones_dropped_on_connect(monkeypatch):
    monkeypatch.setattr(bookstream, "RECONNECT_SECONDS", (0,))
    conn = FakeConnection([], then="hang")
    stream = ScriptedStream(["x"], [conn])
    stream.add(["y"])       # Queued before the first connection, so the subscription carries it, not a command.

    async def scenario():
        task = asyncio.create_task(stream.run())
        await asyncio.sleep(0.02)
        stream.add(["z"])
        stream.remove(["x"])
        await asyncio.sleep(0.02)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert conn.sent == [("subscribe", ["x", "y"]), ("add", ["z"]), ("remove", ["x"])]
    assert stream.wanted == {"y", "z"}


class ClosableConnection(FakeConnection):
    """
    A connection that waits for messages until it is closed, then fails its read, as a websocket does.
    """

    def __init__(self):
        super().__init__([], then="hang")
        self.closed = asyncio.Event()

    async def close(self):
        self.closed.set()

    async def recv(self):
        await self.closed.wait()
        raise ConnectionResetError("closed")


def test_reconnect_closes_the_live_connection_so_the_loop_connects_again(monkeypatch):
    monkeypatch.setattr(bookstream, "RECONNECT_SECONDS", (0,))
    first, second = ClosableConnection(), FakeConnection([], then="hang")
    stream = ScriptedStream(["x"], [first, second])
    stream.stale_seconds = 60                                       # So only the close ends the first connection.

    async def scenario():
        task = asyncio.create_task(stream.run())
        while stream.ws is not first:
            await asyncio.sleep(0.01)
        stream.reconnect()
        while stream.connections:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert stream.logs == ["scripted stream dropped (ConnectionResetError: closed), reconnecting"]
    assert stream.used == [first, second] and len(stream.gaps) == 1

