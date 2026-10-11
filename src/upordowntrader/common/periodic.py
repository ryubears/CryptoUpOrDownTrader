"""
Background work on a timer, started from the recorder's once a second tick.
"""

import asyncio
from upordowntrader.common.log import on_failure


class Periodic:
    """
    Starts work in the background at most once every so many seconds, and
    never while the last one still runs, so a slow venue does not pile up
    calls. The first start is at once. A failure is logged with its
    traceback, and the next start comes on time.
    """

    def __init__(self, seconds, log, what):
        self.seconds = seconds      # The seconds between starts.
        self.log = log
        self.what = what            # The work in words, for the log line when it fails.
        self.last = None            # Wall clock seconds of the last start, None until the first.
        self.running = None         # The work while it runs.

    def again(self):
        """
        Start the work at the next tick, whenever it last started.
        """
        self.last = None

    def tick(self, clock, work):
        """
        Given the wall clock in seconds, start work(), a coroutine, when a start is due and none is running.
        """
        if (self.last is None or clock - self.last >= self.seconds) and (self.running is None or self.running.done()):
            self.last = clock
            self.running = asyncio.create_task(work())
            self.running.add_done_callback(on_failure(self.log, self.what))
