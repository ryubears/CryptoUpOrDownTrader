"""
The one line logger the command line tools use.
"""

from datetime import UTC, datetime


def log(message):
    """
    Print a message with the current UTC time in front, at once rather than buffered, so a log file shows it as it happens.
    """
    print(f"{datetime.now(UTC):%H:%M:%S} {message}", flush=True)
