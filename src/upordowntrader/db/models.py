"""
Data classes shared across the project.
"""

from dataclasses import dataclass


@dataclass
class Market:
    """
    One window's market on a venue: when it trades, the strike it settles against, and how it settled.
    """
    venue: str
    ticker: str                 # 'KXBTC15M-26OCT101515-15'.
    event: str                  # Its event, 'KXBTC15M-26OCT101515', named for when it closes in US Eastern time.
    series: str                 # 'KXBTC15M'.
    open_time: int              # When it opens and closes, in seconds since 1970.
    close_time: int
    strike: float | None        # The BRTI average of the 60 seconds before it opened. None until it opens.
    settlement: float | None    # The BRTI average of the 60 seconds before it closed. None until it settles.
    result: str | None          # 'yes' when the settlement was at least the strike, 'no' otherwise. None until it settles.
    settled_at: float | None    # When the venue settled it, in seconds since 1970.


@dataclass
class Gap:
    """
    A stretch when a venue's feed was down, so its books could not be trusted.
    """
    venue: str
    start_time: float           # When the connection was lost, in seconds since 1970.
    end_time: float | None      # When a new connection was subscribed. None while still down.
