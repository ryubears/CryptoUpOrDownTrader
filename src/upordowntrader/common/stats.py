"""
Summaries of measurements, for the recorder's status line.
"""


def quantile(values, share):
    """
    The value that a share of the values fall at or below, such as 0.9 for
    the 90th percentile, or None when there are none.
    """
    if not values:
        return None
    values = sorted(values)
    return values[min(int(share * len(values)), len(values) - 1)]
