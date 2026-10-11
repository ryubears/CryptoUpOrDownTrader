"""
The sources bitcoin's price history comes from, and the series each one stores.
"""

# Each source's symbol, in its own naming, and bar length in seconds. Coinbase is one of the exchanges BRTI is built from,
# and its shortest candle is 1 minute.
SOURCES = {"coinbase": ("BTC-USD", 60)}
