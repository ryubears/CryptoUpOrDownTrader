"""
The sources bitcoin's price history comes from, and the series each one stores.
"""

# Each source's symbol, in its own naming, and bar length in seconds. Binance's archive has 1 second klines, Coinbase's
# shortest candle is 1 minute.
SOURCES = {"binance": ("BTCUSDT", 1), "coinbase": ("BTC-USD", 60)}
