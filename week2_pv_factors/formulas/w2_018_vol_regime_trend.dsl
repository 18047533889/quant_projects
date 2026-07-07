ema((close / delay(close, 5) - 1) * ts_max(volume, 20) / volume * (divide(median(high - low, 20), high - low) - 1), 5)
