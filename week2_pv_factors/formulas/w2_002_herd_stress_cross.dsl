clip(ema((close / delay(close, 5) - 1) * ts_rank(volume, 8) * (high - low), 12), -3, 3)
