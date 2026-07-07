clip(ema(ts_delta(close, 1) / delay(close, 1) * ts_rank(volume, 6) * (high - low), 12), -2.5, 2.5)
