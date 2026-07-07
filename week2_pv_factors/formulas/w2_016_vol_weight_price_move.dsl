ema(ts_delta(close, 1) / delay(close, 1) * volume / ts_mean(volume, 20), 5)
