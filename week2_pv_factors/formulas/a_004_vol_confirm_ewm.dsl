ema((close / delay(close, 5) - 1) * volume / ts_mean(volume, 20), 10)
