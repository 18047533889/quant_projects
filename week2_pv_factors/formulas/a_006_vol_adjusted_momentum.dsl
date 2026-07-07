ema((close / delay(close, 3) - 1) * volume / ts_mean(volume, 20), 5)
