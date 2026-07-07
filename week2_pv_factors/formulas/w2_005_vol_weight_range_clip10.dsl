ema(ts_delta(close, 1) / delay(close, 1) * volume / ts_mean(volume, 20) * (1 + min(2, max(0, (high - low - ts_mean(high - low, 10)) / ts_std(high - low, 10)))), 10)
