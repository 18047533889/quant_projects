ema(divide(close - low, high - low) * volume / ts_mean(volume, 20), 5)
