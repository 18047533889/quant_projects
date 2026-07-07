ema(close / (high - low), 10) * ema(volume / ts_mean(volume, 25), 15) * ema(close / (close - open), 3)
