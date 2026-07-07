ema(close / (high - low), 8) * ema((close - open) / close, 10) * ema(volume / ts_mean(volume, 25), 15)
